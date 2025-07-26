from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet
import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper, SmallTransformerActionEncoder, SmallTransformerTextEncoder
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.networks.cross_att import CrossAttentiveCritic
from serl_launcher.vision.convernext import ConvNeXtEncoder

class TestAgent(flax.struct.PyTreeNode):

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Forward pass for critic network ensemble"""
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """Forward pass for target critic network"""
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )
    

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """TD3 critic loss function with twin Q-networks"""
        batch_size = batch["rewards"].shape[0]
        
        # Split actions into continuous and gripper parts
        actions= batch["actions"]

        # Predicted Q-values for current actions
        predicted_q = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )
        chex.assert_shape(predicted_q, (batch_size,))
        # Get target actions
        rng, next_action_key = jax.random.split(rng)
        next_actions = batch["next_actions"]
        
        # Get Q-values for next actions
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )
        
        chex.assert_shape(target_next_qs, (batch_size,))

        # Compute target Q-value
        target_q = (
            batch["rewards"]
            + self.config["discount"] * (batch["dones"] < 1) * target_next_qs
        )
        chex.assert_shape(target_q, (batch_size,))
        
        # Compute MSE loss
        critic_loss = jnp.mean((predicted_q - target_q) ** 2)
        jax.debug.print("critic_loss: {}",critic_loss)
        return critic_loss, {
            "critic_loss": critic_loss,
            "predicted_qs": predicted_q,
            "target_qs": target_q,
        }


    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"critic"}),
        **kwargs
    ) -> Tuple["TestAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 50, 14))
        
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
            
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)
            
        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )
        
        # 计算损失函数
        loss_fns = self.loss_fns(batch, **kwargs)
        
        # 只更新指定网络
        assert networks_to_update.issubset(loss_fns.keys())
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})
            
        # 执行梯度更新
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        
        # 延迟更新目标网络
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])
        return self.replace(state=new_state), info

    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        critic_def: nn.Module,
        # Optimizer

        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "critic": critic_def,
        }
        model_def = ModuleDict(networks)
        txs = {
            "critic": make_optimizer(**critic_optimizer_kwargs),
        }
        
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            critic=[observations, actions],
            method=None,
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )
        
        # Config
        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=2,  
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        use_proprio: bool = False,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):

        encoders = {
                image_key: ConvNeXtEncoder()
                for image_key in image_keys
            } 

        critic_def = CrossAttentiveCritic(
            obs_encoder=EncodingWrapper(
                encoder=encoders,
                use_proprio=use_proprio,
                image_keys=image_keys,
                fuse_proprio_images=True,
            ),
            action_encoder=SmallTransformerActionEncoder(),
            text_encoder=SmallTransformerTextEncoder(),
            cross_attn_num_heads=4,
            cross_attn_dropout_rate=0.1,
            cross_attn_use_layer_norm=True,
            mlp_hidden_dims=(64, 1),
            mlp_activations="swish",
            mlp_dropout_rate=0.1,
            mlp_use_layer_norm=True
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            critic_def=critic_def,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )
            
        return agent
    


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "ERROR" 
from flax.training import checkpoints
from absl import app, flags
import tqdm
from serl_launcher.utils.timer_utils import Timer

FLAGS = flags.FLAGS
# flags.DEFINE_multi_string("demo_path", "demo_data/15_demos_2025-07-22_11-37-30.pkl", "Path to the demo data.")
flags.DEFINE_multi_string("demo_path", "demo_data/15_demos_2025-07-22_15-50-46.pkl", "Path to the demo data.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", "checkpoints", "Path to save checkpoints.")


def valer(rng, agent, transitions):
    import matplotlib.pyplot as plt
    ground_truth, target_value, value = [], [], []
    for step in tqdm.tqdm(
        range(len(transitions)), dynamic_ncols=True, desc="valer"
    ):  
        transition = transitions[step]
        transition = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transition)
        ground_truth.append(transition["return"])
        target_value.append(agent.forward_target_critic(observations = transition["observations"], actions = transition["actions"], rng = rng))
        value.append(agent.forward_critic(observations = transition["observations"], actions = transition["actions"], rng = rng))
    loss1 = jnp.mean(jnp.square(jnp.array(ground_truth) - jnp.array(value)))
    loss2 = jnp.mean(jnp.square(jnp.array(ground_truth) - jnp.array(target_value)))
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truth, label="Ground Truth", color='red')
    plt.plot(value, label="Value", color='blue')
    plt.plot(target_value, label="Target Value", color='green')
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Value Comparison (Loss1: {loss1:.4f}, Loss2: {loss2:.4f})")
    plt.show()
    plt.savefig("value_comparison_unseen.png")
    plt.close()
    print(f"Loss1: {loss1}, Loss2: {loss2}")

def make_test_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys,
    encoder_type="resnetv1-10-vit",
    reward_bias=0.0,
    discount=0.95,
    pretrained_policy_path = None,
):
    agent = TestAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        discount=discount,
        reward_bias=reward_bias,
        pretrained_policy_path = pretrained_policy_path,
    )
    return agent

import pickle as pkl
from pi0.src.openpi.training.rl_cfg import rl_config

def main(_):
    
    TASK_ENV = rl_config()
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    sample_obs=TASK_ENV.observation_space.sample()
    sample_action=TASK_ENV.action_space.sample()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_obs)
    sample_action = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_action)

    agent: TestAgent = make_test_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        print("Checkpoint path already exists.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print(f"Loaded previous checkpoint at step {ckpt_number}.")
        else:
            print("No checkpoint found. Starting from scratch.")
            ckpt_number = 0


    for path in FLAGS.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            discount = 0.95
            transitions_rev = transitions[::-1]
            ret = 0.0
            for t in transitions_rev:
                if t["dones"]:
                    ret = t["rewards"]
                else:
                    ret = t["rewards"] + discount * ret
                t["return"] = ret
            transitions = transitions_rev[::-1]
            
    import random
    # random.shuffle(transitions)


    # learner loop
    print("starting learner loop")
    valer(
        rng,
        agent,
        transitions
    )


if __name__ == "__main__":
    app.run(main)