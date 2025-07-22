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
from serl_launcher.common.encoding import EncodingWrapper, SmallTransformerTextEncoder
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import ensemblize
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.networks.cross_att import CrossAttentiveCritic
from serl_launcher.vision.resnet_v1 import resnetv1_configs
from pi0.src.openpi.training.rl_cfg import create_pi0_base_aloha_rl_lora_config
from pi0.src.openpi.models import model
from pi0.src.openpi.shared import nnx_utils

def create_policy():
    config = create_pi0_base_aloha_rl_lora_config()
    rng = jax.random.key(config.seed)
    _, model_rng = jax.random.split(rng)
    policy = config.model.create(model_rng)
    return policy


class TestAgent(flax.struct.PyTreeNode):
    """
    Twin Delayed DDPG (TD3) agent with hybrid policy for dual arm setups.
    - Uses deterministic policy network
    - Maintains two Q-networks (Critic ensemble size = 2)
    - Uses DQN for gripper actions
    """
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

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass for deterministic policy network"""

        if train:
            assert rng is not None, "Must specify rng when training"
        rng_noise, rng_rest = jax.random.split(rng)
        batch_size = observations.shape[0]
        noise = jax.random.normal(rng_noise, (batch_size, 50, 32))
        policy = self.state.policy
        params = grad_params or self.state.params
        policy_with_params = policy.load({"params": params})
        inputs = jax.tree.map(lambda x: x, observations)
        actions = nnx_utils.module_jit(policy_with_params.sample_actions)(rng=rng, observation=model.Observation.from_dict(inputs), noise = noise)

        actions = jax.tree.map(lambda x: x[..., :14], actions)
        return actions
    

    def _compute_next_actions(self, batch, rng):
        """Compute target actions with clipped noise"""
        # Get deterministic actions from policy
        next_actions = self.forward_policy(
            batch["next_observations"], rng=rng
        )
        
        # Add clipped noise
        noise_key, sample_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, next_actions.shape) * self.config["target_policy_noise"]
        noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
        next_actions = next_actions + noise
        
        return next_actions

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """TD3 critic loss function with twin Q-networks"""
        batch_size = batch["rewards"].shape[0]
        
        # Split actions into continuous and gripper parts
        actions= batch["actions"]
        
        # Get target actions
        rng, next_action_key = jax.random.split(rng)
        next_actions = self._compute_next_actions(batch, next_action_key)
        
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
        
        # Predicted Q-values for current actions
        predicted_q = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )
        chex.assert_shape(predicted_q, (batch_size,))
        
        # Compute MSE loss
        critic_loss = jnp.mean((predicted_q - target_q) ** 2)
        
        return critic_loss, {
            "critic_loss": critic_loss,
            "predicted_qs": predicted_q,
            "target_qs": target_q,
        }

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """TD3 policy loss using minimum Q-value"""
        batch_size = batch["rewards"].shape[0]
        
        # Get policy actions
        actions = self.forward_policy(
            batch["observations"], rng=rng, grad_params=params
        )
        
        # Get Q-values from first critic
        predicted_q = self.forward_critic(
            batch["observations"],
            actions,
            rng=rng,
        )
        
        # Use minimum Q-value as policy objective
        policy_loss = -jnp.mean(predicted_q)
        
        return policy_loss, {
            "actor_loss": policy_loss,
        }

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic"}),
        **kwargs
    ) -> Tuple["TD3AgentHybridDualArm", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 14))
        
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
            
        # 每隔一定步数更新策略
        if self.state.step % self.config["policy_update_freq"] == 0:
            new_state = new_state.replace(
                params=new_state.params,
                opt_states={
                    "critic": new_state.opt_states["critic"],
                    # "actor": new_state.opt_states["actor"],
                },
                step=new_state.step + 1,
            )
            
        return self.replace(state=new_state), info
    

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: Data,
        action_chunk: int = 50,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax).
        The internal RNG will not be updated.
        """
        observations = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], observations)
        # For deterministic policy, just take mode or sample with provided seed
        if argmax:
            joint_actions = self.forward_policy(observations, rng=seed, train=False)
        else:
            assert seed is not None, "Must provide seed for sampling"
            noise = jax.random.normal(seed, shape=(observations.shape[0], action_chunk, 14)) * 0.1
            noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
            joint_actions = self.forward_policy(observations, rng=seed, train=False) + noise

        return joint_actions
    


    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.1,
        policy_update_freq: int = 2,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        pretrained_actor_params: Optional[Params] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
        }
        model_def = ModuleDict(networks)
        # Define optimizers
        txs = {
            # "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
        }
        
        rng, init_rng = jax.random.split(rng)
        if pretrained_actor_params is not None:
            all_params = model_def.init(
                init_rng,
                actor=[observations],
                critic=[observations, actions],
            )["params"]
            all_params = all_params.unfreeze()
            all_params["actor"] = pretrained_actor_params  # <<<<<<<< 替换
            params = flax.core.freeze(all_params)
        else:
            print("\npretrained_actor_params is None\n")
            params = model_def.init(
                init_rng,
                actor=[observations],
                critic=[observations, actions],
            )["params"]

        
        
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
            policy = actor_def,  # Store the policy model for sampling actions
        )
        
        # Config
        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=2,  # TD3 uses 2 Q-networks
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_policy_noise=target_policy_noise,
                noise_clip=noise_clip,
                policy_update_freq=policy_update_freq,
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
        critic_ensemble_size: int = 2,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        pretrained_policy_path: Optional[str] = None,
        **kwargs,
    ):

        encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="ViT",
                    num_spatial_blocks=8,
                    bottleneck_dim = 128,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        

        critic_def = partial(
            CrossAttentiveCritic,
            encoder=partial(
                EncodingWrapper,
                encoder=encoders,
                use_proprio=use_proprio,
                image_keys=image_keys,
                fuse_proprio_images = True,
            ),
            text_encoder=partial(
                SmallTransformerTextEncoder,
                vocab_size=30522,
                embed_dim=128,
                num_layers=3,
                num_heads=4,
                pooling="cls",
            ),
            cross_attn_num_heads=4,
            cross_attn_dropout_rate=0.1,
            cross_attn_use_layer_norm=True,
            mlp_hidden_dims=(256, 256),
            mlp_activations="swish",
            mlp_dropout_rate=0.1,
            mlp_use_layer_norm=True
        )
        # critic_def = ensemblize(critic_def, ensemble_size=critic_ensemble_size)(name="critic_ensemble")
        policy_def = create_policy()       
        pretrained_actor_params = None

        
        if pretrained_policy_path is not None:
            pretrained_actor_params = model.restore_params(pretrained_policy_path / "params", dtype=jnp.bfloat16)
        else:
            raise ValueError("pretrained_policy_path must be provided for pixel observations")


        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            pretrained_actor_params=pretrained_actor_params,
            **kwargs,
        )
            
        return agent
    



    