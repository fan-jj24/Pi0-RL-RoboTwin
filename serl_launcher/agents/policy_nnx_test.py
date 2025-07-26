import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
from functools import partial
import glob
import pickle as pkl
import tqdm
from typing import Iterable, Optional, Tuple, FrozenSet, Any
import chex
import distrax
import pdb
import flax
import flax.linen as nn
import flax.nnx as nnx
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_multi_string("demo_path", "demo_data/15_demos_2025-07-22_11-37-30.pkl", "Path to the demo data.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", "/home/anker/robotwin/Pi0-RL-RoboTwin/checkpoint/policy", "Path to save checkpoints.")

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.launcher import make_wandb_logger
from serl_launcher.data.data_store import DynamicNextObsReplayBufferDataStore
from pi0.src.openpi.models import model, pi0
from pi0.src.openpi.training.rl_cfg import rl_config


class NNXToFlax(flax.linen.Module):
    graphdef: Any
    initial_state: Any
    param_path: str = "params"

    @nn.compact
    def __call__(self, sample_rng, observations, *args, **kwargs):
        
        params = self.param(self.param_path, self._init_nnx_model, sample_rng, observations)
        model = nnx.merge(self.graphdef, params)
        return model(sample_rng, observations, **kwargs)

    def _init_nnx_model(self, rng, sample_rng, observations):
        # rng 是 Flax 内部传递的 RNG，通常用于初始化参数的随机性
        # sample_rng 是你传递给 _init_nnx_model 的样本 RNG
        # 初始化 NNX 模型（仅在首次调用时执行）
        model = nnx.merge(self.graphdef, self.initial_state)
        model(sample_rng, observations)  # 触发参数初始化
        graphdef, state = nnx.split(model)
        return state

def create_policy_with_lora(pretrained_policy_path = None):
    policy_config=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    rng = jax.random.key(FLAGS.seed)
    _, model_rng = jax.random.split(rng)
    policy_def = policy_config.create(model_rng)
    freeze_filter = policy_config.get_freeze_filter()
    if pretrained_policy_path is not None:
        pretrained_actor_params = model.restore_params(pretrained_policy_path, dtype=jnp.float32)
    else:
        raise ValueError("pretrained_policy_path must be provided for post training")
    return policy_def, pretrained_actor_params, freeze_filter



class TestAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_policy(
        self,
        sample_rng,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass for policy network"""
        if train:
            assert rng is not None, "Must specify rng when training"
        params = grad_params or self.state.params
        actions = self.state.apply_fn(
            {"params": params},
            sample_rng,
            observations,
            rngs={"dropout": rng} if train else {},
            name="actor",
            train=train,
        )
        actions = jax.tree.map(lambda x: x[..., :14], actions)
        return actions
    
    
    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """TD3 policy loss using minimum Q-value"""
        batch_size = batch["rewards"].shape[0]
        # Get policy actions
        rng, sample_rng = jax.random.split(rng)
        actions = self.forward_policy(
            sample_rng, batch["observations"], rng=rng, grad_params=params
        )
        loss = (batch["actions"] - actions) ** 2
        loss = jnp.mean(loss, axis=-1)  # Average over time
        policy_loss = jnp.mean(loss)
        return policy_loss, {
            "actor_loss": policy_loss,
        }

    def loss_fns(self, batch):
        return {
            "actor": partial(self.policy_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor"}),
        **kwargs
    ) -> Tuple["TestAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 14))
        
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
            
        rng, aug_rng = jax.random.split(self.state.rng)
            
        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )
        
        loss_fns = self.loss_fns(batch, **kwargs)
        assert networks_to_update.issubset(loss_fns.keys())
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})
            
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        # 每隔一定步数更新策略
        if self.state.step % self.config["policy_update_freq"] == 0:
            new_state = new_state.replace(
                params=new_state.params,
                opt_states={
                    "actor": new_state.opt_states["actor"],
                },
                step=new_state.step + 1,
            )
            
        return self.replace(state=new_state), info
    
    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        sample_rng,
        observations: Data,
        action_chunk: int = 50,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        observations = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], observations)
        if argmax:
            joint_actions = self.forward_policy(sample_rng, observations, rng=seed, train=False)[0]
        else:
            assert seed is not None, "Must provide seed for sampling"
            noise = jax.random.normal(seed, shape=(action_chunk, 14)) * 0.1
            noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
            joint_actions = self.forward_policy(sample_rng, observations, rng=seed, train=False)[0] + noise
        return joint_actions
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actor_def,
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        soft_target_update_rate: float = 0.005,
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.1,
        policy_update_freq: int = 2,
        pretrained_actor_params: Optional[Params] = None,
        freeze_filter = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        graphdef, initial_state = nnx.split(actor_def)
        wrapped_model = NNXToFlax(
        graphdef=graphdef,
        initial_state=initial_state,
        param_path="actor"
    )
        networks = {
            "actor": wrapped_model,
        }
        model_def = ModuleDict(networks)
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
        }
        rng, init_rng = jax.random.split(rng)
        sample_rng, create_rng = jax.random.split(rng)
        all_params = model_def.init(
            init_rng, 
            actor=[sample_rng, observations]
            )["params"]

        ACTOR_PARAM_KEY = 'modules_actor' # <-- 这是 ModuleDict 生成的实际键名
        INNER_ACTOR_KEY = 'actor' # NNXToFlax 创建的额外嵌套层
        if ACTOR_PARAM_KEY not in all_params:
            raise KeyError(f"'{ACTOR_PARAM_KEY}' key not found in all_params. Available keys: {list(all_params.keys())}")
        if INNER_ACTOR_KEY not in all_params[ACTOR_PARAM_KEY]:
            raise KeyError(f"'{INNER_ACTOR_KEY}' key not found in all_params['{ACTOR_PARAM_KEY}']. Available keys: {list(all_params[ACTOR_PARAM_KEY].keys())}")
        

        if pretrained_actor_params is not None:
            # 假设 freeze_filter 定义了 *哪些参数应该被替换* (例如 LoRA 参数) 并且 pretrained_actor_params 只包含这些参数, 所以不用管freeze_filter了
            # 注意：这要求 pretrained_actor_params 的键路径与 all_params 中的匹配
            try:
                # 获取实际要更新的参数字典
                target_actor_params = all_params[ACTOR_PARAM_KEY][INNER_ACTOR_KEY]
                # 现在遍历 pretrained_actor_params 并尝试更新 target_actor_params
                # 注意：target_actor_params 的结构可能还是嵌套的，例如
                # {'PaliGemma': {'params': {'lora_a': ..., 'lora_b': ...}}, ...}
                # 而 pretrained_actor_params 是
                # {'PaliGemma': {'lora_a': ..., 'lora_b': ...}, ...}
                # 需要匹配这种结构差异
                
                for module_name, module_pretrained_params in pretrained_actor_params.items():
                    # module_name 例如 'PaliGemma'
                    # module_pretrained_params 例如 {'lora_a': array, 'lora_b': array}
                    if module_name in target_actor_params:
                        target_module = target_actor_params[module_name] # 例如 {'params': {...}}
                        # 假设 target_module 有一个 'params' 子字典来存放实际参数, 在NNXToFlax里没有, 但别的构造可能有
                        if 'params' in target_module and isinstance(target_module['params'], dict):
                            # 如果目标有 'params' 子字典，则再深入更新其内容
                            params_subdict = target_module['params']
                            for param_name, param_value in module_pretrained_params.items():
                                if param_name in params_subdict:
                                    params_subdict[param_name] = param_value
                                    print(f"Loaded pretrained param: {module_name}/params/{param_name}")
                                else:
                                    print(f"Warning: Pretrained param key '{param_name}' not found under '{module_name}/params'. Skipping. Available: {list(params_subdict.keys())}")
                        else:
                            # 如果目标模块没有 'params' 层 ,直接尝试匹配顶层键
                            for param_name, param_value in module_pretrained_params.items():
                                if param_name in target_module:
                                    print(target_module[param_name])
                                    pdb.set_trace()
                                    target_module[param_name] = param_value
                                    print(f"Loaded pretrained param: {module_name}/{param_name}")
                                else:
                                    print(f"Warning: Pretrained param key '{param_name}' not found under '{module_name}'. Skipping. Available: {list(target_module.keys())}")
                    else:
                        print(f"Warning: Pretrained module key '{module_name}' not found in initialized actor params. Skipping. Available: {list(target_actor_params.keys())}")
                        
            except Exception as update_e:
                print(f"Error updating params with pretrained ones: {update_e}")
                raise update_e
            
        params = flax.core.freeze(all_params)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )
        return cls(
            state=state,
            config=dict(
                soft_target_update_rate=soft_target_update_rate,
                target_policy_noise=target_policy_noise,
                noise_clip=noise_clip,
                policy_update_freq=policy_update_freq,
                reward_bias=reward_bias,
                image_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        # Model architecture
        pretrained_policy_path: Optional[str] = None,
        **kwargs,
    ):
        policy_def, pretrained_actor_params, freeze_filter = create_policy_with_lora(pretrained_policy_path=pretrained_policy_path)
        agent = cls.create(
            rng,
            observations,
            actor_def=policy_def,
            pretrained_actor_params=pretrained_actor_params,
            freeze_filter=freeze_filter,
            **kwargs,
        )
        return agent




def learner(agent, replay_buffer, demo_buffer, wandb_logger=None):
    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None

    if latest_ckpt is not None:
        start_step = int(os.path.basename(latest_ckpt)[11:]) + 1
        print(f"Resuming from checkpoint at step {start_step}.")
    else:
        start_step = 0
        print("No checkpoint found. Starting from step 0.")

    step = start_step

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": 16,
        },
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": 16,
        },
    )

    timer = Timer()
    train_critic_networks_to_update = frozenset({"actor"})

    for step in tqdm.tqdm(
        range(start_step, 10000000), dynamic_ncols=True, desc="learner"
    ):  

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_critic_networks_to_update,
            )

        if step % 10 == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and step % 200 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=2
            )

def make_test_agent(
    seed,
    sample_obs,
    reward_bias=0.0,
    discount=0.95,
    pretrained_policy_path = None,
):
    agent = TestAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        discount=discount,
        reward_bias=reward_bias,
        pretrained_policy_path = pretrained_policy_path,
    )
    return agent

def main(_):
    
    TASK_ENV = rl_config()
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    sample_obs=TASK_ENV.observation_space.sample()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_obs)

    agent: TestAgent = make_test_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        pretrained_policy_path = "/home/anker/robotwin/RoboTwin/policy/pi0/checkpoints/pi0_base_aloha_robotwin_lora/pi0_stack_300/30000/params",  
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

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = DynamicNextObsReplayBufferDataStore(
            TASK_ENV.observation_space,
            TASK_ENV.action_space,
            capacity=20000,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="policy_test",
            description="test policy",
        )
        return replay_buffer, wandb_logger

    # learner:
    replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
    demo_buffer = DynamicNextObsReplayBufferDataStore(
        TASK_ENV.observation_space,
        TASK_ENV.action_space,
        capacity=20000,
    )

    for path in FLAGS.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                demo_buffer.insert(transition)
                replay_buffer.insert(transition)
    print(f"demo buffer size: {len(demo_buffer)}")
    print(f"online buffer size: {len(replay_buffer)}")

    if FLAGS.checkpoint_path is not None and os.path.exists(
        os.path.join(FLAGS.checkpoint_path, "buffer")
    ):
        for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    replay_buffer.insert(transition)
        print(
            f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
        )

    if FLAGS.checkpoint_path is not None and os.path.exists(
        os.path.join(FLAGS.checkpoint_path, "demo_buffer")
    ):
        for file in glob.glob(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
        ):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    demo_buffer.insert(transition)
        print(
            f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
        )

    # learner loop
    print("starting learner loop")
    learner(
        agent,
        replay_buffer,
        demo_buffer=demo_buffer,
        wandb_logger=wandb_logger,
    )


if __name__ == "__main__":
    app.run(main)



