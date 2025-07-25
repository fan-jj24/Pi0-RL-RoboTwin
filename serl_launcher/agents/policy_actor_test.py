import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
from functools import partial
import tqdm
from typing import Iterable, Optional, Tuple, FrozenSet, Any
import flax
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
import yaml
import subprocess
from datetime import datetime
import pdb
from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_multi_string("demo_path", str(root_path / "demo_data" / "15_demos_2025-07-22_11-37-30.pkl"), "Path to the demo data.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", str(root_path / "checkpoints"), "Path to save checkpoints.")

import sys
sys.path.insert(0, str(root_path))
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.utils.parmas_utils import merge_lora_weights_in_tree
from pi0.src.openpi.models import model, pi0_nn as pi0
from pi0.src.openpi.training.rl_cfg import rl_config, RoboTwinEnv
from pi0.src.openpi.shared import normalize

def create_policy(pretrained_policy_path = None, lora = True):
    if lora:
        policy_config=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    else:
        policy_config=pi0.Pi0Config(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    if pretrained_policy_path is not None:
        pretrained_actor_params = model.restore_params(pretrained_policy_path, dtype=jnp.float32)
        if not lora:
            pretrained_actor_params = merge_lora_weights_in_tree(pretrained_actor_params)
    else:
        raise ValueError("pretrained_policy_path must be provided for post training")
    policy_def = pi0.Pi0(config=policy_config)
    return policy_def, pretrained_actor_params



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
        target_policy_noise: float = 0.2,
        noise_clip: float = 0.1,
        pretrained_actor_params: Optional[Params] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):

        networks = {
            "actor": actor_def,
        }
        model_def = ModuleDict(networks)
        rng, init_rng = jax.random.split(rng)
        sample_rng, create_rng = jax.random.split(rng)
        all_params = model_def.init(
            init_rng, 
            actor=[sample_rng, observations]
            )["params"]

        ACTOR_PARAM_KEY = 'modules_actor' # <-- 这是 ModuleDict 生成的实际键名
        if pretrained_actor_params is not None:
            # 注意：这要求 pretrained_actor_params 的键路径与 all_params 中的匹配
            # 注意: paligemma 在预训练模型中 (self.PaliGemma = nnx.Dict(llm=llm, img=img)), 而此时的模型是展平的 img 和 llm
            try:
                target_actor_params = all_params[ACTOR_PARAM_KEY]                
                for module_name, module_pretrained_params in pretrained_actor_params.items():
                    # module_name 例如 'PaliGemma', 'state_proj', 'action_in_proj' 等等
                    if module_name in target_actor_params:# 直接尝试匹配顶层键
                        target_actor_params[module_name] = module_pretrained_params
                        print(f"Loaded pretrained param: {module_name}")
                    elif module_name == "PaliGemma":# 特例处理 PaliGemma，因为它包含了 img 和 llm
                        for submodule_name, submodule_pretrained_params in module_pretrained_params.items():
                            if submodule_name in target_actor_params:
                                target_actor_params[submodule_name] = submodule_pretrained_params
                                print(f"Loaded pretrained param: {submodule_name}")
                    else:
                        print(f"Warning: Pretrained module key '{module_name}' not found in initialized actor params. Skipping. Available: {list(target_actor_params.keys())}")
            except Exception as update_e:
                print(f"Error updating params with pretrained ones: {update_e}")
                raise update_e
            
        params = flax.core.freeze(all_params)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs= None,
            rng=create_rng,
        )

        return cls(
            state=state,
            config=dict(
                target_policy_noise=target_policy_noise,
                noise_clip=noise_clip,
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
        policy_def, pretrained_actor_params = create_policy(pretrained_policy_path=pretrained_policy_path, lora = False)
        agent = cls.create(
            rng,
            observations,
            actor_def=policy_def,
            pretrained_actor_params=pretrained_actor_params,
            **kwargs,
        )
        return agent

def actor(agent, sampling_rng):
    output_file = open(str(root_path / "eval_results" / "policy_results.log"), "a")
    norm_stats_dir = str(root_path)
    norm_stats = normalize.load(norm_stats_dir)
    env = RoboTwinEnv(norm_stats=norm_stats)
    save_dir = root_path / "eval_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    def get_camera_config(camera_type):
        camera_config_path = str(root_path / "task_config" / "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        assert camera_type in args, f"camera {camera_type} is not defined"
        return args[camera_type]
    
    if True:
        video_save_dir = save_dir
        camera_config = get_camera_config("D435")
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm.tqdm(range(10), dynamic_ncols=True)
    for step in pbar:
        ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{video_save_dir}/episode{step + 50}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
        obs, task_name, now_seed = env.reset(save_video=True)
        env.task.eval_video_path = video_save_dir
        env.task._set_eval_video_ffmpeg(ffmpeg)
        while True:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                sample_rng=sampling_rng,
                observations=jax.device_put(obs),
                seed=key,
                argmax=True,
            )
            actions = np.asarray(jax.device_get(actions))
            next_obs, reward, done, info = env.step(actions)
            obs = next_obs
            if done:
                result_msg = f"reward: {reward}, task_name: {task_name}, seed: {now_seed}"
                output_file.write(f"{datetime.now()}: {result_msg}\n")  # 写入文件
                output_file.flush() 
                env.task._del_eval_video_ffmpeg()
                env.close_env(env.task)
                break

    output_file.close()
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
        pretrained_policy_path=pretrained_policy_path,
    )
    return agent

def main(_):
    TASK_ENV = rl_config()
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    sample_obs=TASK_ENV.observation_space.sample()
    sample_obs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample_obs)
    agent: TestAgent = make_test_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        pretrained_policy_path = str(root_path / "pretrained_params" / "params_30000"),  
    )
    print("starting actor loop")
    actor(agent, sampling_rng)

if __name__ == "__main__":
    app.run(main)



