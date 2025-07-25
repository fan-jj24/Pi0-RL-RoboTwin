from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.utils.parmas_utils import create_policy

class ActorAgent(flax.struct.PyTreeNode):

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
            joint_actions = self.forward_policy(sample_rng = sample_rng, observations = observations, rng=seed, train=False)
        else:
            assert seed is not None, "Must provide seed for sampling"
            noise = jax.random.normal(seed, shape=(action_chunk, 14)) * 0.1
            noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
            joint_actions = self.forward_policy(sample_rng = sample_rng, observations = observations, rng=seed, train=False) + noise

        return joint_actions
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        # Algorithm config
        target_policy_noise: float = 0.1,
        noise_clip: float = 0.1,
        pretrained_actor_params: Optional[Params] = None,
        **kwargs,
    ):
        networks = {"actor": actor_def}
        model_def = ModuleDict(networks)
        rng, init_rng = jax.random.split(rng)
        sample_rng, create_rng = jax.random.split(rng)
        all_params = model_def.init(
            init_rng, 
            actor=[sample_rng, observations],
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
            txs=None,
            rng=create_rng,
        )
        return cls(
            state=state,
            config=dict(
                target_policy_noise=target_policy_noise,
                noise_clip=noise_clip,
                **kwargs,
            ),
        )
    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        target_policy_noise: float = 0.1,
        noise_clip: float = 0.1,
        pretrained_policy_path: Optional[str] = None,
        **kwargs,
    ):
        policy_def, pretrained_actor_params = create_policy(pretrained_policy_path=pretrained_policy_path,  lora = False)
        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            target_policy_noise=target_policy_noise,
            noise_clip=noise_clip,
            pretrained_actor_params=pretrained_actor_params,
            **kwargs,
        )
        return agent
    