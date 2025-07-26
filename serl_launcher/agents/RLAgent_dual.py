from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet
import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper, SmallTransformerTextEncoder, SmallTransformerActionEncoder
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.cross_att import ensemblize
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.networks.cross_att import CrossAttentiveCritic
from serl_launcher.vision.convernext import ConvNeXtEncoder
from serl_launcher.utils.parmas_utils import merge_lora_weights_in_tree
from pi0.src.openpi.models import model, pi0_nn as pi0


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

class RLAgent(flax.struct.PyTreeNode):

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
    

    def _compute_next_actions(self, batch, rng):
        noise_key, sample_key = jax.random.split(rng)
        next_actions = self.forward_policy(sample_rng = sample_key, observations = batch["next_observations"], rng=rng)
        noise = jax.random.normal(noise_key, next_actions.shape) * self.config["target_policy_noise"]
        noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
        next_actions = next_actions + noise
        return next_actions

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        actions= batch["actions"]
        
        rng, next_action_key = jax.random.split(rng)
        next_actions = self._compute_next_actions(batch, next_action_key)
        target_next_q = self.forward_target_critic(observations = batch["next_observations"], actions = next_actions, rng=rng)
        target_q = (batch["rewards"]+ self.config["discount"] * (batch["dones"] < 1) * target_next_q)
        chex.assert_shape(target_q, (batch_size,))
        
        predicted_q = self.forward_critic(observations = batch["observations"], actions = actions, rng=rng, grad_params=params)
        chex.assert_shape(predicted_q, (batch_size,))

        critic_loss = jnp.mean((predicted_q - target_q) ** 2)
        return critic_loss, {
            "critic_loss": critic_loss,
        }

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, sample_rng = jax.random.split(rng)
        actions = self.forward_policy(sample_rng = sample_rng, observations = batch["observations"], rng=rng, grad_params=params)
        chex.assert_shape(actions, (batch_size, 50, 14))

        predicted_q = self.forward_critic(observations = batch["observations"], actions = actions, rng=rng)
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
    ) -> Tuple["RLAgent", dict]:
        
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 50, 14))
        
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
            
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)
            
        batch = batch.copy(add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]})
        loss_fns = self.loss_fns(batch, **kwargs)
        
        # 只更新指定网络
        assert networks_to_update.issubset(loss_fns.keys())
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})
            
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        
        # soft target update
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])
    
        # Update RNG
        new_state = new_state.replace(rng=rng)
        
        # use the fixed learning rate
        '''# Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]'''
                
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
        target_policy_noise: float = 0.1,
        noise_clip: float = 0.1,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        pretrained_actor_params: Optional[Params] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {"actor": actor_def, "critic": critic_def,}
        model_def = ModuleDict(networks)
        txs = {"actor": make_optimizer(**actor_optimizer_kwargs), "critic": make_optimizer(**critic_optimizer_kwargs)}
        rng, init_rng = jax.random.split(rng)
        sample_rng, create_rng = jax.random.split(rng)
        all_params = model_def.init(
            init_rng, 
            actor=[sample_rng, observations],
            critic=[observations, actions],
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
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        return cls(
            state=state,
            config=dict(
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_policy_noise=target_policy_noise,
                noise_clip=noise_clip,
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
        critic_ensemble_size: int = 2,
        discount = 0.95,
        soft_target_update_rate: float = 0.005,
        target_policy_noise: float = 0.1,
        noise_clip: float = 0.1,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        pretrained_policy_path: Optional[str] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):

        encoders = {
                image_key: ConvNeXtEncoder()
                for image_key in image_keys
        }

        critic_def = CrossAttentiveCritic(
            obs_encoder=EncodingWrapper(
                encoder=encoders,
                use_proprio=True,
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
        # critic_def = ensemblize(critic_def, ensemble_size=critic_ensemble_size)(name="critic_ensemble")
        policy_def, pretrained_actor_params = create_policy(pretrained_policy_path=pretrained_policy_path,  lora = False)

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_policy_noise=target_policy_noise,
            noise_clip=noise_clip,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            pretrained_actor_params=pretrained_actor_params,
            reward_bias=reward_bias,
            **kwargs,
        )
            
        return agent
    