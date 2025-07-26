import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from pi0.src.openpi.models import model as _model
from pi0.src.openpi.models import gemma as _gemma
from pi0.src.openpi.models import siglip as _siglip
from pi0.src.openpi.shared import array_typing as at
from pi0.src.openpi.transforms import pad_to_dim



def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Config:
    dtype: str = "bfloat16"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    def __init__(self, paligemma_variant: _gemma.Variant = "gemma_2b", action_expert_variant: _gemma.Variant = "gemma_300m"):
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        
    
class Pi0(nn.Module):
    config: Pi0Config

    def setup(self):
        self.action_dim, self.action_horizon, self.max_token_len = self.config.action_dim, self.config.action_horizon, self.config.max_token_len
        paligemma_config = _gemma.get_config(self.config.paligemma_variant)
        action_expert_config = _gemma.get_config(self.config.action_expert_variant)

        self.llm = _gemma.Module(configs=[paligemma_config, action_expert_config], embed_dtype=self.config.dtype)
        self.img = _siglip.Module(
            num_classes=paligemma_config.width,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=self.config.dtype,
        )

        self.state_proj = nn.Dense(features=action_expert_config.width)
        self.action_in_proj = nn.Dense(features=action_expert_config.width)
        self.action_time_mlp_in = nn.Dense(features=action_expert_config.width)
        self.action_time_mlp_out = nn.Dense(features=action_expert_config.width)
        self.action_out_proj = nn.Dense(features=self.config.action_dim)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nn.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    

    def input_obs(self, observation):
        if not isinstance(observation, _model.Observation):
            obs_dict = jax.tree.map(lambda x: x, observation)
            obs_dict["image_mask"] = {k: jnp.asarray(v, dtype=bool).reshape(-1) for k, v in obs_dict["image_mask"].items()}
            obs_dict["state"] = jnp.asarray(pad_to_dim(obs_dict["state"], 32), dtype=jnp.float32)
            observation = _model.Observation.from_dict(obs_dict)
        return observation


    def __call__(self, sample_rng, observation, num_steps=10, train=False):
        observation = self.input_obs(observation)
        observation = _model.preprocess_observation(None, observation, train=train)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.llm(
            embedded=[prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.llm(
                embedded=[None, suffix_tokens],
                positions=positions,
                mask=full_attn_mask,
                kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions(self, sample_rng, observation):
        return self.__call__(sample_rng, observation)