import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Sequence
from serl_launcher.networks.mlp import MLP
default_init = nn.initializers.xavier_uniform()

class CrossAttentionBlock(nn.Module):
    num_heads: int
    qkv_features: int = 128
    out_features: Optional[int] = 128
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, query, key_value, train: bool = False):
        if self.use_layer_norm:
            query = nn.LayerNorm()(query)
            key_value = nn.LayerNorm()(key_value)

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.out_features,
            kernel_init=default_init()
        )
        x = attn(query=query, key=key_value, value=key_value) + query
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x
    
class CrossAttentiveCritic(nn.Module):
    # Cross-attention config
    cross_attn_num_heads: int = 4
    cross_attn_qkv_features: Optional[int] = 128
    cross_attn_out_features: Optional[int] = 128
    cross_attn_dropout_rate: float = 0.0
    cross_attn_use_layer_norm: bool = False

    # Final MLP head config
    mlp_hidden_dims: Sequence[int] = (256, 256)
    mlp_activations: Callable = nn.swish
    mlp_activate_final: bool = False
    mlp_use_layer_norm: bool = False
    mlp_dropout_rate: float = 0.0

    # Submodules
    encoder: nn.Module
    action_encoder: nn.Module
    text_encoder: nn.Module

    @nn.compact
    def __call__(self, observations, actions, train: bool = False):
        # 1. Encode observation into tokens [B, N, D_obs]

        obs_emb = self.encoder(observations) #[B, N, D]

        # 2. Encode actions into tokens [B, action_horizon, D_act]

        action_emb = self.action_encoder(actions)  # [B, action_horizon, D_act]

        # 3. Encode instruction into vector [B, D_txt]

        txt_emb = self.text_encoder(
            input_ids=observations["tokenized_prompt"],
            mask=observations["tokenized_prompt_mask"],
            train=train
        )
        txt_emb = jnp.expand_dims(txt_emb, axis=1)  # [B, 1, D_txt]

        # 4. Cross Attention: text as query, obs as key/value

        cross_attn_state = CrossAttentionBlock(
            num_heads=self.cross_attn_num_heads,
            qkv_features=self.cross_attn_qkv_features,
            out_features=self.cross_attn_out_features,
            use_layer_norm=self.cross_attn_use_layer_norm,
            dropout_rate=self.cross_attn_dropout_rate
        )
        state_attended = cross_attn_state(
            query=txt_emb,
            key_value=obs_emb,
            train=train
        ) # [B, 1, D_attn]

        state_attended = nn.swish(state_attended)
        
        # 4. Cross Attention: state as query, actions as key/value

        cross_attn_actions = CrossAttentionBlock(
            num_heads=self.cross_attn_num_heads,
            qkv_features=self.cross_attn_qkv_features,
            out_features=self.cross_attn_out_features,
            use_layer_norm=self.cross_attn_use_layer_norm,
            dropout_rate=self.cross_attn_dropout_rate
        )
        attended = cross_attn_actions(
            query = state_attended,
            key_value = action_emb, # [B, action_horizon, D_act]
            train=train
        ) # [B, 1, D_attn]
        attended = jnp.squeeze(attended, axis=1)  # [B, D_attn]

        # 5. Combine attended context, text embedding
        txt_emb_squeezed = jnp.squeeze(txt_emb, axis=1)  # [B, D_txt]
        combined = jnp.concatenate([attended, txt_emb_squeezed], axis=-1)  # [B, D_total]

        # 6. Final MLP Head
        mlp_head = MLP(
            hidden_dims=self.mlp_hidden_dims,
            activations=self.mlp_activations,
            activate_final=self.mlp_activate_final,
            use_layer_norm=self.mlp_use_layer_norm,
            dropout_rate=self.mlp_dropout_rate
        )
        q_values = mlp_head(combined, train=train)

        return q_values