from typing import Dict, Iterable, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
default_init = nn.initializers.xavier_uniform

class EncodingWrapper(nn.Module):

    encoder: nn.Module
    use_proprio: bool = True
    image_keys: Iterable[str] = ("image",)
    fuse_proprio_images: bool = True

    @nn.compact
    def __call__(
        self,
        observations,
        stop_gradient=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations['image'][image_key]
            image = self.encoder[image_key](image) #[B, N, D]
            if stop_gradient:
                image = jax.lax.stop_gradient(image)
            encoded.append(image)
        obs_tokens = jnp.concatenate(encoded, axis=1)  # [B, 3 * N, D]


        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            state = nn.Dense(
                obs_tokens.shape[-1], kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            state = jnp.expand_dims(state, axis=1)# Add as extra token: [B, 1, D]

            if self.fuse_proprio_images:
                obs_global = jnp.mean(obs_tokens, axis=1, keepdims=True)
                combined = jnp.concatenate([obs_global, state], axis=-1) #[B, 1, D + D]
                combined = nn.Dense(obs_tokens.shape[-1], kernel_init=default_init())(combined) #[B, 1, D]
                combined = nn.gelu(combined)
            else:
                combined = state
            
            obs_tokens = jnp.concatenate([obs_tokens, combined], axis=1) # [B, N + 1, D]
        return obs_tokens # [B, N + 1, D]

class PositionalEncoding(nn.Module):
    """简单的位置编码层"""
    max_len: int = 48
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x):
        pe = self.param('positional_encoding', nn.initializers.normal(1e-6), (self.max_len, self.embed_dim))
        return x + pe[:x.shape[1], :]

class TransformerBlock(nn.Module):
    """单个 Transformer Encoder Block"""
    features: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, train=False):
        x = nn.LayerNorm()(x)
        residual = x
        # Multi-head Self-Attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=default_init()
        )
        x = attn(x, x, x, mask=mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = residual + x
        
        # Feed Forward
        residual = x
        x = nn.LayerNorm()(x)
        y = nn.Dense(self.features * 4, kernel_init=default_init())(x)
        y = nn.gelu(y)
        y = nn.Dense(self.features, kernel_init=default_init())(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        x = residual + y
        
        return x

class SmallTransformerTextEncoder(nn.Module):
    """
    小型 Transformer 文本编码器
    输入: token_ids (integers) [B, L]
    输出: [B, D] 的全局表示
    """
    vocab_size: int = 257152  # PaligemmaTokenizer 默认大小
    embed_dim: int = 256
    num_layers: int = 1
    num_heads: int = 4
    max_seq_length: int = 48
    dropout_rate: float = 0.1
    pooling: str = "cls"  # "cls" or "mean"
    use_positional_encoding: bool = True
    trainable_positional_encoding: bool = True

    def setup(self):
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(max_len=self.max_seq_length, embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, input_ids, train=False, mask = None):
        # Input shape: [B, L], mask shape: [B, L] (optional)
        # Token Embedding
        token_emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            embedding_init=default_init()
        )(input_ids)
        # Positional Encoding
        if self.use_positional_encoding:
            token_emb =self.pos_encoder(token_emb)
            if not self.trainable_positional_encoding:
                token_emb = jax.lax.stop_gradient(token_emb)

        token_emb = nn.Dropout(rate=self.dropout_rate)(token_emb, deterministic=not train)
        # Add CLS token
        if self.pooling == "cls":
            cls_token = self.param('cls_token', 
                                  nn.initializers.normal(1e-6), 
                                  (1, 1, self.embed_dim))
            cls_token = jnp.tile(cls_token, (token_emb.shape[0], 1, 1))
            x = jnp.concatenate([cls_token, token_emb], axis=1)
            if mask is not None:
                cls_mask = jnp.ones((x.shape[0], 1), dtype=mask.dtype)
                mask = jnp.concatenate([cls_mask, mask], axis=1)
        else:
            x = token_emb

        # Convert mask to attention mask format
        if mask is not None:
            attention_mask = mask[:, None, None, :] #[B, 1, 1, L + 1]
            attention_mask = jnp.array(attention_mask, dtype=bool)
        else:
            attention_mask = None

        
            
        # Transformer Blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                features=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, mask = attention_mask, train=train)
        
        # Final Pooling
        if self.pooling == "cls":
            
            return x[:, 0, :]
        elif self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask[:, :, None].astype(x.dtype)  # [B, L, 1]
                x = jnp.sum(x * mask_expanded, axis=1) / jnp.clip(jnp.sum(mask_expanded, axis=1), a_min=1e-6)

            else:
                x = jnp.mean(x, axis=1)

            return x
        else:
            raise ValueError(f"Pooling method {self.pooling} not supported")

class SmallTransformerActionEncoder(nn.Module):
    """
    小型 Transformer 动作编码器
    输入: actions (float) [B, L, 14]
    输出: [B, L, D] 的中间表示
    """
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    trainable_positional_encoding: bool = True
    def setup(self):
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(max_len=50, embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, actions, train=False):
        # Input shape: [B, L, 14]
        token_emb = nn.Dense(
            features=self.embed_dim,
            kernel_init=default_init()
        )(actions)
        if self.use_positional_encoding:
            token_emb =self.pos_encoder(token_emb)
            if not self.trainable_positional_encoding:
                token_emb = jax.lax.stop_gradient(token_emb)        

        # Transformer Blocks
        for _ in range(self.num_layers):
            token_emb = TransformerBlock(
                features=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(token_emb, train=train)

        return token_emb # [B, L, D]