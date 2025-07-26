import flax.linen as nn
import jax.numpy as jnp

class ConvNeXtBlock(nn.Module):
    dim: int  # 当前通道数

    @nn.compact
    def __call__(self, x):
        shortcut = x
        # 深度可分离卷积 (7x7, padding='SAME')
        x = nn.Conv(self.dim, kernel_size=(7, 7), padding='SAME', feature_group_count=self.dim)(x)
        # LayerNorm
        x = nn.LayerNorm()(x)
        # 1x1 卷积扩展通道数 (dim → 4*dim)
        x = nn.Conv(4 * self.dim, kernel_size=(1, 1))(x)
        x = nn.gelu(x)
        # 1x1 卷积恢复通道数 (4*dim → dim)
        x = nn.Conv(self.dim, kernel_size=(1, 1))(x)
        # 残差连接
        x = x + shortcut
        return x

class DownSample(nn.Module):
    dim: int  # 下采样后的通道数

    @nn.compact
    def __call__(self, x):
        # 2x2 卷积，步长2，下采样空间维度
        x = nn.Conv(self.dim, kernel_size=(2, 2), strides=2)(x)
        x = nn.LayerNorm()(x)
        return x

class ConvNeXtEncoder(nn.Module):
    depths: list = (3, 3, 9, 3)  # 每个阶段的块数
    dims: list = (64, 128, 256, 512)  # 每个阶段的通道数

    @nn.compact
    def __call__(self, x):
        # Stem 层：4x4卷积，步长4，下采样到 56x56
        x = nn.Conv(self.dims[0], kernel_size=(4, 4), strides=4)(x)
        x = nn.LayerNorm()(x)

        # Stage 1-4
        for i in range(len(self.depths)):
            # 下采样层（Stage 1-3）
            if i > 0:
                x = DownSample(self.dims[i])(x)

            # ConvNeXt 块
            for _ in range(self.depths[i]):
                x = ConvNeXtBlock(self.dims[i])(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)

        # bottleneck_dim 
        x = nn.gelu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        return x