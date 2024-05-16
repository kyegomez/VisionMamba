"""VisionMambaBlock module."""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def output_head(dim: int, num_classes: int):
    """
    Creates a head for the output layer of a model.

    Args:
        dim (int): The input dimension of the head.
        num_classes (int): The number of output classes.

    Returns:
        nn.Sequential: The output head module.
    """
    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x = self.proj(x)

        # forward con1d
        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.ssm,
        )

        # backward conv1d
        x2 = self.process_direction(
            x,
            self.backward_conv1d,
            self.ssm,
        )

        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip

    def process_direction(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        print(f"Conv1d: {x}")
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x


class Vim(nn.Module):
    """
    Vision Mamba (Vim) model implementation.

    Args:
        dim (int): Dimension of the model.
        dt_rank (int, optional): Rank of the dynamic tensor. Defaults to 32.
        dim_inner (int, optional): Inner dimension of the model. Defaults to None.
        d_state (int, optional): State dimension of the model. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to None.
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of the image patch. Defaults to 16.
        channels (int, optional): Number of image channels. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        depth (int, optional): Number of encoder layers. Defaults to 12.

    Attributes:
        dim (int): Dimension of the model.
        dt_rank (int): Rank of the dynamic tensor.
        dim_inner (int): Inner dimension of the model.
        d_state (int): State dimension of the model.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        channels (int): Number of image channels.
        dropout (float): Dropout rate.
        depth (int): Number of encoder layers.
        to_patch_embedding (nn.Sequential): Sequential module for patch embedding.
        dropout (nn.Dropout): Dropout module.
        cls_token (nn.Parameter): Class token parameter.
        to_latent (nn.Identity): Identity module for latent representation.
        layers (nn.ModuleList): List of encoder layers.
        output_head (output_head): Output head module.

    """

    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        num_classes: int = None,
        image_size: int = 224,
        patch_size: int = 16,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dropout = dropout
        self.depth = depth

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_height,
            ),
            nn.Linear(patch_dim, dim),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=dim,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

        # Output head
        self.output_head = output_head(dim, num_classes)

    def forward(self, x: Tensor):
        # Patch embedding
        b, c, h, w = x.shape

        x = self.to_patch_embedding(x)
        print(f"Patch embedding: {x.shape}")

        # Shape
        b, n, _ = x.shape

        # Cls tokens
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        print(f"Cls tokens: {cls_tokens.shape}")

        # Concatenate
        # x = torch.cat((cls_tokens, x), dim=1)

        # Dropout
        x = self.dropout(x)
        print(x.shape)

        # Forward pass with the layers
        for layer in self.layers:
            x = layer(x)
            print(f"Layer: {x.shape}")

        # Latent
        x = self.to_latent(x)

        # x = reduce(x, "b s d -> b d", "mean")

        # Output head with the cls tokens
        return self.output_head(x)
