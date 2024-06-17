import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

class AdaptiveNorm(nn.Module):
    __constants__ = ['style_channels']
    style_channels: int

    def __init__(self, style_channels: int):
        super().__init__()

        self.style_channels = style_channels

    def forward(self, x: Float[Tensor, "B C *D"], style: Float[Tensor, "B L"]) -> Tensor:
        return super().forward()

# Based on huggingface/diffusers and nvidia/LION

class AdaptiveGroupNorm(AdaptiveNorm):
    def __init__(self, num_groups: int, num_channels: int, style_channels: int, eps: float = 1e-5, affine: bool = False, 
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(style_channels=style_channels)

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype
        )

        self.style_lin = nn.Linear(style_channels, num_channels * 2, **factory_kwargs)

    def forward(self, x: Float[Tensor, "B C *D"], style: Float[Tensor, "B L"]) -> Tensor:
        assert len(style.shape) == 2
        style = self.style_lin(style)
        style = style.view(*style.shape, *[1 for _ in x.shape[2:]])

        scale, shift = style.chunk(2, dim=1)

        out = self.group_norm(x)
        out = out * (1 + scale) + shift
        return out

