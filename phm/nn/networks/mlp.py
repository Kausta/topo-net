from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from .base import BaseNetwork


@phm.register("network-mlp")
class MLP(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        in_ch: int = 200
        out_ch: int = 128

        hidden: list[int] = field(default_factory=lambda: [256, 256])

    cfg: Config

    def configure(self) -> None:
        super().configure()

        in_ch = self.cfg.in_ch
        out_ch = self.cfg.out_ch

        hidden = self.cfg.hidden + [out_ch]
        layers = [nn.Linear(in_ch, hidden[0])]
        for i in range(1, len(hidden)):
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Linear(hidden[i-1], hidden[i])
            ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "B *L"], **kwargs) -> Float[Tensor, "B Dout"]:
        return self.net(x)
