from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from .base import BaseClassifier


@phm.register("classifier-linear")
class LinearClassifier(BaseClassifier):
    @dataclass
    class Config(BaseClassifier.Config):
        in_ch: int = 128
        ch: int = 256
        out_ch: int = 10

        dropout_in: float = 0.0
        dropout_feat: float = 0.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        in_ch = self.cfg.in_ch
        ch = self.cfg.ch
        out_ch = self.cfg.out_ch

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout_in),
            nn.Linear(in_ch, ch),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout_feat),
            nn.Linear(ch, out_ch)
        )

    def forward(self, x: Float[Tensor, "B *L"], **kwargs) -> Float[Tensor, "B Dout"]:
        return self.net(x)
