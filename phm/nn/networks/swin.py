from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from torchvision.models import swin_transformer as swin

from .base import BaseNetwork

SWIN_KEYS = Literal["b", "s", "t"]
SWIN_MAP: Dict[SWIN_KEYS, tuple[Callable[..., swin.SwinTransformer], swin.WeightsEnum]] = {
    "b": (swin.swin_b, swin.Swin_B_Weights.DEFAULT),
    "s": (swin.swin_s, swin.Swin_S_Weights.DEFAULT),
    "t": (swin.swin_t, swin.Swin_T_Weights.DEFAULT),
    "v2_b": (swin.swin_v2_b, swin.Swin_V2_B_Weights.DEFAULT),
    "v2_s": (swin.swin_v2_s, swin.Swin_V2_S_Weights.DEFAULT),
    "v2_t": (swin.swin_v2_t, swin.Swin_V2_T_Weights.DEFAULT),
}


@phm.register("network-swin")
class Swin(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        model: str = "v2_t"  # Any of EFFNET_KEYS
        pretrained: bool = False
        pretrained_frozen: bool = False # Freeze everything except for the final linear classifier

        in_ch: int = 3
        out_ch: int = 256

        inp_map_conv: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.model in SWIN_MAP.keys(
        ), f"Unknown model {self.cfg.model}"
        model_fn, pretrained_weights = SWIN_MAP[self.cfg.model]
        if not self.cfg.pretrained:
            pretrained_weights = None
            num_classes = self.cfg.out_ch
        else:
            num_classes = len(pretrained_weights.meta["categories"])

        self.model = model_fn(weights=pretrained_weights,
                              num_classes=num_classes)
        
        if self.cfg.pretrained:
            if self.cfg.pretrained_frozen:
                for param in self.model.parameters():
                    param.requires_grad = False

            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, self.cfg.out_ch)

        assert self.cfg.in_ch == 3 or self.cfg.inp_map_conv
        if self.cfg.inp_map_conv:
            self.inp_map = nn.Sequential(
                nn.Conv2d(self.cfg.in_ch, 3, kernel_size=5,
                          stride=1, padding=2, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        else:
            self.inp_map = None

    def forward(self, x: Float[Tensor, "B Cin H W"]) -> Float[Tensor, "B Cout"]:
        if self.inp_map is not None:
            x = self.inp_map(x)
        return self.model(x)

