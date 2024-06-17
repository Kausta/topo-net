from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from torchvision.models import efficientnet

from .base import BaseNetwork

EFFNET_KEYS = Literal["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
EFFNET_MAP: Dict[EFFNET_KEYS, tuple[Callable[..., efficientnet.EfficientNet], efficientnet.WeightsEnum]] = {
    "b0": (efficientnet.efficientnet_b0, efficientnet.EfficientNet_B0_Weights.DEFAULT),
    "b1": (efficientnet.efficientnet_b1, efficientnet.EfficientNet_B1_Weights.DEFAULT),
    "b2": (efficientnet.efficientnet_b2, efficientnet.EfficientNet_B2_Weights.DEFAULT),
    "b3": (efficientnet.efficientnet_b3, efficientnet.EfficientNet_B3_Weights.DEFAULT),
    "b4": (efficientnet.efficientnet_b4, efficientnet.EfficientNet_B4_Weights.DEFAULT),
    "b5": (efficientnet.efficientnet_b5, efficientnet.EfficientNet_B5_Weights.DEFAULT),
    "b6": (efficientnet.efficientnet_b6, efficientnet.EfficientNet_B6_Weights.DEFAULT),
    "b7": (efficientnet.efficientnet_b7, efficientnet.EfficientNet_B7_Weights.DEFAULT),
}


@phm.register("network-efficient-net")
class EfficientNet(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        model: str = "b0"  # Any of EFFNET_KEYS
        pretrained: bool = False
        pretrained_frozen: bool = False # Freeze everything except for the final linear classifier

        in_ch: int = 3
        out_ch: int = 256

        inp_map_conv: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.model in EFFNET_MAP.keys(
        ), f"Unknown model {self.cfg.model}"
        model_fn, pretrained_weights = EFFNET_MAP[self.cfg.model]
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

            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, self.cfg.out_ch)

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

