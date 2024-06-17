from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from torchvision.models import mobilenet, WeightsEnum

from .base import BaseNetwork

MNET_KEYS = Literal["v2", "v3_s", "v3_l"]
MNET_MAP: Dict[MNET_KEYS, tuple[Callable[..., Union[mobilenet.MobileNetV2, mobilenet.MobileNetV3]], WeightsEnum]] = {
    "v2": (mobilenet.mobilenet_v2, mobilenet.MobileNet_V2_Weights.DEFAULT),
    "v3_s": (mobilenet.mobilenet_v3_small, mobilenet.MobileNet_V3_Small_Weights.DEFAULT),
    "v3_l": (mobilenet.mobilenet_v3_large, mobilenet.MobileNet_V3_Large_Weights.DEFAULT),
}


@phm.register("network-mobile-net")
class MobileNet(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        model: str = "v2"  # Any of EFFNET_KEYS
        pretrained: bool = False
        pretrained_frozen: bool = False # Freeze everything except for the final linear classifier

        in_ch: int = 3
        out_ch: int = 256

        inp_map_conv: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.model in MNET_MAP.keys(
        ), f"Unknown model {self.cfg.model}"
        model_fn, pretrained_weights = MNET_MAP[self.cfg.model]
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

