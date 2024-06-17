from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from torchvision.models import resnet

from .base import BaseNetwork

RESNET_KEYS = Literal["resnet18", "resnet34",
                      "resnet50", "resnet101", "resnet152"]
RESNET_MAP: Dict[RESNET_KEYS, tuple[Callable[..., resnet.ResNet], resnet.Weights]] = {
    "resnet18": (resnet.resnet18, resnet.ResNet18_Weights.DEFAULT),
    "resnet34": (resnet.resnet34, resnet.ResNet34_Weights.DEFAULT),
    "resnet50": (resnet.resnet50, resnet.ResNet50_Weights.DEFAULT),
    "resnet101": (resnet.resnet101, resnet.ResNet101_Weights.DEFAULT),
    "resnet152": (resnet.resnet152, resnet.ResNet152_Weights.DEFAULT),
}


@phm.register("network-resnet")
class Resnet(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        model: str = "resnet18"  # Any of RESNET_KEYS
        pretrained: bool = False
        pretrained_frozen: bool = False # Freeze everything except for the final linear classifier

        in_ch: int = 3
        out_ch: int = 256

        inp_map_conv: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.model in RESNET_MAP.keys(
        ), f"Unknown model {self.cfg.model}"
        model_fn, pretrained_weights = RESNET_MAP[self.cfg.model]
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
            
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.cfg.out_ch)

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


@phm.register("network-resnet-custom")
class ResnetCustom(BaseNetwork):
    @dataclass
    class Config(BaseNetwork.Config):
        in_ch: int = 3
        out_ch: int = 256

        layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
        strides: list[int] = field(default_factory=lambda: [1, 2, 2, 2])

        block: str = "BasicBlock"  # "BasicBlock" | "Bottleneck"

        groups: int = 1
        width_per_group: int = 64

        inplanes: int = 64

        conv1_kernel_size: int = 7
        conv1_stride: int = 2
        initial_max_pool: bool = True

        global_avg_pool: bool = True
        flat_hw_mul: int = -1

        # "BatchNorm" | "InstanceNorm" | "GroupNorm" | "AdaptiveGroupNorm"
        norm_layer: Optional[str] = None
        norm_affine: bool = True  # Only if norm_layer is not None
        norm_num_groups: int = 8
        norm_style_dim: int = 128

    cfg: Config

    def configure(self) -> None:
        super().configure()

        block_name = self.cfg.block
        if block_name == "BasicBlock":
            block = pnn.resnet.BasicBlock
        elif block_name == "Bottleneck":
            block = pnn.resnet.Bottleneck
        else:
            raise ValueError(f"Unknown block {self.cfg.block}")

        self.model = pnn.resnet.ResNet(
            block=block,
            layers=self.cfg.layers,
            num_classes=self.cfg.out_ch,
            groups=self.cfg.groups,
            width_per_group=self.cfg.width_per_group,
            inplanes=self.cfg.inplanes,
            in_channels=self.cfg.in_ch,
            conv1_kernel_size=self.cfg.conv1_kernel_size,
            conv1_stride=self.cfg.conv1_stride,
            initial_max_pool=self.cfg.initial_max_pool,
            layer_strides=self.cfg.strides,
            global_avg_pool=self.cfg.global_avg_pool,
            flat_hw_mul=self.cfg.flat_hw_mul,
            norm_layer=self._get_norm_layer()
        )

    def _get_norm_layer(self):
        norm_name = self.cfg.norm_layer
        if norm_name is None:
            return None
        elif norm_name == "BatchNorm":
            return lambda ch: nn.BatchNorm2d(ch, affine=self.cfg.norm_affine)
        elif norm_name == "InstanceNorm":
            return lambda ch: nn.InstanceNorm2d(ch, affine=self.cfg.norm_affine)
        elif norm_name == "GroupNorm":
            return lambda ch: nn.GroupNorm(self.cfg.norm_num_groups, ch, affine=self.cfg.norm_affine)
        elif norm_name == "AdaptiveGroupNorm":
            return lambda ch: pnn.AdaptiveGroupNorm(self.cfg.norm_num_groups, ch, self.cfg.norm_style_dim, affine=self.cfg.norm_affine)
        else:
            raise ValueError(f"Unknown norm {norm_name}")

    def forward(self, x: Float[Tensor, "B Cin H W"], **kwargs) -> Float[Tensor, "B Cout"]:
        return self.model(x, **kwargs)
