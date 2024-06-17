import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from torchvision.models.resnet import conv1x1, conv3x3

from .adanorm import AdaptiveNorm

# based on torchvision


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            def act_layer(): return nn.ReLU(inplace=True)
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.act1 = act_layer()

        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm_layer(planes)
        self.act2 = act_layer()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        style = None
        if isinstance(self.norm1, AdaptiveNorm):
            assert "style" in kwargs
            style = kwargs["style"]

        identity = x

        out = self.conv1(x)
        out = self.norm1(out, style) if style is not None else self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out, style) if style is not None else self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x, **kwargs)

        out += identity
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            def act_layer(): return nn.ReLU(inplace=True)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = norm_layer(width)
        self.act1 = act_layer()

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width)
        self.act2 = act_layer()

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = norm_layer(planes * self.expansion)
        self.act3 = act_layer()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        style = None
        if isinstance(self.norm1, AdaptiveNorm):
            assert "style" in kwargs
            style = kwargs["style"]

        identity = x

        out = self.conv1(x)
        out = self.norm1(out, style) if style is not None else self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out, style) if style is not None else self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out, style) if style is not None else self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x, **kwargs)

        out += identity
        out = self.act3(out)

        return out


class Downsample(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = conv1x1(inplanes, planes, stride)
        self.norm = norm_layer(planes)

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        style = None
        if isinstance(self.norm, AdaptiveNorm):
            assert "style" in kwargs
            style = kwargs["style"]

        out = self.conv(x)
        out = self.norm(out, style) if style is not None else self.norm(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,

        inplanes: int = 64,
        in_channels: int = 3,
        conv1_kernel_size: int = 7,
        conv1_stride: int = 2,
        initial_max_pool: bool = True,
        layer_strides: List[int] = [1, 2, 2, 2],

        global_avg_pool: bool = True,
        flat_hw_mul: int = -1,

        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            def act_layer(): return nn.ReLU(inplace=True)
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=conv1_kernel_size,
                               stride=conv1_stride, padding=conv1_kernel_size // 2, bias=False)
        self.norm1 = norm_layer(self.inplanes)
        self.act1 = act_layer()

        self.initial_max_pool = initial_max_pool
        if self.initial_max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer_modules = []
        assert len(layers) > 0
        assert len(layers) == len(layer_strides)
        ch = self.inplanes
        for layer, stride in zip(layers, layer_strides):
            layer_modules.append(self._make_layer(
                block, ch, layer, stride=stride))
            ch *= 2
        self.layers = nn.ModuleList(layer_modules)
        if global_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flat_hw_mul = 1
        else:
            self.avgpool = None
            assert flat_hw_mul > 0
        self.fc = nn.Linear((ch // 2) * block.expansion * flat_hw_mul, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample(
                self.inplanes, planes * block.expansion, stride
            )
        previous_dilation = 1

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=1,
                    norm_layer=norm_layer,
                )
            )

        return nn.ModuleList(layers)

    def _forward_impl(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        # See note [TorchScript super()]

        style = None
        if isinstance(self.norm1, AdaptiveNorm):
            assert "style" in kwargs
            style = kwargs["style"]

        x = self.conv1(x)
        x = self.norm1(x, style) if style is not None else self.norm1(x)
        x = self.act1(x)
        if self.initial_max_pool:
            x = self.maxpool(x)

        for layer in self.layers:
            for block in layer:
                x = block(x, **kwargs)

        if self.avgpool is not None:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        return self._forward_impl(x, **kwargs)

