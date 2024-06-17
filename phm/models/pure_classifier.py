from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
import phm.nn as pnn
import phm.nn.functional as PF
from phm.util.typing import *

from .base import BaseModel


@phm.register("pure-classifier")
class PureClassifier(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        network_type: str = ""
        network: dict = field(default_factory=dict)

        classifier_type: Optional[str] = None
        classifier: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.network = phm.find(self.cfg.network_type)(self.cfg.network)

        if self.cfg.classifier_type is not None:
            self.classifier = phm.find(
                self.cfg.classifier_type)(self.cfg.classifier)
        else:
            self.classifier = None

    def forward(self, x: Float[Tensor, "B Cin *D"], **kwargs) -> tuple[Float[Tensor, "B Dout"], dict[str, Tensor]]:

        features = self.network(x)
        if self.classifier is not None:
            pred = self.classifier(features)
        else:
            pred = features

        return pred, dict()
