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


@phm.register("concat-classifier")
class ConcatClassifier(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        network_type: str = ""
        network: dict = field(default_factory=dict)

        aux_network_type: str = ""
        aux_network: dict = field(default_factory=dict)

        classifier_type: str = ""
        classifier: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.network = phm.find(self.cfg.network_type)(self.cfg.network)
        self.aux_network = phm.find(self.cfg.aux_network_type)(self.cfg.aux_network)
        self.classifier = phm.find(self.cfg.classifier_type)(self.cfg.classifier)

    def forward(self, x: Float[Tensor, "B Cin *D"], **kwargs) -> tuple[Float[Tensor, "B Dout"], dict[str, Tensor]]:
        assert "aux" in kwargs
        
        features = self.network(x)

        aux_x = kwargs["aux"]
        aux_features = self.aux_network(aux_x)
        
        features = torch.cat((features, aux_features), dim=-1)
        pred = self.classifier(features)

        return pred, dict()
