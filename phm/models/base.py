from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import phm
from phm.util.base import BaseModule
from phm.util.typing import *


class BaseModel(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self) -> None:
        super().configure()
