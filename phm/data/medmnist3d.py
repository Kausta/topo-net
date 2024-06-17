import os
from dataclasses import dataclass, field
import glob
from pathlib import Path
import h5py
import pandas as pd
import random
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms as transforms

import phm
from phm import register
from phm.util.base import Updateable
from phm.util.config import parse_structured
from phm.util.misc import get_rank
from phm.util.typing import *

import medmnist
from medmnist.dataset import MedMNIST3D

@dataclass
class MedMnist3dDataModuleConfig:
    dataset: str = "adrenalmnist3d" # in DATASET_MAPPING
    dataroot: str = str("/data2/medmnist/")
    download: bool = True
    mmap_mode: Optional[str] = None

    as_rgb: bool = True
    size: Optional[int] = None # None, 28, 64

    transform: bool = True

    train_data_percent: Optional[float] = None
    train_data_percent_seed: int = 0

    train_data_num: Optional[int] = None

    train_batch_size: int = 4
    val_batch_size: int = 4
    test_batch_size: int = 4
    
    train_workers: int = 8
    val_workers: int = 8
    test_workers: int = 8

    has_betti_data: bool = False
    betti_data_csv: str = ""
    betti_data_partial: bool = False
    betti_data_partial_csv: str = ""


DATASET_MAPPING = {
    "organmnist3d": ("OrganMNIST3D", medmnist.OrganMNIST3D),
    "nodulemnist3d": ("NoduleMNIST3D",medmnist.NoduleMNIST3D),
    "adrenalmnist3d": ("AdrenalMNIST3D",medmnist.AdrenalMNIST3D),
    "fracturemnist3d": ("FractureMNIST3D",medmnist.FractureMNIST3D),
    "vesselmnist3d": ("VesselMNIST3D",medmnist.VesselMNIST3D),
    "synapsemnist3d": ("SynapseMNIST3D",medmnist.SynapseMNIST3D),
}

class Transform3D:
    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)


class MedMnist3dDataset(Dataset, Updateable):
    def __init__(self, cfg: MedMnist3dDataModuleConfig, split: str):
        super().__init__()

        self.cfg = cfg

        if self.cfg.transform:
            if split == "train":
                data_transform = Transform3D(mul='random') 
            else:
                data_transform = Transform3D(mul='0.5')
        else:
            data_transform = Transform3D()
        
        assert cfg.dataset in DATASET_MAPPING
        dataset_name: str
        dataset_cls: Type[MedMNIST3D] 
        dataset_name, dataset_cls = DATASET_MAPPING[cfg.dataset]
        self.dataset: MedMNIST3D = dataset_cls(
            split=split,
            transform=data_transform,
            root=self.cfg.dataroot,
            download=self.cfg.download,
            as_rgb=self.cfg.as_rgb,
            size=self.cfg.size,
            mmap_mode=self.cfg.mmap_mode
        )

        self.betti_partial = False
        if split == "train" and self.cfg.betti_data_partial:
            self.betti_partial = True

        self.betti_data = None
        if self.cfg.has_betti_data:
            csv_base = self.cfg.betti_data_csv
            if self.betti_partial:
                csv_base = self.cfg.betti_data_partial_csv
            csv_file = csv_base.format(split=split, size=self.cfg.size, name=dataset_name)
            self.betti_data = pd.read_csv(csv_file).to_numpy()

        self.indices = None
        if split == "train" and self.cfg.train_data_percent is not None:
            assert 0.001 <= self.cfg.train_data_percent <= 0.999
            indices = list(range(len(self.dataset)))
            cutoff_point = int(self.cfg.train_data_percent * len(indices))

            rng = random.Random(self.cfg.train_data_percent_seed)
            rng.shuffle(indices)

            self.indices = indices[:cutoff_point]
        elif split == "train" and self.cfg.train_data_num is not None:
            indices = list(range(len(self.dataset)))
            cutoff_point = min(self.cfg.train_data_num, len(self.dataset))

            rng = random.Random(self.cfg.train_data_percent_seed)
            rng.shuffle(indices)

            self.indices = indices[:cutoff_point]

    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            orig_index = index
            index = self.indices[index]
        
        img: np.ndarray
        target: np.ndarray
        img, target = self.dataset[index]
       
        data = {
            "input": img,
            'target': target.astype(np.int64)
        }

        if self.betti_data is not None:
            betti_index = index
            if self.betti_partial:
                betti_index = orig_index
            data["betti"] = self.betti_data[betti_index].astype(np.float32)

        return data

@register("medmnist3d-datamodule")
class MedMnist3dDataModule(pl.LightningDataModule):
    cfg: MedMnist3dDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MedMnist3dDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MedMnist3dDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MedMnist3dDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MedMnist3dDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, shuffle=False, num_workers=0) -> DataLoader:
        return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, 
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.train_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, 
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.val_workers
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg.test_batch_size,
            num_workers=self.cfg.test_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg.test_batch_size,
            num_workers=self.cfg.test_workers
        )
