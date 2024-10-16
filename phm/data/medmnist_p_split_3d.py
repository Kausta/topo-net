import random

from phm.util.typing import *

import medmnist
from medmnist.dataset import MedMNIST3D

DATASET_MAPPING = {
    "adrenalmnist3d": ("AdrenalMNIST3D", medmnist.AdrenalMNIST3D),
    "fracturemnist3d": ("FractureMNIST3D",medmnist.FractureMNIST3D),
    "organmnist3d": ("OrganMNIST3D",medmnist.OrganMNIST3D),
    "nodulemnist3d": ("NoduleMNIST3D",medmnist.NoduleMNIST3D),
    "synapsemnist3d": ("SynapseMNIST3D",medmnist.SynapseMNIST3D),
    "vesselmnist3d": ("VesselMNIST3D",medmnist.VesselMNIST3D),
}

size = 64
dataroot: str = "/data2/medmnist/"
download = True
mmap_mode = None

as_rgb = True

train_data_percent_seed: int = 0

NUM_ELEMS = [50, 100, 200, 40, 80, 150, 250, 500]

def get_dataset_len(dataset_name: str) -> int:
    dataset_cls: Type[MedMNIST3D] 
    _, dataset_cls = DATASET_MAPPING[dataset_name]
    dataset: MedMNIST3D = dataset_cls(
        split="train",
        transform=None,
        root=dataroot,
        download=download,
        as_rgb=as_rgb,
        size=size,
        mmap_mode=mmap_mode
    )
    return len(dataset)

def get_train_indices(dataset_len, num_elems) -> List[int]:
    assert num_elems >= 0

    indices = list(range(dataset_len))
    
    cutoff_point = min(num_elems, dataset_len)

    rng = random.Random(train_data_percent_seed)
    rng.shuffle(indices)

    cut_indices = indices[:cutoff_point]

    return cut_indices


if __name__ == "__main__":
    out_root = "./splits/train"
    for d_name in DATASET_MAPPING.keys():
        d_len = get_dataset_len(d_name)
        for num_elems in NUM_ELEMS:
            out_file = f"{out_root}_{d_name}_{num_elems}N.txt"
            indices = get_train_indices(d_len, num_elems)
            with open(out_file, "w") as f:
                f.writelines(map(lambda x: f"{x}\n", indices))
