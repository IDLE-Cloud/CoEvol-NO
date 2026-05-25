"""ShapeNetCar dataset loading utilities.

Adapted from Car-Design-ShapeNetCar/dataset/.
Requires: torch_geometric
"""

import torch
import numpy as np
import os.path as osp

try:
    from torch_geometric.data import Data, Dataset as PyGDataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


if HAS_PYG:
    class CarDataset(PyGDataset):
        def __init__(self, root, split='train', transform=None):
            super().__init__(root, transform=transform)
            self.split = split
            self.data_list = self._load_data()

        def _load_data(self):
            # Expects .pt files in root/split/
            split_dir = osp.join(self.root, self.split)
            if not osp.exists(split_dir):
                return []
            files = sorted([f for f in os.listdir(split_dir) if f.endswith('.pt')])
            return [torch.load(osp.join(split_dir, f), map_location='cpu') for f in files]

        def len(self):
            return len(self.data_list)

        def get(self, idx):
            return self.data_list[idx]
else:
    class CarDataset:
        def __init__(self, root, split='train'):
            self.root = root
            self.split = split
            self.data_list = []

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]
