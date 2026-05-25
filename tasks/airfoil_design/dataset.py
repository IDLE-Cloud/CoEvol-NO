"""AirfRANS dataset loading utilities.

Copied from Airfoil-Design-AirfRANS/dataset/dataset.py with path cleanup.
Requires: torch_geometric, pyvista
"""

import torch
import numpy as np
import os.path as osp

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def Dataset(data_list, norm=False, sample=None, coef_norm=None, my_path='.'):
    dataset = []
    coef_norms = []

    for idx in range(len(data_list)):
        path = data_list[idx]
        if not path.endswith('.vtk'):
            continue
        try:
            import pyvista as pv
            mesh = pv.read(osp.join(my_path, path))
        except Exception:
            continue

        pos = torch.tensor(mesh.points, dtype=torch.float)
        x = torch.tensor(np.stack([mesh['Velocity'][:, 0], mesh['Velocity'][:, 1],
                                    mesh['Pressure'][:, 0], mesh['Turbulent_viscosity'][:, 0],
                                    mesh['Distance_function'][:, 0]], axis=-1), dtype=torch.float)
        y = torch.tensor(np.stack([mesh['Velocity'][:, 0], mesh['Velocity'][:, 1],
                                    mesh['Pressure'][:, 0], mesh['Turbulent_viscosity'][:, 0]], axis=-1), dtype=torch.float)
        surf = torch.tensor(mesh['Surface_flag'] == 1)

        pos_input = pos[:, :2]
        x_input = x[:, 2:]
        full_input = torch.cat([pos_input, x_input], dim=-1)

        data = Data(x=x_input, pos=pos_input, y=y, surf=surf)
        dataset.append(data)
        coef_norms.append(x_input)

    if norm and len(coef_norms) > 0:
        all_x = torch.cat(coef_norms, dim=0)
        coef_norm = (all_x.mean(0), all_x.std(0))

    if coef_norm is not None:
        for data in dataset:
            data.x = (data.x - coef_norm[0]) / coef_norm[1]

    return dataset, coef_norm
