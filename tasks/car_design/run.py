"""Car design task (ShapeNetCar dataset) runner.

Usage:
    python tasks/car_design/run.py --config configs/car/default.yaml --data_path /path/to/data

Requires: torch_geometric
"""

import argparse
import logging
import os
import os.path as osp
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm

from coevol_no import CoEvolNO, CoEvolNOLatent, CoEvolNOSequence
from tasks.car_design.dataset import CarDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    'CoEvolNO': CoEvolNO,
    'CoEvolNOLatent': CoEvolNOLatent,
    'CoEvolNOSequence': CoEvolNOSequence,
}


class CarModel(torch.nn.Module):
    """Wraps CoEvol models for ShapeNetCar (cfd_data, geom_data) interface."""

    def __init__(self, backbone, out_dim=4):
        super().__init__()
        self.backbone = backbone
        self.output_proj = nn.Sequential(
            nn.Linear(backbone.output_feature_dim, backbone.output_feature_dim * 4),
            nn.GELU(),
            nn.Linear(backbone.output_feature_dim * 4, out_dim),
        )

    def forward(self, data):
        x = data.x
        out_features = self.backbone(pos=x)
        return self.output_proj(out_features)


def train_epoch(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()
    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        targets = data.y
        loss_press = criterion_func(out[data.surf, -1], targets[data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


@torch.no_grad()
def test_epoch(device, model, test_loader):
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        targets = data.y
        loss_press = criterion_func(out[data.surf, -1], targets[data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


def main():
    parser = argparse.ArgumentParser('Car Design (ShapeNetCar)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='CoEvolNO')
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--save_path', type=str, default='results/car')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})

    train_dataset = CarDataset(args.data_path, split='train')
    test_dataset = CarDataset(args.data_path, split='test')

    backbone = MODEL_REGISTRY[args.model_name](
        coord_dim=model_cfg.get('coord_dim', 7),
        in_channels=model_cfg.get('in_channels', 0),
        depth=model_cfg.get('depth', 8),
        dim_lat=model_cfg.get('dim_lat', 256),
        dim_tok=model_cfg.get('dim_tok', 256),
        num_latents=model_cfg.get('num_latents', 128),
        num_heads=model_cfg.get('num_heads', 8),
        mlp_ratio=model_cfg.get('mlp_ratio', 1.0),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
    )
    model = CarModel(backbone, out_dim=4).to(device)

    lr = train_cfg.get('lr', 1e-3)
    epochs = train_cfg.get('epochs', 500)
    batch_size = train_cfg.get('batch_size', 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=(len(train_dataset) // batch_size + 1) * epochs,
        final_div_factor=1000.)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    reg = args.weight

    for epoch in tqdm(range(epochs)):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        loss_press, loss_velo = train_epoch(device, model, train_loader, optimizer, scheduler, reg=reg)
        train_loss = loss_velo + reg * loss_press
        del train_loader

        if epoch % 10 == 0 or epoch == epochs - 1:
            test_loader = DataLoader(test_dataset, batch_size=1)
            test_press, test_velo = test_epoch(device, model, test_loader)
            test_loss = test_velo + reg * test_press
            del test_loader
            logger.info(f"Epoch {epoch} train: {train_loss:.5f} test: {test_loss:.5f}")
        else:
            logger.info(f"Epoch {epoch} train: {train_loss:.5f}")

    torch.save(model, osp.join(args.save_path, 'model'))


if __name__ == '__main__':
    main()
