"""Airfoil design task (AirfRANS dataset) runner.

Usage:
    python tasks/airfoil_design/run.py --config configs/airfoil/full.yaml --data_path /path/to/Dataset

Requires: torch_geometric, pyvista
"""

import argparse
import json
import logging
import os
import os.path as osp
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm

from coevol_no import CoEvolNO, CoEvolNOLatent, CoEvolNOSequence
from tasks.airfoil_design.dataset import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    'CoEvolNO': CoEvolNO,
    'CoEvolNOLatent': CoEvolNOLatent,
    'CoEvolNOSequence': CoEvolNOSequence,
}


class AirfoilModel(torch.nn.Module):
    """Wraps CoEvol models for AirfRANS forward(data) interface."""

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


def train_epoch(device, model, train_loader, optimizer, scheduler, criterion='MSE_weighted', reg=1):
    model.train()
    avg_loss = 0
    avg_loss_surf = 0
    avg_loss_vol = 0
    iters = 0
    for data in train_loader:
        data = data.clone().to(device)
        optimizer.zero_grad()
        out = model(data)
        targets = data.y
        loss_fn = nn.MSELoss(reduction='none')
        loss_per_var = loss_fn(out, targets).mean(dim=0)
        loss_surf = loss_fn(out[data.surf, :], targets[data.surf, :]).mean()
        loss_vol = loss_fn(out[~data.surf, :], targets[~data.surf, :]).mean()
        if criterion == 'MSE_weighted':
            (loss_vol + reg * loss_surf).backward()
        else:
            loss_per_var.mean().backward()
        optimizer.step()
        scheduler.step()
        avg_loss += loss_per_var.mean().item()
        avg_loss_surf += loss_surf.item()
        avg_loss_vol += loss_vol.item()
        iters += 1
    return avg_loss / iters, avg_loss_surf / iters, avg_loss_vol / iters


@torch.no_grad()
def test_epoch(device, model, test_loader):
    model.eval()
    avg_loss = 0
    avg_loss_surf = 0
    avg_loss_vol = 0
    iters = 0
    for data in test_loader:
        data = data.clone().to(device)
        out = model(data)
        targets = data.y
        loss_fn = nn.MSELoss(reduction='none')
        loss_surf = loss_fn(out[data.surf, :], targets[data.surf, :]).mean()
        loss_vol = loss_fn(out[~data.surf, :], targets[~data.surf, :]).mean()
        avg_loss += loss_fn(out, targets).mean().item()
        avg_loss_surf += loss_surf.item()
        avg_loss_vol += loss_vol.item()
        iters += 1
    return avg_loss / iters, avg_loss_surf / iters, avg_loss_vol / iters


def main():
    parser = argparse.ArgumentParser('Airfoil Design (AirfRANS)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='CoEvolNO')
    parser.add_argument('--task', type=str, default='full', choices=['full', 'scarce'])
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, default='results/airfoil')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(osp.join(args.data_path, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    manifest_train = manifest[args.task + '_train']
    test_dataset_list = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
    n = int(.1 * len(manifest_train))
    train_dataset_list = manifest_train[:-n]
    val_dataset_list = manifest_train[-n:]

    train_dataset, coef_norm = Dataset(train_dataset_list, norm=True, sample=None, my_path=args.data_path)
    val_dataset = Dataset(val_dataset_list, sample=None, coef_norm=coef_norm, my_path=args.data_path)

    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})

    backbone = MODEL_REGISTRY[args.model_name](
        coord_dim=model_cfg.get('coord_dim', 7),
        in_channels=model_cfg.get('in_channels', 0),
        depth=model_cfg.get('depth', 8),
        dim_lat=model_cfg.get('dim_lat', 256),
        dim_tok=model_cfg.get('dim_tok', 256),
        num_latents=model_cfg.get('num_latents', 32),
        num_heads=model_cfg.get('num_heads', 8),
        mlp_ratio=model_cfg.get('mlp_ratio', 1.0),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
        # PC attention
        x_exact_update=model_cfg.get('x_exact_update', False),
        s_approximate=model_cfg.get('s_approximate', False),
        s_loss_type=model_cfg.get('s_loss_type', 'dot product'),
        s_momentum_beta=model_cfg.get('s_momentum_beta', 0.9),
        x_loss_type=model_cfg.get('x_loss_type', 'dot product'),
        x_momentum_beta=model_cfg.get('x_momentum_beta', 0.0),
        # PCFFN
        use_pc_ffn=model_cfg.get('use_pc_ffn', False),
        pc_ffn_loss_type=model_cfg.get('pc_ffn_loss_type', 'dot product'),
        pc_ffn_momentum_beta=model_cfg.get('pc_ffn_momentum_beta', 0.9),
        pc_ffn_analytical=model_cfg.get('pc_ffn_analytical', True),
        # Positional / misc
        final_norm=model_cfg.get('final_norm', True),
        unified_pos=model_cfg.get('unified_pos', False),
        ref=model_cfg.get('ref', 8),
    )
    model = AirfoilModel(backbone, out_dim=4).to(device)

    lr = train_cfg.get('lr', 1e-3)
    epochs = train_cfg.get('epochs', 500)
    batch_size = train_cfg.get('batch_size', 4)
    subsampling = train_cfg.get('subsampling', 5000)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=(len(train_dataset) // batch_size + 1) * epochs)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    val_iter = train_cfg.get('val_iter', 10)

    for epoch in tqdm(range(epochs)):
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), subsampling)
            idx = torch.tensor(idx)
            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]
            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size=batch_size, shuffle=True)
        del train_dataset_sampled

        train_loss, loss_surf, loss_vol = train_epoch(
            device, model, train_loader, optimizer, scheduler,
            criterion='MSE_weighted', reg=args.weight)
        del train_loader

        if epoch % val_iter == val_iter - 1:
            val_loss, val_surf, val_vol = test_epoch(device, model, DataLoader(val_dataset, batch_size=1))
            logger.info(f"Epoch {epoch} train: {train_loss:.5f} val_vol: {val_vol:.5f} val_surf: {val_surf:.5f}")
        else:
            logger.info(f"Epoch {epoch} train: {train_loss:.5f} surf: {loss_surf:.5f} vol: {loss_vol:.5f}")

    torch.save(model, osp.join(args.save_path, 'model'))


if __name__ == '__main__':
    main()
