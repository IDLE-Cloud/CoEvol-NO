"""Unified entry point for PDE benchmark experiments.

Usage:
    python tasks/pde_benchmarks/run.py --config configs/pde/pipe.yaml
    python tasks/pde_benchmarks/run.py --config configs/pde/darcy.yaml --data_path /path/to/data
"""

import argparse
import logging
import yaml

import torch

from coevol_no import CoEvolNO, CoEvolNOLatent, CoEvolNOSequence, OperatorNet
from tasks.pde_benchmarks.train import (
    set_seed, load_darcy, load_ns, load_elasticity, load_airfoil_pde, load_pipe, train_pde
)


MODEL_REGISTRY = {
    'CoEvolNO': CoEvolNO,
    'CoEvolNOLatent': CoEvolNOLatent,
    'CoEvolNOSequence': CoEvolNOSequence,
    'TransolverPC': None,  # lazy import
}


def build_model(cfg, data_info):
    model_name = cfg.get('model', {}).get('name', 'CoEvolNO')
    variant = cfg.get('model', {}).get('variant', None)
    model_cfg = cfg.get('model', {})
    res = data_info.get('resolution', 64)

    # --- PC-Transolver baseline ---
    if model_name == 'TransolverPC':
        from baselines.pc_transolver import TransolverPC
        from coevol_no.wrapper import OperatorNet as _OperatorNet

        n_hidden = model_cfg.get('dim_tok', 128)
        branch = TransolverPC(
            space_dim=data_info.get('coord_dim', 2),
            n_layers=model_cfg.get('depth', 8),
            n_hidden=n_hidden,
            n_head=model_cfg.get('num_heads', 8),
            mlp_ratio=model_cfg.get('mlp_ratio', 1.0),
            fun_dim=data_info.get('in_channels', 1),
            out_dim=n_hidden,
            slice_num=model_cfg.get('num_latents', 32),
            H=res, W=res,
            use_pc_attn=model_cfg.get('use_pc_attn', True),
            use_pc_ffn=model_cfg.get('use_pc_ffn', False),
            loss_type=model_cfg.get('s_loss_type', 'dot product'),
            momentum_beta=model_cfg.get('s_momentum_beta', 0.9),
            init_values=1e-5,
        )
        return _OperatorNet(branch, num_basis=n_hidden, resolution=res)

    # --- CoEvol-NO family ---
    common_kwargs = {
        'img_size': (res, res),
        'in_channels': data_info.get('in_channels', 1),
        'coord_dim': data_info.get('coord_dim', 2),
        'depth': model_cfg.get('depth', 8),
        'dim_lat': model_cfg.get('dim_lat', 128),
        'dim_tok': model_cfg.get('dim_tok', 128),
        'num_latents': model_cfg.get('num_latents', 128),
        'num_heads': model_cfg.get('num_heads', 8),
        'mlp_ratio': model_cfg.get('mlp_ratio', 1.0),
        'drop_path_rate': model_cfg.get('drop_path_rate', 0.1),
        # PC attention
        'x_exact_update': model_cfg.get('x_exact_update', False),
        's_approximate': model_cfg.get('s_approximate', False),
        's_loss_type': model_cfg.get('s_loss_type', 'dot product'),
        's_momentum_beta': model_cfg.get('s_momentum_beta', 0.9),
        'x_loss_type': model_cfg.get('x_loss_type', 'dot product'),
        'x_momentum_beta': model_cfg.get('x_momentum_beta', 0.0),
        # PCFFN
        'use_pc_ffn': model_cfg.get('use_pc_ffn', False),
        'pc_ffn_loss_type': model_cfg.get('pc_ffn_loss_type', 'dot product'),
        'pc_ffn_momentum_beta': model_cfg.get('pc_ffn_momentum_beta', 0.9),
        'pc_ffn_analytical': model_cfg.get('pc_ffn_analytical', True),
        # Positional / misc
        'final_norm': model_cfg.get('final_norm', True),
        'unified_pos': model_cfg.get('unified_pos', False),
        'ref': model_cfg.get('ref', 8),
    }

    if model_name == 'CoEvolNO' and variant:
        from coevol_no.models import CoEvolNOVariant
        branch = CoEvolNOVariant.create(variant, **common_kwargs)
    else:
        ModelClass = MODEL_REGISTRY[model_name]
        branch = ModelClass(**common_kwargs)

    use_wrapper = model_cfg.get('use_wrapper', True)
    if use_wrapper:
        model = OperatorNet(
            branch=branch,
            num_basis=model_cfg.get('dim_tok', 128),
            resolution=res,
        )
    else:
        model = branch
    return model


LOADERS = {
    'darcy': load_darcy,
    'ns': load_ns,
    'elasticity': load_elasticity,
    'airfoil': load_airfoil_pde,
    'pipe': load_pipe,
}


def main():
    parser = argparse.ArgumentParser('PDE Benchmark Runner')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--data_path', type=str, default=None, help='Override data path')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get('data', {})
    task = data_cfg.get('task', 'darcy')
    data_path = args.data_path or data_cfg.get('data_path', '')
    ntrain = data_cfg.get('ntrain', 1000)
    ntest = data_cfg.get('ntest', 200)
    downsample = data_cfg.get('downsample', 1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler()])

    set_seed(data_cfg.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    loader_kwargs = {'data_path': data_path, 'ntrain': ntrain, 'ntest': ntest}
    if task in ('darcy', 'ns'):
        loader_kwargs['downsample'] = downsample
    if task == 'airfoil':
        loader_kwargs['downsamplex'] = data_cfg.get('downsamplex', 1)
        loader_kwargs['downsampley'] = data_cfg.get('downsampley', 1)
    if task == 'pipe':
        loader_kwargs['downsamplex'] = data_cfg.get('downsamplex', 1)
        loader_kwargs['downsampley'] = data_cfg.get('downsampley', 1)

    data_info = LOADERS[task](**loader_kwargs)
    model = build_model(cfg, data_info)
    print(model)
    print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_cfg = cfg.get('training', {})
    train_cfg['save_name'] = data_cfg.get('save_name', task)
    train_pde(model, data_info, train_cfg)


if __name__ == '__main__':
    main()
