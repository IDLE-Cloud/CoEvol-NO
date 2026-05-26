# CoEvol-NO: Co-Evolution Neural Operator with Predictor-Corrector

English | **[中文](README.md)**

Official implementation of **CoEvol-NO** (ICML 2026).

## Overview

CoEvol-NO introduces a co-evolution framework for neural operators. The core idea is to maintain a set of learnable latent states S and a mesh sequence X, and update them bidirectionally through a Predictor-Corrector (PC) mechanism, achieving linear complexity O(NMC) where M << N.

### Three Evolution Paradigms

| Paradigm | Model Class | Description |
|----------|-------------|-------------|
| **Co-Evolution** | `CoEvolNO` | Bidirectional PC update of S and X (full model) |
| **State-Evol** | `CoEvolNOLatent` | Encode → Evolve → Decode (latent-only evolution) |
| **Coords-Evol** | `CoEvolNOSequence` | Per-layer local latents (no persistent state across layers) |

### Key Features

- Predictor-Corrector (PC) mechanism with exact Jacobian gradient via autograd
- Momentum + LayerScale for stable deep evolution training
- Linear complexity attention: O(NMC), M << N
- Supports both structured grids and irregular meshes
- Branch operator network wrapper for PDE tasks

### Predictor-Corrector (PC) Update Rule

```
Predictor:   S_pred = CrossAttn(Q=S, K=X, V=X)
Loss:        L_S = -<S, S_pred>  (dot product)  or  ||S - S_pred||² / 2  (L2)
Exact grad:  ∇_S L_S = (S - S_pred) - J_S^T (S - S_pred)   [exact]
             ∇_S L_S ≈ S - S_pred                              [first-order approx]
Momentum:    m_t = β · m_{t-1} + ∇_S L_S
Update:      S_t = S_{t-1} - η · m_t
```

### Ablation Variants

Four configurations controlled by `x_exact_update` and `s_approximate`:

| Variant | S Update | X Update | Description |
|---------|----------|----------|-------------|
| `dual_exact` | Exact gradient | Exact gradient | Full model |
| `s_exact` | Exact gradient | First-order approx | Default CoEvol-NO |
| `x_exact` | First-order approx | Exact gradient | Ablation study |
| `first_order` | First-order approx | First-order approx | Ablation study |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### PDE Standard Benchmarks

```bash
# Darcy Flow
bash scripts/run_darcy.sh /path/to/AI4PDE 0

# Navier-Stokes
bash scripts/run_ns.sh /path/to/NavierStokes_V1e-5_N1200_T20.mat 0

# Elasticity
bash scripts/run_elasticity.sh /path/to/AI4PDE 0

# Pipe Flow
bash scripts/run_pipe.sh /path/to/AI4PDE/pipe 0

# Airfoil (PDE benchmark)
bash scripts/run_airfoil.sh /path/to/AI4PDE/airfoil/naca 0
```

### Industrial Tasks

```bash
# Airfoil Design (AirfRANS dataset)
python tasks/airfoil_design/run.py --config configs/airfoil/full.yaml --data_path /path/to/Dataset

# Car Design (ShapeNetCar dataset)
python tasks/car_design/run.py --config configs/car/default.yaml --data_path /path/to/ShapeNetCar
```

### Model API

```python
from coevol_no import CoEvolNO, CoEvolNOLatent, CoEvolNOSequence, OperatorNet

# Primary model (structured grid input)
model = CoEvolNO(
    img_size=(64, 64), in_channels=1, coord_dim=2,
    depth=8, num_latents=128, dim_lat=128, dim_tok=128,
    num_heads=8, mlp_ratio=1.0, drop_path_rate=0.1,
)

# Irregular mesh input (Airfoil, Car tasks)
model = CoEvolNO(
    in_channels=0, coord_dim=7,  # use coord_dim=3 for car task
    depth=8, num_latents=32, dim_lat=256, dim_tok=256,
)

# Create ablation variants via factory
from coevol_no.models import CoEvolNOVariant
model = CoEvolNOVariant.create('dual_exact', coord_dim=2, depth=8)

# Branch wrapper for PDE tasks
branch = CoEvolNO(in_channels=1, coord_dim=2, depth=8)
model = OperatorNet(branch=branch, num_basis=128, resolution=64)
```

## Analytical Gradients

CoEvol-NO provides analytical (closed-form) PC gradient computations that replace `torch.autograd.grad(create_graph=True)` by explicitly deriving backward pass formulas, avoiding second-order graph construction overhead.

### S/X Analytical Gradients

```python
from coevol_no.analytical import (
    compute_s_gradient_analytical,
    compute_x_gradient_analytical,
)

# Drop-in replacement for attn._compute_s_gradient()
grad_S, S_pred = compute_s_gradient_analytical(attn, x_lat, x_tok)

# Drop-in replacement for attn._compute_x_gradient()
grad_X, X_pred = compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S)
```

### PCFFN: Predictor-Corrector FFN

The standard residual FFN can be replaced with a PC update. Enable via `use_pc_ffn=True`; analytical gradient is used by default.

```python
from coevol_no import CoEvolNO

# Enable PCFFN (analytical gradient by default)
model = CoEvolNO(
    img_size=(64, 64), in_channels=1, depth=8,
    use_pc_ffn=True,              # enable PCFFN
    pc_ffn_loss_type='dot product',
    pc_ffn_momentum_beta=0.9,
    pc_ffn_analytical=True,       # analytical gradient (default)
)

# Standalone PCFFN usage
from coevol_no import PCFFN
pcffn = PCFFN(dim=128, hidden_dim=128, analytical=True)
x_out, momentum = pcffn(x, momentum)
```

### Equivalence Verification

```bash
python tests/test_analytical.py   # S/X gradient equivalence (17 tests, all pass)
python tests/test_pcffn.py        # PCFFN equivalence (all pass)
```

### Speed Benchmark

CPU results (B=4, N=1024, C=128):

| Operation | Autograd | Analytical | Speedup |
|-----------|----------|------------|---------|
| S gradient (exact) | 155ms | 51ms | **3.0x** |
| X gradient (exact) | 228ms | 89ms | **2.6x** |
| PCFFN (exact) | 252ms | 174ms | **1.5x** |
| X gradient (first-order) | 12ms | 13ms | ~1.0x |

```bash
python tests/benchmark_analytical.py
```

## Project Structure

```
CoEvol-NO/
├── coevol_no/              # Core library
│   ├── models.py           # CoEvolNO, CoEvolNOLatent, CoEvolNOSequence
│   ├── attention.py        # DualExactStateAttention (core PC module)
│   ├── blocks.py           # DualExactBlock, PCFFN, LatentBlock, SequenceBlock
│   ├── layers.py           # LayerScale, Newton-Schulz orthogonalization
│   ├── analytical.py       # Analytical gradients (explicit backward for S/X/FFN)
│   └── wrapper.py          # OperatorNet (Branch wrapper)
├── baselines/              # Baseline models
│   ├── transolver/         # Transolver (2D and irregular mesh variants)
│   ├── pc_transolver/      # PC-Transolver (Transolver + PC mechanism)
│   └── operators/          # PerceiverIO, ISAB, ClusterAttention, LNO
├── utils/                  # Utilities
│   ├── loss.py             # TestLoss (relative Lp error)
│   ├── normalizer.py       # UnitTransformer, UnitGaussianNormalizer
│   └── visualization.py    # TensorVisualizer (PCA / heatmap)
├── tasks/                  # Task-specific training code
│   ├── pde_benchmarks/     # Darcy, NS, Elasticity, Airfoil, Pipe
│   ├── airfoil_design/     # AirfRANS industrial airfoil task
│   └── car_design/         # ShapeNetCar industrial car task
├── configs/                # Per-task YAML configuration files
└── scripts/                # Shell launch scripts
```

## Baselines

Included baselines for comparison:

| Baseline | Description |
|----------|-------------|
| **Transolver** | Physics-Attention slice attention (2D + irregular mesh variants) |
| **PC-Transolver** | Transolver + PC mechanism (Attention/FFN PC independently switchable) |
| **PerceiverIO** | Perceiver IO (with optional cross-attention) |
| **ISAB** | Set Transformer (Induced Set Attention Block) |
| **ClusterAttention** | LSH-clustered attention |
| **LNO** | Latent Neural Operator |

## Requirements

Core dependencies:
- Python >= 3.8
- PyTorch >= 2.0
- einops, timm, numpy, scipy, h5py, tqdm, matplotlib, pyyaml, scikit-learn

Optional (for industrial tasks):
- `torch_geometric` (Airfoil, Car tasks)
- `pyvista` (Airfoil task data loading)

## Citation

```bibtex
@inproceedings{coevolno2026,
  title={CoEvol-NO: Co-Evolution Neural Operator with Predictor-Corrector},
  author={},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgments

The Transolver baseline is adapted from [Transolver](https://github.com/thuml/Transolver).
