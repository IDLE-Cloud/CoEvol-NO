# CoEvol-NO: 基于预测-校正机制的多演算神经算子

**[English](README_EN.md)** | 中文

**CoEvol-NO** (ICML 2026) 官方实现。

## 概述

CoEvol-NO 提出了一种多演算（Co-Evolution）框架用于神经算子。核心思想是：维护一组可学习的隐状态 S 和网格序列 X，通过预测-校正（Predictor-Corrector, PC）机制进行双向演化更新，实现线性复杂度 O(NMC)，其中 M << N。

### 三种演化范式

| 范式 | 模型类 | 描述 |
|------|--------|------|
| **Co-Evolution** | `CoEvolNO` | S 和 X 双向 PC 更新（完整模型） |
| **State-Evol** | `CoEvolNOLatent` | 编码 → 演化 → 解码（仅隐状态演化） |
| **Coords-Evol** | `CoEvolNOSequence` | 每层独立局部隐状态（无跨层共享状态） |

### 核心特性

- 预测-校正（PC）机制：通过 autograd 计算精确 Jacobian 梯度
- 动量 + LayerScale：稳定深层演化训练
- 线性复杂度注意力：O(NMC)，M << N
- 同时支持结构化网格和非结构化网格输入
- Branch 算子网络封装，用于 PDE 任务

### 预测-校正（PC）更新规则

```
预测器:   S_pred = CrossAttn(Q=S, K=X, V=X)
校正损失: L_S = -<S, S_pred>  (点积)  或  ||S - S_pred||² / 2  (L2)
精确梯度: ∇_S L_S = (S - S_pred) - J_S^T (S - S_pred)   [精确]
          ∇_S L_S ≈ S - S_pred                              [一阶近似]
动量更新: m_t = β · m_{t-1} + ∇_S L_S
状态更新: S_t = S_{t-1} - η · m_t
```

### 消融变体

通过 `x_exact_update` 和 `s_approximate` 控制四种配置：

| 变体 | S 更新方式 | X 更新方式 | 说明 |
|------|-----------|-----------|------|
| `dual_exact` | 精确梯度 | 精确梯度 | 完整模型 |
| `s_exact` | 精确梯度 | 一阶近似 | 默认 CoEvol-NO |
| `x_exact` | 一阶近似 | 精确梯度 | 消融实验 |
| `first_order` | 一阶近似 | 一阶近似 | 消融实验 |

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### PDE 标准基准实验

```bash
# Darcy Flow（达西流）
bash scripts/run_darcy.sh /path/to/AI4PDE 0

# Navier-Stokes（纳维-斯托克斯方程）
bash scripts/run_ns.sh /path/to/NavierStokes_V1e-5_N1200_T20.mat 0

# Elasticity（弹性力学）
bash scripts/run_elasticity.sh /path/to/AI4PDE 0

# Pipe Flow（管道流）
bash scripts/run_pipe.sh /path/to/AI4PDE/pipe 0

# Airfoil（翼型绕流，PDE 基准）
bash scripts/run_airfoil.sh /path/to/AI4PDE/airfoil/naca 0
```

### 工业任务

```bash
# 翼型设计（AirfRANS 数据集）
python tasks/airfoil_design/run.py --config configs/airfoil/full.yaml --data_path /path/to/Dataset

# 汽车设计（ShapeNetCar 数据集）
python tasks/car_design/run.py --config configs/car/default.yaml --data_path /path/to/ShapeNetCar
```

### 模型 API

```python
from coevol_no import CoEvolNO, CoEvolNOLatent, CoEvolNOSequence, OperatorNet

# 主模型（结构化网格输入）
model = CoEvolNO(
    img_size=(64, 64), in_channels=1, coord_dim=2,
    depth=8, num_latents=128, dim_lat=128, dim_tok=128,
    num_heads=8, mlp_ratio=1.0, drop_path_rate=0.1,
)

# 非结构化网格输入（翼型、汽车等任务）
model = CoEvolNO(
    in_channels=0, coord_dim=7,  # 汽车任务用 coord_dim=3
    depth=8, num_latents=32, dim_lat=256, dim_tok=256,
)

# 通过工厂类创建消融变体
from coevol_no.models import CoEvolNOVariant
model = CoEvolNOVariant.create('dual_exact', coord_dim=2, depth=8)

# Branch 封装（PDE 任务）
branch = CoEvolNO(in_channels=1, coord_dim=2, depth=8)
model = OperatorNet(branch=branch, num_basis=128, resolution=64)
```

## 解析梯度

CoEvol-NO 提供解析（analytical）版本的 PC 梯度计算，通过显式推导反向传播公式避免 `torch.autograd.grad(create_graph=True)` 的二阶图构建开销。

### S/X 解析梯度

```python
from coevol_no.analytical import (
    compute_s_gradient_analytical,
    compute_x_gradient_analytical,
)

# 替代 attn._compute_s_gradient() 的解析版本
grad_S, S_pred = compute_s_gradient_analytical(attn, x_lat, x_tok)

# 替代 attn._compute_x_gradient() 的解析版本
grad_X, X_pred = compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S)
```

### PCFFN：Predictor-Corrector FFN

标准 FFN 的残差连接可替换为 PC 更新。通过 `use_pc_ffn=True` 启用，默认使用解析梯度（`analytical=True`）。

```python
from coevol_no import CoEvolNO

# 启用 PCFFN（默认使用解析梯度）
model = CoEvolNO(
    img_size=(64, 64), in_channels=1, depth=8,
    use_pc_ffn=True,              # 启用 PCFFN
    pc_ffn_loss_type='dot product', # PC 损失类型
    pc_ffn_momentum_beta=0.9,     # 动量系数
    pc_ffn_analytical=True,       # 使用解析梯度（默认）
)

# 也可单独使用 PCFFN
from coevol_no import PCFFN
pcffn = PCFFN(dim=128, hidden_dim=128, analytical=True)
x_out, momentum = pcffn(x, momentum)
```

### 等价性验证

```bash
python test_analytical.py   # S/X 梯度等价性（17 项全通过）
python test_pcffn.py        # PCFFN 等价性（全部通过）
```

### 速度基准

CPU 测试结果（B=4, N=1024, C=128）：

| 操作 | Autograd | 解析版本 | 加速比 |
|------|----------|----------|--------|
| S 梯度（精确） | 155ms | 51ms | **3.0x** |
| X 梯度（精确） | 228ms | 89ms | **2.6x** |
| PCFFN（精确） | 252ms | 174ms | **1.5x** |
| X 梯度（一阶近似） | 12ms | 13ms | ~1.0x |

```bash
python benchmark_analytical.py
```

## 项目结构

```
CoEvol-NO/
├── coevol_no/              # 核心库
│   ├── models.py           # CoEvolNO, CoEvolNOLatent, CoEvolNOSequence
│   ├── attention.py        # StateAttention, DualExactStateAttention（核心 PC 模块）
│   ├── blocks.py           # DualExactBlock, PCFFN, LatentBlock, SequenceBlock
│   ├── layers.py           # LayerScale, Newton-Schulz 正交化
│   ├── analytical.py       # 解析梯度（S/X/FFN 的显式反向公式）
│   └── wrapper.py          # OperatorNet（Branch 封装）
├── baselines/              # 基线模型
│   ├── transolver/         # Transolver（2D 和非结构化网格版本）
│   └── operators/          # PerceiverIO, ISAB, ClusterAttention, LNO
├── utils/                  # 工具函数
│   ├── loss.py             # TestLoss（相对 Lp 误差）
│   ├── normalizer.py       # UnitTransformer, UnitGaussianNormalizer
│   └── visualization.py    # TensorVisualizer（PCA / 热力图）
├── tasks/                  # 任务训练代码
│   ├── pde_benchmarks/     # Darcy, NS, Elasticity, Airfoil, Pipe
│   ├── airfoil_design/     # AirfRANS 工业翼型设计任务
│   └── car_design/         # ShapeNetCar 工业汽车设计任务
├── configs/                # 每个任务的 YAML 配置文件
└── scripts/                # Shell 启动脚本
```

## 基线模型

包含以下基线模型用于对比实验：

| 基线 | 说明 |
|------|------|
| **Transolver** | Physics-Attention 切片注意力（2D + 非结构化网格） |
| **PerceiverIO** | 感知器 IO（可选交叉注意力） |
| **ISAB** | 集合 Transformer（Induced Set Attention） |
| **ClusterAttention** | LSH 聚类注意力 |
| **LNO** | 潜在神经算子（Latent Neural Operator） |

## 依赖

核心依赖：
- Python >= 3.8
- PyTorch >= 2.0
- einops, timm, numpy, scipy, h5py, tqdm, matplotlib, pyyaml, scikit-learn

工业任务额外依赖：
- `torch_geometric`（翼型、汽车任务）
- `pyvista`（翼型任务数据加载）

## 引用

```bibtex
@inproceedings{coevolno2026,
  title={CoEvol-NO: Co-Evolution Neural Operator with Predictor-Corrector},
  author={},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## 致谢

Transolver 基线代码改编自 [Transolver](https://github.com/thuml/Transolver)。
