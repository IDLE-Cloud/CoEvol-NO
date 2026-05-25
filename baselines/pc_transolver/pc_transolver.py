"""PC-Transolver: Transolver + Predictor-Corrector Framework.

Applies PC updates to Transolver's Attention and/or FFN sub-blocks.

Original Transolver Block:
    fx = Attn(LN(fx)) + fx    (residual)
    fx = MLP(LN(fx)) + fx     (residual)

PC-Transolver Block:
    fx = LN(fx) - eta * momentum(grad)   (PC update)

Configurations:
    use_pc_attn=True,  use_pc_ffn=False  -> PC on Attention only
    use_pc_attn=False, use_pc_ffn=True   -> PC on FFN only
    use_pc_attn=True,  use_pc_ffn=True   -> PC on both
    use_pc_attn=False, use_pc_ffn=False  -> Original Transolver (baseline)
"""

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_

from baselines.transolver.physics_attention import Physics_Attention_Structured_Mesh_2D
from baselines.transolver.embedding import timestep_embedding

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act = ACTIVATION[act] if act in ACTIVATION else act
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class TransolverBlockPC(nn.Module):
    """Transolver block with optional PC updates on Attention and/or FFN."""

    def __init__(self, num_heads, hidden_dim, dropout, act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32, H=85, W=85,
                 use_pc_attn=False, use_pc_ffn=False,
                 loss_type='dot product', momentum_beta=0.9, init_values=1e-5):
        super().__init__()
        self.last_layer = last_layer
        self.use_pc_attn = use_pc_attn
        self.use_pc_ffn = use_pc_ffn
        self.loss_type = loss_type
        self.momentum_beta = momentum_beta

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, H=H, W=W,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim,
                        n_layers=0, res=False, act=act)

        self.eta_attn = nn.Parameter(init_values * torch.ones(hidden_dim))
        self.eta_ffn = nn.Parameter(init_values * torch.ones(hidden_dim))

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def _pc_update(self, x, predictor, momentum, eta):
        if momentum is None:
            momentum = torch.zeros_like(x)

        with torch.enable_grad():
            X_param = x.clone().requires_grad_(True)
            pred = predictor(X_param)

            if self.loss_type == 'l2':
                loss = torch.sum((X_param - pred) ** 2) / 2.0
            elif self.loss_type == 'dot product':
                loss = -torch.einsum('bnc,bnc->b', X_param, pred).sum()
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            grad = torch.autograd.grad(loss, X_param, create_graph=True)[0]

        momentum = self.momentum_beta * momentum + grad
        return x - eta * momentum, momentum

    def forward(self, fx, momentum_attn=None, momentum_ffn=None):
        x_norm = self.ln_1(fx)
        if self.use_pc_attn:
            fx, momentum_attn = self._pc_update(x_norm, self.Attn, momentum_attn, self.eta_attn)
        else:
            fx = self.Attn(x_norm) + fx

        x_norm = self.ln_2(fx)
        if self.use_pc_ffn:
            fx, momentum_ffn = self._pc_update(x_norm, self.mlp, momentum_ffn, self.eta_ffn)
        else:
            fx = self.mlp(x_norm) + fx

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))

        return fx, momentum_attn, momentum_ffn


class TransolverPC(nn.Module):
    """Transolver with optional Predictor-Corrector updates.

    Same interface as Transolver2D: forward(x, fx, T) -> fx
    """

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, Time_Input=False, act='gelu', mlp_ratio=1,
                 fun_dim=1, out_dim=1, slice_num=32, ref=8,
                 unified_pos=False, H=85, W=85,
                 use_pc_attn=False, use_pc_ffn=False,
                 loss_type='dot product', momentum_beta=0.9,
                 init_values=1e-5):
        super().__init__()
        self.__name__ = 'Transolver_PC_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref * ref, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)

        if Time_Input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([
            TransolverBlockPC(
                num_heads=n_head, hidden_dim=n_hidden,
                dropout=dropout, act=act, mlp_ratio=mlp_ratio,
                out_dim=out_dim, slice_num=slice_num, H=H, W=W,
                last_layer=(i == n_layers - 1),
                use_pc_attn=use_pc_attn, use_pc_ffn=use_pc_ffn,
                loss_type=loss_type, momentum_beta=momentum_beta,
                init_values=init_values,
            )
            for i in range(n_layers)
        ])

        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)

        pos = torch.sqrt(
            torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)
        ).reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        # Handle OperatorNet convention: when x is None, fx holds positional input
        if x is None and fx is not None:
            x, fx = fx, None

        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(
                x.shape[0], self.H * self.W, self.ref * self.ref)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        momentum_attn = None
        momentum_ffn = None

        for block in self.blocks:
            if block.last_layer:
                fx = block(fx, momentum_attn, momentum_ffn)
            else:
                fx, momentum_attn, momentum_ffn = block(fx, momentum_attn, momentum_ffn)

        return fx
