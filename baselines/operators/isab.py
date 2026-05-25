import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.eta = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.eta


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, mlp_ratio=4, drop_path=0., init_values=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.dim_V = dim_V

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.ln_q = nn.LayerNorm(dim_V)
        self.ln_k = nn.LayerNorm(dim_V)
        self.ln_post_attn = nn.LayerNorm(dim_V)

        self.ls_tok1 = LayerScale(dim_V, init_values)
        self.drop_path_tok1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_tok = nn.Sequential(
            nn.Linear(dim_V, int(dim_V * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim_V * mlp_ratio), dim_V)
        )
        self.ls_tok2 = LayerScale(dim_V, init_values)
        self.drop_path_tok2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, Q, K):
        q_proj = self.fc_q(Q)
        k_proj = self.fc_k(K)
        v_proj = self.fc_v(K)

        q = self.ln_q(q_proj)
        k = self.ln_k(k_proj)
        v = self.ln_k(v_proj)

        B, N, _ = q.shape
        _, M, _ = k.shape
        h = self.num_heads

        q = q.view(B, N, h, -1).transpose(1, 2)
        k = k.view(B, M, h, -1).transpose(1, 2)
        v = v.view(B, M, h, -1).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else 0.1)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.dim_V)
        delta_tok_ca = self.fc_o(attn_out)

        q_proj = q_proj + self.drop_path_tok1(self.ls_tok1(delta_tok_ca))

        delta_mlp = self.mlp_tok(self.ln_post_attn(q_proj))
        q_proj = q_proj + self.drop_path_tok2(self.ls_tok2(delta_mlp))
        return q_proj


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.1)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.1)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class ISABAttentionBranch(nn.Module):
    def __init__(self, img_size=(64, 64), in_channels=1, coord_dim=2,
                 depth=6, num_latents=64, dim_lat=256, dim_tok=128,
                 num_heads=8, mlp_ratio=4.0, drop_path_rate=0.1):
        super().__init__()
        self.size = img_size[0]
        self.img_size = img_size
        self.in_channels = in_channels
        self.coord_dim = coord_dim
        self.output_feature_dim = dim_tok

        self.preprocess = nn.Linear(in_channels + coord_dim, dim_tok)
        self.latents = nn.Parameter(torch.zeros(1, num_latents, dim_tok))
        trunc_normal_(self.latents, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ISAB(dim_out=dim_tok, dim_in=dim_tok, num_heads=num_heads, num_inds=num_latents, ln=True)
            for i in range(depth)
        ])
        self.placeholder = nn.Parameter((1 / (dim_tok)) * torch.rand(dim_tok, dtype=torch.float))
        self.norm = nn.LayerNorm(dim_tok)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, shape, device):
        B, H, W, _ = shape
        gridx = torch.tensor(np.linspace(0, 1, H), dtype=torch.float, device=device).reshape(1, H, 1, 1).repeat([B, 1, W, 1])
        gridy = torch.tensor(np.linspace(0, 1, W), dtype=torch.float, device=device).reshape(1, 1, W, 1).repeat([B, H, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, u=None, pos=None, t=None):
        if u is not None:
            x = u.unsqueeze(-1)
            u = x.reshape(x.shape[0], self.size, self.size, self.in_channels)
            B, H, W, C = u.shape
            grid = self.get_grid((B, H, W, self.coord_dim), u.device)
            x_with_coords = torch.cat([u, grid], dim=-1)
            u = x_with_coords.reshape(B, H * W, -1)
            x_tok = self.preprocess(u)
        else:
            u = self.preprocess(pos)
            x_tok = u + self.placeholder[None, None, :]

        for block in self.blocks:
            x_tok = block(x_tok)
        return x_tok
