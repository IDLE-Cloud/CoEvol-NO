import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSHClusteredAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_clusters=32, num_hashes=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_clusters = num_clusters
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.register_buffer("hash_planes", torch.randn(self.num_heads, self.head_dim, self.num_clusters))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        hash_planes = F.normalize(self.hash_planes, dim=1)
        scores = torch.einsum('bhnd,hdc->bhnc', F.normalize(q, dim=-1), hash_planes)
        cluster_ids = torch.argmax(scores, dim=-1)

        q_flat = q.reshape(B * self.num_heads, N, self.head_dim)
        ids_flat = cluster_ids.reshape(B * self.num_heads, N)
        ids_expanded = ids_flat.unsqueeze(-1).expand(-1, -1, self.head_dim)

        q_grouped_flat = torch.zeros(B * self.num_heads, self.num_clusters, self.head_dim, device=x.device)
        counts_flat = torch.zeros(B * self.num_heads, self.num_clusters, device=x.device)
        q_grouped_flat.scatter_add_(1, ids_expanded, q_flat)
        ones = torch.ones_like(ids_flat, dtype=torch.float)
        counts_flat.scatter_add_(1, ids_flat, ones)
        counts_flat = torch.clamp(counts_flat, min=1.0)
        q_grouped_flat = q_grouped_flat / counts_flat.unsqueeze(-1)
        q_grouped = q_grouped_flat.reshape(B, self.num_heads, self.num_clusters, self.head_dim)

        attn = (q_grouped @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v_grouped = attn @ v

        v_grouped_flat = v_grouped.reshape(B * self.num_heads, self.num_clusters, self.head_dim)
        ids_gather = ids_flat.unsqueeze(-1).expand(-1, -1, self.head_dim)
        x_out_flat = torch.gather(v_grouped_flat, 1, ids_gather)
        x_out = x_out_flat.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj_drop(self.proj(x_out))


class ClusterBlock(nn.Module):
    def __init__(self, dim, num_heads, num_clusters=32, mlp_ratio=4., drop=0., init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LSHClusteredAttention(dim, num_heads=num_heads, num_clusters=num_clusters, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class ClusterAttentionNO(nn.Module):
    def __init__(self, img_size=(64, 64), in_channels=1, out_channels=1, coord_dim=2,
                 dim=128, num_clusters=64, depth=4, num_heads=4):
        super().__init__()
        self.coord_dim = coord_dim
        self.img_size = img_size
        self.size = img_size[0]
        self.in_channels = in_channels

        self.fc_in = nn.Linear(in_channels + coord_dim, dim)
        self.placeholder = nn.Parameter(((1 / dim)) * torch.rand(dim, dtype=torch.float))
        self.blocks = nn.ModuleList([
            ClusterBlock(dim, num_heads, num_clusters=num_clusters) for _ in range(depth)])
        self.fc_out = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, 1))

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
            x = self.fc_in(x_with_coords.reshape(B, H * W, -1))
        else:
            x = self.fc_in(pos) + self.placeholder[None, None, :]

        for blk in self.blocks:
            x = blk(x)
        return self.fc_out(x)
