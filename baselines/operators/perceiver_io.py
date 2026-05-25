import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange

ACTIVATIONS = {'gelu': nn.GELU(approximate='tanh'), 'silu': nn.SiLU()}


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.eta = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.eta


class PerceiverEncoder(nn.Module):
    def __init__(self, channel_dim, num_heads=8, num_latents=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.latent_q = nn.Parameter(torch.randn(num_latents, channel_dim))
        nn.init.normal_(self.latent_q, mean=0.0, std=1.0)
        self.ln = nn.LayerNorm(channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):
        x = self.ln(x)
        q = rearrange(self.latent_q, 'm (h d) -> h m d', h=self.num_heads)
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = rearrange(y, 'b h n d -> b n (h d)')
        return self.out_proj(y)


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, channel_dim, num_heads=None):
        super().__init__()
        self.num_heads = channel_dim // 16 if num_heads is None else num_heads
        self.head_dim = channel_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim, bias=False)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [rearrange(z, 'b n (h d) -> b h n d', h=self.num_heads) for z in [q, k, v]]
        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(rearrange(y, 'b h n d -> b n (h d)'))


class SelfAttentionBlock(nn.Module):
    def __init__(self, channel_dim, num_heads=None, mlp_ratio=4.0, act=None, init_values=1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.att = MultiHeadedSelfAttention(channel_dim, num_heads)
        self.mlp = MLPBlock(in_dim=channel_dim, hidden_dim=int(channel_dim * mlp_ratio), out_dim=channel_dim, act=act)
        self.ls1 = LayerScale(channel_dim, init_values)
        self.ls2 = LayerScale(channel_dim, init_values)

    def forward(self, x):
        x = x + self.ls1(self.att(self.ln1(x)))
        x = x + self.ls2(self.mlp(self.ln2(x)))
        return x


class PerceiverDecoder(nn.Module):
    def __init__(self, channel_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x, y):
        x = self.ln1(x)
        y = self.ln2(y)
        q = rearrange(self.q_proj(y), 'b m (h d) -> b h m d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        z = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(rearrange(z, 'b h n d -> b n (h d)'))


class PerceiverIO(nn.Module):
    def __init__(self, in_dim, img_size=(64, 64), channel_dim=128, coord_dim=2,
                 num_blocks=8, num_heads=8, num_latents=128, mlp_ratio=1.0,
                 act=None, cross_attn=False):
        super().__init__()
        self.img_size = img_size
        self.size = img_size[0]
        self.in_channels = in_dim
        self.coord_dim = coord_dim
        self.in_proj = nn.Linear(in_dim + coord_dim, channel_dim)
        self.placeholder = nn.Parameter((1 / (channel_dim)) * torch.rand(channel_dim, dtype=torch.float))

        self.encoder = PerceiverEncoder(channel_dim=channel_dim, num_heads=num_heads, num_latents=num_latents)
        self.cross_attn = cross_attn
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(channel_dim=channel_dim, num_heads=num_heads, act=act, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.decoder = PerceiverDecoder(channel_dim=channel_dim, num_heads=num_heads)
        self.out_proj = nn.Sequential(
            nn.Linear(channel_dim, channel_dim * 4),
            nn.GELU(),
            nn.Linear(channel_dim * 4, 1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

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
            x = self.in_proj(u)
            z = self.encoder(x)
        else:
            u = self.in_proj(pos)
            x = u + self.placeholder[None, None, :]
            z = self.encoder(x)

        for block in self.blocks:
            z = block(z)

        x = self.decoder(z, x)
        x = self.out_proj(x)
        return x
