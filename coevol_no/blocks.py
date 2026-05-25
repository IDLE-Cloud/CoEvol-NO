"""Block wrappers that compose attention modules with LayerNorm, FFN, and residuals.

Block hierarchy:
    DualExactBlock  — wraps DualExactStateAttention (CoEvol-NO)
    LatentBlock     — wraps StateAttentionLatent (State-Evol ablation)
    SequenceBlock   — per-layer local latents (Coords-Evol ablation)

Each block follows the Pre-Norm Transformer pattern:
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
with LayerScale and DropPath for stable deep training.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp

from coevol_no.layers import LayerScale
from coevol_no.attention import DualExactStateAttention, StateAttentionLatent


# ===========================================================================
# DualExactBlock: Full CoEvol-NO (dual exact gradients for S and X)
# ===========================================================================

class DualExactBlock(nn.Module):
    """Block wrapping DualExactStateAttention with FFN on the token side.

    The primary block of CoEvol-NO.  Both S and X are updated via
    Predictor-Corrector with (optionally) exact gradients.
    """

    def __init__(self, dim_lat, dim_tok, num_heads=8, mlp_ratio=4.,
                 drop_path=0., init_values=1e-5, qkv_bias=True,
                 act_layer=nn.GELU,
                 # PC parameters
                 x_exact_update=True, x_loss_type='dot product',
                 x_momentum_beta=0.9,
                 s_approximate=False, s_loss_type='dot product'):
        super().__init__()
        self.norm_lat = nn.LayerNorm(dim_lat)
        self.norm_tok = nn.LayerNorm(dim_tok)

        # Core dual-exact PC attention
        self.cross_attn = DualExactStateAttention(
            dim_lat=dim_lat, dim_tok=dim_tok, num_heads=num_heads,
            qkv_bias=qkv_bias, drop_path=drop_path,
            s_loss_type=s_loss_type, x_exact_update=x_exact_update,
            x_loss_type=x_loss_type, x_momentum_beta=x_momentum_beta,
            s_approximate=s_approximate,
        )

        # Token FFN (standard)
        self.norm_tok2 = nn.LayerNorm(dim_tok)
        self.mlp_tok = Mlp(in_features=dim_tok, hidden_features=int(dim_tok * mlp_ratio), act_layer=act_layer)
        self.ls_tok2 = LayerScale(dim_tok, init_values)
        self.drop_path_tok2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_lat, x_tok, momentum_s, momentum_x):
        """Forward pass: dual-exact PC update + FFN.

        Args:
            x_lat: Latent State S, shape ``(B, M, C_lat)``.
            x_tok: Token Sequence X, shape ``(B, N, C_tok)``.
            momentum_s: S momentum from previous layer.
            momentum_x: X momentum from previous layer.

        Returns:
            x_lat, x_tok, momentum_s, momentum_x: Updated states and momenta.
        """
        # Dual-exact PC update
        x_lat, x_tok, momentum_s, momentum_x = self.cross_attn(
            self.norm_lat(x_lat), self.norm_tok(x_tok), momentum_s, momentum_x)

        # Token FFN (residual)
        x_tok = x_tok + self.drop_path_tok2(self.ls_tok2(self.mlp_tok(self.norm_tok2(x_tok))))

        return x_lat, x_tok, momentum_s, momentum_x


# ===========================================================================
# LatentBlock: State-Evol ablation (Encoder/Evolution/Decoder)
# ===========================================================================

class LatentBlock(nn.Module):
    """Generic block for asymmetric Q/KV with PC update.

    Used in three roles:
    - Encoder:  Q=latent(S), KV=token(X)   → encode X into S
    - Evolution: Q=latent(S), KV=latent(S)  → self-evolve S
    - Decoder:  Q=token(X), KV=latent(S)    → decode S back to X
    """

    def __init__(self, dim_q, dim_kv, num_heads=8, mlp_ratio=4.,
                 drop_path=0., init_values=1e-5, qkv_bias=True,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.attn = StateAttentionLatent(
            dim_q=dim_q, dim_kv=dim_kv, num_heads=num_heads,
            qkv_bias=qkv_bias, drop_path=drop_path, init_values=init_values,
        )

        self.norm_mlp = nn.LayerNorm(dim_q)
        self.mlp = Mlp(in_features=dim_q, hidden_features=int(dim_q * mlp_ratio), act_layer=act_layer)
        self.ls_mlp = LayerScale(dim_q, init_values)
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, momentum):
        """Forward: PC update on Q attending to KV, then FFN.

        Args:
            x_q: Query tensor (being updated), shape ``(B, M, C_q)``.
            x_kv: Key-Value tensor, shape ``(B, N, C_kv)``.
            momentum: Momentum from previous layer.

        Returns:
            x_q, momentum: Updated Q and momentum.
        """
        x_q, momentum = self.attn(self.norm_q(x_q), self.norm_kv(x_kv), momentum)
        x_q = x_q + self.drop_path_mlp(self.ls_mlp(self.mlp(self.norm_mlp(x_q))))
        return x_q, momentum


# ===========================================================================
# SequenceBlock: Coords-Evol ablation (per-layer local latents)
# ===========================================================================

class SequenceBlock(nn.Module):
    """Block with per-layer local latents (Coords-Evol ablation).

    Unlike DualExactBlock, latents are re-initialized each layer (no persistent
    state across layers).  Uses a simplified first-order gradient update.
    """

    def __init__(self, dim_lat, dim_tok, num_heads=8, mlp_ratio=4.,
                 drop_path=0., init_values=1e-5, num_latents=128,
                 qkv_bias=True, act_layer=nn.GELU):
        super().__init__()
        self.norm_lat = nn.LayerNorm(dim_lat)
        self.norm_tok = nn.LayerNorm(dim_tok)
        self.num_heads = num_heads

        # S encoding path (S <- X)
        self.q_lat_proj = nn.Linear(dim_lat, dim_lat, bias=qkv_bias)
        self.k_tok_proj = nn.Linear(dim_tok, dim_lat, bias=qkv_bias)
        self.v_tok_proj = nn.Linear(dim_tok, dim_lat, bias=qkv_bias)
        self.scale_lat = (dim_lat // num_heads) ** -0.5

        # X decoding path (X <- S)
        self.q_tok_proj = nn.Linear(dim_tok, dim_tok, bias=qkv_bias)
        self.k_lat_proj = nn.Linear(dim_lat, dim_tok, bias=qkv_bias)
        self.v_lat_proj = nn.Linear(dim_lat, dim_tok, bias=qkv_bias)
        self.proj_tok = nn.Linear(dim_tok, dim_tok)
        self.scale_tok = (dim_tok // num_heads) ** -0.5

        self.momentum_beta = 0.9
        self.ls_lat = LayerScale(dim_lat, init_values)

        # Token FFN
        self.ls_tok1 = LayerScale(dim_tok, init_values)
        self.drop_path_tok1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_tok2 = nn.LayerNorm(dim_tok)
        self.mlp_tok = Mlp(in_features=dim_tok, hidden_features=int(dim_tok * mlp_ratio), act_layer=act_layer)
        self.ls_tok2 = LayerScale(dim_tok, init_values)
        self.drop_path_tok2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Per-layer local latents (not shared across layers)
        self.latents = nn.Parameter(torch.zeros(1, num_latents, dim_lat))
        nn.init.normal_(self.latents, mean=0.0, std=0.1)

    def forward(self, x_tok, momentum):
        """Forward: first-order PC update with per-layer latents.

        Args:
            x_tok: Token Sequence X, shape ``(B, N, C_tok)``.
            momentum: Momentum from previous layer.

        Returns:
            x_tok, momentum, x_lat: Updated X, momentum, and local latents.
        """
        import torch.nn.functional as F

        B, N, C_tok = x_tok.shape
        x_lat = self.latents.expand(B, -1, -1)
        M, C_lat = x_lat.shape[1], x_lat.shape[2]
        head_dim_lat = C_lat // self.num_heads
        head_dim_tok = C_tok // self.num_heads

        if momentum is None:
            momentum = torch.zeros_like(x_lat)

        x_lat_n = self.norm_lat(x_lat)
        x_tok_n = self.norm_tok(x_tok)

        # ========== Step 1: S encoding (S <- X, first-order) ==========
        q_lat = x_lat_n.reshape(B, M, self.num_heads, head_dim_lat).permute(0, 2, 1, 3)
        k = self.k_tok_proj(x_tok_n).reshape(B, N, self.num_heads, head_dim_lat).permute(0, 2, 1, 3)
        v = self.v_tok_proj(x_tok_n).reshape(B, N, self.num_heads, head_dim_lat).permute(0, 2, 1, 3)

        delta_lat_val = F.scaled_dot_product_attention(q_lat, k, v)
        delta_lat_flat = delta_lat_val.permute(0, 2, 1, 3).reshape(B, M, C_lat)
        grad_direct = x_lat_n - delta_lat_flat

        # S momentum + update (first-order)
        momentum = self.momentum_beta * momentum + grad_direct
        delta_S = momentum.to(self.ls_lat.eta.dtype)
        x_lat_final = x_lat - self.ls_lat(delta_S)

        # ========== Step 2: X decoding (X <- S) ==========
        q_tok = self.q_tok_proj(x_tok_n).reshape(B, N, self.num_heads, head_dim_tok).permute(0, 2, 1, 3)
        k_lat = self.k_lat_proj(x_lat_final).reshape(B, M, self.num_heads, head_dim_tok).permute(0, 2, 1, 3)
        v_lat = self.v_lat_proj(delta_S).reshape(B, M, self.num_heads, head_dim_tok).permute(0, 2, 1, 3)

        delta_tok = F.scaled_dot_product_attention(q_tok, k_lat, v_lat)
        delta_tok = delta_tok.transpose(1, 2).reshape(B, N, C_tok)
        out_tok = self.proj_tok(delta_tok)

        # X residual + FFN
        x_tok = x_tok + self.drop_path_tok1(self.ls_tok1(out_tok))
        x_tok = x_tok + self.drop_path_tok2(self.ls_tok2(self.mlp_tok(self.norm_tok2(x_tok))))

        return x_tok, momentum, x_lat
