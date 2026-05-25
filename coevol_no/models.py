"""Top-level model classes for CoEvol-NO.

Three paradigm variants:
    CoEvolNO         — Co-Evolution: bidirectional PC update of S and X (full model)
    CoEvolNOLatent   — State-Evol: Encode → Evolve → Decode (latent-only)
    CoEvolNOSequence — Coords-Evol: per-layer local latents (no persistent state)

Plus a variant factory:
    CoEvolNOVariant  — creates ablation models with different PC configurations
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import numpy as np

from coevol_no.layers import LayerScale
from coevol_no.blocks import DualExactBlock, LatentBlock, SequenceBlock


# ===========================================================================
# Shared utilities
# ===========================================================================

def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def _get_grid(shape, device):
    """Generate a 2D coordinate grid in [0, 1].

    Args:
        shape: ``(B, H, W, coord_dim)`` — only H, W are used.
        device: torch device.

    Returns:
        Coordinate grid of shape ``(B, H, W, 2)`` with (x, y) channels.
    """
    B, H, W, _ = shape
    gridx = torch.tensor(np.linspace(0, 1, H), dtype=torch.float, device=device).reshape(1, H, 1, 1).repeat([B, 1, W, 1])
    gridy = torch.tensor(np.linspace(0, 1, W), dtype=torch.float, device=device).reshape(1, 1, W, 1).repeat([B, H, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)


# ===========================================================================
# CoEvolNO: Full Co-Evolution model (primary contribution)
# ===========================================================================

class CoEvolNO(nn.Module):
    """CoEvol-NO: Co-Evolution with Predictor-Corrector for Neural Operator.

    The primary model from the paper.  Supports:
    - PDE grid input: ``forward(u, pos, t)`` with u shape ``(B, C, H, W)``
    - Irregular mesh input: ``forward(pos=...)`` with pos shape ``(B, N, coord_dim)``

    Ablation variants are controlled by ``x_exact_update`` and ``s_approximate``:
        dual_exact:   x_exact_update=True,  s_approximate=False
        s_exact:      x_exact_update=False, s_approximate=False  (default)
        x_exact:      x_exact_update=True,  s_approximate=True
        first_order:  x_exact_update=False, s_approximate=True
    """

    def __init__(self,
                 # Input parameters
                 img_size=(64, 64), in_channels=1, coord_dim=2,
                 # Core model parameters
                 depth=8, num_latents=128, dim_lat=128, dim_tok=128,
                 num_heads=8, mlp_ratio=1.0, drop_path_rate=0.1,
                 # Predictor-Corrector parameters
                 x_exact_update=True, x_loss_type='dot product',
                 x_momentum_beta=0.0, s_approximate=False,
                 s_loss_type='dot product', s_momentum_beta=0.9,
                 # PCFFN parameters
                 use_pc_ffn=False, pc_ffn_loss_type='dot product',
                 pc_ffn_momentum_beta=0.9, pc_ffn_analytical=True,
                 # Positional
                 unified_pos=False, ref=8,
                 # Other
                 final_norm=True):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.size = self.img_size[0]
        self.in_channels = in_channels
        self.coord_dim = coord_dim
        self.output_feature_dim = dim_tok
        self.unified_pos = unified_pos
        self.ref = ref

        # Input preprocessing: [u(x), x] -> dim_tok
        if self.unified_pos:
            self.preprocess = nn.Linear(in_channels + coord_dim + ref * ref, dim_tok)
        else:
            self.preprocess = nn.Linear(in_channels + coord_dim, dim_tok)

        self.placeholder = nn.Parameter((1 / dim_tok) * torch.rand(dim_tok, dtype=torch.float))

        # Learnable latent vectors (shared across layers)
        self.latents = nn.Parameter(torch.zeros(1, num_latents, dim_lat))
        trunc_normal_(self.latents, std=.02)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Stack DualExactBlocks
        self.blocks = nn.ModuleList([
            DualExactBlock(
                dim_lat=dim_lat, dim_tok=dim_tok, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop_path=dpr[i],
                x_exact_update=x_exact_update, x_loss_type=x_loss_type,
                x_momentum_beta=x_momentum_beta, s_approximate=s_approximate,
                s_loss_type=s_loss_type, s_momentum_beta=s_momentum_beta,
                use_pc_ffn=use_pc_ffn, pc_ffn_loss_type=pc_ffn_loss_type,
                pc_ffn_momentum_beta=pc_ffn_momentum_beta,
                pc_ffn_analytical=pc_ffn_analytical,
            )
            for i in range(depth)
        ])

        # Final normalization
        self.final_norm = final_norm
        self.norm = nn.LayerNorm(dim_tok) if final_norm else nn.Identity()

        self.apply(_init_weights)

    def _get_unified_pos(self, pos):
        """Compute distance-based positional encoding for irregular meshes.

        For each point in pos, computes distances to a reference grid
        to create a unified positional representation.
        """
        batchsize = pos.shape[0]
        gridx = torch.tensor(np.linspace(-2, 4, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(pos.device).reshape(batchsize, self.ref ** 2, 2)
        return torch.sqrt(
            torch.sum((pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
        ).reshape(batchsize, pos.shape[1], self.ref * self.ref).contiguous()

    def forward(self, u=None, pos=None, t=None):
        """Forward pass.

        Args:
            u: Input field, shape ``(B, C_in, H, W)`` or ``(B, H*W, C_in)``.
               Used for PDE grid tasks.
            pos: Position encoding, shape ``(B, N, coord_dim)``.
                 Used for irregular mesh tasks.  Either u or pos must be provided.
            t: Time step (reserved for future temporal tasks).

        Returns:
            x_tok: Output features, shape ``(B, N, dim_tok)`` .
        """
        if u is not None:
            # Handle flat input: (B, H*W, C) -> (B, C, H, W)
            if u.dim() == 3:
                B, P, C = u.shape
                H = W = int(math.sqrt(P))
                u = u.permute(0, 2, 1).reshape(B, C, H, W)

            x = u.unsqueeze(-1) if u.dim() == 3 else u
            u = x.reshape(x.shape[0], self.size, self.size, self.in_channels)
            u = u.permute(0, 3, 1, 2)
            B, C, H, W = u.shape

            # Prepend coordinate grid
            grid = _get_grid((B, H, W, self.coord_dim), u.device)
            u_permuted = u.permute(0, 2, 3, 1)
            x_with_coords = torch.cat([u_permuted, grid], dim=-1)
            u_flat = x_with_coords.reshape(B, H * W, -1)
            x_tok = self.preprocess(u_flat)
        elif pos is not None:
            # Irregular mesh input
            if self.unified_pos and pos.dim() == 3 and pos.shape[-1] == 2:
                new_pos = self._get_unified_pos(pos)
                pos = torch.cat([pos, new_pos], dim=-1)
            x_tok = self.preprocess(pos) + self.placeholder[None, None, :]
        else:
            raise ValueError("Either u or pos must be provided")

        B = x_tok.shape[0]

        # Initialize latent state S (shared across layers)
        x_lat = self.latents.expand(B, -1, -1)

        # Initialize momenta
        momentum_s = None
        momentum_x = None
        momentum_ffn = None

        # Co-Evolution through all blocks
        for block in self.blocks:
            x_lat, x_tok, momentum_s, momentum_x, momentum_ffn = block(
                x_lat, x_tok, momentum_s, momentum_x, momentum_ffn)

        # Final normalization
        x_tok = self.norm(x_tok)

        return x_tok


# ===========================================================================
# CoEvolNOLatent: State-Evol ablation (Encode → Evolve → Decode)
# ===========================================================================

class CoEvolNOLatent(nn.Module):
    """State-Evol ablation: Encode → Evolve → Decode.

    Three phases:
    - Encoder:  latent queries token (X → S), captures information into latent
    - Evolution: latent self-evolves (S → S), disconnected from X
    - Decoder:  token queries latent (S → X), decodes back to output

    This variant isolates the latent evolution from the token sequence,
    demonstrating the importance of bidirectional co-evolution.
    """

    def __init__(self, img_size=(64, 64), in_channels=1, coord_dim=2,
                 depth=6, num_latents=64, dim_lat=256, dim_tok=128,
                 num_heads=8, mlp_ratio=2.0, drop_path_rate=0.1,
                 unified_pos=False, ref=8, **kwargs):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.in_channels = in_channels
        self.coord_dim = coord_dim
        self.dim_lat = dim_lat
        self.dim_tok = dim_tok
        self.output_feature_dim = dim_tok
        self.unified_pos = unified_pos
        self.ref = ref

        # Input preprocessing
        self.preprocess = nn.Linear(in_channels + coord_dim, dim_tok)

        # Learnable latent vectors
        self.latents = nn.Parameter(torch.zeros(1, num_latents, dim_lat))
        trunc_normal_(self.latents, std=.02)

        # Encoder: X → S
        self.encoder_block = LatentBlock(dim_q=dim_lat, dim_kv=dim_tok, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Evolution: S → S (self-evolution, disconnected from X)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.evolution_blocks = nn.ModuleList([
            LatentBlock(dim_q=dim_lat, dim_kv=dim_lat, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])

        # Decoder: S → X
        self.decoder_block = LatentBlock(dim_q=dim_tok, dim_kv=dim_lat, num_heads=num_heads, mlp_ratio=mlp_ratio)

        self.norm = nn.LayerNorm(dim_tok)

        self.apply(_init_weights)

    def forward(self, u=None, pos=None, t=None):
        """Forward: Encode → Evolve → Decode.

        Args:
            u: Input field, shape ``(B, C, H, W)`` or ``(B, H*W, C)``.
            pos: Position encoding, shape ``(B, N, coord_dim)``.
            t: Time step (reserved).

        Returns:
            Output features, shape ``(B, N, dim_tok)`` .
        """
        if u is not None:
            if u.dim() == 3:
                B, P, C = u.shape
                H = W = int(math.sqrt(P))
                u = u.permute(0, 2, 1).reshape(B, C, H, W)

            x = u.reshape(u.shape[0], self.img_size[0], self.img_size[1], self.in_channels)
            u_perm = x.permute(0, 3, 1, 2)
            B, C, H, W = u_perm.shape

            grid = _get_grid((B, H, W, self.coord_dim), u.device)
            u_perm = u_perm.permute(0, 2, 3, 1)
            x_with_coords = torch.cat([u_perm, grid], dim=-1)
            x_tok = self.preprocess(x_with_coords.reshape(B, H * W, -1))
        else:
            x_tok = self.preprocess(pos)

        B = x_tok.shape[0]
        x_lat = self.latents.expand(B, -1, -1)

        # Encode: X → S
        x_lat, _ = self.encoder_block(x_lat, x_tok, None)

        # Evolve: S → S
        momentum_evo = None
        for block in self.evolution_blocks:
            x_lat, momentum_evo = block(x_lat, x_lat, momentum_evo)

        # Decode: S → X
        x_out, _ = self.decoder_block(x_tok, x_lat, None)

        return x_out


# ===========================================================================
# CoEvolNOSequence: Coords-Evol ablation (per-layer local latents)
# ===========================================================================

class CoEvolNOSequence(nn.Module):
    """Coords-Evol ablation: per-layer local latents, no persistent state.

    Unlike CoEvolNO, latents are re-initialized each layer and never shared.
    This demonstrates the importance of persistent latent state across layers.
    Uses a simplified first-order gradient approximation (no autograd).
    """

    def __init__(self, img_size=(64, 64), in_channels=1, coord_dim=2,
                 depth=6, num_latents=128, dim_lat=256, dim_tok=128,
                 num_heads=8, mlp_ratio=1.0, drop_path_rate=0.1,
                 unified_pos=False, ref=8, **kwargs):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.size = self.img_size[0]
        self.in_channels = in_channels
        self.coord_dim = coord_dim
        self.output_feature_dim = dim_tok
        self.unified_pos = unified_pos
        self.ref = ref

        self.preprocess = nn.Linear(in_channels + coord_dim, dim_tok)
        self.placeholder = nn.Parameter((1 / dim_tok) * torch.rand(dim_tok, dtype=torch.float))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SequenceBlock(
                dim_lat=dim_lat, dim_tok=dim_tok, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], num_latents=num_latents,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim_tok)

        self.apply(_init_weights)

    def forward(self, u=None, pos=None, t=None):
        """Forward: per-layer local latents with first-order PC update.

        Args:
            u: Input field, shape ``(B, C, H, W)`` or ``(B, H*W, C)``.
            pos: Position encoding, shape ``(B, N, coord_dim)``.
            t: Time step (reserved).

        Returns:
            Output features, shape ``(B, N, dim_tok)`` .
        """
        if u is not None:
            if u.dim() == 3:
                B, P, C = u.shape
                H = W = int(math.sqrt(P))
                u = u.permute(0, 2, 1).reshape(B, C, H, W)

            x = u.unsqueeze(-1) if u.dim() == 3 else u
            u = x.reshape(x.shape[0], self.size, self.size, self.in_channels)
            u = u.permute(0, 3, 1, 2)
            B, C, H, W = u.shape

            grid = _get_grid((B, H, W, self.coord_dim), u.device)
            u_perm = u.permute(0, 2, 3, 1)
            x_with_coords = torch.cat([u_perm, grid], dim=-1)
            x_tok = self.preprocess(x_with_coords.reshape(B, H * W, -1))
        else:
            x_tok = self.preprocess(pos) + self.placeholder[None, None, :]

        momentum = None
        for block in self.blocks:
            x_tok, momentum, _ = block(x_tok, momentum)

        x_tok = self.norm(x_tok)

        return x_tok


# ===========================================================================
# CoEvolNOVariant: Factory for creating ablation variants
# ===========================================================================

class CoEvolNOVariant:
    """Factory for creating ablation variants of CoEvol-NO.

    Variants:
        'dual_exact':  S exact, X exact  (full model)
        's_exact':     S exact, X first-order  (default CoEvol-NO)
        'x_exact':     S first-order, X exact
        'first_order': S first-order, X first-order

    Usage:
        model = CoEvolNOVariant.create('dual_exact', img_size=(64, 64))
        model = CoEvolNOVariant.create('s_exact', coord_dim=7, depth=8)
    """

    CONFIGS = {
        'dual_exact': {'x_exact_update': True, 's_approximate': False},
        's_exact':    {'x_exact_update': False, 's_approximate': False},
        'x_exact':    {'x_exact_update': True, 's_approximate': True},
        'first_order': {'x_exact_update': False, 's_approximate': True},
    }

    @staticmethod
    def create(variant, **kwargs):
        """Create a CoEvolNO instance with the specified variant configuration.

        Args:
            variant: One of 'dual_exact', 's_exact', 'x_exact', 'first_order'.
            **kwargs: Additional arguments passed to CoEvolNO constructor.

        Returns:
            CoEvolNO instance with the specified PC configuration.
        """
        if variant not in CoEvolNOVariant.CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(CoEvolNOVariant.CONFIGS.keys())}")
        config = {**kwargs, **CoEvolNOVariant.CONFIGS[variant]}
        return CoEvolNO(**config)
