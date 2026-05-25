"""Shared primitives for CoEvol-NO.

LayerScale: per-channel learnable scaling used in residual connections.
zeropower_via_newtonschulz5: Newton-Schulz iteration for orthogonal projection (Muon optimizer).
"""

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Per-channel learnable scaling: ``x * eta``.

    Stabilizes deep residual networks by initializing ``eta`` near zero
    so that new paths start as near-identity and gradually learn.
    """

    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.eta = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.eta


def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iterative approximation to the orthogonal projection.

    Approximates ``orth(G)`` using a polynomial iteration with per-step
    optimized coefficients for fast convergence.  Used by the Muon optimizer
    to orthogonalize gradient updates.

    Args:
        G: input tensor of shape ``(..., M, N)``.
        steps: number of Newton-Schulz iterations (1-5).

    Returns:
        Orthogonalized tensor of the same shape.
    """
    if steps == 0:
        return G
    assert G.ndim >= 3

    coeffs_data = [
        [3.87796501, -6.09156166, 2.79297994],
        [3.96680788, -6.20261014, 2.71123643],
        [3.72638362, -6.32660637, 2.76360346],
        [3.20363086, -5.95273990, 3.11181014],
        [3.29856763, -5.51437019, 3.41495988],
    ]
    current_coeffs = coeffs_data[:steps]
    coeffs = torch.tensor(current_coeffs, device=G.device, dtype=torch.float32)

    X = G.to(torch.bfloat16)
    norm = torch.norm(X, dim=(-2, -1), keepdim=True).clamp(min=1e-7)
    X = X / norm

    for i in range(steps):
        a, b, c = coeffs[i]
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X * norm
