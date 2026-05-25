"""Analytical gradient computations for CoEvol-NO.

Explicit gradient formulas that replace torch.autograd, computing the same
PC gradients via direct mathematical derivation of the attention backward pass.

Three gradient computations:
    1. S PC gradient  — gradient of correction loss w.r.t. latent state S
    2. X PC gradient  — gradient of correction loss w.r.t. token sequence X
    3. FFN gradient   — gradient through the MLP sub-block (norm → mlp → layerscale)

Derivation:
    Predictor: S_pred = softmax(Q K^T / sqrt(d)) V,  Q = W_q S
    Dot-product loss:  L = -<S, S_pred>
        grad_S = -S_pred - W_q^T @ VJP_attn(S)
    L2 loss:  L = ||S - S_pred||^2 / 2
        grad_S = (S - S_pred) - W_q^T @ VJP_attn(S - S_pred)

    Where VJP_attn(v) is the vector-Jacobian product of attention w.r.t. Q:
        dA = v @ V^T
        dz = A * (dA - (dA * A).sum(-1, keepdim=True))    [softmax backward]
        dQ = scale * dz @ K
"""

import math
import torch
import torch.nn.functional as F


# ===========================================================================
# Core VJP helper
# ===========================================================================

def _attention_vjp_q(A, K, V, upstream, scale):
    """Vector-Jacobian product of attention output Y = A @ V w.r.t. query Q.

    Args:
        A: Attention weights after softmax, shape ``(B, H, M, N)``.
        K: Key tensor, shape ``(B, H, N, d)``.
        V: Value tensor, shape ``(B, H, N, d)``.
        upstream: Upstream gradient g = dL/dY, shape ``(B, H, M, d)``.
        scale: Attention scale ``1/sqrt(d)``.

    Returns:
        dL/dQ, shape ``(B, H, M, d)``.
    """
    dA = upstream @ V.transpose(-2, -1)
    dz = A * (dA - (dA * A).sum(-1, keepdim=True))
    return scale * dz @ K


# ===========================================================================
# S PC gradient
# ===========================================================================

def compute_s_gradient_analytical(attn, x_lat, x_tok):
    """Analytical S PC gradient (replaces ``_compute_s_gradient`` autograd).

    Args:
        attn: ``DualExactStateAttention`` module.
        x_lat: Latent state S, shape ``(B, M, C_lat)``.
        x_tok: Token sequence X, shape ``(B, N, C_tok)``.

    Returns:
        (grad_S, S_pred) with shapes ``(B, M, C_lat)`` each.
    """
    B, M, C_lat = x_lat.shape
    N = x_tok.shape[1]
    H = attn.num_heads
    d = C_lat // H

    k_tok = attn.k_tok_proj(x_tok).reshape(B, N, H, d).permute(0, 2, 1, 3)
    v_tok = attn.v_tok_proj(x_tok).reshape(B, N, H, d).permute(0, 2, 1, 3)
    q_lat = attn.q_lat_proj(x_lat).reshape(B, M, H, d).permute(0, 2, 1, 3)

    scores = (q_lat @ k_tok.transpose(-2, -1)) * attn.scale_lat
    A = scores.softmax(dim=-1)
    S_pred = (A @ v_tok).transpose(1, 2).reshape(B, M, C_lat)

    if attn.s_approximate:
        if attn.s_loss_type == 'dot product':
            return -S_pred, S_pred
        return x_lat - S_pred, S_pred

    if attn.s_loss_type == 'dot product':
        upstream = (-x_lat).reshape(B, M, H, d).permute(0, 2, 1, 3)
        direct_grad = -S_pred
    else:
        diff = x_lat - S_pred
        upstream = (-diff).reshape(B, M, H, d).permute(0, 2, 1, 3)
        direct_grad = diff

    dQ = _attention_vjp_q(A, k_tok, v_tok, upstream, attn.scale_lat)
    dQ_flat = dQ.permute(0, 2, 1, 3).reshape(B, M, C_lat)
    dS_jacobian = dQ_flat @ attn.q_lat_proj.weight

    return direct_grad + dS_jacobian, S_pred


# ===========================================================================
# X PC gradient
# ===========================================================================

def compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S):
    """Analytical X PC gradient (replaces ``_compute_x_gradient`` autograd).

    Args:
        attn: ``DualExactStateAttention`` module.
        x_lat: Updated latent state S, shape ``(B, M, C_lat)``.
        x_tok: Token sequence X, shape ``(B, N, C_tok)``.
        delta_S: S update direction, shape ``(B, M, C_lat)``.

    Returns:
        (grad_X, X_pred).  When first-order, X_pred is None.
    """
    B, N, C_tok = x_tok.shape
    M = x_lat.shape[1]
    H = attn.num_heads
    d = C_tok // H

    if not attn.x_exact_update:
        q_tok = attn.q_tok_proj(x_tok).reshape(B, N, H, d).permute(0, 2, 1, 3)
        k_lat = attn.k_lat_proj(x_lat).reshape(B, M, H, d).permute(0, 2, 1, 3)
        v_lat = attn.v_lat_proj(delta_S).reshape(B, M, H, d).permute(0, 2, 1, 3)
        delta_tok = F.scaled_dot_product_attention(q_tok, k_lat, v_lat)
        return delta_tok.transpose(1, 2).reshape(B, N, C_tok), None

    q_tok = attn.q_tok_proj(x_tok).reshape(B, N, H, d).permute(0, 2, 1, 3)
    k_lat = attn.k_lat_proj(x_lat).reshape(B, M, H, d).permute(0, 2, 1, 3)
    v_lat = attn.v_lat_proj(delta_S).reshape(B, M, H, d).permute(0, 2, 1, 3)

    scores = (q_tok @ k_lat.transpose(-2, -1)) * attn.scale_tok
    A = scores.softmax(dim=-1)
    X_pred = (A @ v_lat).transpose(1, 2).reshape(B, N, C_tok)

    if attn.x_loss_type == 'dot product':
        upstream = (-x_tok).reshape(B, N, H, d).permute(0, 2, 1, 3)
        direct_grad = -X_pred
    else:
        diff = x_tok - X_pred
        upstream = (-diff).reshape(B, N, H, d).permute(0, 2, 1, 3)
        direct_grad = diff

    dQ = _attention_vjp_q(A, k_lat, v_lat, upstream, attn.scale_tok)
    dQ_flat = dQ.permute(0, 2, 1, 3).reshape(B, N, C_tok)
    dX_jacobian = dQ_flat @ attn.q_tok_proj.weight

    return direct_grad + dX_jacobian, X_pred


# ===========================================================================
# FFN gradient helpers (used by PCFFN in blocks.py)
# ===========================================================================

def _gelu_derivative(x):
    """Derivative of the standard GELU activation."""
    return (
        0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        + x * torch.exp(-x ** 2 / 2.0) / math.sqrt(2.0 * math.pi)
    )


def _layernorm_backward(grad_y, x, ln_module):
    """Gradient of LayerNorm output w.r.t. input x.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
        grad_y: dL/dy, shape ``(B, N, C)``.
        x: Original input, shape ``(B, N, C)``.
        ln_module: ``nn.LayerNorm`` module.

    Returns:
        dL/dx, shape ``(B, N, C)``.
    """
    gamma = ln_module.weight
    eps = ln_module.eps

    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = torch.sqrt(var + eps)
    x_hat = (x - mean) / std

    grad_x_hat = grad_y * gamma
    return (1.0 / std) * (
        grad_x_hat
        - grad_x_hat.mean(dim=-1, keepdim=True)
        - x_hat * (grad_x_hat * x_hat).mean(dim=-1, keepdim=True)
    )
