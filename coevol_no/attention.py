"""Attention modules for CoEvol-NO.

Two attention variants implementing the Predictor-Corrector (PC) mechanism:

1. DualExactStateAttention — PC update for both S and X.
   The core module of CoEvol-NO. X PC update is optional (x_exact_update).

2. StateAttentionLatent — Generic asymmetric attention (dim_q != dim_kv).
   Used by the State-Evol ablation (Encoder/Evolution/Decoder pattern).

Predictor-Corrector update rule:
    Predictor:  S_pred = CrossAttn(Q=S, K=X, V=X)
    Loss:       L_S = -<S, S_pred>  (dot product)  or  ||S - S_pred||^2 / 2  (l2)
    Gradient:   ∇_S L_S = (S - S_pred) - J_S^T (S - S_pred)     [exact]
                ∇_S L_S ≈ S - S_pred                                 [first-order approx]
    Momentum:   m_t = β * m_{t-1} + ∇_S L_S
    Update:     S_t = S_{t-1} - η * m_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from coevol_no.layers import LayerScale, zeropower_via_newtonschulz5


# ===========================================================================
# DualExactStateAttention: Dual exact gradient for both S and X (full CoEvol-NO)
# ===========================================================================

class DualExactStateAttention(nn.Module):
    """Dual exact gradient State Attention: updates both S and X via PC.

    S update: Predictor = CrossAttn(Q=S, K=X, V=X), exact gradient correction.
    X update: Predictor = CrossAttn(Q=X, K=S, V=delta_S), exact or first-order.

    Supports four ablation modes via ``x_exact_update`` and ``s_approximate``:
        dual_exact:   S exact, X exact  (full model)
        s_exact:      S exact, X first-order  (default CoEvol-NO)
        x_exact:      S first-order, X exact
        first_order:  S first-order, X first-order
    """

    def __init__(self, dim_lat, dim_tok, num_heads=8, drop_path=0.,
                 init_values=1e-5, qkv_bias=False,
                 # S update parameters
                 s_loss_type='dot product', s_momentum_beta=0.9, s_eta_init=1e-5,
                 # X update parameters
                 x_loss_type='dot product', x_momentum_beta=0.0, x_eta_init=1e-5,
                 # Ablation switches
                 x_exact_update=True, s_approximate=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim_lat = dim_lat // num_heads
        head_dim_tok = dim_tok // num_heads

        # ========== S update parameters ==========
        self.s_loss_type = s_loss_type
        self.s_momentum_beta = s_momentum_beta
        self.s_approximate = s_approximate

        # S encoding path (S <- X)
        self.q_lat_proj = nn.Linear(dim_lat, dim_lat, bias=qkv_bias)
        self.k_tok_proj = nn.Linear(dim_tok, dim_lat, bias=qkv_bias)
        self.v_tok_proj = nn.Linear(dim_tok, dim_lat, bias=qkv_bias)
        self.scale_lat = head_dim_lat ** -0.5

        # S LayerScale and DropPath
        self.eta_s = nn.Parameter(s_eta_init * torch.ones(dim_lat))
        self.drop_path_s = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ========== X update parameters ==========
        self.x_loss_type = x_loss_type
        self.x_momentum_beta = x_momentum_beta
        self.x_exact_update = x_exact_update

        # X decoding path (X <- S), this is the Predictor
        self.q_tok_proj = nn.Linear(dim_tok, dim_tok, bias=qkv_bias)
        self.k_lat_proj = nn.Linear(dim_lat, dim_tok, bias=qkv_bias)
        self.v_lat_proj = nn.Linear(dim_lat, dim_tok, bias=qkv_bias)
        self.proj_tok = nn.Linear(dim_tok, dim_tok)
        self.scale_tok = head_dim_tok ** -0.5

        # X LayerScale and DropPath
        self.eta_x = nn.Parameter(x_eta_init * torch.ones(dim_tok))
        self.drop_path_x = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _compute_s_gradient(self, x_lat, x_tok):
        """Compute S's exact gradient via autograd.

        Predictor: S_pred = CrossAttn(Q=S, K=X, V=X)
        Loss:      L_S = ||S - S_pred||^2 / 2  or  -<S, S_pred>
        Gradient:  ∇_S L_S = (S - S_pred) - J_S^T (S - S_pred)
        """
        B, M, C_lat = x_lat.shape
        N = x_tok.shape[1]

        # Prepare K, V from X
        k_tok = self.k_tok_proj(x_tok).reshape(B, N, self.num_heads, C_lat // self.num_heads).permute(0, 2, 1, 3)
        v_tok = self.v_tok_proj(x_tok).reshape(B, N, self.num_heads, C_lat // self.num_heads).permute(0, 2, 1, 3)

        with torch.enable_grad():
            S_param = x_lat.clone().requires_grad_(True)

            if self.s_approximate:
                # First-order: detach Q from the graph
                q_lat = self.q_lat_proj(x_lat).reshape(B, M, self.num_heads, C_lat // self.num_heads).permute(0, 2, 1, 3)
            else:
                # Exact: Q depends on S, autograd captures Jacobian
                q_lat = self.q_lat_proj(S_param).reshape(B, M, self.num_heads, C_lat // self.num_heads).permute(0, 2, 1, 3)

            # Predictor: S_pred = softmax(Q K^T / sqrt(d)) V
            attn_s = (q_lat @ k_tok.transpose(-2, -1)) * self.scale_lat
            attn_s = attn_s.softmax(dim=-1)
            S_pred = (attn_s @ v_tok).transpose(1, 2).reshape(B, M, C_lat)

            # Correction Loss
            if self.s_loss_type == 'l2':
                loss_s = torch.sum((S_param - S_pred) ** 2) / 2.0
            elif self.s_loss_type == 'dot product':
                loss_s = -torch.einsum('bmd,bmd->b', S_param, S_pred).sum()
            else:
                raise ValueError(f"Unknown s_loss_type: {self.s_loss_type}")

            # Exact gradient (includes Jacobian term via autograd)
            grad_S = torch.autograd.grad(loss_s, S_param, create_graph=True)[0]

        return grad_S, S_pred

    def _compute_x_gradient(self, x_lat, x_tok, delta_S):
        """Compute X's gradient.  Exact (with Jacobian) or first-order approximation.

        Predictor: X_pred = CrossAttn(Q=X, K=S, V=delta_S)
        Loss:      L_X = ||X - X_pred||^2 / 2  or  -<X, X_pred>
        Gradient:  ∇_X L_X = (X - X_pred) - J_X^T (X - X_pred)  [exact]
                   ∇_X L_X ≈ X - X_pred                             [first-order]

        Note: S uses updated x_lat, V uses delta_S (change amount).
        """
        B, N, C_tok = x_tok.shape
        M = x_lat.shape[1]
        head_dim = C_tok // self.num_heads

        if not self.x_exact_update:
            # First-order approximation: direct cross-attention output
            q_tok = self.q_tok_proj(x_tok).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            k_lat = self.k_lat_proj(x_lat).reshape(B, M, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_lat = self.v_lat_proj(delta_S).reshape(B, M, self.num_heads, head_dim).permute(0, 2, 1, 3)

            delta_tok_attn = F.scaled_dot_product_attention(q_tok, k_lat, v_lat)
            delta_tok = delta_tok_attn.transpose(1, 2).reshape(B, N, C_tok)
            return delta_tok, None  # None = no exact gradient computed

        # Exact gradient update
        with torch.enable_grad():
            X_param = x_tok.clone().requires_grad_(True)

            # Predictor: X_pred = CrossAttn(Q=X, K=S, V=delta_S)
            q_tok = self.q_tok_proj(X_param).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            k_lat = self.k_lat_proj(x_lat).reshape(B, M, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_lat = self.v_lat_proj(delta_S).reshape(B, M, self.num_heads, head_dim).permute(0, 2, 1, 3)

            attn_x = (q_tok @ k_lat.transpose(-2, -1)) * self.scale_tok
            attn_x = attn_x.softmax(dim=-1)
            X_pred = (attn_x @ v_lat).transpose(1, 2).reshape(B, N, C_tok)

            # Correction Loss
            if self.x_loss_type == 'l2':
                loss_x = torch.sum((X_param - X_pred) ** 2) / 2.0
            elif self.x_loss_type == 'dot product':
                loss_x = -torch.einsum('bnd,bnd->b', X_param, X_pred).sum()
            else:
                raise ValueError(f"Unknown x_loss_type: {self.x_loss_type}")

            # Exact gradient (includes Jacobian term via autograd)
            grad_X = torch.autograd.grad(loss_x, X_param, create_graph=True)[0]

        return grad_X, X_pred

    def forward(self, x_lat, x_tok, momentum_s_in=None, momentum_x_in=None):
        """Dual exact gradient forward pass.

        Args:
            x_lat: Latent State S, shape ``(B, M, C_lat)``
            x_tok: Token Sequence X, shape ``(B, N, C_tok)``
            momentum_s_in: S momentum from previous layer.
            momentum_x_in: X momentum from previous layer.

        Returns:
            x_lat_final: Updated S.
            x_tok_final: Updated X.
            momentum_s_out: Updated S momentum.
            momentum_x_out: Updated X momentum.
        """
        B, M, C_lat = x_lat.shape
        B, N, C_tok = x_tok.shape

        # ========== Initialize momentum ==========
        if momentum_s_in is None:
            momentum_s_in = torch.zeros_like(x_lat)
        if momentum_x_in is None:
            momentum_x_in = torch.zeros_like(x_tok)

        # ========== Step 1: S exact gradient update ==========
        grad_S, _ = self._compute_s_gradient(x_lat, x_tok)

        # S momentum update
        momentum_s_out = self.s_momentum_beta * momentum_s_in + grad_S
        delta_S = momentum_s_out

        # S final update
        x_lat_final = x_lat - self.drop_path_s(self.eta_s * delta_S)

        # ========== Step 2: X gradient update ==========
        grad_X, _ = self._compute_x_gradient(x_lat_final, x_tok, delta_S)

        if self.x_exact_update:
            # X exact gradient: momentum update
            momentum_x_out = self.x_momentum_beta * momentum_x_in + grad_X
            delta_X = momentum_x_out
            x_tok_final = x_tok - self.drop_path_x(self.eta_x * delta_X)
        else:
            # X first-order: optional momentum accumulation
            momentum_x_out = self.x_momentum_beta * momentum_x_in + self.proj_tok(grad_X)
            x_tok_final = x_tok + self.drop_path_x(self.eta_x * momentum_x_out)

        return x_lat_final, x_tok_final, momentum_s_out, momentum_x_out


# ===========================================================================
# StateAttentionLatent: asymmetric attention for State-Evol ablation
# ===========================================================================

class StateAttentionLatent(nn.Module):
    """Generic PC attention with asymmetric Q/KV dimensions.

    Used by the State-Evol (Encode→Evolve→Decode) ablation where
    ``dim_q`` and ``dim_kv`` can differ.  For example:
    - Encoder: Q=latent (dim_lat), KV=token (dim_tok)
    - Evolution: Q=latent (dim_lat), KV=latent (dim_lat)
    - Decoder: Q=token (dim_tok), KV=latent (dim_lat)

    Predictor-Corrector update:
        Predictor:  Q_pred = CrossAttn(Q, KV, KV)
        Gradient:   ∇_Q L  via autograd (exact or first-order)
    """

    def __init__(self, dim_q, dim_kv, num_heads=8, drop_path=0.,
                 init_values=1e-5, qkv_bias=False, loss_type='dot product',
                 momentum_beta=0.9, muon_steps=0, approximate=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_q // num_heads

        self.loss_type = loss_type
        self.momentum_beta = momentum_beta
        self.muon_steps = muon_steps
        self.approximate = approximate

        self.q_proj = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.scale = head_dim ** -0.5

        self.eta = nn.Parameter(init_values * torch.ones(dim_q))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, momentum_in=None):
        """Forward: PC update for Q attending to KV.

        Args:
            x_q: Query tensor (the one being updated), shape ``(B, M, C_q)``.
            x_kv: Key-Value tensor, shape ``(B, N, C_kv)``.
            momentum_in: Momentum from previous layer.

        Returns:
            x_q_final: Updated Q.
            momentum_out: Updated momentum.
        """
        B, M, C_q = x_q.shape
        N = x_kv.shape[1]

        if momentum_in is None:
            momentum_in = torch.zeros_like(x_q)

        # Prepare K, V from x_kv
        k = self.k_proj(x_kv).reshape(B, N, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)

        with torch.enable_grad():
            Q_param = x_q.clone().requires_grad_(True)

            if self.approximate:
                # First-order: detach Q from the graph
                q = self.q_proj(x_q).reshape(B, M, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
            else:
                # Exact: Q depends on the parameter, autograd captures Jacobian
                q = self.q_proj(Q_param).reshape(B, M, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)

            # Predictor: Q_pred = softmax(Q K^T / sqrt(d)) V
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            Q_pred = (attn @ v).transpose(1, 2).reshape(B, M, C_q)

            # Correction Loss
            if self.loss_type == 'l2':
                loss = torch.sum((Q_param - Q_pred) ** 2) / 2.0
            elif self.loss_type == 'dot product':
                loss = -torch.einsum('bmd,bmd->b', Q_param, Q_pred).sum()
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # Exact gradient via autograd
            grad = torch.autograd.grad(loss, Q_param, create_graph=True)[0]

        # Momentum update
        momentum_out = self.momentum_beta * momentum_in + grad
        update_dir = momentum_out

        # Optional: orthogonalize via Newton-Schulz (Muon)
        if self.muon_steps > 0:
            update_dir = zeropower_via_newtonschulz5(update_dir, steps=self.muon_steps)

        # Final update
        delta_S = update_dir.to(self.eta.dtype)
        x_q_final = x_q - self.drop_path(self.eta * delta_S)

        return x_q_final, momentum_out
