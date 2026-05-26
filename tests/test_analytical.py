"""Tests verifying analytical gradient computations match autograd.

Run: python tests/test_analytical.py
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coevol_no.attention import DualExactStateAttention
from coevol_no.blocks import DualExactBlock, PCFFN
from coevol_no.analytical import (
    compute_s_gradient_analytical,
    compute_x_gradient_analytical,
)

ATOL = 1e-4
passed = 0
failed = 0


def check(name, auto, anal):
    global passed, failed
    diff = (auto - anal).abs().max().item()
    ok = diff < ATOL
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max diff = {diff:.2e}")
    if ok:
        passed += 1
    else:
        failed += 1
    return ok


# ===========================================================================
# S gradient tests
# ===========================================================================

def test_s_gradient_dot_product_exact():
    attn = DualExactStateAttention(
        dim_lat=128, dim_tok=128, num_heads=8,
        s_loss_type='dot product', s_approximate=False,
        x_exact_update=True,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 128)
    x_tok = torch.randn(2, 64, 128)

    grad_auto, pred_auto = attn._compute_s_gradient(x_lat, x_tok)
    grad_anal, pred_anal = compute_s_gradient_analytical(attn, x_lat, x_tok)

    check("S_pred (dot, exact)", pred_auto, pred_anal)
    check("S grad (dot, exact)", grad_auto, grad_anal)


def test_s_gradient_l2_exact():
    attn = DualExactStateAttention(
        dim_lat=128, dim_tok=128, num_heads=8,
        s_loss_type='l2', s_approximate=False,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 128)
    x_tok = torch.randn(2, 64, 128)

    grad_auto, _ = attn._compute_s_gradient(x_lat, x_tok)
    grad_anal, _ = compute_s_gradient_analytical(attn, x_lat, x_tok)

    check("S grad (L2, exact)", grad_auto, grad_anal)


def test_s_gradient_approximate():
    for loss_type in ['dot product', 'l2']:
        attn = DualExactStateAttention(
            dim_lat=128, dim_tok=128, num_heads=8,
            s_loss_type=loss_type, s_approximate=True,
        )
        attn.eval()
        x_lat = torch.randn(2, 32, 128)
        x_tok = torch.randn(2, 64, 128)

        grad_auto, _ = attn._compute_s_gradient(x_lat, x_tok)
        grad_anal, _ = compute_s_gradient_analytical(attn, x_lat, x_tok)

        check(f"S grad ({loss_type}, approx)", grad_auto, grad_anal)


def test_s_gradient_asymmetric_dims():
    """Test with dim_lat != dim_tok."""
    attn = DualExactStateAttention(
        dim_lat=256, dim_tok=128, num_heads=8,
        s_loss_type='dot product', s_approximate=False,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 256)
    x_tok = torch.randn(2, 64, 128)

    grad_auto, pred_auto = attn._compute_s_gradient(x_lat, x_tok)
    grad_anal, pred_anal = compute_s_gradient_analytical(attn, x_lat, x_tok)

    check("S_pred (asymmetric)", pred_auto, pred_anal)
    check("S grad (asymmetric)", grad_auto, grad_anal)


# ===========================================================================
# X gradient tests
# ===========================================================================

def test_x_gradient_dot_product_exact():
    attn = DualExactStateAttention(
        dim_lat=128, dim_tok=128, num_heads=8,
        x_loss_type='dot product', x_exact_update=True,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 128)
    x_tok = torch.randn(2, 64, 128)
    delta_S = torch.randn(2, 32, 128)

    grad_auto, pred_auto = attn._compute_x_gradient(x_lat, x_tok, delta_S)
    grad_anal, pred_anal = compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S)

    check("X_pred (dot, exact)", pred_auto, pred_anal)
    check("X grad (dot, exact)", grad_auto, grad_anal)


def test_x_gradient_l2_exact():
    attn = DualExactStateAttention(
        dim_lat=128, dim_tok=128, num_heads=8,
        x_loss_type='l2', x_exact_update=True,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 128)
    x_tok = torch.randn(2, 64, 128)
    delta_S = torch.randn(2, 32, 128)

    grad_auto, _ = attn._compute_x_gradient(x_lat, x_tok, delta_S)
    grad_anal, _ = compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S)

    check("X grad (L2, exact)", grad_auto, grad_anal)


def test_x_gradient_first_order():
    attn = DualExactStateAttention(
        dim_lat=128, dim_tok=128, num_heads=8,
        x_exact_update=False,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 128)
    x_tok = torch.randn(2, 64, 128)
    delta_S = torch.randn(2, 32, 128)

    grad_auto, _ = attn._compute_x_gradient(x_lat, x_tok, delta_S)
    grad_anal, _ = compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S)

    check("X grad (first-order)", grad_auto, grad_anal)


def test_x_gradient_asymmetric_dims():
    attn = DualExactStateAttention(
        dim_lat=256, dim_tok=128, num_heads=8,
        x_loss_type='dot product', x_exact_update=True,
    )
    attn.eval()
    x_lat = torch.randn(2, 32, 256)
    x_tok = torch.randn(2, 64, 128)
    delta_S = torch.randn(2, 32, 256)

    grad_auto, _ = attn._compute_x_gradient(x_lat, x_tok, delta_S)
    grad_anal, _ = compute_x_gradient_analytical(attn, x_lat, x_tok, delta_S)

    check("X grad (asymmetric)", grad_auto, grad_anal)


# ===========================================================================
# PCFFN tests (analytical vs autograd)
# ===========================================================================

def test_pcffn_dot_product():
    torch.manual_seed(0)
    auto = PCFFN(dim=128, hidden_dim=128, loss_type='dot product', analytical=False)
    torch.manual_seed(0)
    anal = PCFFN(dim=128, hidden_dim=128, loss_type='dot product', analytical=True)
    anal.load_state_dict(auto.state_dict())
    auto.eval(); anal.eval()
    x = torch.randn(2, 64, 128)
    out_a, _ = auto(x.clone(), None)
    out_b, _ = anal(x.clone(), None)
    check("PCFFN (dot product)", out_a, out_b)


def test_pcffn_l2():
    torch.manual_seed(0)
    auto = PCFFN(dim=128, hidden_dim=128, loss_type='l2', analytical=False)
    torch.manual_seed(0)
    anal = PCFFN(dim=128, hidden_dim=128, loss_type='l2', analytical=True)
    anal.load_state_dict(auto.state_dict())
    auto.eval(); anal.eval()
    x = torch.randn(2, 64, 128)
    out_a, _ = auto(x.clone(), None)
    out_b, _ = anal(x.clone(), None)
    check("PCFFN (L2)", out_a, out_b)


def test_pcffn_large():
    torch.manual_seed(0)
    auto = PCFFN(dim=256, hidden_dim=256, loss_type='dot product', analytical=False)
    torch.manual_seed(0)
    anal = PCFFN(dim=256, hidden_dim=256, loss_type='dot product', analytical=True)
    anal.load_state_dict(auto.state_dict())
    auto.eval(); anal.eval()
    x = torch.randn(2, 256, 256)
    out_a, _ = auto(x.clone(), None)
    out_b, _ = anal(x.clone(), None)
    check("PCFFN (large)", out_a, out_b)


# ===========================================================================
# End-to-end block test
# ===========================================================================

def test_full_block_consistency():
    """Verify analytical S + X gradients produce same block output as autograd."""
    block = DualExactBlock(
        dim_lat=128, dim_tok=128, num_heads=8, mlp_ratio=1.0,
        drop_path=0., x_exact_update=False, s_approximate=False,
    )
    block.eval()

    x_lat = torch.randn(2, 32, 128)
    x_tok = torch.randn(2, 64, 128)

    # Autograd version (original forward)
    x_lat_a, x_tok_a, ms_a, mx_a, mf_a = block(x_lat.clone(), x_tok.clone(), None, None)

    # Manual forward using analytical gradients
    x_lat_n = block.norm_lat(x_lat)
    x_tok_n = block.norm_tok(x_tok)
    attn = block.cross_attn

    grad_S, _ = compute_s_gradient_analytical(attn, x_lat_n, x_tok_n)
    ms = grad_S
    delta_S = ms
    x_lat_final = x_lat_n - attn.drop_path_s(attn.eta_s * delta_S)

    grad_X, _ = compute_x_gradient_analytical(attn, x_lat_final, x_tok_n, delta_S)
    mx = attn.x_momentum_beta * torch.zeros_like(x_tok_n) + attn.proj_tok(grad_X)
    x_tok_final = x_tok_n + attn.drop_path_x(attn.eta_x * mx)

    x_tok_final = x_tok_final + block.ls_tok2(block.mlp_tok(block.norm_tok2(x_tok_final)))

    check("Block S output", x_lat_a, x_lat_final)
    check("Block X output", x_tok_a, x_tok_final)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    torch.manual_seed(42)

    print("=" * 60)
    print("Analytical Gradient Equivalence Tests")
    print("=" * 60)

    print("\nS gradient tests:")
    test_s_gradient_dot_product_exact()
    test_s_gradient_l2_exact()
    test_s_gradient_approximate()
    test_s_gradient_asymmetric_dims()

    print("\nX gradient tests:")
    test_x_gradient_dot_product_exact()
    test_x_gradient_l2_exact()
    test_x_gradient_first_order()
    test_x_gradient_asymmetric_dims()

    print("\nPCFFN tests:")
    test_pcffn_dot_product()
    test_pcffn_l2()
    test_pcffn_large()

    print("\nEnd-to-end block test:")
    test_full_block_consistency()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
