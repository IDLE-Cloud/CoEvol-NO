"""PCFFN equivalence and speed tests: autograd vs analytical gradient.

Run: python test_pcffn.py
"""

import time
import torch
from coevol_no.blocks import PCFFN

B, N, C = 4, 1024, 128


def make_pair(loss_type, seed=0):
    torch.manual_seed(seed)
    auto = PCFFN(dim=C, hidden_dim=C, loss_type=loss_type, analytical=False)
    torch.manual_seed(seed)
    anal = PCFFN(dim=C, hidden_dim=C, loss_type=loss_type, analytical=True)
    anal.load_state_dict(auto.state_dict())
    auto.eval(); anal.eval()
    return auto, anal


def check(name, a, b):
    diff = (a - b).abs().max().item()
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  [{status}] {name}: {diff:.2e}")
    return diff < 1e-4


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(B, N, C)
    all_pass = True

    print("=" * 60)
    print("PCFFN: Autograd vs Analytical Equivalence")
    print("=" * 60)

    # --- Dot product ---
    auto, anal = make_pair('dot product')
    out_a, mom_a = auto(x.clone(), None)
    out_b, mom_b = anal(x.clone(), None)
    all_pass &= check("output (dot)", out_a, out_b)
    all_pass &= check("momentum (dot)", mom_a, mom_b)

    # --- L2 ---
    auto2, anal2 = make_pair('l2')
    out_a2, mom_a2 = auto2(x.clone(), None)
    out_b2, mom_b2 = anal2(x.clone(), None)
    all_pass &= check("output (L2)", out_a2, out_b2)
    all_pass &= check("momentum (L2)", mom_a2, mom_b2)

    # --- Multi-layer consistency ---
    print("\nMulti-layer consistency:")
    auto3, anal3 = make_pair('dot product')
    mom_a = mom_b = None
    for i in range(4):
        out_a, mom_a = auto3(x.clone(), mom_a)
        out_b, mom_b = anal3(x.clone(), mom_b)
    all_pass &= check(f"4-layer output", out_a, out_b)

    # --- Speed ---
    print(f"\n{'='*60}")
    print(f"Speed: B={B}, N={N}, C={C}, hidden={C}")
    print(f"{'='*60}")

    auto_b, anal_b = make_pair('dot product')
    for _ in range(5):
        auto_b(x.clone(), None)
        anal_b(x.clone(), None)

    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        auto_b(x.clone(), None)
    t_auto = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        anal_b(x.clone(), None)
    t_anal = (time.perf_counter() - t0) / reps * 1000

    print(f"  Autograd:  {t_auto:.2f} ms")
    print(f"  Analytical: {t_anal:.2f} ms")
    print(f"  Speedup: {t_auto/t_anal:.2f}x")

    print(f"\n{'='*60}")
    print(f"Result: {'ALL PASS' if all_pass else 'FAILED'}")
    print(f"{'='*60}")
