"""Speed and memory benchmarks: analytical vs autograd gradient computation.

Run: python benchmark_analytical.py
"""

import time
import numpy as np
import torch

from coevol_no.attention import DualExactStateAttention
from coevol_no.blocks import DualExactBlock, PCFFN
from coevol_no.analytical import (
    compute_s_gradient_analytical,
    compute_x_gradient_analytical,
)


def bench(fn, warmup=10, repeats=100):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.std()


def peak_mb(fn):
    if not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def row(name, ma, sa, mb, sb, mema=None, memb=None):
    sp = ma / mb if mb > 0 else float('inf')
    s = f"  {name:35s} auto={ma:8.2f}±{sa:5.2f}ms  anal={mb:8.2f}±{sb:5.2f}ms  speedup={sp:.2f}x"
    if mema is not None and memb is not None:
        s += f"  mem={mema:.0f}->{memb:.0f}MB"
    print(s)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        ("Small (PDE debug)",    2,  64, 1024, 128, 8),
        ("Medium (PDE 64x64)",   4, 128, 4096, 128, 8),
        ("Large (irregular)",    4,  32, 5000, 256, 8),
    ]

    for label, B, M, N, C, H in configs:
        print(f"\n{'='*80}")
        print(f"{label}: B={B}, M={M}, N={N}, C={C}, H={H}")
        print(f"{'='*80}")

        xl = torch.randn(B, M, C, device=device)
        xt = torch.randn(B, N, C, device=device)

        # S gradient (exact, dot product)
        attn_s = DualExactStateAttention(
            dim_lat=C, dim_tok=C, num_heads=H,
            s_loss_type='dot product', s_approximate=False,
        ).to(device).eval()

        ma, sa = bench(lambda: attn_s._compute_s_gradient(xl, xt))
        mb, sb = bench(lambda: compute_s_gradient_analytical(attn_s, xl, xt))
        mema = peak_mb(lambda: attn_s._compute_s_gradient(xl, xt))
        memb = peak_mb(lambda: compute_s_gradient_analytical(attn_s, xl, xt))
        row("S gradient (exact)", ma, sa, mb, sb, mema, memb)

        # X gradient (exact, dot product)
        attn_x = DualExactStateAttention(
            dim_lat=C, dim_tok=C, num_heads=H,
            x_loss_type='dot product', x_exact_update=True,
        ).to(device).eval()
        dS = torch.randn(B, M, C, device=device)

        ma, sa = bench(lambda: attn_x._compute_x_gradient(xl, xt, dS))
        mb, sb = bench(lambda: compute_x_gradient_analytical(attn_x, xl, xt, dS))
        mema = peak_mb(lambda: attn_x._compute_x_gradient(xl, xt, dS))
        memb = peak_mb(lambda: compute_x_gradient_analytical(attn_x, xl, xt, dS))
        row("X gradient (exact)", ma, sa, mb, sb, mema, memb)

        # X gradient (first-order)
        attn_xfo = DualExactStateAttention(
            dim_lat=C, dim_tok=C, num_heads=H, x_exact_update=False,
        ).to(device).eval()

        ma, sa = bench(lambda: attn_xfo._compute_x_gradient(xl, xt, dS))
        mb, sb = bench(lambda: compute_x_gradient_analytical(attn_xfo, xl, xt, dS))
        row("X gradient (first-order)", ma, sa, mb, sb)

        # PCFFN benchmark
        pcffn_auto = PCFFN(dim=C, hidden_dim=C, analytical=False).to(device).eval()
        pcffn_anal = PCFFN(dim=C, hidden_dim=C, analytical=True).to(device).eval()
        pcffn_anal.load_state_dict(pcffn_auto.state_dict())

        ma, sa = bench(lambda: pcffn_auto(xt.clone(), None))
        mb, sb = bench(lambda: pcffn_anal(xt.clone(), None))
        mema = peak_mb(lambda: pcffn_auto(xt.clone(), None))
        memb = peak_mb(lambda: pcffn_anal(xt.clone(), None))
        row("PCFFN (exact)", ma, sa, mb, sb, mema, memb)
