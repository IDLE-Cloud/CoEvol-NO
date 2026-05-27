"""Scalability benchmark: analytical vs autograd across sequence lengths.

Compares speed and memory from N=4K to N=1M, for both inference-only
and training (forward+backward) modes.

Run:
    python tests/benchmark_scaling.py                 # CPU smoke test
    python tests/benchmark_scaling.py --gpu           # full GPU benchmark
    python tests/benchmark_scaling.py --gpu --max_N 524288  # limit max N

Results saved to: results/benchmark_scaling_<device>.csv
"""

import sys
import os
import csv
import gc
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from coevol_no.attention import DualExactStateAttention
from coevol_no.blocks import DualExactBlock, PCFFN
from coevol_no.analytical import (
    compute_s_gradient_analytical,
    compute_x_gradient_analytical,
)

# ── Sequence lengths to test ──
SEQ_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# ── Fixed hyperparameters ──
B = 1          # batch size (keep 1 for large N)
M = 32         # latent count
C = 128        # hidden dim
H = 8          # heads
WARMUP = 5
REPEATS = 20


# ===========================================================================
# Timing & memory utilities
# ===========================================================================

class Timer:
    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == 'cuda'

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self._t0) * 1000  # ms


def bench_fn(fn, device, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.type == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return np.mean(times), np.std(times)


def bench_with_memory(fn, device, warmup=WARMUP, repeats=REPEATS):
    """Benchmark timing + measure peak memory separately."""
    t_mean, t_std = bench_fn(fn, device, warmup, repeats)
    mem = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        fn()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return t_mean, t_std, mem


# ===========================================================================
# Benchmark: S gradient
# ===========================================================================

def bench_s_gradient(N, device, mode='inference'):
    """Benchmark S gradient: autograd vs analytical.

    mode='inference': just compute the gradient (no parameter backward)
    mode='training':  compute gradient, then backward through model params
    """
    xl = torch.randn(B, M, C, device=device)
    xt = torch.randn(B, N, C, device=device)

    attn_auto = DualExactStateAttention(
        dim_lat=C, dim_tok=C, num_heads=H,
        s_loss_type='dot product', s_approximate=False,
    ).to(device)

    attn_anal = DualExactStateAttention(
        dim_lat=C, dim_tok=C, num_heads=H,
        s_loss_type='dot product', s_approximate=False,
    ).to(device)
    attn_anal.load_state_dict(attn_auto.state_dict())
    attn_auto.eval()
    attn_anal.eval()

    if mode == 'inference':
        fn_auto = lambda: attn_auto._compute_s_gradient(xl, xt)
        fn_anal = lambda: compute_s_gradient_analytical(attn_anal, xl, xt)
    else:
        # Training: forward + backward through model params
        def fn_auto():
            attn_auto.zero_grad()
            grad, pred = attn_auto._compute_s_gradient(xl, xt)
            loss = grad.sum()
            loss.backward()
            return grad, pred

        def fn_anal():
            attn_anal.zero_grad()
            grad, pred = compute_s_gradient_analytical(attn_anal, xl, xt)
            loss = grad.sum()
            loss.backward()
            return grad, pred

    t_auto, t_std_auto, mem_auto = bench_with_memory(fn_auto, device)
    t_anal, t_std_anal, mem_anal = bench_with_memory(fn_anal, device)

    del xl, xt, attn_auto, attn_anal
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return t_auto, t_std_auto, t_anal, t_std_anal, mem_auto, mem_anal


# ===========================================================================
# Benchmark: X gradient
# ===========================================================================

def bench_x_gradient(N, device, mode='inference'):
    xl = torch.randn(B, M, C, device=device)
    xt = torch.randn(B, N, C, device=device)
    dS = torch.randn(B, M, C, device=device)

    attn_auto = DualExactStateAttention(
        dim_lat=C, dim_tok=C, num_heads=H,
        x_loss_type='dot product', x_exact_update=True,
    ).to(device)

    attn_anal = DualExactStateAttention(
        dim_lat=C, dim_tok=C, num_heads=H,
        x_loss_type='dot product', x_exact_update=True,
    ).to(device)
    attn_anal.load_state_dict(attn_auto.state_dict())
    attn_auto.eval()
    attn_anal.eval()

    if mode == 'inference':
        fn_auto = lambda: attn_auto._compute_x_gradient(xl, xt, dS)
        fn_anal = lambda: compute_x_gradient_analytical(attn_anal, xl, xt, dS)
    else:
        def fn_auto():
            attn_auto.zero_grad()
            grad, pred = attn_auto._compute_x_gradient(xl, xt, dS)
            loss = grad.sum()
            loss.backward()

        def fn_anal():
            attn_anal.zero_grad()
            grad, pred = compute_x_gradient_analytical(attn_anal, xl, xt, dS)
            loss = grad.sum()
            loss.backward()

    t_auto, t_std_auto, mem_auto = bench_with_memory(fn_auto, device)
    t_anal, t_std_anal, mem_anal = bench_with_memory(fn_anal, device)

    del xl, xt, dS, attn_auto, attn_anal
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return t_auto, t_std_auto, t_anal, t_std_anal, mem_auto, mem_anal


# ===========================================================================
# Benchmark: PCFFN
# ===========================================================================

def bench_pcffn(N, device, mode='inference'):
    xt = torch.randn(B, N, C, device=device)

    pcffn_auto = PCFFN(dim=C, hidden_dim=C, analytical=False).to(device)
    pcffn_anal = PCFFN(dim=C, hidden_dim=C, analytical=True).to(device)
    pcffn_anal.load_state_dict(pcffn_auto.state_dict())
    pcffn_auto.eval()
    pcffn_anal.eval()

    if mode == 'inference':
        fn_auto = lambda: pcffn_auto(xt.clone(), None)
        fn_anal = lambda: pcffn_anal(xt.clone(), None)
    else:
        def fn_auto():
            pcffn_auto.zero_grad()
            out, _ = pcffn_auto(xt.clone(), None)
            out.sum().backward()

        def fn_anal():
            pcffn_anal.zero_grad()
            out, _ = pcffn_anal(xt.clone(), None)
            out.sum().backward()

    t_auto, t_std_auto, mem_auto = bench_with_memory(fn_auto, device)
    t_anal, t_std_anal, mem_anal = bench_with_memory(fn_anal, device)

    del xt, pcffn_auto, pcffn_anal
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return t_auto, t_std_auto, t_anal, t_std_anal, mem_auto, mem_anal


# ===========================================================================
# Benchmark: Full block (end-to-end)
# ===========================================================================

def bench_block(N, device, mode='inference'):
    xl = torch.randn(B, M, C, device=device)
    xt = torch.randn(B, N, C, device=device)

    block_auto = DualExactBlock(
        dim_lat=C, dim_tok=C, num_heads=H, mlp_ratio=1.0,
        drop_path=0., x_exact_update=False, s_approximate=False,
        use_pc_ffn=True, pc_ffn_analytical=False,
    ).to(device)

    block_anal = DualExactBlock(
        dim_lat=C, dim_tok=C, num_heads=H, mlp_ratio=1.0,
        drop_path=0., x_exact_update=False, s_approximate=False,
        use_pc_ffn=True, pc_ffn_analytical=True,
    ).to(device)
    block_anal.load_state_dict(block_auto.state_dict())
    block_auto.eval()
    block_anal.eval()

    if mode == 'inference':
        fn_auto = lambda: block_auto(xl, xt, None, None, None)
        fn_anal = lambda: block_anal(xl, xt, None, None, None)
    else:
        def fn_auto():
            block_auto.zero_grad()
            out = block_auto(xl, xt, None, None, None)
            # out is (x_lat, x_tok, ms, mx, mf)
            loss = out[0].sum() + out[1].sum()
            loss.backward()

        def fn_anal():
            block_anal.zero_grad()
            out = block_anal(xl, xt, None, None, None)
            loss = out[0].sum() + out[1].sum()
            loss.backward()

    t_auto, t_std_auto, mem_auto = bench_with_memory(fn_auto, device)
    t_anal, t_std_anal, mem_anal = bench_with_memory(fn_anal, device)

    del xl, xt, block_auto, block_anal
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return t_auto, t_std_auto, t_anal, t_std_anal, mem_auto, mem_anal


# ===========================================================================
# Main
# ===========================================================================

BENCHMARKS = {
    'S_grad': bench_s_gradient,
    'X_grad': bench_x_gradient,
    'PCFFN': bench_pcffn,
    'Block': bench_block,
}


def run_all(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    device_name = 'cpu'
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(0).replace(' ', '_')

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"Config: B={B}, M={M}, C={C}, H={H}")
    print(f"Sequence lengths: {SEQ_LENGTHS}")
    print(f"Modes: inference, training")
    print()

    # Filter N by max_N
    seq_lengths = [n for n in SEQ_LENGTHS if n <= args.max_N]

    results = []
    header = ['op', 'N', 'mode', 't_auto_ms', 't_auto_std', 't_anal_ms', 't_anal_std',
              'speedup', 'mem_auto_mb', 'mem_anal_mb', 'mem_ratio']

    for bench_name, bench_fn in BENCHMARKS.items():
        for N in seq_lengths:
            for mode in ['inference', 'training']:
                label = f"{bench_name} N={N:>8d} {mode:10s}"
                print(f"  {label} ... ", end='', flush=True)

                try:
                    t_a, t_a_s, t_b, t_b_s, mem_a, mem_b = bench_fn(N, device, mode)
                    speedup = t_a / t_b if t_b > 0 else float('inf')
                    mem_ratio = (mem_a / mem_b) if (mem_a and mem_b and mem_b > 0) else None

                    mem_a_str = f"{mem_a:.0f}" if mem_a else "n/a"
                    mem_b_str = f"{mem_b:.0f}" if mem_b else "n/a"
                    print(f"auto={t_a:8.2f}ms  anal={t_b:8.2f}ms  "
                          f"speedup={speedup:.2f}x  "
                          f"mem={mem_a_str}->{mem_b_str}MB")

                    results.append({
                        'op': bench_name, 'N': N, 'mode': mode,
                        't_auto_ms': f"{t_a:.4f}", 't_auto_std': f"{t_a_s:.4f}",
                        't_anal_ms': f"{t_b:.4f}", 't_anal_std': f"{t_b_s:.4f}",
                        'speedup': f"{speedup:.4f}",
                        'mem_auto_mb': f"{mem_a:.1f}" if mem_a else '',
                        'mem_anal_mb': f"{mem_b:.1f}" if mem_b else '',
                        'mem_ratio': f"{mem_ratio:.3f}" if mem_ratio else '',
                    })
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() or 'memory' in str(e).lower():
                        print("OOM - skipping")
                        results.append({
                            'op': bench_name, 'N': N, 'mode': mode,
                            't_auto_ms': 'OOM', 't_auto_std': '', 't_anal_ms': 'OOM',
                            't_anal_std': '', 'speedup': '', 'mem_auto_mb': 'OOM',
                            'mem_anal_mb': '', 'mem_ratio': '',
                        })
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"ERROR: {e}")
                        raise

    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = f'results/benchmark_scaling_{device_name}.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY (device={device})")
    print(f"{'='*100}")
    print(f"{'Op':>8s} {'N':>8s} {'Mode':>10s} | "
          f"{'Auto(ms)':>10s} {'Anal(ms)':>10s} {'Speedup':>8s} | "
          f"{'Mem_auto':>10s} {'Mem_anal':>10s} {'MemRatio':>8s}")
    print('-' * 100)
    for r in results:
        if r['t_auto_ms'] == 'OOM':
            print(f"{r['op']:>8s} {r['N']:>8d} {r['mode']:>10s} | {'OOM':>10s}")
        else:
            print(f"{r['op']:>8s} {r['N']:>8d} {r['mode']:>10s} | "
                  f"{float(r['t_auto_ms']):>10.2f} {float(r['t_anal_ms']):>10.2f} "
                  f"{float(r['speedup']):>7.2f}x | "
                  f"{r['mem_auto_mb']:>10s} {r['mem_anal_mb']:>10s} {r['mem_ratio']:>8s}")

    plot_results(out_path)


def plot_results(csv_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['t_auto_ms'] == 'OOM':
                continue
            rows.append(r)
    if not rows:
        print("No data to plot.")
        return

    ops = list(dict.fromkeys(r['op'] for r in rows))
    modes = ['inference', 'training']
    colors = {'S_grad': '#2563eb', 'X_grad': '#dc2626', 'PCFFN': '#16a34a', 'Block': '#9333ea'}
    markers = {'inference': 'o', 'training': 's'}

    # ── Figure 1: Speedup vs N (separate inference / training) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mode in zip(axes, modes):
        for op in ops:
            data = [(int(r['N']), float(r['speedup']))
                    for r in rows if r['op'] == op and r['mode'] == mode]
            data.sort()
            if not data:
                continue
            ns, sps = zip(*data)
            ax.plot(ns, sps, color=colors.get(op, 'gray'),
                    marker=markers[mode], markersize=5, linewidth=1.5, label=op)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Sequence Length (N)')
        ax.set_ylabel('Speedup (auto / anal)')
        ax.set_title(f'Speedup: Analytical vs Autograd ({mode})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    out1 = csv_path.replace('.csv', '_speedup.png')
    fig.savefig(out1, dpi=150)
    print(f"Saved: {out1}")
    plt.close(fig)

    # ── Figure 2: Raw time vs N (log-log) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mode in zip(axes, modes):
        for op in ops:
            data_a = [(int(r['N']), float(r['t_auto_ms']))
                      for r in rows if r['op'] == op and r['mode'] == mode]
            data_b = [(int(r['N']), float(r['t_anal_ms']))
                      for r in rows if r['op'] == op and r['mode'] == mode]
            data_a.sort(); data_b.sort()
            if not data_a:
                continue
            ns_a, ta = zip(*data_a)
            ns_b, tb = zip(*data_b)
            ax.plot(ns_a, ta, color=colors.get(op, 'gray'), linestyle='-',
                    linewidth=1.5, label=f'{op} (autograd)')
            ax.plot(ns_b, tb, color=colors.get(op, 'gray'), linestyle='--',
                    linewidth=1.5, label=f'{op} (analytical)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Sequence Length (N)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Wall-clock Time ({mode})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = csv_path.replace('.csv', '_time.png')
    fig.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close(fig)

    # ── Figure 3: Memory (GPU only) ──
    mem_rows = [r for r in rows if r['mem_auto_mb'] and r['mem_anal_mb']]
    if mem_rows:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, mode in zip(axes, modes):
            for op in ops:
                data_a = [(int(r['N']), float(r['mem_auto_mb']))
                          for r in mem_rows if r['op'] == op and r['mode'] == mode]
                data_b = [(int(r['N']), float(r['mem_anal_mb']))
                          for r in mem_rows if r['op'] == op and r['mode'] == mode]
                data_a.sort(); data_b.sort()
                if not data_a:
                    continue
                ns_a, ma = zip(*data_a)
                ns_b, mb = zip(*data_b)
                ax.plot(ns_a, ma, color=colors.get(op, 'gray'), linestyle='-',
                        linewidth=1.5, label=f'{op} (autograd)')
                ax.plot(ns_b, mb, color=colors.get(op, 'gray'), linestyle='--',
                        linewidth=1.5, label=f'{op} (analytical)')
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Sequence Length (N)')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title(f'GPU Memory ({mode})')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out3 = csv_path.replace('.csv', '_memory.png')
        fig.savefig(out3, dpi=150)
        print(f"Saved: {out3}")
        plt.close(fig)
    else:
        print("(Memory plot skipped — no GPU data)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analytical vs Autograd Scaling Benchmark')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--max_N', type=int, default=1048576, help='Max sequence length')
    parser.add_argument('--plot_only', type=str, default=None,
                        help='Only generate plots from existing CSV (skip benchmark)')
    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.plot_only)
    else:
        run_all(args)
