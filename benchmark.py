"""Latency + peak-memory benchmark: fused Triton kernel vs. PyTorch.

Compares against:
  - "pytorch_naive": matmul -> softmax -> matmul (materialises N x N scores)
  - "pytorch_sdpa":  torch.nn.functional.scaled_dot_product_attention

Run:  python benchmark.py --dtype fp16 --batch 4 --heads 16 --head-dim 64
"""
import argparse
import csv

import torch
import torch.nn.functional as F
import triton

from fused_attention import fused_attention, reference_attention

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def measure(fn, warmup=10, rep=50):
    """Returns (median latency ms, peak extra memory MB) for one call."""
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
    return ms, peak_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--causal", action="store_true")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    dtype = DTYPES[args.dtype]
    print(f"GPU: {torch.cuda.get_device_name()}  dtype={args.dtype}  B={args.batch} "
          f"H={args.heads} D={args.head_dim} causal={args.causal}\n")
    header = ["seq_len", "impl", "latency_ms", "peak_mem_mb"]
    rows = []
    print(f"{'N':>6} {'impl':>14} {'ms':>9} {'peak MB':>10} {'speedup':>8} {'mem saved':>10}")

    for N in args.seq_lens:
        torch.manual_seed(0)
        shape = (args.batch, args.heads, N, args.head_dim)
        q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))

        impls = {
            "pytorch_naive": lambda: reference_attention(q, k, v, causal=args.causal),
            "pytorch_sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal),
            "triton_fused": lambda: fused_attention(q, k, v, causal=args.causal),
        }
        results = {}
        for name, fn in impls.items():
            try:
                results[name] = measure(fn)
            except torch.cuda.OutOfMemoryError:
                results[name] = (float("nan"), float("nan"))
                torch.cuda.empty_cache()

        base_ms, base_mb = results["pytorch_naive"]
        for name, (ms, mb) in results.items():
            sp = base_ms / ms if ms == ms else float("nan")
            saved = 100 * (1 - mb / base_mb) if base_mb == base_mb and base_mb > 0 else float("nan")
            print(f"{N:>6} {name:>14} {ms:>9.3f} {mb:>10.1f} {sp:>7.2f}x {saved:>9.1f}%")
            rows.append([N, name, f"{ms:.4f}", f"{mb:.1f}"])
        print()

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
