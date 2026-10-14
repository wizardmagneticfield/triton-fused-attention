# Triton Fused Attention Kernel

A FlashAttention-style forward pass written in [Triton](https://github.com/triton-lang/triton). `softmax(QKᵀ / √d) V` runs in **one** block-tiled kernel:

1. Each program loads a `BLOCK_M × d` tile of queries into SRAM.
2. It streams over keys and values in `BLOCK_N` chunks, computing `QKᵀ` for that chunk.
3. An **online softmax** keeps a running max `m` and running sum `l`, rescaling the accumulator as new chunks arrive, so the full `N × N` score matrix never touches global memory.
4. The chunk's probabilities are multiplied by `V` and accumulated; one division by `l` at the end.

Memory grows as O(N) instead of O(N²), and you skip two large round-trips to HBM.

Supports fp16 / bf16 / fp32, causal masking, head dims 16–128, and sequence lengths that aren't block multiples.

## Setup

Needs an NVIDIA GPU (Colab's free T4 works).

```bash
pip install -r requirements.txt
```

## Test correctness

```bash
pytest -v -s test_fused_attention.py
```

Compares against an fp32 PyTorch reference for N = 512 … 8192, fp16 / bf16 / fp32, causal and
non-causal, plus ragged lengths and every supported head dim (bf16 needs an Ampere or newer GPU).

No GPU? Triton's interpreter runs the kernel on the CPU, which is slow but works for the small tests:

```bash
TRITON_INTERPRET=1 pytest -v test_fused_attention.py -k "ragged or small"
```

## Benchmark

```bash
python benchmark.py --dtype fp16 --batch 4 --heads 16 --head-dim 64
python benchmark.py --dtype fp16 --causal
```

Reports median latency (via `triton.testing.do_bench`) and peak extra memory for:
- `pytorch_naive`: matmul → softmax → matmul
- `pytorch_sdpa`: PyTorch's built-in `scaled_dot_product_attention` (which uses its own FlashAttention kernel, so it's a tough baseline)
- `triton_fused`: this kernel

## Results

> Fill in from **your** runs.

GPU: _____  dtype: _____  B/H/D: _____

| Seq len | Naive (ms) | SDPA (ms) | Triton (ms) | Speedup vs naive | Memory saved vs naive |
|---|---|---|---|---|---|
| 1024 | | | | | |
| 4096 | | | | | |
| 8192 | | | | | |

Max abs error vs fp32 reference: fp16 ____, fp32 ____

## What I learned

_Fill in: why online softmax works, what block sizes did to performance, how you compare to SDPA and why._

## Next steps

- Backward pass
- `triton.autotune` over BLOCK_M / BLOCK_N / num_warps / num_stages
- Cross-attention (different Q and KV lengths)
