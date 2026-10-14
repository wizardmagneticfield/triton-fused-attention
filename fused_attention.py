"""Fused attention forward pass in Triton (FlashAttention-style).

softmax(Q K^T * scale) V is computed in a single kernel. Each program
handles a BLOCK_M slice of queries and streams over the keys/values in
BLOCK_N chunks, keeping a running max and running sum (online softmax),
so the full N x N attention matrix is never written to global memory.
"""

import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O,
    sm_scale,
    stride_b, stride_n,          # q/k/v/o share the same (B*H, N, D) contiguous layout
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    PRECISION: tl.constexpr,     # "ieee" for strict fp32, "tf32" allows tensor cores
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    base = bh * stride_b
    q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Work in base 2: exp(x) = exp2(x * log2(e))
    qk_scale = sm_scale * 1.4426950408889634

    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)   # skip blocks above the diagonal
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        # K loaded transposed: (HEAD_DIM, BLOCK_N)
        k = tl.load(K + base + cols[None, :] * stride_n + offs_d[:, None],
                    mask=cols[None, :] < N_CTX, other=0.0)
        qk = tl.dot(q, k, input_precision=PRECISION) * qk_scale

        mask = cols[None, :] < N_CTX
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cols[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V + base + cols[:, None] * stride_n + offs_d[None, :],
                    mask=cols[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_n + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _block_sizes(dtype: torch.dtype, head_dim: int):
    # fp32 tiles take 2x the shared memory, so use smaller key blocks.
    if dtype == torch.float32:
        return 64, 32 if head_dim >= 128 else 64
    return 128 if head_dim <= 64 else 64, 64


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = False, sm_scale: float | None = None,
                    allow_tf32: bool = False) -> torch.Tensor:
    """q, k, v: (batch, heads, seq_len, head_dim) CUDA tensors, fp16/bf16/fp32."""
    assert q.shape == k.shape == v.shape, "self-attention shapes only"
    # TRITON_INTERPRET=1 runs the kernel on the CPU (slow, but handy for debugging).
    assert q.is_cuda or os.environ.get("TRITON_INTERPRET") == "1", "inputs must be CUDA tensors"
    assert q.dtype == k.dtype == v.dtype
    B, H, N, D = q.shape
    assert D in (16, 32, 64, 128), "head_dim must be 16, 32, 64 or 128"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q_ = q.reshape(B * H, N, D).contiguous()
    k_ = k.reshape(B * H, N, D).contiguous()
    v_ = v.reshape(B * H, N, D).contiguous()
    o = torch.empty_like(q_)

    BLOCK_M, BLOCK_N = _block_sizes(q.dtype, D)
    precision = "tf32" if (q.dtype == torch.float32 and allow_tf32) else "ieee"
    num_warps = 4 if D <= 64 else 8
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    _attn_fwd_kernel[grid](
        q_, k_, v_, o, sm_scale,
        q_.stride(0), q_.stride(1), N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal, PRECISION=precision,
        num_warps=num_warps, num_stages=2,
    )
    return o.reshape(B, H, N, D)


def reference_attention(q, k, v, causal=False, sm_scale=None):
    """Plain PyTorch attention that materialises the full N x N score matrix."""
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        N = q.shape[-2]
        mask = torch.ones(N, N, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)
