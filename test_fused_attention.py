"""Correctness tests: Triton kernel vs. a float32 PyTorch reference.

Run:  pytest -v test_fused_attention.py
No GPU: TRITON_INTERPRET=1 pytest -v test_fused_attention.py -k "ragged or small"
"""
import os

import pytest
import torch

from fused_attention import fused_attention, reference_attention

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu" and not INTERPRET,
                                reason="needs a CUDA GPU (or TRITON_INTERPRET=1)")

# atol per dtype. The reference is always computed in fp32 from the same inputs;
# fp16/bf16 round the probabilities and the output, so they get looser bounds.
TOLERANCES = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 3e-2}


def bf16_supported():
    return DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64])
def test_matches_reference(dtype, seq_len, causal, head_dim):
    if INTERPRET:
        pytest.skip("too slow for the CPU interpreter; run on a GPU")
    if dtype == torch.bfloat16 and not bf16_supported():
        pytest.skip("bf16 tl.dot needs an Ampere (sm_80) or newer GPU")
    torch.manual_seed(0)
    B, H = 1, 4
    q, k, v = (torch.randn(B, H, seq_len, head_dim, device=DEVICE, dtype=dtype) for _ in range(3))

    out = fused_attention(q, k, v, causal=causal)
    ref = reference_attention(q.float(), k.float(), v.float(), causal=causal)

    max_err = (out.float() - ref).abs().max().item()
    print(f"{dtype} N={seq_len} causal={causal}: max abs err = {max_err:.2e}")
    assert max_err < TOLERANCES[dtype]


@pytest.mark.parametrize("seq_len", [100, 777])  # not a multiple of the block size
def test_ragged_lengths(seq_len):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, seq_len, 32, device=DEVICE, dtype=torch.float16) for _ in range(3))
    for causal in (False, True):
        out = fused_attention(q, k, v, causal=causal)
        ref = reference_attention(q.float(), k.float(), v.float(), causal=causal)
        assert (out.float() - ref).abs().max().item() < TOLERANCES[torch.float16]


@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_small_all_head_dims(head_dim, dtype):
    """Every supported head dim, causal and not, at a size the interpreter can handle."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 150, head_dim, device=DEVICE, dtype=dtype) for _ in range(3))
    for causal in (False, True):
        out = fused_attention(q, k, v, causal=causal)
        ref = reference_attention(q.float(), k.float(), v.float(), causal=causal)
        assert (out.float() - ref).abs().max().item() < TOLERANCES[dtype]
