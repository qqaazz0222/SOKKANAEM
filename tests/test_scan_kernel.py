"""T2-10: the fused Triton scan must be a drop-in for the reference scan.

The chunked PyTorch scan in ssm.py is the definition of correct here — it is
what training differentiates through. The kernel only replaces it at
inference, so the two must agree numerically, and the one property the whole
method rests on (Delta=0 copies the state exactly) must survive verbatim.

CUDA-only: skipped on machines without a GPU, where the fallback is the only
path anyway.
"""
import pytest
import torch

from sokkanaem import scan_triton
from sokkanaem.ssm import SelectiveSSM

cuda_only = pytest.mark.skipif(
    not (torch.cuda.is_available() and scan_triton.HAVE_TRITON),
    reason="needs CUDA + triton")


@cuda_only
@pytest.mark.parametrize("B,L,D", [(1, 64, 96), (1, 256, 96), (2, 64, 96),
                                   (1, 13, 96)])
def test_kernel_matches_reference_scan(B, L, D):
    torch.manual_seed(0)
    m = SelectiveSSM(D, d_state=16).cuda().eval()
    u = torch.randn(B, L, D, device="cuda")
    mask = (torch.rand(B, L, device="cuda") < 0.4).float()
    h0 = torch.randn(B, m.d_inner, 16, device="cuda") * 0.1

    with torch.no_grad():
        with torch.enable_grad():          # grad mode on -> reference path
            y_ref, h_ref = m(u, mask, h0)
        y_k, h_k = m(u, mask, h0)          # no_grad -> triton path

    assert (y_ref - y_k).abs().max() < 1e-4 * y_ref.abs().max().clamp(min=1.0)
    assert (h_ref - h_k).abs().max() < 1e-4 * h_ref.abs().max().clamp(min=1.0)


@cuda_only
def test_gated_off_tokens_copy_the_state_bit_exactly():
    """Delta=0 gives exp(0)=1 and a zero input term, so the recurrence has to
    hand back h0 unchanged — not close, identical. This is the claim the
    kernel is not allowed to soften."""
    torch.manual_seed(0)
    m = SelectiveSSM(96, d_state=16).cuda().eval()
    u = torch.randn(1, 32, 96, device="cuda")
    h0 = torch.randn(1, m.d_inner, 16, device="cuda")

    with torch.no_grad():
        _, h = m(u, torch.zeros(1, 32, device="cuda"), h0)

    assert torch.equal(h, h0)


@cuda_only
def test_training_still_uses_the_differentiable_path():
    """The kernel is forward-only; a graph must never route through it."""
    m = SelectiveSSM(96, d_state=16).cuda()
    u = torch.randn(1, 16, 96, device="cuda", requires_grad=True)
    y, _ = m(u)
    y.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
