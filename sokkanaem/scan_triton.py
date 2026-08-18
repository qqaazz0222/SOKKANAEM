"""Fused selective-scan kernel (Triton).

The reference scan in ssm.py is chunked and materialises a
(B, C, C, P, S) pairwise-decay tensor per chunk. That is exact and
differentiable, but at inference it is pure memory traffic: for the spatial
path (L=64, P=384, S=16) it moves ~25 MB per chunk to do 400 kMAC of work,
and it dominates the frame (71% of a sparse step, measured).

This kernel keeps the recurrence in registers instead:

    h_l = h_{l-1} * exp(dt_l * A) + (dt_l * x_l) * B_l
    y_l = sum_s h_l * C_l

Δ-gating stays *exactly* exact here — dt_l = 0 gives exp(0) = 1 and a zero
input term, so h is copied bit-for-bit, which is the property the whole
method rests on (and what tests/test_arch.py pins).

Forward only: training keeps the differentiable chunked path. The scan is
sequential in L and parallel over (batch, channel), so occupancy comes from
P/BLOCK_P programs — small, but the work per program is small too.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                   # CPU-only installs
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _scan_kernel(
        dt_ptr, x_ptr, A_ptr, B_ptr, C_ptr, h0_ptr, y_ptr, hl_ptr,
        L, P, S,
        s_dt_b, s_dt_l, s_bc_b, s_bc_l, s_h_b, s_h_p,
        HAS_H0: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)

        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        offs_s = tl.arange(0, BLOCK_S)
        mask_p = offs_p < P
        mask_s = offs_s < S
        mask_ps = mask_p[:, None] & mask_s[None, :]

        # A is (P, S), shared by every batch and step
        A = tl.load(A_ptr + offs_p[:, None] * S + offs_s[None, :],
                    mask=mask_ps, other=0.0)

        if HAS_H0:
            h = tl.load(h0_ptr + pid_b * s_h_b + offs_p[:, None] * s_h_p
                        + offs_s[None, :], mask=mask_ps, other=0.0)
        else:
            h = tl.zeros([BLOCK_P, BLOCK_S], dtype=tl.float32)

        for l in range(L):
            dt = tl.load(dt_ptr + pid_b * s_dt_b + l * s_dt_l + offs_p,
                         mask=mask_p, other=0.0)
            xv = tl.load(x_ptr + pid_b * s_dt_b + l * s_dt_l + offs_p,
                         mask=mask_p, other=0.0)
            Bv = tl.load(B_ptr + pid_b * s_bc_b + l * s_bc_l + offs_s,
                         mask=mask_s, other=0.0)
            Cv = tl.load(C_ptr + pid_b * s_bc_b + l * s_bc_l + offs_s,
                         mask=mask_s, other=0.0)

            # dt = 0 (gated-off token) => decay is exactly 1, input term is
            # exactly 0 => h is unchanged bit-for-bit.
            decay = tl.exp(dt[:, None] * A)
            h = h * decay + (dt * xv)[:, None] * Bv[None, :]

            y = tl.sum(h * Cv[None, :], axis=1)
            tl.store(y_ptr + pid_b * s_dt_b + l * s_dt_l + offs_p, y,
                     mask=mask_p)

        tl.store(hl_ptr + pid_b * s_h_b + offs_p[:, None] * s_h_p
                 + offs_s[None, :], h, mask=mask_ps)


def selective_scan(dt, x, A, Bp, Cp, h0=None):
    """dt, x: (B, L, P) - A: (P, S) - Bp, Cp: (B, L, S) - h0: (B, P, S)|None.

    Returns (y, h_last) with y: (B, L, P), h_last: (B, P, S).
    Same contract as the chunked scan in ssm.py, forward only.
    """
    B, L, P = dt.shape
    S = A.shape[1]
    dt, x, Bp, Cp = (t.contiguous() for t in (dt, x, Bp, Cp))
    A = A.contiguous()
    y = torch.empty_like(dt)
    h_last = torch.empty((B, P, S), device=dt.device, dtype=dt.dtype)
    if h0 is not None:
        h0 = h0.contiguous()

    BLOCK_P = 32 if P >= 32 else triton.next_power_of_2(P)
    grid = (B, triton.cdiv(P, BLOCK_P))
    _scan_kernel[grid](
        dt, x, A, Bp, Cp, h0 if h0 is not None else dt, y, h_last,
        L, P, S,
        dt.stride(0), dt.stride(1), Bp.stride(0), Bp.stride(1),
        h_last.stride(0), h_last.stride(1),
        HAS_H0=h0 is not None,
        BLOCK_P=BLOCK_P, BLOCK_S=triton.next_power_of_2(S),
        num_warps=4,
    )
    return y, h_last


def usable(dt):
    """Forward-only CUDA fast path; anything else falls back to the
    differentiable reference scan."""
    return (HAVE_TRITON and dt.is_cuda and not torch.is_grad_enabled()
            and dt.dtype in (torch.float32, torch.float16, torch.bfloat16))
