"""Selective SSM with Δ-gating (IDEA.md §3.2).

Pure-PyTorch implementation. The scan is chunked (segment-sum, Mamba-2
style): within a chunk the recurrence is closed-form via pairwise decay
factors exp(Lcum_i − Lcum_j), i ≥ j — exponents are always ≤ 0 (A < 0,
Δ ≥ 0), so this is numerically stable with no clamping. Chunks carry the
state sequentially, so the Python loop runs L/CHUNK times instead of L.
The math is exact — identical to the step-by-step recurrence.

Discretization (Mamba simplified ZOH):
    Ābar = exp(Δ · A),  B̄x = Δ · B · x
    h_i  = Ābar_i * h_{i-1} + B̄x_i
Gating: Δ̃ = mask · Δ.  mask=0 ⇒ Ābar=1, B̄x=0 ⇒ h_i = h_{i-1} exactly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """One selective-scan mixer. Used for both spatial and temporal axes."""

    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(dim, 2 * self.d_inner)  # x, z(gate)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.bc_proj = nn.Linear(self.d_inner, 2 * d_state)
        self.out_proj = nn.Linear(self.d_inner, dim)

        # A: diagonal, negative real (S4D-real init)
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # dt bias init so softplus(dt) starts in [1e-3, 1e-1]
        dt = torch.exp(torch.rand(self.d_inner)
                       * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))

    def _params(self, u, mask):
        """u: (B, L, D). Returns x, z, dt, Bp, Cp with Δ-gating applied."""
        x, z = self.in_proj(u).chunk(2, dim=-1)          # (B, L, d_inner)
        dt = F.softplus(self.dt_proj(x))                  # (B, L, d_inner)
        if mask is not None:
            dt = dt * mask.unsqueeze(-1)                  # Δ-gating: the core trick
        Bp, Cp = self.bc_proj(x).chunk(2, dim=-1)         # (B, L, d_state)
        return x, z, dt, Bp, Cp

    def _finish(self, y, x, z):
        y = y + self.D * x
        return self.out_proj(y * F.silu(z))

    def forward(self, u, mask=None, h0=None):
        """Scan over L. u: (B, L, D), mask: (B, L) 0/1 or None,
        h0: (B, d_inner, d_state) or None. Returns (out, h_last)."""
        B, L, _ = u.shape
        x, z, dt, Bp, Cp = self._params(u, mask)
        A = -torch.exp(self.A_log)                        # (d_inner, d_state)

        h = h0 if h0 is not None else u.new_zeros(B, self.d_inner, self.d_state)
        s = dt.unsqueeze(-1) * A                          # (B, L, P, S), all ≤ 0
        b = (dt * x).unsqueeze(-1) * Bp.unsqueeze(2)      # (B, L, P, S); mask=0 ⇒ 0
        # ponytail: CHUNK=16 caps the (B, C, C, P, S) pairwise tensor; raise
        # for fewer launches at the cost of memory (training keeps it for grad).
        CHUNK = 16
        ys = []
        for c0 in range(0, L, CHUNK):
            sc, bc = s[:, c0:c0 + CHUNK], b[:, c0:c0 + CHUNK]
            C = sc.shape[1]
            lc = sc.cumsum(1)                             # (B, C, P, S)
            # pairwise decay exp(lc_i - lc_j), j ≤ i: exponents ≤ 0 ⇒ stable.
            # j > i must be -inf *before* exp — those exponents are ≥ 0 and
            # overflow to inf, and inf * 0 = NaN if masked after.
            tri = torch.ones(C, C, dtype=torch.bool, device=u.device).tril()
            w = (lc.unsqueeze(2) - lc.unsqueeze(1)) \
                .masked_fill(~tri.view(1, C, C, 1, 1), -torch.inf).exp()
            hc = lc.exp() * h.unsqueeze(1) \
                + torch.einsum("bijps,bjps->bips", w, bc)  # (B, C, P, S)
            ys.append(torch.einsum("bcps,bcs->bcp", hc, Cp[:, c0:c0 + CHUNK]))
            h = hc[:, -1]
        y = torch.cat(ys, dim=1)
        return self._finish(y, x, z), h

    def step(self, u, mask=None, h=None):
        """Single time step (temporal streaming). u: (B, D), mask: (B,) or None."""
        out, h = self.forward(u.unsqueeze(1),
                              None if mask is None else mask.unsqueeze(1), h)
        return out.squeeze(1), h


class BiSpatialSSM(nn.Module):
    """Bidirectional raster scan over patches within one frame.
    # ponytail: 2 directions, not VMamba's 4-way cross-scan; add if accuracy demands.
    """

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.fwd = SelectiveSSM(dim, d_state)
        self.bwd = SelectiveSSM(dim, d_state)

    def forward(self, u):
        yf, _ = self.fwd(u)
        yb, _ = self.bwd(u.flip(1))
        return yf + yb.flip(1)
