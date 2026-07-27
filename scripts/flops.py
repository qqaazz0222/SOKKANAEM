"""FLOPs vs change rate (IDEA.md §4.2's core efficiency figure).

Analytic MAC count, derived from the config so it tracks dim/depth changes.
The point is to state honestly *which* MACs a static patch actually removes:

  TemporalBlock, Δ-gating, static token: Δ̃=0 removes dt_proj, the B half of
  bc_proj and the state recurrence — but NOT in_proj/out_proj/C, because the
  block still reads the retained state with the current frame's C and adds
  D·x. Exactness is a property of the *state*, not a licence to skip the
  readout.
  SpatialBlock: unaffected by Δ-gating (no temporal state). Only the opt-in
  static-patch output cache (§4.5, an approximation) skips it.
  embed / decoder: dense, always paid.

Run: python scripts/flops.py [--dim 384] [--size 256]
"""
import argparse


def macs(dim=192, depth=4, d_state=16, p=16, size=256, chunk=16,
         decoder="conv", scan_directions=2, local_conv=False, bins=0,
         dpt_width=64):
    """Per-frame MACs, split into the pieces that scale with active% and the
    pieces that do not. Returns a dict of component -> (fixed, per_token)."""
    N = (size // p) ** 2
    di = 2 * dim  # d_inner (expand=2)
    n_t = (depth + 1) // 2   # TemporalBlocks (interleaved T,S,T,S,...)
    n_s = depth // 2         # SpatialBlocks

    in_proj = dim * 2 * di
    dt_proj = di * di
    bc_proj = di * 2 * d_state
    out_proj = di * dim
    recur = 3 * di * d_state          # decay, input term, readout
    pairwise = chunk * di * d_state   # chunked segment-sum (spatial scan only)

    # per token, one SelectiveSSM
    ssm_all = in_proj + dt_proj + bc_proj + out_proj + recur
    # what a static token still costs under Δ-gating: readout path only
    ssm_static = in_proj + (bc_proj // 2) + out_proj + di * d_state

    embed = N * 3 * p * p * dim
    # decoder: dim->128 @ N, up4 -> 128->64, up4 -> 64->32 -> 1, at full res
    r2 = N * 16
    r3 = N * 256  # = size*size, full resolution
    if decoder == "shuffle":  # v6: channel work at patch res + pixel-shuffle
        dec = (N * dim * 128 * 9 + N * 128 * (p * p) * 9
               + r3 * (1 * 16 * 9 + 16 * 1 * 9))
    elif decoder == "dpt":    # v8: multi-scale fusion + RGB skip stem
        w = dpt_width
        widths = (w, 32, 16)                     # fusion at 1/8, 1/4, 1/2
        res = {k: (size // k) ** 2 for k in (2, 4, 8)}
        dec = depth * N * dim * w                # per-block 1x1 reassemble
        dec += (res[2] * 3 * 16 * 9 + res[4] * 16 * 32 * 9
                + res[8] * 32 * w * 9)           # stride-2 RGB stem
        for cin, c, k in zip((w,) + widths[:-1], widths, (8, 4, 2)):
            dec += res[k] * (cin * c + c * c * 9)   # reduce 1x1 + mix 3x3
        dec += res[2] * (widths[-1] * widths[-1] * 9
                         + widths[-1] * max(bins, 1) * 9)   # head
    else:  # original: 64->32 and 32->1 convs run at FULL resolution
        dec = (N * dim * 128 * 9 + r2 * 128 * 64 * 9
               + r3 * 64 * 32 * 9 + r3 * 32 * 1 * 9)

    return {
        "N": N,
        "embed": embed,
        "decoder": dec,
        "temporal_active": n_t * ssm_all,          # per active token
        "temporal_static": n_t * ssm_static,       # per static token
        # per token: 2 scans per direction pair (4-way cross-scan doubles it)
        "spatial": n_s * scan_directions * (ssm_all + pairwise),
        # depthwise 3x3 x2 on the token grid: dense, but ~0.6% of one scan
        "local": n_s * 2 * dim * 9 if local_conv else 0,
    }


def curve(m, active, spatial_cache=False, temporal_cache=False):
    """Total per-frame MACs at a given active fraction."""
    N = m["N"]
    a, s = N * active, N * (1 - active)
    total = m["embed"] + m["decoder"] + N * m["local"]
    # temporal_cache reuses a static token's block *output*, so it drops the
    # readout that Δ-gating alone cannot skip (58.5% of an active token)
    total += a * m["temporal_active"]
    total += 0 if temporal_cache else s * m["temporal_static"]
    total += (a if spatial_cache else N) * m["spatial"]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--decoder", default="conv",
                    choices=["conv", "shuffle", "dpt"])
    ap.add_argument("--scan-directions", type=int, default=2, choices=[2, 4])
    ap.add_argument("--local-conv", action="store_true")
    ap.add_argument("--bins", type=int, default=0)
    args = ap.parse_args()

    m = macs(dim=args.dim, depth=args.depth, d_state=args.d_state,
             size=args.size, decoder=args.decoder,
             scan_directions=args.scan_directions,
             local_conv=args.local_conv, bins=args.bins)
    full = curve(m, 1.0)
    N = m["N"]
    print(f"dim={args.dim} depth={args.depth} d_state={args.d_state} "
          f"size={args.size} decoder={args.decoder} "
          f"scan={args.scan_directions}way local_conv={args.local_conv} "
          f"bins={args.bins} tokens={N}")
    print(f"\nper-frame breakdown at active=100%: {full/1e9:.3f} GMAC")
    for k, per_tok in (("embed", None), ("decoder", None),
                       ("temporal", m["temporal_active"]),
                       ("spatial", m["spatial"]), ("local", m["local"])):
        v = m[k] if per_tok is None else N * per_tok
        print(f"  {k:<9} {v/1e9:7.3f} GMAC  ({v/full*100:5.1f}%)")
    frac = m["temporal_static"] / m["temporal_active"]
    print(f"\nstatic token still costs {frac*100:.1f}% of an active token "
          f"inside a TemporalBlock (Δ-gating removes dt_proj + B + recurrence "
          f"only)")

    print("\nactive%   GMAC (Δ-gate)  vs full   GMAC (+sp cache)  vs full   "
          "GMAC (+both)  vs full   ideal")
    print("-" * 96)
    for a in (1.0, 0.8, 0.56, 0.4, 0.23, 0.166, 0.112, 0.05, 0.0):
        g = curve(m, a)
        gc = curve(m, a, spatial_cache=True)
        gb = curve(m, a, spatial_cache=True, temporal_cache=True)
        print(f"{a*100:6.1f}   {g/1e9:8.3f}      {g/full*100:5.1f}%   "
              f"{gc/1e9:11.3f}      {gc/full*100:5.1f}%   "
              f"{gb/1e9:8.3f}      {gb/full*100:5.1f}%   {a*100:5.1f}%")
    print("\n'ideal' = the compute-proportional-to-change-rate claim taken "
          "literally. The gap is the honest cost of a dense embed+decoder "
          "plus a readout that Δ-gating cannot skip.")


if __name__ == "__main__":
    main()
