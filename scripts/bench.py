"""Wall-clock benchmark: what the analytic MAC saving is actually worth.

scripts/flops.py counts MACs. On a 4090 the older model was kernel-launch
bound, so MACs and FPS moved apart — every efficiency claim about the current
checkpoint needs the measured curve, not the counted one. This reports, for
the real v8/arm2 model: latency and FPS against active ratio, with the caches
on and off, fp32 or fp16, plus peak VRAM and the per-stream persistent state
that decides how many cameras fit on one GPU.

Active ratio is FORCED by a detector stub rather than driven by content: the
x axis is then exact and the same for every configuration, which is what a
latency-vs-change-rate curve needs. Detector cost itself is excluded (it is a
patch-wise MSE, ~0 next to the backbone) — the numbers are backbone+decoder.

Run:
    python scripts/bench.py --ckpt work_dirs/arm2-binloss/latest.pt
    python scripts/bench.py --ckpt work_dirs/arm2-binloss/latest.pt --half
    python scripts/bench.py --ckpt ... --streams 4 --cache off --compile
"""
import argparse
import time

import torch

from sokkanaem import checkpoint_config, from_checkpoint


class FixedRatio:
    """ChangeDetector stand-in that activates a fixed fraction of patches.

    Frame 0 is full (a stream must start dense — same contract as the real
    detector's keyframe), every later frame draws an iid mask at `ratio`."""

    def __init__(self, ratio, p):
        self.ratio, self.p = ratio, p
        self.tau_on = self.tau_off = 0.0

    def __call__(self, frame, det):
        B, _, H, W = frame.shape
        n = (H // self.p) * (W // self.p)
        if det is None:
            return torch.ones(B, n, device=frame.device), 1
        m = (torch.rand(B, n, device=frame.device) < self.ratio).float()
        return m, det + 1

    def is_keyframe(self, det):
        return det is None


def state_bytes(state):
    """Bytes that persist between frames for ONE stream batch: SSM hidden
    states, spatial/temporal output caches, previous frame."""
    total = 0
    for v in state.values():
        for t in (v if isinstance(v, list) else [v]):
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def bench(model, size, ratio, streams, dtype, iters, repeat=3, warmup=20):
    """Fastest of `repeat` timings: anything else on the GPU only ever makes
    a run slower, and this box shares its 4090 with other jobs (a contended
    measurement once read 47 ms on a path that costs 19)."""
    runs = [_timed(model, size, ratio, streams, dtype, iters, warmup)
            for _ in range(repeat)]
    return min(runs, key=lambda r: r["ms"])


@torch.no_grad()
def _timed(model, size, ratio, streams, dtype, iters, warmup=20):
    dev = next(model.parameters()).device
    model.detector = FixedRatio(ratio, model.p)
    frame = torch.rand(streams, 3, size, size, device=dev, dtype=dtype)
    state, active, t0 = None, [], 0.0
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        _, state, info = model.step(frame, state)
        if i >= warmup:
            active.append(info["active_ratio"])
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    got = sum(active) / len(active)
    assert abs(got - ratio) < 0.05, f"forced mask off target: {got} != {ratio}"
    return {"ms": dt * 1000, "fps": streams / dt, "active": got,
            "vram": torch.cuda.max_memory_allocated() / 2**20,
            "state": state_bytes(state) / 2**20 / streams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=None,
                    help="default = the checkpoint's training size")
    ap.add_argument("--streams", type=int, default=1, help="batched streams")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=3,
                    help="timings per point; the fastest is reported")
    ap.add_argument("--active", type=float, nargs="+",
                    default=[0.0, 0.05, 0.15, 0.3, 0.5, 1.0])
    ap.add_argument("--cache", default="both", choices=["both", "on", "off"],
                    help="spatial+temporal output caches (the sparse path)")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--bucket", type=int, default=0,
                    help="round the gathered active-token count up to a "
                         "multiple of this (0 = off). Static shapes are what "
                         "the sparse path needs to be compilable at all")
    ap.add_argument("--compile", action="store_true",
                    help="CUDA-graph the full-compute path; without "
                         "--bucket the sparse path has a new shape every "
                         "frame and stays eager")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "wall-clock numbers need the GPU"
    if args.size is None:
        args.size = checkpoint_config(args.ckpt).get("size", 256)
    dtype = torch.half if args.half else torch.float
    configs = {"both": [True, False], "on": [True], "off": [False]}[args.cache]

    print(f"{args.ckpt}  {args.size}px  streams {args.streams}  "
          f"{'fp16' if args.half else 'fp32'}  "
          f"{'compiled' if args.compile else 'eager'}  "
          f"bucket {args.bucket or 'off'}  "
          f"{torch.cuda.get_device_name(0)}")
    hdr = "cache  active%   ms/frame     FPS   peakVRAM_MB  state_MB/stream"
    print(hdr + "\n" + "-" * len(hdr))
    for cache in configs:
        model = from_checkpoint(args.ckpt, "cuda", spatial_cache=cache,
                                temporal_cache=cache,
                                bucket=args.bucket).eval().to(dtype)
        if args.compile:
            model.compile_sparse() if cache and args.bucket else \
                model.enable_cuda_graphs()
        for ratio in args.active:
            r = bench(model, args.size, ratio, args.streams, dtype,
                      args.iters, args.repeat)
            print(f"{'on ' if cache else 'off'}    {r['active']*100:5.1f}  "
                  f"{r['ms']:9.3f}  {r['fps']:7.1f}  {r['vram']:11.1f}  "
                  f"{r['state']:15.2f}")


if __name__ == "__main__":
    main()
