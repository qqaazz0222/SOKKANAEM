"""Latency and energy per frame on an edge device, dense path against sparse.

The desktop GPU cannot answer whether the MAC reduction of Delta-gating becomes
time: after the fused kernel it is overhead-bound, latency is flat in activity,
and compiled dense beats the sparse path at every activity level (Section 5.7).
That question needs hardware where arithmetic is the constraint. This script is
the same measurement on such hardware, and it is deliberately dependency-light
and Python 3.6-compatible so it runs on a JetPack 4.6 image (torch 1.10,
Python 3.6) as well as on a 64-bit Raspberry Pi OS (torch 2.x, CPU only).

What each device can and cannot close:
  - Raspberry Pi 4 (CPU): the cleanest test that MAC reduction converts to time,
    with no kernel-launch confound. Says nothing about the fused kernel.
  - Jetson Nano B01 (Maxwell, sm_53): compute-bound GPU, but Triton cannot
    target sm_53, so the fused scan is unavailable and the reference chunked
    scan runs instead. Report it as the reference implementation.
  - Jetson Orin and later: the only class that can measure the reported
    implementation.

Power comes from whatever the device exposes, via --power-cmd: a shell command
printing watts on stdout. Defaults are wired for tegrastats (Jetson) and
nvidia-smi (desktop); a Pi needs an external meter and its own command.

    # Raspberry Pi 4, CPU, no power rail
    python scripts/edge_bench.py --ckpt final.pt --device cpu --threads 4

    # Jetson Nano B01, GPU, on-board rails
    python scripts/edge_bench.py --ckpt final.pt --device cuda --power tegra
"""
from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sokkanaem import from_checkpoint            # noqa: E402

# tegrastats prints rails in two shapes across generations:
#   Nano/TX2:  "POM_5V_IN 4104/4104"      (mW, no unit)
#   Xavier/Orin: "VDD_IN 4000mW/4000mW"
# board-in rail first, GPU rail only as a fallback: searching one alternation
# would pick whichever appears earliest in the line, which differs by generation
TEGRA_W = [re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)(?:mW)?/"),
           re.compile(r"(?:VDD_SYS_GPU|POM_5V_GPU)\s+(\d+)(?:mW)?/")]

# Reading the INA3221 sysfs node directly beats parsing tegrastats: no
# subprocess, no sudo, and no hang. `tegrastats --interval 100 | head -1`
# deadlocks -- head exits after one line and tegrastats does not die on
# SIGPIPE, so the parent waits forever and every sample is silently lost.
# Channel 0 is the board input rail (POM_5V_IN / VDD_IN) on every generation
# that exposes it; JetPack 4.x reports mW under ina3221x, newer L4T reports
# uW under hwmon.
RAILS = [("/sys/bus/i2c/drivers/ina3221x/*/iio:device0/in_power0_input", 1e-3),
         ("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power1_input", 1e-6),
         ("/sys/class/hwmon/hwmon*/power1_input", 1e-6)]


def find_rail():
    """(path, scale-to-watts) of the first readable power rail, or (None, 0)."""
    import glob
    for pattern, scale in RAILS:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as f:
                    float(f.read().strip())
                return path, scale
            except Exception:
                continue
    return None, 0.0


class FixedRatio(object):
    """Force a known active fraction, so the x axis is identical everywhere.

    A detector-driven sweep changes both the mask and which frames are active;
    this changes only the fraction, which is what the compute claim is about.
    """

    def __init__(self, ratio, p):
        self.ratio, self.p = ratio, p
        self.keyframe_every = 10 ** 9

    def __call__(self, frame, det):
        b, _, h, w = frame.shape
        n = (h // self.p) * (w // self.p)
        k = int(round(self.ratio * n))
        m = torch.zeros(b, n, device=frame.device, dtype=frame.dtype)
        if k:
            m[:, :k] = 1.0
        return m, {"prev": frame}

    def is_keyframe(self, det):
        return det is None


class Power(object):
    """Mean watts over the measured window, from a sysfs rail or a command.

    source is either ("rail", path, scale) or ("cmd", shell_command, None).
    Failures are counted, not swallowed: a silent zero once cost a whole
    measurement round.
    """

    def __init__(self, source, period=0.05):
        self.source, self.period = source, period
        self.samples, self.errors = [], 0
        self._stop = None

    def _read(self):
        kind, spec, scale = self.source
        if kind == "rail":
            with open(spec) as f:
                return float(f.read().strip()) * scale
        out = subprocess.check_output(spec, shell=True,
                                      stderr=subprocess.STDOUT)
        out = out.decode("utf-8", "replace")
        m = next((r.search(out) for r in TEGRA_W if r.search(out)), None)
        if m:
            return float(m.group(1)) / 1000.0
        return float(out.strip().splitlines()[0])

    def start(self):
        if self.source is None:
            return self
        self._stop = threading.Event()

        def poll():
            while not self._stop.is_set():
                try:
                    self.samples.append(self._read())
                except Exception:
                    self.errors += 1
                self._stop.wait(self.period)

        self._t = threading.Thread(target=poll)
        self._t.daemon = True
        self._t.start()
        return self

    def stop(self):
        if self._stop is not None:
            self._stop.set()
            self._t.join(2)
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def timed(model, size, ratio, dtype, device, iters, warmup, power_src):
    model.detector = FixedRatio(ratio, model.p)
    model.dense_above = 0.0            # the sweep forces the ratio, so the
    frame = torch.rand(1, 3, size, size, device=device, dtype=dtype)
    state, active, t0, pw = None, [], 0.0, None
    with torch.no_grad():
        for i in range(warmup + iters):
            if i == warmup:
                if device == "cuda":
                    torch.cuda.synchronize()
                pw = Power(power_src).start()
                t0 = time.time()
            _, state, info = model.step(frame, state)
            if i >= warmup:
                active.append(info["active_ratio"])
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    watts = pw.stop() if pw is not None else 0.0
    got = sum(active) / len(active)
    return {"ms": dt * 1000, "fps": 1.0 / dt, "active": got, "watts": watts,
            "mj": watts * dt * 1000,
            "power_errors": pw.errors if pw is not None else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads; set it, or a Pi silently uses all four "
                         "and the number stops meaning anything")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--active", type=float, nargs="+",
                    default=[0.05, 0.15, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--power", default="none",
                    choices=["none", "tegra", "nvidia-smi", "cmd", "probe"],
                    help="tegra reads the INA3221 sysfs rail directly; probe "
                         "only lists what rails exist and exits")
    ap.add_argument("--power-cmd", default=None,
                    help="shell command printing watts (or tegrastats output)")
    args = ap.parse_args()

    if args.power == "none":
        power_src = None
    elif args.power == "tegra":
        path, scale = find_rail()
        if path is None:
            ap.error("no readable INA3221 rail found under /sys. Run with "
                     "--power probe to see what exists, or use --power cmd "
                     "with an external meter.")
        power_src = ("rail", path, scale)
    elif args.power == "nvidia-smi":
        power_src = ("cmd", "nvidia-smi --query-gpu=power.draw "
                            "--format=csv,noheader,nounits", None)
    elif args.power == "probe":
        import glob
        found, readable = [], []
        print("rail candidates:")
        for pattern, scale in RAILS:
            for hit in sorted(glob.glob(pattern)):
                found.append(hit)
                try:
                    with open(hit) as f:
                        raw = f.read().strip()
                    readable.append(hit)
                    print("  %s = %s (x%g -> %.3f W)"
                          % (hit, raw, scale, float(raw) * scale))
                except Exception as e:
                    print("  %s NOT readable: %s" % (hit, e))
        if readable:
            print("\nusable. run with --power tegra")
        elif found:
            # the rail exists and the process cannot read it: a permission
            # problem, not a hardware one. sysfs modes reset at boot, so
            # opening the node beats running the whole benchmark as root --
            # a --user torch install is invisible to root's interpreter.
            print("\nrails exist but are not readable by this user. Open them:")
            for hit in found:
                print("  sudo chmod a+r %s" % hit)
            print("  (resets on reboot; add a udev rule to persist)")
        else:
            print("\nno power rail on this board -- use --power cmd with an "
                  "external meter")
        return
    else:
        if not args.power_cmd:
            ap.error("--power cmd needs --power-cmd")
        power_src = ("cmd", args.power_cmd, None)
    if args.threads:
        torch.set_num_threads(args.threads)

    dtype = torch.half if args.half else torch.float
    print("device=%s dtype=%s size=%d threads=%s torch=%s"
          % (args.device, "fp16" if args.half else "fp32", args.size,
             torch.get_num_threads(), torch.__version__))
    if args.device == "cuda":
        print("gpu=%s" % torch.cuda.get_device_name(0))
    try:
        from sokkanaem import scan_triton
        print("fused scan available: %s" % scan_triton.HAVE_TRITON)
    except Exception as e:
        print("fused scan unavailable: %s" % e)

    if power_src is not None:
        print("power source: %s" % (power_src[1],))
    print("\ncache  active%   ms/frame      FPS    watts   mJ/frame")
    print("-" * 56)
    for cache in (True, False):
        model = from_checkpoint(args.ckpt, args.device, spatial_cache=cache,
                                temporal_cache=cache).eval().to(dtype)
        for ratio in args.active:
            r = timed(model, args.size, ratio, dtype, args.device,
                      args.iters, args.warmup, power_src)
            note = ""
            if power_src is not None and not r["watts"]:
                note = "   <- power read failed %dx" % r["power_errors"]
            print("%-5s  %6.1f  %9.2f  %7.1f  %7.2f  %9.2f%s"
                  % ("on" if cache else "off", r["active"] * 100, r["ms"],
                     r["fps"], r["watts"], r["mj"], note))


if __name__ == "__main__":
    main()
