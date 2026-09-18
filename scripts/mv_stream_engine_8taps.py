"""Streaming partial refresh, adapted from scripts/mv_stream_engine.py for the 8-taps+band
winning arm (mambavision_da2_hr_8taps_band8k_ft, appendix O) instead of the original 4-tap
recipe. New file; scripts/mv_stream_engine.py and its OUT tree are read from (attention_partial
only) but never written to.

Mechanism is UNCHANGED from the original engine -- this only widens which taps feed the
decoder, from {1,3,5,7} to all of {0..7}:

  prefix   stage-2's 4 Mamba mixer blocks (0-3), dense every frame -- a scan output depends
           on its whole prefix, so there is no way to skip a token here. ALL FOUR of their
           outputs are now taps (the original engine only kept blocks 1 and 3).
  attend   stage-2's 4 attention blocks (4-7), partially refreshed on the highest-drift
           tiles the Q0 way. ALL FOUR cached outputs are now taps (the original kept only
           blocks 5 and 7).
  decode   the 8-tap decoder (scripts/mv_decoder_probe_bigdecoder.py's build_wide_decoder(8)),
           fed [mamba0, mamba1, mamba2, mamba3, attn4, attn5, attn6, attn7] in block order --
           the same order AllTapsBackbone.features() produces during training, so the
           transplanted decoder.proj indices line up.

This is the concrete test of "Mamba can't skip, attention can": if partial-refresh quality
holds up here the same way appendix G found for the plain 4-tap arm, the claim is that
skip-compute works for the attention half of this backbone regardless of how many Mamba taps
feed the decoder -- not a Mamba-specific mechanism.

Usage (repository root, `mambavision` env):
  python scripts/mv_stream_engine_8taps.py --frames N   # N limits frames per sequence (smoke)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mv_decoder_probe import BEHAVE, CalibrationHead, depth_at
from scripts.mv_decoder_probe_bigdecoder import AllTapsBackbone, OUT, build_wide_decoder
from scripts.mv_stream_profile import attention_partial

ARM = "mambavision_da2_hr_8taps_band8k_ft"
STREAM_OUT = OUT / "stream_8taps_band8k"
REPEATS, WARMUP = 3, 16


def load_model(arm=ARM):
    AllTapsBackbone.student = OUT / arm / "backbone.pt"
    backbone = AllTapsBackbone().cuda().eval()
    decoder = build_wide_decoder(backbone.dim, 8)
    decoder.load_state_dict(torch.load(OUT / arm / "decoder.pt", map_location="cpu", weights_only=False)["ema"])
    head = CalibrationHead(backbone.dim)
    head.load_state_dict(torch.load(OUT / arm / "calib.pt", map_location="cpu", weights_only=False)["head"])
    return backbone, decoder.cuda().eval(), head.cuda().eval()


class MambaVisionStream8Taps:
    def __init__(self, model, tiles, refresh_every=8, grid=4):
        self.backbone, self.decoder, self.head = model
        self.tiles, self.refresh_every = tiles, refresh_every
        m = self.backbone.model
        self.level = m.levels[2]
        self.size, self.side = self.backbone.size, self.backbone.size // 16
        self.n = self.side * self.side
        attn = self.level.blocks[4].mixer
        dev = "cuda"
        axis = torch.arange(self.side, device=dev) * grid // self.side
        self.tile_ids = (axis[:, None] * grid + axis[None, :]).flatten()
        self.tile_counts = torch.bincount(self.tile_ids, minlength=grid * grid).float()
        self.frame = torch.zeros(1, 3, 256, 256, device=dev)
        self.anchor = torch.zeros(1, self.n, 320, device=dev)
        self.caches = [(torch.zeros(1, attn.num_heads, self.n, attn.head_dim, device=dev),
                        torch.zeros(1, attn.num_heads, self.n, attn.head_dim, device=dev),
                        torch.zeros(1, self.n, 320, device=dev)) for _ in range(4)]
        self.sel_mask = torch.zeros(self.n, dtype=torch.bool, device=dev)
        self.full_idx = torch.arange(self.n, device=dev)
        counts = [int(c) for c in self.tile_counts.tolist()]
        self.counts = [] if not tiles else sorted({sum(c) for c in itertools.combinations(counts, tiles)})
        self.idx_bufs = {n: torch.zeros(n, dtype=torch.long, device=dev) for n in self.counts}
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.outputs = {}, {}
        self.age = None
        self._capture_all()

    def _prefix(self):
        m = self.backbone.model
        x = F.interpolate(self.frame, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = m.levels[1](m.levels[0](m.patch_embed((x - self.backbone.mean) / self.backbone.std)))
        x = self.backbone.window_partition(x, self.level.window_size)
        taps = []
        for i in range(4):
            x = self.level.blocks[i](x)
            taps.append(x)                    # all 4 Mamba-mixer taps, not just {1,3}
        return (x,) + tuple(taps)

    def _attention(self, idx):
        tokens = self.outputs["prefix"][0]
        self.anchor.index_copy_(1, idx, tokens.index_select(1, idx))
        x = tokens.index_select(1, idx)
        for j, i in enumerate(range(4, 8)):
            x = attention_partial(self.level.blocks[i], x, idx, *self.caches[j])
        return x

    def _score(self):
        tokens = self.outputs["prefix"][0]
        drift = (tokens - self.anchor).square().mean(-1).flatten()
        tile = torch.zeros_like(self.tile_counts).scatter_add_(0, self.tile_ids, drift) / self.tile_counts
        chosen = tile.topk(self.tiles, sorted=False).indices
        selected = torch.zeros_like(self.tile_counts, dtype=torch.bool).index_fill_(0, chosen, True)
        self.sel_mask.copy_(selected[self.tile_ids])

    def _decode(self):
        _, mamba0, mamba1, mamba2, mamba3 = self.outputs["prefix"]
        as_map = lambda t: t.transpose(1, 2).reshape(1, 320, self.side, self.side)
        feats = [as_map(mamba0), as_map(mamba1), as_map(mamba2), as_map(mamba3),
                 as_map(self.caches[0][2]), as_map(self.caches[1][2]),
                 as_map(self.caches[2][2]), as_map(self.caches[3][2])]
        depth = depth_at(self.decoder, feats, self.frame, self.size)
        return self.head(depth, feats[-1].mean((2, 3)))

    def _capture(self, name, body):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            self.outputs[name] = body()
        self.graphs[name] = graph

    @torch.no_grad()
    def _capture_all(self):
        self.frame.uniform_()
        self._capture("prefix", self._prefix)
        self.graphs["prefix"].replay()
        self._capture("dense", lambda: self._attention(self.full_idx))
        self.graphs["dense"].replay()
        if self.tiles:
            self._capture("score", self._score)
            for n in self.counts:
                self.idx_bufs[n].copy_(torch.arange(n, device="cuda"))
                self._capture(("partial", n), lambda n=n: self._attention(self.idx_bufs[n]))
        self._capture("decode", self._decode)
        torch.cuda.synchronize()

    @torch.no_grad()
    def reset(self):
        self.age = None

    @torch.no_grad()
    def step(self, frame):
        self.frame.copy_(frame)
        self.graphs["prefix"].replay()
        refresh = self.tiles is None or self.age is None or self.age + 1 >= self.refresh_every
        if refresh:
            self.graphs["dense"].replay()
            self.age, active = 0, self.n
        elif self.tiles == 0:
            self.age, active = self.age + 1, 0
        else:
            self.graphs["score"].replay()
            idx = self.sel_mask.nonzero(as_tuple=True)[0]
            active = int(idx.numel())
            self.idx_bufs[active].copy_(idx)
            self.graphs[("partial", active)].replay()
            self.age += 1
        self.graphs["decode"].replay()
        return self.outputs["decode"].clone(), active / self.n, refresh


def timed(engine, rgb):
    engine.reset()
    for i in range(WARMUP):
        engine.step(rgb[i % len(rgb)][None])
    engine.reset()
    torch.cuda.synchronize()
    preds, times, actives = [], [], []
    for i in range(len(rgb)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        depth, active, _ = engine.step(rgb[i][None])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        preds.append(depth.cpu())
        actives.append(active)
    return torch.cat(preds), times, actives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=0, help="limit frames per sequence (smoke)")
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--out", default=None)
    parser.add_argument("--refresh-every", default="8", help="comma list of keyframe intervals")
    parser.add_argument("--tiles", default="12,8,4",
                        help="comma list of active tile budgets; 0 = attention blocks only at keyframes")
    args = parser.parse_args()
    out = Path(args.out) if args.out else STREAM_OUT
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = load_model(args.arm)
    arms = {"dense": (None, 8)}
    for k in map(int, args.refresh_every.split(",")):
        for t in map(int, args.tiles.split(",")):
            arms[f"t{t}" if k == 8 else f"t{t}_k{k}"] = (t, k)
    engines = {name: MambaVisionStream8Taps(model, tiles, refresh_every=k) for name, (tiles, k) in arms.items()}
    names = list(engines)
    orders = [names, list(reversed(names)), names[2:] + names[:2]]
    results = {}
    for sequence, (source, start, end) in BEHAVE.items():
        data = torch.load(ROOT / source / f"{sequence}.pt", map_location="cpu", weights_only=False)
        end = start + min(end - start, args.frames) if args.frames else end
        rgb = (data["rgb"][start:end].permute(0, 3, 1, 2).float() / 255).cuda()
        del data
        row = results[sequence] = {name: {"runs": []} for name in names}
        stored = {}
        for repeat, order in enumerate(orders[:REPEATS]):
            for name in order:
                pred, times, actives = timed(engines[name], rgb)
                if repeat == 0:
                    stored[name] = pred
                    (out / name).mkdir(exist_ok=True)
                    torch.save(pred, out / name / f"{sequence}_predictions.pt")
                    row[name]["mean_active_ratio"] = float(np.mean(actives))
                else:
                    if not torch.equal(pred, stored[name]):
                        raise RuntimeError(f"{sequence} {name}: repeat {repeat} differs from repeat 0")
                row[name]["runs"].append({"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))})
                print("RUN", sequence, repeat, name, round(float(np.mean(times)), 3), flush=True)
        for name in names:
            runs = row[name]["runs"]
            row[name]["mean_ms"] = float(np.median([r["mean_ms"] for r in runs]))
            row[name]["p95_ms"] = float(np.median([r["p95_ms"] for r in runs]))
        probe = OUT / args.arm / "predictions_calibrated" / f"{sequence}_predictions.pt"
        if probe.exists():
            saved = torch.load(probe, map_location="cpu")[:len(stored["dense"])]
            row["dense"]["max_abs_vs_batched_probe_prediction"] = float((stored["dense"] - saved).abs().max())
        (out / "timing.json").write_text(json.dumps(results, indent=2) + "\n")
        del rgb, stored
    combined = {name: {"mean_ms": float(np.mean([results[s][name]["mean_ms"] for s in results])),
                       "p95_ms": float(np.mean([results[s][name]["p95_ms"] for s in results])),
                       "mean_active_ratio": float(np.mean([results[s][name]["mean_active_ratio"] for s in results]))}
                for name in names}
    (out / "timing.json").write_text(json.dumps({"per_sequence": results, "combined": combined}, indent=2) + "\n")
    print("\n%-6s %9s %9s %8s" % ("arm", "mean_ms", "p95_ms", "active"))
    for name, c in combined.items():
        print("%-6s %9.3f %9.3f %8.3f" % (name, c["mean_ms"], c["p95_ms"], c["mean_active_ratio"]))
    print("dense vs batched probe prediction max abs:",
          {s: results[s]["dense"].get("max_abs_vs_batched_probe_prediction") for s in results})
    print("STREAM_DONE")


if __name__ == "__main__":
    main()
