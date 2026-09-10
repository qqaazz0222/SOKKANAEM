"""Sequential development profiling and matched feature/output/motion studies."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qstream_study import evaluate, quality_gate
from scripts.qdelta_study import HybridStream
from sokkanaem.model import from_checkpoint
from sokkanaem.protocol import file_record, runtime_versions
from sokkanaem.qstream import Q0ExactStream
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.qquality import decode_with_grad, depth_retention, motion_grid, align_state
from sokkanaem.detector import ChangeDetector


class MotionStream(HybridStream):
    def __init__(self, exact, adapter, warp=False, refresh_every=4, reset_memory=False, no_delta=False):
        super().__init__(exact, adapter, adaptive=False)
        self.warp = warp
        self.refresh_every = refresh_every
        self.reset_memory = reset_memory
        self.no_delta = no_delta
        self.detector.keyframe_every = refresh_every

    @torch.no_grad()
    def step(self, frame, state=None):
        image = F.interpolate(frame, (518, 518), mode="bilinear", align_corners=False)
        force = state is None or state["age"] >= self.refresh_every-1
        score = None if force else F.avg_pool2d((image-state["anchor"]).square().mean(1, keepdim=True), 14)[:, 0]
        mask, det = self.detector.gate(score, 1, 37, 37, frame.device, None if state is None else state["det"])
        if force:
            features = self.adapter.refresh(self.exact.encode(frame))
            state = {"features": features, "anchor": image.clone(), "age": 0, "det": det}
            updated = 0
        else:
            features = state["features"]
            if self.warp:
                grid, _ = motion_grid(state["previous"], frame, features["grid"])
                features = align_state(features, grid)
            if self.reset_memory:
                features = {**features, "h": torch.zeros_like(features["h"])}
            updated = 0
            if not self.no_delta:
                features, info = self.adapter.update(frame, mask, features)
                updated = info["updated_patches"]
            state = {**state, "features": features, "age": state["age"]+1, "det": det}
        state["previous"] = frame.clone()
        pred = self.exact.decode(state["features"])
        return pred, state, {"full_refresh": force, "backbone_calls": int(force),
                             "active_ratio": float(mask.mean()), "updated_patches": updated,
                             "cache_age": state["age"], "warp": self.warp}


def load_exact(record):
    assert file_record(record["path"]) == record
    model = from_checkpoint(record["path"], "cuda").eval().requires_grad_(False)
    weights = torch.load(record["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(weights.get("ema") or weights.get("model") or weights, strict=True)
    return Q0ExactStream(model)


@torch.no_grad()
def profile(exact, adapter, dev):
    """Diagnostic synchronized component timings, not additive E2E speed claims."""
    times = {}
    def measure(name, fn):
        torch.cuda.synchronize(); start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.setdefault(name, []).append((time.perf_counter()-start)*1000)
        return result
    for i in range(80):
        s = dev[i % len(dev)]
        t = 1 + i % (len(s["rgb"])-1)
        frame, previous = s["rgb"][t:t+1].cuda(), s["rgb"][t-1:t].cuda()
        encoded = exact.encode(previous)
        state = adapter.refresh(encoded)
        x, size = measure("preprocess", lambda: exact.q._prep(frame))
        out = measure("encoder", lambda: exact.q.net.backbone.forward_with_filtered_kwargs(
            x, output_hidden_states=True, output_attentions=False))
        current = {"features": list(out.feature_maps), "pooled": out.hidden_states[-1][:, 1:].mean(1),
                   "grid": encoded["grid"], "size": size}
        measure("cache_refresh", lambda: adapter.refresh(current))
        for ratio in (.25, 1.):
            mask = torch.zeros(1, 1369, device="cuda"); mask[:, :round(1369*ratio)] = 1
            measure("ssm_update_active_"+str(ratio), lambda: adapter.update(frame, mask, state))
        disp = measure("decoder", lambda: exact.q._decode(current["features"], current["grid"]))
        measure("calibration", lambda: exact.q._calibrate(disp, current["pooled"], size))
        grid, _ = measure("motion_match", lambda: motion_grid(previous, frame, (37, 37)))
        measure("warp_features_and_h", lambda: align_state(state, grid))
        measure("dense_end_to_end", lambda: exact.step(frame))
    return {"note": "FP32 GPU resident; separate synchronized wall timings; each component's first16 samples discarded. Instrumentation changes scheduling. Do not sum as end-to-end latency.",
            "components": {k: {"mean_ms": statistics.mean(v[16:]), "median_ms": statistics.median(v[16:]),
                                 "samples_ms": v[16:]} for k, v in times.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("profile", "feature", "output"), required=True)
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("Use a new output directory")
    parent = Path("work_dirs/qdelta_20260908")
    pp = json.loads((parent/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = QDeltaSSM().cuda()
    adapter.load_state_dict(torch.load(parent/"adapter.pt", map_location="cuda", weights_only=False)["adapter"], strict=True)
    dev_path = Path("work_dirs/qstream_20260908/development_data.pt")
    dev = torch.load(dev_path, weights_only=False)
    cache = torch.load(parent/"teacher_cache.pt", weights_only=False)
    train_paths = {str(Path(p).resolve()) for s in cache for pair in s["pairs"] for p in pair}
    dev_paths = {str(Path(p).resolve()) for s in dev for pair in s["pairs"] for p in pair}
    assert not train_paths & dev_paths
    args.out.mkdir(parents=True)
    protocol = {"role": "development_only_no_final_no_promotion", "checkpoint": pp["checkpoint"],
                "initial_adapter": file_record(parent/"adapter.pt"), "teacher_cache": file_record(parent/"teacher_cache.pt"),
                "development": file_record(dev_path), "mode": args.mode, "warp": args.warp,
                "steps": args.steps, "seed": 736, "lr": .0003, "clips": len(cache),
                "objective": "featureL1 + .1 pooledL1" + (" + 5 logdepthL1 + 10 multiscale_loggradientL1" if args.mode == "output" else ""),
                "training": "matched continuation of same600step adapter; same clip order; one refresh plus3 updates; no GT loss",
                "motion": "backward RGB local match74px radius2; bilinear feature AND h transport; no learned confidence or certified occlusion",
                "policies": ["periodicK4", "periodicK2", "K4_h_reset", "K4_no_delta"],
                "quality_gate": "unchanged scripts.qstream_study.quality_gate; speed requires mean latency reduction>=5%",
                "runtime": runtime_versions(), "gpu": torch.cuda.get_device_name(),
                "code": [file_record(p) for p in [__file__, "sokkanaem/qquality.py", "sokkanaem/qdelta_ssm.py",
                          "sokkanaem/qstream.py", "scripts/qstream_study.py", "scripts/qdelta_study.py",
                          "sokkanaem/qmodel.py", "sokkanaem/ssm.py", "sokkanaem/detector.py",
                          "scripts/quality_refinement.py", "sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json", protocol)
    if args.mode == "profile":
        result = profile(exact, adapter.eval(), dev)
        q.write(args.out/"profile.json", result)
        print({k: v["mean_ms"] for k, v in result["components"].items()}, flush=True)
        return
    targets = []
    if args.mode == "output":
        with torch.no_grad():
            for s in cache:
                targets.append(exact.decode(exact.encode(s["rgb"].cuda().float())).cpu())
        torch.save(targets, args.out/"train_q0_depth.pt")
    opt = torch.optim.AdamW(adapter.parameters(), lr=.0003, weight_decay=.0001)
    detector = ChangeDetector(patch_size=14, tau_on=.001, tau_off=.0005, keyframe_every=4, dilate=True)
    gen = torch.Generator().manual_seed(736)
    history = []; start = time.perf_counter()
    for step in range(1, args.steps+1):
        ci = int(torch.randint(len(cache), (), generator=gen)); s = cache[ci]
        rgb = s["rgb"].cuda().float(); fs = [x.cuda().float() for x in s["features"]]; pooled = s["pooled"].cuda()
        state = adapter.refresh({"features": [x[:1] for x in fs], "pooled": pooled[:1], "grid": s["grid"], "size": s["size"]})
        images = F.interpolate(rgb, (518, 518), mode="bilinear", align_corners=False)
        _, det = detector.gate(None, 1, 37, 37, rgb.device, None)
        losses = []; components = []
        for t in range(1, len(rgb)):
            score = F.avg_pool2d((images[t:t+1]-images[:1]).square().mean(1, keepdim=True), 14)[:, 0]
            mask, det = detector.gate(score, 1, 37, 37, rgb.device, det)
            if args.warp:
                grid, _ = motion_grid(rgb[t-1:t], rgb[t:t+1], state["grid"])
                state = align_state(state, grid)
            state, _ = adapter.update(rgb[t:t+1], mask, state)
            feature = sum((a-b[t:t+1]).abs().mean() for a, b in zip(state["features"], fs))/len(fs)
            feature = feature + .1*(state["pooled"]-pooled[t:t+1]).abs().mean()
            depth = feature.new_zeros(()); edge = feature.new_zeros(())
            if args.mode == "output":
                pred = decode_with_grad(exact, state)
                depth, edge = depth_retention(pred, targets[ci][t:t+1].cuda())
            losses.append(feature+5*depth+10*edge)
            components.append(torch.stack((feature.detach(), depth.detach(), edge.detach())))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        else: norm = torch.tensor(0.)
        if step == 1 or step % 50 == 0:
            row = {"step": step, "loss": float(loss.detach()), "components": torch.stack(components).mean(0).tolist(),
                   "grad_norm": float(norm), "elapsed_s": time.perf_counter()-start}
            history.append(row); q.write(args.out/"history.json", history)
            print(args.mode, "warp", args.warp, row, flush=True)
    assert all(p.grad is None for p in exact.parameters())
    adapter.eval()
    torch.save({"adapter": adapter.state_dict(), "initial": protocol["initial_adapter"], "steps": args.steps,
                "mode": args.mode, "warp": args.warp}, args.out/"adapter.pt")
    baseline, preds = evaluate(exact, dev)
    torch.save(preds, args.out/"dense_predictions.pt")
    results = {"baseline": baseline, "policies": {}, "history": history, "promoted": False}
    policies = [("periodicK4", {}), ("periodicK2", {"refresh_every": 2}),
                ("K4_h_reset", {"reset_memory": True}), ("K4_no_delta", {"no_delta": True})]
    for name, config in policies:
        measured, preds = evaluate(MotionStream(exact, adapter, warp=args.warp, **config), dev)
        measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
        measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
        measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
        results["policies"][name] = measured
        torch.save(preds, args.out/(name+"_predictions.pt"))
        q.write(args.out/"results.json", results)
        print(name, measured["metrics"]["balanced"], "ms", measured["mean_ms"], "gate", measured["quality_gate"], flush=True)
    for record in protocol["code"] + [protocol["checkpoint"], protocol["initial_adapter"]]:
        assert file_record(record["path"]) == record


if __name__ == "__main__": main()
