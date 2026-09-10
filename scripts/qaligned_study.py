"""Matched scratch SSM/MLP training on DIS-aligned features, development only."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record, FINAL_TEST_SEQUENCES, runtime_versions
from sokkanaem.qaligned_adapter import AlignedAdapter
from sokkanaem.qflow import RGBFlowStream, gray_image, correspondence
from sokkanaem.qflow_features import RGBFlowFeatureStream, transport_features
from sokkanaem.qquality import decode_with_grad, depth_retention
from sokkanaem.qflat_state import flat_mask, flat_excess_loss
from sokkanaem.detector import ChangeDetector


def check_records(records):
    for r in records: assert file_record(r["path"]) == r, r["path"]


def train_cache(exact, train, out):
    template = AlignedAdapter().cuda().eval()
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    detector = ChangeDetector(patch_size=14, tau_on=.001, tau_off=.0005, keyframe_every=2, dilate=True)
    cache = []
    with torch.no_grad():
        for ci, s in enumerate(train):
            rgb = s["rgb"].cuda().float(); enc = exact.encode(rgb); teacher = exact.decode(enc)
            images = F.interpolate(rgb, (518, 518), mode="bilinear", align_corners=False)
            flat, valid = flat_mask(s["gt"].cuda(), s["valid"].cuda())
            for t in range(1, len(rgb), 2):
                previous = {**enc, "features": [f[t-1:t] for f in enc["features"]], "pooled": enc["pooled"][t-1:t]}
                state = template.refresh(previous)
                flow, reliable = correspondence(gray_image(rgb[t-1:t]), gray_image(rgb[t:t+1]), engine)
                state, _ = transport_features(state, flow, reliable)
                _, det = detector.gate(None, 1, 37, 37, rgb.device, None)
                score = F.avg_pool2d((images[t:t+1]-images[t-1:t]).square().mean(1, keepdim=True), 14)[:, 0]
                mask, _ = detector.gate(score, 1, 37, 37, rgb.device, det)
                cache.append({"features": [f.cpu().half() for f in state["features"]], "pooled": state["pooled"].cpu(),
                              "grid": state["grid"], "size": state["size"], "mask": mask.cpu(), "rgb": rgb[t:t+1].cpu(),
                              "teacher_features": [f[t:t+1].cpu().half() for f in enc["features"]],
                              "teacher": teacher[t:t+1].cpu(), "flat": flat[t:t+1].cpu(), "valid": valid[t:t+1].cpu(),
                              "pairs": s["pairs"][t], "previous_pairs": s["pairs"][t-1], "source": s["source"]})
            if (ci+1) % 32 == 0: print("cache", ci+1, "/", len(train), flush=True)
    torch.save(cache, out/"training_cache.pt")
    return cache


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    if args.steps < 1: raise ValueError("Positive training budget required")
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cudnn.benchmark = False
    pp = json.loads(Path("work_dirs/qquality_feature_20260908/protocol.json").read_text())
    exact = load_exact(pp["checkpoint"])
    train_path = Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    train = torch.load(train_path, weights_only=False)
    dev_paths = [Path("work_dirs/qstream_20260908/development_data.pt"), Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev = {k: torch.load(p, weights_only=False) for k,p in zip(("L32", "L256"), dev_paths)}
    tf = {str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df = {str(Path(p).resolve()) for ss in dev.values() for s in ss for pair in s["pairs"] for p in pair}
    assert not tf & df
    assert not any(set(Path(p).parts) & set(FINAL_TEST_SEQUENCES) for p in tf | df)
    models = {}
    for mode in ("ssm", "mlp"):
        torch.manual_seed(738); models[mode] = AlignedAdapter(mode).cuda()
    for name in ("rgb", "context", "fuse", "readout"):
        for a,b in zip(getattr(models["ssm"],name).parameters(), getattr(models["mlp"],name).parameters()): assert torch.equal(a,b)
    args.out.mkdir(parents=True)
    protocol = {"role": "training_only_fit_reused_development_screen_no_final_no_promotion", "checkpoint": pp["checkpoint"],
                "training_data": file_record(train_path), "development": [file_record(p) for p in dev_paths],
                "training_clips": len(train), "training_frames": sum(len(s["rgb"]) for s in train),
                "steps_each": args.steps, "seed": 738, "lr": .0003, "batch": 1,
                "initialization": "fresh cores, bitwise identical common RGB/context/fuse/readout initialization; readout zero; no warm-start asymmetry",
                "parameters": {m: sum(p.numel() for p in net.parameters()) for m,net in models.items()},
                "objective": "1 all-patch featureL1 + 5Q0logdepth + 10Q0multiscalegradient + 20trainingGTflat_excess",
                "cache": "384 K2 pairs from192clips; DIS-aligned features and teacherfeatures storedFP16, RGB/depth/pooled FP32; training h always zero",
                "memory": "K2 resets h, so this compares one-step SSM parameterization vs MLP, NOT temporal memory; SSM A has zero data gradient at h0=0",
                "gate": "unchanged quality_gate AND mean latency reduction>=5pct",
                "timing": "synchronized GPU-resident wall latency, includes CPU DIS and transfers; excludes initialIO/H2D; RTX4090 not edge",
                "runtime": runtime_versions(), "opencv": cv2.__version__,
                "code": [file_record(p) for p in [__file__, "sokkanaem/qaligned_adapter.py", "sokkanaem/qflow_features.py", "sokkanaem/qflow.py",
                          "sokkanaem/qdelta_ssm.py", "sokkanaem/ssm.py", "sokkanaem/qquality.py", "sokkanaem/qflat_state.py",
                          "scripts/qquality_study.py", "scripts/qstream_study.py", "sokkanaem/qstream.py", "sokkanaem/qmodel.py",
                          "sokkanaem/detector.py", "scripts/quality_refinement.py", "sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json", protocol)
    cache = train_cache(exact, train, args.out)
    assert len(cache) == sum(len(s["rgb"])//2 for s in train)
    q.write(args.out/"cache_record.json", file_record(args.out/"training_cache.pt"))
    order = torch.randint(len(cache), (args.steps,), generator=torch.Generator().manual_seed(738)).tolist()
    q.write(args.out/"training_order.json", order)
    for mode, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        history = []
        for step, i in enumerate(order, 1):
            row = cache[i]
            state = model.refresh({"features": [f.cuda().float() for f in row["features"]], "pooled": row["pooled"].cuda(),
                                   "grid": row["grid"], "size": row["size"]})
            current, info = model.update(row["rgb"].cuda(), row["mask"].cuda(), state)
            pred = decode_with_grad(exact, current)
            feature = sum((f[:,1:]-t[:,1:].cuda().float()).abs().mean() for f,t in zip(current["features"], row["teacher_features"]))/4
            depth, edge = depth_retention(pred, row["teacher"].cuda())
            flat = flat_excess_loss(pred, row["teacher"].cuda(), row["valid"].cuda(), row["flat"].cuda())
            loss = feature+5*depth+10*edge+20*flat+0*model.readout.weight.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if step == 1 or step % 200 == 0 or step == args.steps:
                item = {"step": step, "loss": float(loss.detach()), "feature": float(feature.detach()), "depth": float(depth.detach()),
                        "edge": float(edge.detach()), "flat": float(flat.detach()), "updated_patches": info["updated_patches"]}
                history.append(item); print(mode, item, flush=True); q.write(args.out/(mode+"_history.json"), history)
        model.eval().requires_grad_(False)
        torch.save({"adapter": model.state_dict(), "mode": mode}, args.out/(mode+"_adapter.pt"))
    results = {"promoted": False}; parity = []
    for length, samples in dev.items():
        baseline, dense = evaluate(exact, samples); results[length] = {"baseline": baseline}
        torch.save(dense, args.out/(length+"_dense_predictions.pt"))
        wrappers = {"output_flow": RGBFlowStream(exact), "aligned_no_update": RGBFlowFeatureStream(exact, models["ssm"], no_delta=True)}
        wrappers.update({mode: RGBFlowFeatureStream(exact, model) for mode,model in models.items()})
        for name, wrapper in wrappers.items():
            measured, preds = evaluate(wrapper, samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            count = 0
            for ci, tele in enumerate(measured["telemetry"]):
                for t, info in enumerate(tele["frames"]):
                    if info["full_refresh"]:
                        assert torch.equal(preds[ci][t], dense[ci][t]); count += 1
            parity.append({"length": length, "model": name, "comparisons": count, "max_abs": 0.})
            results[length][name] = measured
            torch.save(preds, args.out/(length+"_"+name+"_predictions.pt"))
            q.write(args.out/"results.json", results); q.write(args.out/"parity.json", parity)
            print(length, name, measured["metrics"]["balanced"], "ms", measured["mean_ms"], measured["quality_gate"], flush=True)
    check_records(protocol["code"]+protocol["development"]+[protocol["checkpoint"],protocol["training_data"]])
    q.write(args.out/"integrity.json", {"input_hashes_unchanged": True, "full_refresh_parity_comparisons": sum(r["comparisons"] for r in parity)})


if __name__ == "__main__": main()
