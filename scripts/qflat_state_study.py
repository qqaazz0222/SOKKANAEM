"""Matched K2 continuation: output loss vs GT-flat loss, reset vs refresh carry."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.detector import ChangeDetector
from sokkanaem.protocol import file_record
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.qquality import decode_with_grad, depth_retention
from sokkanaem.qflat_state import flat_mask, flat_excess_loss, refresh_memory, PersistentStream


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flat", action="store_true"); ap.add_argument("--carry", action="store_true")
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    root = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((root/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = QDeltaSSM().cuda()
    adapter.load_state_dict(torch.load(root/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    train_path = Path("work_dirs/qdelta_20260908/teacher_cache.pt")
    cache = torch.load(train_path, weights_only=False)
    gt_path = Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    gt_cache = torch.load(gt_path, weights_only=False)
    lookup = {tuple(str(p) for pair in s["pairs"] for p in pair): s for s in gt_cache}
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"),
             Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev = {name: torch.load(path, weights_only=False) for name, path in zip(("L32", "L256"), paths)}
    tf = {str(Path(p).resolve()) for s in cache for pair in s["pairs"] for p in pair}
    df = {str(Path(p).resolve()) for data in dev.values() for s in data for pair in s["pairs"] for p in pair}
    assert not tf & df
    args.out.mkdir(parents=True)
    protocol = {"role": "reused_development_only_no_final_no_promotion", "checkpoint": pp["checkpoint"],
                "initial_adapter": file_record(root/"adapter.pt"), "train_cache": file_record(train_path),
                "training_gt": file_record(gt_path), "dev": [file_record(p) for p in paths],
                "flat": args.flat, "carry": args.carry, "steps": 600, "seed": 736, "lr": .0001,
                "objective": "featureL1+.1pooledL1+5logdepth+10loggradient" + ("+20GTflat_Q0excess_gradient" if args.flat else ""),
                "training": "same48clips192frames, K2 matched train/inference,4frame TBPTT crosses one refresh; no synthetic concatenation",
                "carry_rule": ".9h at refresh only where consecutive RGB patchMSE<=.002; force reset every32frames; fresh Q0 features/pooled unchanged",
                "limitations": "heuristic same-coordinate memory retention, not motion/occlusion validity proof; flat masks trainingGT only",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qflat_state.py", "sokkanaem/qquality.py",
                          "sokkanaem/qdelta_ssm.py", "sokkanaem/qstream.py", "scripts/qquality_study.py",
                          "scripts/qstream_study.py", "sokkanaem/sharpness.py", "scripts/quality_refinement.py"]]}
    q.write(args.out/"protocol.json", protocol)
    targets = []; masks = []; valids = []
    with torch.no_grad():
        for s in cache:
            original = lookup[tuple(str(p) for pair in s["pairs"] for p in pair)]
            torch.testing.assert_close(s["rgb"].float(), original["rgb"].float(), rtol=0, atol=0)
            targets.append(exact.decode(exact.encode(s["rgb"].cuda().float())).cpu())
            mask, valid = flat_mask(original["gt"].cuda(), original["valid"].cuda())
            masks.append(mask.cpu()); valids.append(valid.cpu())
    torch.save({"depth": targets, "flat_mask": masks, "valid": valids}, args.out/"targets.pt")
    opt = torch.optim.AdamW(adapter.parameters(), lr=.0001, weight_decay=.0001)
    gen = torch.Generator().manual_seed(736); history = []
    detector = ChangeDetector(patch_size=14, tau_on=.001, tau_off=.0005, keyframe_every=2, dilate=True)
    for step in range(1, 601):
        ci = int(torch.randint(len(cache), (), generator=gen)); s = cache[ci]
        rgb = s["rgb"].cuda().float(); fs = [a.cuda().float() for a in s["features"]]; pooled = s["pooled"].cuda()
        images = F.interpolate(rgb, (518, 518), mode="bilinear", align_corners=False)
        state = None; det = None; losses = []; parts = []
        for t in range(len(rgb)):
            refresh = t % 2 == 0
            score = None if refresh else F.avg_pool2d((images[t:t+1]-images[t-1:t]).square().mean(1, keepdim=True), 14)[:, 0]
            mask, det = detector.gate(score, 1, 37, 37, rgb.device, det)
            if refresh:
                fresh = adapter.refresh({"features": [a[t:t+1] for a in fs], "pooled": pooled[t:t+1], "grid": s["grid"], "size": s["size"]})
                state, _ = refresh_memory(fresh, state, rgb[t:t+1], rgb[t-1:t] if t else None, args.carry)
                continue
            state, _ = adapter.update(rgb[t:t+1], mask, state)
            feature = sum((a-b[t:t+1]).abs().mean() for a,b in zip(state["features"], fs))/len(fs)
            feature = feature+.1*(state["pooled"]-pooled[t:t+1]).abs().mean()
            pred = decode_with_grad(exact, state); target = targets[ci][t:t+1].cuda()
            depth, edge = depth_retention(pred, target)
            flat = flat_excess_loss(pred, target, valids[ci][t:t+1].cuda(), masks[ci][t:t+1].cuda())
            losses.append(feature+5*depth+10*edge+(20*flat if args.flat else 0))
            parts.append(torch.stack((feature.detach(), depth.detach(), edge.detach(), flat.detach())))
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True); opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach()), "components": torch.stack(parts).mean(0).tolist()}
            history.append(row); print(row, flush=True)
    assert all(p.grad is None for p in exact.parameters())
    adapter.eval(); torch.save({"adapter": adapter.state_dict(), "flat": args.flat, "carry": args.carry}, args.out/"adapter.pt")
    q.write(args.out/"history.json", history)
    results = {"promoted": False}
    for name, samples in dev.items():
        baseline, _ = evaluate(exact, samples)
        results[name] = {"baseline": baseline}
        for carry in (False, True):
            measured, preds = evaluate(PersistentStream(exact, adapter, carry), samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            key = "carry" if carry else "reset"; results[name][key] = measured
            torch.save(preds, args.out/(name+"_"+key+"_predictions.pt")); q.write(args.out/"results.json", results)
            print(name, key, measured["metrics"]["balanced"], "ms", measured["mean_ms"], measured["quality_gate"], flush=True)
    for record in protocol["code"]+[protocol["checkpoint"], protocol["initial_adapter"]]:
        assert file_record(record["path"]) == record


if __name__ == "__main__": main()
