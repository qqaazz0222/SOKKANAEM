"""Matched current-only vs two-frame error repair on pooled-held Q0+SSM K2."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact, MotionStream
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qseparated import SeparatedDelta
from sokkanaem.qquality import depth_retention
from sokkanaem.qflat_state import flat_mask, flat_excess_loss
from sokkanaem.qstream import ChangeAwareQ0
from sokkanaem.qtemporal_refiner import TemporalErrorRefiner, TemporalRefinedStream, repair_target


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temporal", action="store_true"); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    root = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((root/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = SeparatedDelta(hold_cls=True, hold_pooled=True).cuda().eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(root/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    train_path = Path("work_dirs/qdelta_20260908/teacher_cache.pt")
    train = torch.load(train_path, weights_only=False)
    gt_path = Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    gt_cache = torch.load(gt_path, weights_only=False)
    lookup = {tuple(p for pair in s["pairs"] for p in pair): s for s in gt_cache}
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"), Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev = {name: torch.load(p, weights_only=False) for name, p in zip(("L32", "L256"), paths)}
    tf = {str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df = {str(Path(p).resolve()) for ss in dev.values() for s in ss for pair in s["pairs"] for p in pair}
    assert not tf & df
    refiner = TemporalErrorRefiner(args.temporal).cuda()
    args.out.mkdir(parents=True)
    protocol = {"role": "reused_development_not_final_no_promotion", "checkpoint": pp["checkpoint"],
                "adapter": file_record(root/"adapter.pt"), "train_cache": file_record(train_path),
                "training_gt": file_record(gt_path), "development": [file_record(p) for p in paths],
                "temporal": args.temporal, "steps": 1200, "seed": 736, "lr": .0005,
                "parameters": sum(p.numel() for p in refiner.parameters()),
                "objective": "5Q0logdepth+10Q0loggradient+20GTflat_excess+.1BCErepair_region",
                "repair_target": "trainingQ0 logdepth error>.015 OR summed loggradient error>.02",
                "base": "same frozen featureK2 adapter; hold pooled metric input and unused postencoderCLS; no h carry",
                "control": "same11input channels/network/init/order; previousRGB=currentRGB and previousDepth=coarse so explicit temporal evidence removed",
                "bound": "+/-.1 logdepth times sigmoid gate; final clamp to Q0 depth interval",
                "scope": "two-frame output repair; no motion groundtruth/occlusion oracle or proof of temporal SSM memory; no hard edge mask",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qtemporal_refiner.py", "sokkanaem/qseparated.py",
                          "sokkanaem/qdelta_ssm.py", "sokkanaem/qflat_state.py", "sokkanaem/qquality.py",
                          "sokkanaem/qstream.py", "scripts/qquality_study.py", "scripts/qstream_study.py",
                          "sokkanaem/sharpness.py", "scripts/quality_refinement.py"]]}
    q.write(args.out/"protocol.json", protocol)
    cache = []
    with torch.no_grad():
        for s in train:
            gt_sample = lookup[tuple(p for pair in s["pairs"] for p in pair)]
            rgb = s["rgb"].cuda().float()
            teacher = exact.decode(exact.encode(rgb))
            mask, valid = flat_mask(gt_sample["gt"].cuda(), gt_sample["valid"].cuda())
            stream = MotionStream(exact, adapter, refresh_every=2); state = None; previous_depth = None
            for t in range(len(rgb)):
                coarse, state, info = stream.step(rgb[t:t+1], state)
                if not info["full_refresh"]:
                    cache.append({"current": rgb[t:t+1].cpu(), "previous": rgb[t-1:t].cpu(),
                                  "coarse": coarse.cpu(), "previous_depth": previous_depth.cpu(), "teacher": teacher[t:t+1].cpu(),
                                  "flat": mask[t:t+1].cpu(), "valid": valid[t:t+1].cpu(),
                                  "repair": repair_target(coarse, teacher[t:t+1]).cpu(), "pairs": s["pairs"][t],
                                  "previous_pairs": s["pairs"][t-1]})
                previous_depth = coarse
    assert len(cache) == 96
    torch.save(cache, args.out/"training_cache.pt")
    opt = torch.optim.AdamW(refiner.parameters(), lr=.0005)
    gen = torch.Generator().manual_seed(736); history = []
    for step in range(1, 1201):
        idx = torch.randint(len(cache), (4,), generator=gen).tolist()
        b = {k: torch.cat([cache[i][k] for i in idx]).cuda() for k in
             ("current", "previous", "coarse", "previous_depth", "teacher", "flat", "valid", "repair")}
        pred, logits = refiner(b["current"], b["previous"], b["coarse"], b["previous_depth"])
        pred = pred.clamp(min=exact.q.d_min, max=exact.q.d_max)
        depth, edge = depth_retention(pred, b["teacher"])
        flat = flat_excess_loss(pred, b["teacher"], b["valid"], b["flat"])
        gate = F.binary_cross_entropy_with_logits(logits, b["repair"])
        loss = 5*depth+10*edge+20*flat+.1*gate
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1., error_if_nonfinite=True); opt.step()
        if step == 1 or step % 200 == 0:
            row = {"step": step, "loss": float(loss.detach()), "components": [float(x.detach()) for x in (depth, edge, flat, gate)]}
            history.append(row); print(row, flush=True)
    refiner.eval(); torch.save({"refiner": refiner.state_dict(), "temporal": args.temporal}, args.out/"refiner.pt")
    q.write(args.out/"history.json", history)
    result = {"promoted": False}
    for name, samples in dev.items():
        baseline, _ = evaluate(exact, samples); result[name] = {"baseline": baseline}
        kwargs = {"d_min": exact.q.d_min, "d_max": exact.q.d_max}
        removed = TemporalErrorRefiner(False).cuda().eval(); removed.load_state_dict(refiner.state_dict())
        models = {"ssm_base": MotionStream(exact, adapter, refresh_every=2),
                  "ssm_repair": TemporalRefinedStream(MotionStream(exact, adapter, refresh_every=2), refiner, **kwargs),
                  "temporal_removed": TemporalRefinedStream(MotionStream(exact, adapter, refresh_every=2), removed, **kwargs),
                  "hold_repair": TemporalRefinedStream(ChangeAwareQ0(exact, refresh_every=2, periodic_only=True), refiner, **kwargs)}
        for key, model in models.items():
            measured, preds = evaluate(model, samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            result[name][key] = measured
            torch.save(preds, args.out/(name+"_"+key+"_predictions.pt")); q.write(args.out/"results.json", result)
            print(name, key, measured["metrics"]["balanced"], "ms", measured["mean_ms"], measured["quality_gate"], flush=True)
    for r in protocol["code"]+[protocol["checkpoint"], protocol["adapter"]]: assert file_record(r["path"]) == r


if __name__ == "__main__": main()
