"""Matched output-local vs unrestricted refiner; frozen earlier feature K2 base."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact, MotionStream
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.qquality import depth_retention
from sokkanaem.qflat_state import flat_mask, flat_excess_loss
from sokkanaem.qrgb_refiner import RefinedStream
from sokkanaem.qstream import ChangeAwareQ0
from sokkanaem.qlocal import LocalRefiner, editable_mask


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protected", action="store_true"); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    parent = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((parent/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = QDeltaSSM().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(parent/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    cache_path = Path("work_dirs/qquality_rgb_20260908/train_cache.pt")
    cache = [s for s in torch.load(cache_path, weights_only=False) if s["k"] == 2]
    gt_path = Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    gt_cache = torch.load(gt_path, weights_only=False); lookup = {}
    for s in gt_cache:
        for t, pair in enumerate(s["pairs"]): lookup[tuple(pair)] = (s["gt"][t:t+1], s["valid"][t:t+1])
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"), Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev = {name: torch.load(p, weights_only=False) for name, p in zip(("L32", "L256"), paths)}
    tf = {str(Path(p).resolve()) for s in cache for p in s["pairs"]}
    df = {str(Path(p).resolve()) for ss in dev.values() for s in ss for pair in s["pairs"] for p in pair}
    assert not tf & df and len(cache) == 96
    args.out.mkdir(parents=True)
    protocol = {"role": "reused_development_screen_no_promotion", "checkpoint": pp["checkpoint"],
                "adapter": file_record(parent/"adapter.pt"), "train_cache": file_record(cache_path), "training_gt": file_record(gt_path),
                "development": [file_record(p) for p in paths], "protected": args.protected, "steps": 600, "seed": 736,
                "lr": .001, "train_skipped_frames": 96, "parameters": 3217,
                "objective": "5 Q0logdepth+10 Q0loggradient+20GTflat_Q0excess", "bound": "+/- .03 logdepth",
                "mask": "RGBgradient>.12 or predictedlogdepthgradient>.02, dilated5x5, output on mask copied exactly; no inferenceGT",
                "base_reason": "Earlier feature K2 retained boundaries better than flat-trained feature candidates; same unchanged base for both local/unrestricted controls",
                "limits": "Protected pixels retain coarse predictions, NOT groundtruth boundaries; normalized global metrics need not remain constant; no long SSM carry in this K2 base",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qlocal.py", "sokkanaem/qrgb_refiner.py",
                          "sokkanaem/qflat_state.py", "sokkanaem/qquality.py", "scripts/qquality_study.py",
                          "scripts/qstream_study.py", "sokkanaem/qstream.py", "sokkanaem/qdelta_ssm.py",
                          "sokkanaem/sharpness.py", "scripts/quality_refinement.py"]]}
    q.write(args.out/"protocol.json", protocol)
    with torch.no_grad():
        for s in cache:
            gt, valid = lookup[tuple(s["pairs"])]
            mask, valid = flat_mask(gt.cuda(), valid.cuda())
            s["flat_mask"] = mask.cpu(); s["valid"] = valid.cpu()
    refiner = LocalRefiner(args.protected).cuda(); opt = torch.optim.AdamW(refiner.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(736); history = []
    for step in range(1, 601):
        indices = torch.randint(len(cache), (4,), generator=gen).tolist()
        batch = {k: torch.cat([cache[i][k] for i in indices]).cuda() for k in ("rgb", "depth", "teacher", "flat_mask", "valid")}
        pred = refiner(batch["rgb"], batch["depth"])
        depth, edge = depth_retention(pred, batch["teacher"])
        flat = flat_excess_loss(pred, batch["teacher"], batch["valid"], batch["flat_mask"])
        loss = 5*depth+10*edge+20*flat
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1., error_if_nonfinite=True); opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach()), "components": [float(x.detach()) for x in (depth, edge, flat)]}
            history.append(row); print(row, flush=True)
    refiner.eval(); torch.save({"refiner": refiner.state_dict(), "protected": args.protected}, args.out/"refiner.pt")
    q.write(args.out/"history.json", history)
    result = {"promoted": False}
    for name, samples in dev.items():
        baseline, _ = evaluate(exact, samples); result[name] = {"baseline": baseline}
        policies = {"ssm_unrefined": MotionStream(exact, adapter, refresh_every=2),
                    "ssm_refined": RefinedStream(MotionStream(exact, adapter, refresh_every=2), refiner),
                    "hold_refined": RefinedStream(ChangeAwareQ0(exact, refresh_every=2, periodic_only=True), refiner)}
        for key, model in policies.items():
            measured, preds = evaluate(model, samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            result[name][key] = measured
            torch.save(preds, args.out/(name+"_"+key+"_predictions.pt")); q.write(args.out/"results.json", result)
            print(name, key, measured["metrics"]["balanced"], measured["quality_gate"], flush=True)
    for r in protocol["code"]+[protocol["checkpoint"], protocol["adapter"]]: assert file_record(r["path"]) == r


if __name__ == "__main__": main()
