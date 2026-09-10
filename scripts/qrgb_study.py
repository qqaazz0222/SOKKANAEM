"""Stage4: bounded current-RGB refinement of the no-warp feature candidate."""
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
from sokkanaem.qrgb_refiner import QRGBRefiner, RefinedStream


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    parent = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((parent/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = QDeltaSSM().cuda().eval()
    adapter.load_state_dict(torch.load(parent/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    adapter.requires_grad_(False)
    train = torch.load("work_dirs/qdelta_20260908/teacher_cache.pt", weights_only=False)
    dev = torch.load(pp["development"]["path"], weights_only=False)
    train_paths = {str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    dev_paths = {str(Path(p).resolve()) for s in dev for pair in s["pairs"] for p in pair}
    assert not train_paths & dev_paths
    args.out.mkdir(parents=True)
    protocol = {"role": "development_only_no_promotion", "parent": file_record(parent/"protocol.json"),
                "adapter": file_record(parent/"adapter.pt"), "checkpoint": pp["checkpoint"],
                "train_cache": file_record("work_dirs/qdelta_20260908/teacher_cache.pt"),
                "development": pp["development"], "seed": 736, "steps": 600, "lr": .001,
                "objective": "5 metric logdepth L1 +10 multiscale loggradient L1 to exact frozenQ0; no GT",
                "model": "3217parameter RGB/logdepth/gradient CNN; bounded +/- .06 logresidual; full refresh output untouched",
                "training": "48clips,192frames; cached K2 and K4 rollout; train skipped frames only; frozen adapter and Q0",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qrgb_refiner.py", "sokkanaem/qquality.py",
                          "scripts/qquality_study.py", "scripts/qstream_study.py", "sokkanaem/qdelta_ssm.py",
                          "sokkanaem/qstream.py", "scripts/quality_refinement.py", "sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json", protocol)
    cache = []
    with torch.no_grad():
        for s in train:
            rgb = s["rgb"].cuda().float()
            teacher = exact.decode(exact.encode(rgb))
            for k in (2, 4):
                model = MotionStream(exact, adapter, refresh_every=k); state = None
                for t in range(len(rgb)):
                    pred, state, info = model.step(rgb[t:t+1], state)
                    if not info["full_refresh"]:
                        cache.append({"rgb": rgb[t:t+1].cpu(), "depth": pred.cpu(), "teacher": teacher[t:t+1].cpu(),
                                      "pairs": s["pairs"][t], "k": k})
    torch.save(cache, args.out/"train_cache.pt")
    refiner = QRGBRefiner().cuda(); opt = torch.optim.AdamW(refiner.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(736); history = []
    for step in range(1, 601):
        indices = torch.randint(len(cache), (4,), generator=gen).tolist()
        batch = {key: torch.cat([cache[i][key] for i in indices]).cuda() for key in ("rgb", "depth", "teacher")}
        pred = refiner(batch["rgb"], batch["depth"])
        depth, edge = depth_retention(pred, batch["teacher"]); loss = 5*depth+10*edge
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1., error_if_nonfinite=True); opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach()), "depth": float(depth.detach()), "edge": float(edge.detach())}
            history.append(row); print(row, flush=True)
    refiner.eval(); torch.save({"refiner": refiner.state_dict()}, args.out/"refiner.pt")
    q.write(args.out/"history.json", history)
    baseline, _ = evaluate(exact, dev)
    results = {"baseline": baseline, "policies": {}, "parameters": sum(p.numel() for p in refiner.parameters()), "promoted": False}
    for k in (2, 4):
        measured, preds = evaluate(RefinedStream(MotionStream(exact, adapter, refresh_every=k), refiner), dev)
        measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
        measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
        measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
        results["policies"]["K"+str(k)] = measured
        torch.save(preds, args.out/("K"+str(k)+"_predictions.pt")); q.write(args.out/"results.json", results)
        print(k, measured["metrics"]["balanced"], measured["quality_gate"], flush=True)
    for record in protocol["code"]+[protocol["adapter"], protocol["checkpoint"]]:
        assert file_record(record["path"]) == record


if __name__ == "__main__": main()
