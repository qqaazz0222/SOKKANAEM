"""Stage5: train-only teacher-error proxy for full refresh, fixed thresholds."""
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
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.qquality import depth_retention
from sokkanaem.qrisk import risk_features, RefreshRisk, RiskStream


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    parent = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((parent/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = QDeltaSSM().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(parent/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    train_path = Path("work_dirs/qdelta_20260908/teacher_cache.pt")
    cache = torch.load(train_path, weights_only=False)
    short_path = Path(pp["development"]["path"])
    long_path = Path("work_dirs/qquality_audit_20260908/long_development_data.pt")
    samples = {"L32": torch.load(short_path, weights_only=False), "L256": torch.load(long_path, weights_only=False)}
    tf = {str(Path(p).resolve()) for s in cache for pair in s["pairs"] for p in pair}
    df = {str(Path(p).resolve()) for data in samples.values() for s in data for pair in s["pairs"] for p in pair}
    assert not tf & df
    args.out.mkdir(parents=True)
    protocol = {"role": "development_only_post_L256_diagnostic_no_promotion", "checkpoint": pp["checkpoint"],
                "adapter": file_record(parent/"adapter.pt"), "train_cache": file_record(train_path),
                "development": [file_record(short_path), file_record(long_path)], "seed": 736, "steps": 1000,
                "target": "Q0 mean absolute logdepth error +2 multiscale loggradient error, no GT",
                "train_split": "first12 clips/source fit;last4/source holdout error diagnostic; adapter itself saw all48",
                "thresholds": [.02, .04, .06], "max_refresh_interval": 4,
                "caveat": "Small RGB statistics regressor is a risk PROXY, not calibrated uncertainty; L256 already opened; no independent generalization claim",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qrisk.py", "scripts/qquality_study.py",
                          "scripts/qstream_study.py", "sokkanaem/qquality.py", "sokkanaem/qdelta_ssm.py",
                          "sokkanaem/qstream.py", "scripts/quality_refinement.py", "sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json", protocol)
    rows = []; source_counts = {}
    with torch.no_grad():
        for s in cache:
            src = s["source"]; ci = source_counts.get(src, 0); source_counts[src] = ci+1
            rgb = s["rgb"].cuda().float(); teacher = exact.decode(exact.encode(rgb))
            model = MotionStream(exact, adapter); state = None
            for t in range(len(rgb)):
                x = None if state is None else risk_features(rgb[t:t+1], state["anchor"], state["age"]+1)
                pred, state, info = model.step(rgb[t:t+1], state)
                if not info["full_refresh"]:
                    depth, edge = depth_retention(pred, teacher[t:t+1])
                    rows.append({"x": x.cpu(), "y": (depth+2*edge).reshape(1, 1).cpu(),
                                 "fit": ci < 12, "source": src, "pairs": s["pairs"][t]})
    torch.save(rows, args.out/"risk_labels.pt")
    fit = [r for r in rows if r["fit"]]; held = [r for r in rows if not r["fit"]]
    x = torch.cat([r["x"] for r in fit]).cuda(); y = torch.cat([r["y"] for r in fit]).cuda()
    risk = RefreshRisk().cuda(); opt = torch.optim.AdamW(risk.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(736)
    for step in range(1, 1001):
        idx = torch.randint(len(x), (32,), generator=gen).cuda()
        loss = F.smooth_l1_loss(risk(x[idx])/.05, y[idx]/.05)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 200 == 0: print("risk", step, float(loss.detach()), flush=True)
    risk.eval(); torch.save({"risk": risk.state_dict()}, args.out/"risk.pt")
    diagnostic = {}
    with torch.no_grad():
        for name, data in (("fit", fit), ("heldout_training_clips", held)):
            xx = torch.cat([r["x"] for r in data]).cuda(); yy = torch.cat([r["y"] for r in data]).cuda()
            prediction = risk(xx)
            diagnostic[name] = {"n": len(data), "mae": float((prediction-yy).abs().mean()),
                                "target_mean": float(yy.mean()), "predicted_mean": float(prediction.mean())}
    results = {"risk_diagnostic": diagnostic, "L32": {}, "L256": {}, "promoted": False}
    for length, data in samples.items():
        baseline, _ = evaluate(exact, data); results[length]["baseline"] = baseline
        for threshold in protocol["thresholds"]:
            measured, preds = evaluate(RiskStream(exact, adapter, risk, threshold), data)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            key = "risk_"+str(threshold); results[length][key] = measured
            torch.save(preds, args.out/(length+"_"+key+"_predictions.pt")); q.write(args.out/"results.json", results)
            print(length, key, measured["metrics"]["balanced"], "skip", measured["skip_fraction"], "ms", measured["mean_ms"], measured["quality_gate"], flush=True)
    for record in protocol["code"]+[protocol["adapter"], protocol["checkpoint"]]:
        assert file_record(record["path"]) == record


if __name__ == "__main__": main()
