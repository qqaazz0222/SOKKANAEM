"""Stage-two fixed-budget screen. Reuses stage-one frozen data, never final test."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.edge_local_refiner import EdgeLocalRefiner, edge_local_loss
from sokkanaem.protocol import file_record, check_final_test
from sokkanaem.data import load_manifest
from sokkanaem.model import from_checkpoint

spec = importlib.util.spec_from_file_location("screen", Path(__file__).with_name("quality_refinement.py"))
screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh directory")
    protocol = json.loads((args.parent / "protocol.json").read_text())
    assert check_final_test(protocol["manifest"]["path"]) == "development_validation"
    assert file_record(protocol["checkpoint"]["path"]) == protocol["checkpoint"]
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    train = torch.load(args.parent / "train_cache.pt", weights_only=False)
    dev = torch.load(args.parent / "development_cache.pt", weights_only=False)
    baseline = screen.evaluate(dev)
    record = {"role": "adaptive_second_development_screen_not_final",
              "parent": file_record(args.parent / "protocol.json"),
              "train_cache": file_record(args.parent / "train_cache.pt"),
              "dev_cache": file_record(args.parent / "development_cache.pt"),
              "steps": args.steps, "seed": 20260908, "lr": .001, "batch": 8,
              "variants": {"local_edge_0.5": .5, "local_edge_2.0": 2.},
              "code": [file_record(p) for p in [__file__, "scripts/quality_refinement.py",
                         "sokkanaem/edge_local_refiner.py", "sokkanaem/bounded_refiner.py"]]}
    screen.write(args.out / "protocol.json", record)
    # Historical candidates: exact same original RGB preprocessing, selected clips,
    # streaming reset, scoring resolution, raw and clip-median gauge.
    available = {}
    for src, ds in load_manifest(protocol["manifest"]["path"]):
        for i in range(len(ds)):
            available[tuple(tuple(p) for p in screen.paths(ds, i))] = (ds, i)
    historical = {}
    for name in ("d1-boundary3-s0", "v11-teacher-s0"):
        path = Path("work_dirs") / name / "latest.pt"
        model = from_checkpoint(path, "cuda").eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for sample in dev:
                ds, idx = available[tuple(tuple(p) for p in sample["pairs"])]
                rgb, _, _ = ds[idx]
                state, preds = None, []
                for frame in rgb:
                    p, state, _ = model.step(frame[None].cuda(), state)
                    preds.append(p)
                rows.append({"source": sample["source"], "scene": sample["scene"],
                             "metrics": screen.score(torch.cat(preds), sample)})
        historical[name] = {"checkpoint": file_record(path), "metrics": screen.aggregate(rows)}
        print(name, historical[name]["metrics"]["balanced"], flush=True)
        del model
    screen.write(args.out / "historical_comparison.json", historical)
    tensors = {k: torch.cat([s[k] for s in train]).cuda().float()
               for k in ("rgb", "coarse", "features", "gt", "valid")}
    results = {"baseline": baseline, "historical": historical, "variants": {}, "promoted": False}
    for name, weight in record["variants"].items():
        torch.manual_seed(record["seed"])
        model = EdgeLocalRefiner().cuda()
        assert screen.evaluate(dev, model)["balanced"] == baseline["balanced"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        start = time.monotonic(); history = []
        for step in range(1, args.steps+1):
            idx = torch.randint(len(tensors["rgb"]), (8,), device="cuda")
            x = {k: v[idx] for k, v in tensors.items()}
            p, delta = model(x["rgb"], x["coarse"], x["features"])
            loss, parts = edge_local_loss(p, x["coarse"], x["gt"], x["valid"], delta, weight)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 200 == 0:
                row = {"step": step, "loss": float(loss.detach()), **parts}
                history.append(row); print(name, row, flush=True)
        model.eval(); measured = screen.evaluate(dev, model)
        outcome = {"metrics": measured, "gate": screen.gate(baseline, measured),
                   "history": history, "seconds": time.monotonic()-start,
                   "parameters": sum(p.numel() for p in model.parameters())}
        results["variants"][name] = outcome
        torch.save({"refiner": model.state_dict(), "kind": "edge_local", "max_log_delta": .15,
                    "base_checkpoint_sha256": protocol["checkpoint"]["sha256"],
                    "variant": name, "steps": args.steps}, args.out / f"{name}.pt")
        screen.write(args.out / f"{name}.json", outcome)
        print(name, measured["balanced"], outcome["gate"], flush=True)
    screen.write(args.out / "results.json", results)


if __name__ == "__main__":
    main()
