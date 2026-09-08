"""Freeze/verify the paper baseline, evidence ledger and data-use inventory.

Only reads checkpoints on CPU. No training, model inference or git mutation.
Use --inspect first; --write creates immutable records; --verify checks hashes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tomllib

import torch

from sokkanaem import SOKKANAEM, checkpoint_config, from_checkpoint
from sokkanaem.protocol import (EVALUATION_FILES, FINAL_TEST_SEQUENCES, VERSION,
                                file_record, runtime_versions)

BASE = Path("work_dirs/v11-longclip-spread-s0/latest.pt")
OUT = Path("paper/freeze")


def record(path):
    path = Path(path)
    return file_record(path) if path.is_file() else {"path": str(path), "missing": True}


def lineage():
    path, rows, seen = BASE, [], set()
    while str(path) not in seen:
        seen.add(str(path))
        sidecar = path.with_name("config.toml")
        cfg = tomllib.loads(sidecar.read_text()) if sidecar.exists() else {}
        item = {"checkpoint": record(path), "config": record(sidecar), "sidecar": cfg}
        if not path.exists():
            item["ancestry_status"] = "missing checkpoint; cannot establish earlier initialization"
            rows.append(item)
            break
        obj = torch.load(path, map_location="cpu", weights_only=True)
        meta = obj.get("meta", {})
        args = meta.get("args", {})
        item.update(step=obj.get("step"), checkpoint_meta=meta,
                    weight_selection="ema" if obj.get("ema") else "model")
        resume = args.get("resume") or cfg.get("resume")
        item["resume_evidence"] = "checkpoint metadata" if args.get("resume") else "sidecar or absent"
        item["resume"] = resume
        item["ancestry_status"] = "recorded parent" if resume else "no parent recorded; not proof of fresh initialization"
        rows.append(item)
        del obj
        if not resume:
            break
        path = Path(resume)
    return rows


def data_inventory():
    validation, manifests = set(), []
    for path in sorted(Path("manifests").glob("acc_real_L*.json")):
        m = json.loads(path.read_text())
        counts = Counter()
        for c in m["clips"]:
            name = Path(c["pairs"][0][0]).parent.parent.name
            validation.add(name)
            counts[f"{c['source']}/{name}"] += 1
        manifests.append({**record(path), "role": "development_validation",
                          "clips": len(m["clips"]), "sequence_clips": dict(counts)})
    roots = {"tum": Path("/home/hyunsu/dataset_ssd/tum_static"),
             "bonn": Path("/home/hyunsu/dataset_ssd/bonn/rgbd_bonn_dataset")}
    sequences = []
    for source, root in roots.items():
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not (path / "rgb").is_dir() or not (path / "depth").is_dir():
                continue
            sequences.append({"source": source, "sequence": path.name,
                              "root": str(path), "role": "development_validation"
                              if path.name in validation else "training_pool"})
    # Audit historical configs/manifests for an explicit reference to each
    # reservation. Absence is project-record evidence, not proof about third-
    # party pretraining. Existing dataset roots are separately inventoried.
    paths = sorted(Path("configs").glob("*.toml")) + sorted(Path("work_dirs").glob("*/config.toml"))
    hits = {name: [] for name in FINAL_TEST_SEQUENCES}
    for path in paths:
        content = path.read_text()
        for name in hits:
            if name in content or name.removeprefix("rgbd_dataset_freiburg3_") in content:
                hits[name].append(str(path))
    if any(hits.values()):
        raise RuntimeError(f"reserved final-test sequences occur in historical configs: {hits}")
    return {"protocol": VERSION, "legacy_manifests": manifests,
            "existing_real_sequences": sequences,
            "classification_basis": "main v8/v9/v10/v11 data roots and holdout rules; historical holdouts repeatedly used for selection",
            "reserved_final_test": list(FINAL_TEST_SEQUENCES),
            "historical_config_files_checked": len(paths), "reserved_name_hits": hits,
            "scope": "new sequences in the same TUM scene, with camera motion; not an unseen-room or fixed-camera test",
            "pretraining_overlap": "unknown for external baselines",
            "final_test": json.loads((OUT / "final_test.json").read_text())}


def evidence_ledger():
    paths = set(Path("work_dirs/v11-longclip-spread-s0").glob("scores*.json"))
    paths.update(Path("work_dirs/acc").glob("*.json"))
    rows = []
    for path in sorted(paths):
        d = json.loads(path.read_text())
        manifest = d.get("manifest")
        rows.append({**record(path), "model_claimed_by_dump": d.get("model"),
                     "manifest_claimed_by_dump": manifest,
                     "manifest_snapshot": record(manifest) if manifest else None,
                     "gauges": list(d.get("pooled", {})),
                     "params_m": d.get("params_m"), "infer_size": d.get("infer_size"),
                     "effective_size": d.get("effective_size"),
                     "status": "versioned" if d.get("provenance") else "legacy_provenance_incomplete",
                     "role": "development evidence; not final-test results"})
    return {"records": rows,
            "caveat": "hashing old dumps now preserves their bytes; it does not establish which weight/code bytes originally produced them",
            "legacy_timing_records": [record(p) for p in (
                "work_dirs/t2-10-bench.log", "work_dirs/tier2-bench.log", "EDGE_BENCH.md")],
            "timing_status": "legacy, no linked checkpoint hash; remeasure before claiming the frozen run's deployment speed"}


def snapshot():
    torch.set_num_threads(2)
    history = lineage()
    model = from_checkpoint(BASE, "cpu").eval()
    cfg = checkpoint_config(BASE)
    import inspect
    resolved = {k: v.default for k, v in inspect.signature(SOKKANAEM).parameters.items()
                if v.default is not inspect.Parameter.empty}
    resolved.update(cfg.get("model", {}))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    baseline = {"protocol": VERSION, "frozen_on": "2026-09-06",
                "model_id": "sokkanaem-v11-s0-256", "checkpoint": record(BASE),
                "config": record(BASE.with_name("config.toml")),
                "params": sum(p.numel() for p in model.parameters()),
                "resolved_model_kwargs": resolved, "input_size": cfg["size"],
                "weight_selection": history[0]["weight_selection"],
                "git_commit_at_freeze": git, "lineage": history,
                "environment_at_freeze": runtime_versions(),
                "preparation_code": [record("scripts/paper_freeze.py"),
                                     record("scripts/prepare_paper_test.py")],
                "code": [record(p) for p in EVALUATION_FILES],
                "operating_point": {"tau_on": 0.05, "tau_off": 0.025,
                                    "keyframe_every": 30, "gmc": False},
                "training_claim": "v8 60k -> v9 60k -> v10 25k -> v11 8k; 153k recorded stage steps, not independently verified optimizer-update totals",
                "final_test_inference_performed": False,
                "limitations": ["legacy score/timing files lack weight hashes",
                                "no external baseline pretraining non-overlap guarantee"]}
    return {"baseline.json": baseline, "data_split.json": data_inventory(),
            "evidence_ledger.json": evidence_ledger()}


def verify():
    checked = set()
    def visit(value):
        if isinstance(value, dict):
            if "sha256" in value and "path" in value and not value.get("missing"):
                key = (value["path"], value["sha256"])
                if key not in checked:
                    actual = file_record(value["path"])
                    if actual["sha256"] != value["sha256"]:
                        raise RuntimeError(f"frozen artifact changed: {value['path']}")
                    checked.add(key)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for name in ("baseline.json", "data_split.json", "evidence_ledger.json", "final_test.json"):
        visit(json.loads((OUT / name).read_text()))
    final = json.loads((OUT / "final_test.json").read_text())
    visit({"path": final["inventory"], "sha256": final["inventory_sha256"]})
    inventory = json.loads(Path(final["inventory"]).read_text())
    root = Path(inventory["root"])
    for frames in inventory["sequences"].values():
        for frame in frames:
            for kind in ("rgb", "depth"):
                visit({"path": str(root / frame[kind]), "sha256": frame[f"{kind}_sha256"]})
    for download in inventory["downloads"]:
        visit({"path": download["archive"], "sha256": download["archive_sha256"]})
    print(f"verified {len(checked)} unique frozen file/hash pairs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()
    docs = snapshot()
    b = docs["baseline.json"]
    print(f"baseline {b['params']} parameters; {b['checkpoint']['sha256']}")
    for row in b["lineage"]:
        print(f"  {row['checkpoint']['path']} step={row.get('step')} parent={row.get('resume')}")
    print("data roles:", dict(Counter(s["role"] for s in docs["data_split.json"]["existing_real_sequences"])))
    print(f"legacy evidence files: {len(docs['evidence_ledger.json']['records'])}")
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for name in docs:
            if (OUT / name).exists():
                raise FileExistsError(f"freeze already exists: {OUT / name}; create a new version instead")
        for name, content in docs.items():
            (OUT / name).write_text(json.dumps(content, indent=2, default=str) + "\n")
        print(f"created {len(docs)} freeze records in {OUT}")


if __name__ == "__main__":
    main()
