"""Acquire the preregistered, sequence-disjoint TUM test set without inference.

The four sequences are fixed before download or model evaluation. They test
moving-camera transfer within TUM, not an unseen room or a fixed-camera domain.
Downloads and extraction never overwrite an existing completed archive/sequence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


SEQUENCES = tuple(f"rgbd_dataset_freiburg3_{s}" for s in (
    "sitting_xyz", "sitting_rpy", "walking_xyz", "walking_rpy"))
BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sealed_write(path, content):
    """An idempotent write, refusing to alter a previously sealed artifact."""
    text = json.dumps(content, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to change sealed artifact: {path}")
    if not path.exists():
        path.write_text(text)


def download_record(name, root):
    receipt = json.loads((root / f".{name}.complete.json").read_text())
    archive = root / "archives" / f"{name}.tgz"
    digest = sha256(archive)
    if digest != receipt["sha256"]:
        raise RuntimeError(f"archive no longer matches receipt: {archive}")
    return {"sequence": name, "url": receipt["url"], "archive": str(archive),
            "archive_sha256": digest, "archive_bytes": archive.stat().st_size}


def acquire(name, root):
    archive = root / "archives" / f"{name}.tgz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}.tgz"
    if not archive.exists():
        part = archive.with_suffix(".tgz.part")
        print(f"download {name}", flush=True)
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "3",
            "--connect-timeout", "30", "--speed-time", "60",
            "--speed-limit", "1024", "--continue-at", "-",
            "--silent", "--show-error", "--output", str(part), url,
        ], check=True)
        part.rename(archive)
    digest = sha256(archive)
    target = root / name
    marker = root / f".{name}.complete.json"
    if target.exists():
        if not marker.exists() or json.loads(marker.read_text())["sha256"] != digest:
            raise RuntimeError(f"existing unverified extraction: {target}")
    else:
        print(f"extract {name}", flush=True)
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                parts = Path(member.name).parts
                if (not parts or parts[0] != name or ".." in parts
                        or Path(member.name).is_absolute()
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f"unexpected archive member: {member.name}")
            for member in members:
                destination = root / member.name
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
        marker.write_text(json.dumps({"url": url, "sha256": digest}, indent=2) + "\n")
    print(f"ready {name} sha256={digest}", flush=True)
    return {"sequence": name, "url": url, "archive": str(archive),
            "archive_sha256": digest, "archive_bytes": archive.stat().st_size}


def make_manifests(root, records):
    import torch
    from sokkanaem.data import ClipDataset, _pair_by_timestamp

    torch.set_num_threads(2)
    seqs = {name: _pair_by_timestamp(root / name / "rgb", root / name / "depth")
            for name in SEQUENCES}
    # Decode every paired RGB/depth once and hash the originals. This checks
    # data integrity without running a model, ranking clips or selecting on GT.
    inventory, valid_counts = {}, {}
    reader = ClipDataset([], 5000.0, size=256)
    for name, pairs in seqs.items():
        if len(pairs) < 256:
            raise ValueError(f"too few paired frames: {name}: {len(pairs)}")
        frames = []
        for rgb, dep in pairs:
            reader._rgb(rgb)
            depth = reader._depth(dep)
            n = int((torch.isfinite(depth) & (depth > 0)).sum())
            valid_counts[(rgb, dep)] = n
            frames.append({"rgb": str(Path(rgb).relative_to(root)),
                           "depth": str(Path(dep).relative_to(root)),
                           "rgb_sha256": sha256(rgb), "depth_sha256": sha256(dep),
                           "valid_px_256": n})
        inventory[name] = frames
        print(f"verified {name}: {len(frames)} paired frames", flush=True)
    out = Path("paper/freeze")
    out.mkdir(parents=True, exist_ok=True)
    inv_path = out / "final_test_files.json"
    sealed_write(inv_path, {"root": str(root), "sequences": inventory,
                           "downloads": records})
    manifests = []
    for length in (8, 32, 256):
        clips = []
        for name, pairs in seqs.items():
            for start in range(0, len(pairs) - length + 1, length):
                chosen = pairs[start:start + length]
                clips.append({"source": "tum", "sequence": name,
                              "start": start, "pairs": chosen,
                              "valid_px": sum(valid_counts[tuple(p)] for p in chosen),
                              "px": length * 256 * 256})
        path = Path(f"manifests/paper_test_tum_L{length}.json")
        content = {"version": 2, "role": "final_test", "created": "2026-09-06",
                   "protocol": "sokkanaem-paper-v1", "clip_len": length,
                   "frame_stride": 1, "size": 256, "sequences": list(SEQUENCES),
                   "scope": "sequence-disjoint TUM moving-camera transfer; not scene-disjoint",
                   "selection": "all timestamp-matched frames, disjoint tiles, discard only incomplete tail",
                   "inventory": str(inv_path), "inventory_sha256": sha256(inv_path),
                   "sources": {"tum": {"spec": f"tum:{root}", "scale": 5000.0,
                                          "mode": "u16"}}, "clips": clips}
        text = json.dumps(content, indent=1) + "\n"
        if path.exists() and path.read_text() != text:
            raise RuntimeError(f"refusing to change sealed manifest: {path}")
        if not path.exists():
            path.write_text(text)
        manifests.append({"path": str(path), "sha256": sha256(path), "clips": len(clips)})
        print(f"sealed {path}: {len(clips)} clips", flush=True)
    sealed_write(out / "final_test.json", {
        "status": "sealed_not_evaluated", "sequences": list(SEQUENCES),
        "inventory": str(inv_path), "inventory_sha256": sha256(inv_path),
        "manifests": manifests, "model_inference_performed": False,
        "pretraining_overlap": "unknown for external pretrained baselines",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/paper_test/tum"))
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.download:
        if shutil.disk_usage(root).free < 7 * 1024 ** 3:
            raise RuntimeError("need at least 7 GiB free for archives and extraction")
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = list(pool.map(lambda name: acquire(name, root), SEQUENCES))
    else:
        records = [download_record(name, root) for name in SEQUENCES]
    make_manifests(root, records)


if __name__ == "__main__":
    main()
