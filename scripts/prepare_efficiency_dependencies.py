"""Restore the two pinned source checkouts and official MiDaS weights for tasks 9–11."""
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record

ROOT = Path(__file__).resolve().parents[1] / "work_dirs/paper_dependencies"
REPOS = {
    "MiDaS-v2_1": ("https://github.com/isl-org/MiDaS.git", "94afff70fe0873c3af70355e5388e4d482bfc807"),
    "gen-efficientnet-pytorch": ("https://github.com/rwightman/gen-efficientnet-pytorch.git", "771ce082b2ce6d033f55b3d47c1f77389ad3c180"),
}
WEIGHT = "midas_v21_small_256.pt"
URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/" + WEIGHT
SHA = "70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca"


def git(*args):
    return subprocess.run(["git", *map(str, args)], text=True, capture_output=True, check=True).stdout.strip()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in REPOS.items():
        target = ROOT / name
        if not target.exists():
            git("clone", "--depth", "1", "--no-checkout", url, target)
            git("-C", target, "fetch", "--depth", "1", "origin", revision)
            git("-C", target, "checkout", "--detach", revision)
        if git("-C", target, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"existing source revision differs; not overwriting {target}")
        if git("-C", target, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"existing tracked source edits; not overwriting {target}")
        print(f"verified {name}: {revision}")
    target = ROOT / WEIGHT
    if not target.exists():
        partial = ROOT / (WEIGHT + ".download")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("xb") as out:
            while data := response.read(1024 * 1024):
                out.write(data)
        if file_record(partial)["sha256"] != SHA:
            raise RuntimeError(f"download hash mismatch; preserved for inspection: {partial}")
        if target.exists():
            raise RuntimeError("destination appeared during download; not overwriting")
        partial.rename(target)
    if file_record(target)["sha256"] != SHA:
        raise RuntimeError("MiDaS weight hash differs; not overwriting")
    print(f"verified {WEIGHT}: {SHA}")


if __name__ == "__main__":
    main()
