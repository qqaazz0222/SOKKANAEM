"""Actual development predictions, fixed midpoint+1 skipped frame per scene."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replace", action="store_true", help="Regenerate this renderer's three existing artifacts")
    args = ap.parse_args()
    if args.out.exists():
        if not args.replace or {p.name for p in args.out.iterdir()} != {"comparison.png", "comparison.pdf", "provenance.json"}:
            raise FileExistsError("New output required, or explicit replacement of this renderer's artifacts only")
    root = Path("work_dirs/qflow_20260909")
    data_path = Path("work_dirs/qquality_audit_20260908/long_development_data.pt")
    data = torch.load(data_path, weights_only=False)
    names = ("dense", "hold_k2", "flow_nearest", "flow_gated")
    paths = [root/("L256_"+n+"_predictions.pt") for n in names]
    predictions = [torch.load(p, weights_only=False) for p in paths]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": "#22354d"})
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#d2d7dc")
    fig, axes = plt.subplots(len(data), 6, figsize=(15, 10))
    fig.subplots_adjust(left=.11, right=.995, top=.96, bottom=.15, wspace=.025, hspace=.11)
    for j, title in enumerate(("RGB", "GT", "Q0 dense", "Hold K2", "Flow nearest", "Flow gated")):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")
    selection = []
    for i, s in enumerate(data):
        t = len(s["rgb"])//2+1
        assert t % 2 == 1
        images = [s["rgb"][t].permute(1, 2, 0).float().numpy(),
                  np.ma.array(s["gt"][t, 0].numpy(), mask=~s["valid"][t, 0].numpy())]
        images += [p[i][t, 0].numpy() for p in predictions]
        for j, img in enumerate(images):
            ax = axes[i, j]
            if j == 0: ax.imshow(img, interpolation="nearest")
            else: ax.imshow(img, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_color("#dce4ec")
        label = Path(s["scene"]).name.replace("rgbd_dataset_freiburg3_", "").replace("rgbd_bonn_", "")
        axes[i, 0].set_ylabel(label.replace("_", "\n")+f"\nt={t}", rotation=0, ha="right", va="center", labelpad=8)
        selection.append({"clip": i, "frame": t, "pair": s["pairs"][t]})
    ca = fig.add_axes([.36, .09, .38, .012])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 5), cmap=cmap), cax=ca,
                 orientation="horizontal", label="Raw metric depth (m); above 5 m saturated")
    fig.text(.11, .008, "Reused development, midpoint+1 (skipped) frame per scene; no visual best-case selection or GT scale alignment.\n"
             "All Q0 inputs: 518 px. Flow: grayscale 128 px. Flow controls contain no SSM. Gray: invalid GT.", fontsize=9, color="#526477")
    args.out.mkdir(parents=True, exist_ok=args.replace)
    artifacts = []
    for ext in ("png", "pdf"):
        path = args.out/("comparison."+ext); fig.savefig(path, dpi=170, bbox_inches="tight"); artifacts.append(file_record(path))
    plt.close(fig)
    write(args.out/"provenance.json", {"source": file_record(__file__), "inputs": [file_record(data_path)]+[file_record(p) for p in paths],
                                      "selection_rule": "midpoint+1, all4 existingL256clips", "selection": selection, "artifacts": artifacts})
    print(args.out)


if __name__ == "__main__": main()
