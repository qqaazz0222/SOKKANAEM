"""Render real cached development predictions; no generated depth images."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.edge_local_refiner import EdgeLocalRefiner
from sokkanaem.protocol import file_record
from sokkanaem.sharpness import grad_mag

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work_dirs/quality_refinement_20260908"
LOCAL = ROOT / "work_dirs/quality_refinement_local_20260908"
OUT = ROOT / "paper/quality_visualizations_20260908"


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Do not overwrite an existing visualization audit")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#22354d", "axes.labelcolor": "#22354d",
                         "axes.edgecolor": "#dce4ec", "savefig.facecolor": "white"})
    results = json.loads((LOCAL / "results.json").read_text())
    protocol = json.loads((LOCAL / "protocol.json").read_text())
    assert file_record(BASE / "development_cache.pt")["sha256"] == protocol["dev_cache"]["sha256"]
    cache = torch.load(BASE / "development_cache.pt", weights_only=False)
    weight_path = LOCAL / "local_edge_0.5.pt"
    weights = torch.load(weight_path, weights_only=False, map_location="cpu")
    model = EdgeLocalRefiner().eval()
    model.load_state_dict(weights["refiner"])
    chosen, seen = [], set()
    for i, sample in enumerate(cache):
        if sample["scene"] in seen:
            continue
        seen.add(sample["scene"])
        t = len(sample["gt"]) // 2
        with torch.no_grad():
            refined, delta = model(sample["rgb"][t:t+1].float(), sample["coarse"][t:t+1],
                                   sample["features"][t:t+1].float())
        gt, valid = sample["gt"][t:t+1], sample["valid"][t:t+1]
        gradient, defined = grad_mag(gt.clamp_min(.001).log(), valid)
        # Select a 64px crop using GT only. No search over candidate improvements.
        mass = torch.nn.functional.avg_pool2d(gradient*defined, 64, 1)[0, 0]
        flat = int(mass.argmax()); y, x = divmod(flat, mass.shape[1])
        name = Path(sample["scene"]).name.replace("rgbd_dataset_freiburg3_", "").replace("rgbd_bonn_", "")
        chosen.append({"cache_index": i, "frame": t, "name": name,
                       "pair": sample["pairs"][t], "crop_xywh": [x, y, 64, 64],
                       "rgb": sample["rgb"][t].float().permute(1, 2, 0).numpy(),
                       "gt": gt[0, 0].numpy(), "valid": valid[0, 0].numpy(),
                       "base": sample["coarse"][t, 0].numpy(),
                       "refined": refined[0, 0].numpy(), "delta": delta[0, 0].numpy()})
    depth_cmap = plt.get_cmap("viridis").copy(); depth_cmap.set_bad("#d2d7dc")
    error_cmap = plt.get_cmap("magma").copy(); error_cmap.set_bad("#d2d7dc")
    diff_cmap = plt.get_cmap("RdBu_r").copy(); diff_cmap.set_bad("#d2d7dc")
    artifacts = []
    def save(fig, stem):
        for ext in ("png", "pdf"):
            path = OUT / f"{stem}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            artifacts.append(file_record(path))
        plt.close(fig)
    def masked(x, s):
        return np.ma.array(x, mask=~s["valid"])
    def render(zoom):
        names = ["RGB", "Ground truth", "v11 baseline", "Local refinement", "Baseline error", "Refined error"]
        if zoom:
            names = ["RGB crop", "Ground truth", "v11 baseline", "Local refinement", "Depth correction", "Error change"]
        fig, axes = plt.subplots(4, 6, figsize=(15, 10))
        fig.subplots_adjust(wspace=.055, hspace=.17, left=.12, right=.99, bottom=.15, top=.96)
        for j, name in enumerate(names):
            axes[0, j].set_title(name, fontsize=11, fontweight="bold", pad=12)
        for row, s in enumerate(chosen):
            x,y,w,h = s["crop_xywh"]
            crop = lambda a: a[y:y+h, x:x+w] if zoom else a
            arrays = [s["rgb"], masked(s["gt"], s), s["base"], s["refined"],
                      masked(np.abs(s["base"]-s["gt"]), s),
                      masked(np.abs(s["refined"]-s["gt"]), s)]
            if zoom:
                arrays[-2] = s["refined"]-s["base"]
                arrays[-1] = masked(np.abs(s["refined"]-s["gt"])-np.abs(s["base"]-s["gt"]), s)
            for col, (ax, a) in enumerate(zip(axes[row], arrays)):
                if col == 0:
                    ax.imshow(crop(a), interpolation="nearest")
                elif col < 4:
                    ax.imshow(crop(a), cmap=depth_cmap, vmin=0, vmax=5, interpolation="nearest")
                else:
                    ax.imshow(crop(a), cmap=diff_cmap if zoom else error_cmap,
                              vmin=-.1 if zoom else 0, vmax=.1 if zoom else .5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_linewidth(.6)
                if not zoom and col < 4:
                    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor="#f1ae43",linewidth=1.2))
            axes[row,0].set_ylabel(s["name"].replace("_", "\n")+f"\nt={s['frame']}",
                                   rotation=0, ha="right", va="center", labelpad=12, fontsize=9)
        cax = fig.add_axes([.24,.088,.30,.013])
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,5), cmap=depth_cmap), cax=cax,
                     orientation="horizontal", label="Depth (m); values above 5 m saturated")
        cax = fig.add_axes([.66,.088,.27,.013])
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-.1 if zoom else 0,.1 if zoom else .5),
                                          cmap=diff_cmap if zoom else error_cmap), cax=cax,
                     orientation="horizontal", label="Signed change (m); limits saturated" if zoom else "Absolute depth error (m); >0.5 m saturated")
        note = "Raw metric depth; no GT alignment. Gray = invalid GT. First clip per scene, midpoint frame; GT-only crop selection."
        if zoom:
            note += "\nError change: blue = lower error, red = higher error. Depth correction: blue = nearer, red = farther."
        else:
            note += "\nAmber boxes identify the crops. Local refinement: lambda=0.5. Development examples, not final-test evidence."
        fig.text(.12,.017,note,fontsize=9,color="#526477",va="bottom")
        save(fig, "boundary_crops" if zoom else "depth_comparison")
    render(False); render(True)
    keys = [("absrel_raw","AbsRel"),("boundary_f1","Boundary F1"),
            ("overshoot","Overshoot"),("flat_tv","Flat-region TV")]
    values = [results["baseline"]["balanced"]] + [results["variants"][n]["metrics"]["balanced"]
                for n in ("local_edge_0.5", "local_edge_2.0")]
    fig, axes = plt.subplots(1,4,figsize=(13,3.9))
    fig.subplots_adjust(bottom=.30,wspace=.32)
    for ax,(key,label) in zip(axes,keys):
        vals = [v[key] for v in values]
        ax.bar(range(3),vals,color=["#22354d","#087f8c","#d59432"],width=.62)
        ax.set_title(label+(" (higher is better)" if key=="boundary_f1" else " (lower is better)"),fontsize=10)
        ax.set_xticks(range(3),["v11","Local 0.5","Local 2.0"],fontsize=9)
        ax.set_ylim(0,max(vals)*1.22)
        for j,v in enumerate(vals): ax.text(j,v+max(vals)*.025,f"{v:.4f}",ha="center",fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    fig.text(.12,.06,"16 development clips / 128 frames; source-balanced means. No uncertainty estimate.\nBoth candidates fail the overshoot non-regression gate; neither replaces the paper model.",fontsize=10,color="#526477")
    save(fig,"metric_comparison")
    np.savez_compressed(OUT/"selected_predictions.npz", **{
        f"scene{i}_{k}":s[k] for i,s in enumerate(chosen)
        for k in ("rgb","gt","valid","base","refined","delta")})
    artifacts.append(file_record(OUT/"selected_predictions.npz"))
    audit = {"role":"development_visualization_not_final", "checkpoint":file_record(weight_path),
             "cache":file_record(BASE/"development_cache.pt"), "results":file_record(LOCAL/"results.json"),
             "source":file_record(__file__), "model_code":[file_record(ROOT/p) for p in
             ("sokkanaem/edge_local_refiner.py","sokkanaem/bounded_refiner.py")],
             "selection_rule":"first cached clip per scene; t=T//2; crop maximum GT log-gradient mass, 64x64",
             "selection":[{k:v for k,v in s.items() if k not in ("rgb","gt","valid","base","refined","delta")} for s in chosen],
             "artifacts":artifacts}
    (OUT/"provenance.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"output":str(OUT),"scenes":len(chosen),"artifacts":len(artifacts)},indent=2))


if __name__ == "__main__":
    main()
