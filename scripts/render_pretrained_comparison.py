"""Fixed first-clip/midpoint development visualization, genuine predictions."""
import json
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record


def main():
    root=Path("work_dirs/pretrained_quality_20260908")
    out=Path("paper/pretrained_quality_visuals_20260908")
    if out.exists():raise FileExistsError(out)
    p=json.loads((root/"protocol.json").read_text())
    data=torch.load(p["dev_data"]["path"],weights_only=False)
    names=("native224","q0_224","q0_518")
    preds={n:torch.load(root/(n+"_predictions.pt"),weights_only=False) for n in names}
    selected=[];seen=set()
    for i,s in enumerate(data):
        if s["scene"] not in seen:selected.append((i,len(s["rgb"])//2));seen.add(s["scene"])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":"#22354d","axes.labelcolor":"#22354d"})
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#d2d7dc")
    fig,axes=plt.subplots(4,5,figsize=(13,10))
    fig.subplots_adjust(left=.14,right=.99,top=.96,bottom=.13,wspace=.045,hspace=.15)
    for j,title in enumerate(("RGB","GT","Native 224","Q0 224","Q0 518 reference")):
        axes[0,j].set_title(title,fontweight="bold",fontsize=11)
    for row,(i,t) in enumerate(selected):
        s=data[i]
        images=[s["rgb"][t].permute(1,2,0).float().numpy(),
                np.ma.array(s["gt"][t,0].numpy(),mask=~s["valid"][t,0].numpy())]
        images += [preds[n][i][t,0].numpy() for n in names]
        for j,img in enumerate(images):
            ax=axes[row,j]
            if j==0:ax.imshow(img,interpolation="nearest")
            else:ax.imshow(img,cmap=cmap,vmin=0,vmax=5,interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color("#dce4ec")
        label=Path(s["scene"]).name.replace("rgbd_dataset_freiburg3_","").replace("rgbd_bonn_","")
        axes[row,0].set_ylabel(label.replace("_","\n")+f"\nt={t}",rotation=0,ha="right",va="center",labelpad=10,fontsize=9)
    ca=fig.add_axes([.35,.075,.40,.012])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,5),cmap=cmap),cax=ca,orientation="horizontal",label="Raw metric depth (m); values above 5 m saturated")
    fig.text(.14,.012,"Development frames, first clip per scene / midpoint frame. Gray: invalid GT. No GT scale alignment.\nNative224 vs Q0_224: matched input resolution. Q0_518: unequal-compute reference, NOT a fair speed comparison.",fontsize=9,color="#526477")
    out.mkdir(parents=True)
    artifacts=[]
    for ext in ("png","pdf"):
        f=out/("comparison."+ext);fig.savefig(f,dpi=180,bbox_inches="tight");artifacts.append(file_record(f))
    plt.close(fig)
    audit={"role":"development_visualization_not_final","source":file_record(__file__),
           "inputs":[file_record(root/(n+"_predictions.pt")) for n in names]+[p["dev_data"]],
           "selection":[{"clip_index":i,"frame":t,"pair":data[i]["pairs"][t]} for i,t in selected],"artifacts":artifacts}
    (out/"provenance.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(out)


if __name__=="__main__":main()
