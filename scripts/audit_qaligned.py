"""Training-set objective audit and actual trained zero-history SSM algebra."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from sokkanaem.protocol import file_record
from sokkanaem.qaligned_adapter import AlignedAdapter, zero_state_formula
from sokkanaem.qquality import decode_with_grad, depth_retention
from sokkanaem.qflat_state import flat_excess_loss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    root = Path("work_dirs/qaligned_20260909"); pp = json.loads((root/"protocol.json").read_text())
    cache_record = json.loads((root/"cache_record.json").read_text())
    records = pp["code"]+pp["development"]+[pp["checkpoint"],pp["training_data"],cache_record]
    for r in records: assert file_record(r["path"]) == r
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    models = {}
    for mode in ("ssm", "mlp"):
        model = AlignedAdapter(mode).cuda().eval()
        model.load_state_dict(torch.load(root/(mode+"_adapter.pt"), map_location="cuda", weights_only=False)["adapter"], strict=True)
        models[mode] = model
    cache = torch.load(cache_record["path"], weights_only=False)
    sums = {name: torch.zeros(4, dtype=torch.float64) for name in ("aligned_no_update", "ssm", "mlp")}
    core_rows = []
    for i, row in enumerate(cache):
        state = models["ssm"].refresh({"features": [f.cuda().float() for f in row["features"]], "pooled": row["pooled"].cuda(),
                                        "grid": row["grid"], "size": row["size"]})
        rgb = row["rgb"].cuda(); mask = row["mask"].cuda()
        with torch.no_grad():
            for name in sums:
                current = state if name == "aligned_no_update" else models[name].update(rgb, mask, state)[0]
                pred = decode_with_grad(exact, current)
                feature = sum((f[:,1:]-t[:,1:].cuda().float()).abs().mean() for f,t in zip(current["features"],row["teacher_features"]))/4
                depth, edge = depth_retention(pred, row["teacher"].cuda())
                flat = flat_excess_loss(pred, row["teacher"].cuda(), row["valid"].cuda(), row["flat"].cuda())
                sums[name] += torch.tensor([float(x) for x in (feature, depth, edge, flat)], dtype=torch.float64)
        # Fixed every48th training pair: algebra audit, not dev model selection.
        if i % 48 == 0:
            model = models["ssm"]; idx = (mask.flatten() > .5).nonzero().flatten()
            image = F.interpolate(rgb, (518,518), mode="bilinear", align_corners=False)
            local = model.rgb(image).flatten(2).transpose(1,2)
            context = model.context(state["features"][-1][:,1:])
            u = model.fuse(torch.cat((local,context),-1)).reshape(-1,32)[idx]
            actual, h = model.ssm.step(u, h=state["h"][idx])
            direct, hd = zero_state_formula(model.ssm,u)
            torch.testing.assert_close(actual,direct,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(h,hd,rtol=1e-5,atol=1e-6)
            grad = torch.autograd.grad(actual.sum(),model.ssm.A_log,allow_unused=True)[0]
            norm = 0. if grad is None else float(grad.abs().max())
            assert norm < 1e-7
            core_rows.append({"training_pair":i,"active_patches":len(idx),"output_max_abs":float((actual-direct).abs().max().detach()),
                              "hidden_max_abs":float((h-hd).abs().max().detach()),"A_data_gradient_max_abs":norm})
        if (i+1) % 96 == 0: print("audit training pairs",i+1,"/",len(cache),flush=True)
    means = {}
    for name, value in sums.items():
        value /= len(cache)
        means[name] = dict(zip(("feature","depth","edge","flat"),value.tolist()))
        means[name]["objective"] = float(value @ torch.tensor([1,5,10,20],dtype=torch.float64))
    args.out.mkdir(parents=True)
    result = {"role":"posthoc_training_objective_and_algebra_audit_not_independent_evaluation", "source":file_record(__file__),
              "inputs":[file_record(root/(m+"_adapter.pt")) for m in models]+[cache_record],
              "training_pairs":len(cache),"matched_all_pair_mean":means,"zero_history_core":core_rows,
              "note":"Different randomly sampled per-step losses are not a learning curve. These means use all same384trainingpairs. Zero-history algebra tests do not show memory benefit."}
    q.write(args.out/"audit.json",result)
    for r in records: assert file_record(r["path"]) == r
    print(means,flush=True); print("algebra comparisons",len(core_rows),flush=True)


if __name__ == "__main__": main()
