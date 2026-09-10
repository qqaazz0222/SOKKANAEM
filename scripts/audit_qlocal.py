"""Audit actual protected pixels and where GT-flat excess remains (not inference)."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from sokkanaem.protocol import file_record
from sokkanaem.qlocal import editable_mask
from sokkanaem.qflat_state import flat_mask
from sokkanaem.sharpness import norm_disp, grad_mag
from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingReassembleStage


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    torch.set_num_threads(4)
    result = {"role": "posthoc_GT_audit_not_runtime_mask", "source": file_record(__file__),
              "decoder_source": file_record(inspect.getsourcefile(DepthAnythingReassembleStage)), "data": {}}
    for run in ("qseparated", "qlocal_unrestricted", "qlocal_protected"):
        p = json.loads(Path("work_dirs/"+run+"_20260909/protocol.json").read_text())
        for r in p["code"]+[p["checkpoint"],p["adapter"]]+p["development"]:
            assert file_record(r["path"]) == r
    for length, data_path in (("L32", "work_dirs/qstream_20260908/development_data.pt"),
                              ("L256", "work_dirs/qquality_audit_20260908/long_development_data.pt")):
        samples = torch.load(data_path, weights_only=False)
        root = Path("work_dirs/qlocal_protected_20260909")
        base = torch.load(root/(length+"_ssm_unrefined_predictions.pt"), weights_only=False)
        pred = torch.load(root/(length+"_ssm_refined_predictions.pt"), weights_only=False)
        teacher = torch.load("work_dirs/qquality_audit_20260908/"+length+"_dense_predictions.pt", weights_only=False)
        counts = {"skipped_pixels": 0, "editable_pixels": 0, "flat_pixels": 0, "editable_flat_pixels": 0,
                  "flat_positive_excess_sum": 0., "protected_positive_excess_sum": 0., "refresh_frames": 0,
                  "protected_pixel_max_abs_change": 0., "refresh_max_abs_change": 0.}
        for s, before, after, target in zip(samples, base, pred, teacher):
            torch.testing.assert_close(after[::2], target[::2], rtol=0, atol=0)
            counts["refresh_frames"] += len(after[::2])
            for start in range(0, len(before), 32):
                sl = slice(start+1, min(start+32, len(before)), 2)
                b = before[sl].cuda(); a = after[sl].cuda(); t = target[sl].cuda()
                edit = editable_mask(s["rgb"][sl].cuda(), b)
                torch.testing.assert_close(a[~edit], b[~edit], rtol=0, atol=0)
                flat, valid = flat_mask(s["gt"][sl].cuda(), s["valid"][sl].cuda())
                gb, _ = grad_mag(norm_disp(b, valid), valid); gt, _ = grad_mag(norm_disp(t, valid), valid)
                excess = (gb-gt).relu()*flat
                counts["skipped_pixels"] += edit.numel(); counts["editable_pixels"] += int(edit.sum())
                counts["flat_pixels"] += int(flat.sum()); counts["editable_flat_pixels"] += int((flat & edit).sum())
                counts["flat_positive_excess_sum"] += float(excess.sum())
                counts["protected_positive_excess_sum"] += float((excess*(~edit)).sum())
        counts["editable_pixel_fraction"] = counts["editable_pixels"]/counts["skipped_pixels"]
        counts["editable_GTflat_fraction"] = counts["editable_flat_pixels"]/counts["flat_pixels"]
        counts["protected_share_of_positive_excess"] = counts["protected_positive_excess_sum"]/max(counts["flat_positive_excess_sum"], 1e-12)
        result["data"][length] = counts
        sr = Path("work_dirs/qseparated_20260909")
        for a, b in (("unchanged", "hold_cls"), ("hold_pooled", "hold_both")):
            xs = torch.load(sr/(length+"_"+a+"_predictions.pt"), weights_only=False)
            ys = torch.load(sr/(length+"_"+b+"_predictions.pt"), weights_only=False)
            for x, y in zip(xs, ys): torch.testing.assert_close(x, y, rtol=0, atol=0)
        print(length, counts, flush=True)
    args.out.mkdir(parents=True); q.write(args.out/"audit.json", result)


if __name__ == "__main__": main()
