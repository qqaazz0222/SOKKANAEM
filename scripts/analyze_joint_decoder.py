"""Read-only model ablation, writing a separate analysis artifact."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.joint_decoder_study import evaluate
from scripts.quality_refinement import write
from sokkanaem.joint_detail_decoder import JointDetailDecoder
from sokkanaem.model import from_checkpoint
from sokkanaem.protocol import file_record


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);args=ap.parse_args()
    dest=args.run/"detail_ablation.json"
    if dest.exists():raise FileExistsError(dest)
    torch.set_num_threads(4)
    p=json.loads((args.run/"protocol.json").read_text())
    base=from_checkpoint(p["checkpoint"]["path"],"cuda").eval()
    model=JointDetailDecoder(base.decoder,True).cuda().eval();del base
    weight=args.run/"joint_detail.pt"
    model.load_state_dict(torch.load(weight,weights_only=False)["state_dict"])
    data=torch.load(args.run/"dev_tokens.pt",weights_only=False)
    with torch.no_grad():
        enabled=evaluate(model,data)
        detail=model.detail;model.detail=None
        disabled=evaluate(model,data)
    result={"role":"same_trained_decoder_detail_path_removal_development",
            "checkpoint":file_record(weight),"source":file_record(__file__),
            "enabled":enabled,"disabled":disabled,
            "difference_enabled_minus_disabled":{k:enabled["balanced"][k]-disabled["balanced"][k]
                                                for k in enabled["balanced"]},
            "detail_parameters":sum(x.numel() for x in detail.parameters()),
            "note":"Post-training inference removal, not a substitute for independent training ablation."}
    write(dest,result);print(json.dumps(result["difference_enabled_minus_disabled"],indent=2))


if __name__=="__main__":main()
