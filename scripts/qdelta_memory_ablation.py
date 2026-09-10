"""Same trained adapter, reset temporal hidden state every frame for diagnosis."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qdelta_study import HybridStream
from scripts.qstream_study import evaluate
from scripts.quality_refinement import write
from sokkanaem.qstream import Q0ExactStream
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.model import from_checkpoint
from sokkanaem.protocol import file_record


class ResetMemory(HybridStream):
    @torch.no_grad()
    def step(self,frame,state=None):
        if state is not None:
            state={**state,"features":{**state["features"],"h":torch.zeros_like(state["features"]["h"])}}
        return super().step(frame,state)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);args=ap.parse_args()
    out=args.run/"memory_ablation.json"
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4)
    protocol=json.loads((args.run/"protocol.json").read_text())
    parent=Path(protocol["parent"]["path"]).parent
    model=from_checkpoint(protocol["checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    exact=Q0ExactStream(model);adapter=QDeltaSSM().cuda().eval()
    adapter.load_state_dict(torch.load(args.run/"adapter.pt",weights_only=False)["adapter"])
    samples=torch.load(parent/"development_data.pt",weights_only=False)
    # Periodic policy gives memory an opportunity to operate on 75% of frames.
    reset,preds=evaluate(ResetMemory(exact,adapter,adaptive=False),samples)
    trained=json.loads((args.run/"results.json").read_text())["policies"]["periodic_ssm4"]
    result={"role":"same_weight_recurrent_state_reset_development_ablation",
            "source":file_record(__file__),"adapter":file_record(args.run/"adapter.pt"),
            "reset":reset,"carry_minus_reset":{k:trained["metrics"]["balanced"][k]-reset["metrics"]["balanced"][k]
                                                   for k in trained["metrics"]["balanced"]},
            "note":"Caches retained, only SSM hidden state reset. Not a separately trained nonrecurrent control or long-stream test."}
    write(out,result);print(result["carry_minus_reset"])


if __name__=="__main__":main()
