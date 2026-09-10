"""K4 matched sequence training: carry SSM / reset SSM / MLP, no final data."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record, FINAL_TEST_SEQUENCES, runtime_versions
from sokkanaem.qaligned_adapter import AlignedAdapter
from sokkanaem.qrecurrent_flow import advance, RecurrentFlowStream, K4OutputFlow
from sokkanaem.qflow import RGBFlowStream, gray_image, correspondence
from sokkanaem.qquality import decode_with_grad, depth_retention
from sokkanaem.qflat_state import flat_mask, flat_excess_loss
from sokkanaem.detector import ChangeDetector


def build_cache(exact, train, out):
    cache = []; engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    detector = ChangeDetector(patch_size=14, tau_on=.001, tau_off=.0005, keyframe_every=4, dilate=True)
    with torch.no_grad():
        for ci,s in enumerate(train):
            rgb = s["rgb"].cuda().float(); assert len(rgb) == 4
            encoded = exact.encode(rgb); teacher = exact.decode(encoded)
            images = F.interpolate(rgb, (518,518), mode="bilinear", align_corners=False)
            flat, valid = flat_mask(s["gt"].cuda(), s["valid"].cuda())
            _, det = detector.gate(None,1,37,37,rgb.device,None)
            flows = []; reliable = []; masks = []
            for t in range(1,4):
                flow, confidence = correspondence(gray_image(rgb[t-1:t]),gray_image(rgb[t:t+1]),engine)
                score = F.avg_pool2d((images[t:t+1]-images[:1]).square().mean(1,keepdim=True),14)[:,0]
                mask, det = detector.gate(score,1,37,37,rgb.device,det)
                flows.append(flow); reliable.append(confidence); masks.append(mask.cpu())
            cache.append({"features":[f.cpu().half() for f in encoded["features"]], "pooled":encoded["pooled"].cpu(),
                          "grid":encoded["grid"], "size":encoded["size"], "rgb":rgb.cpu(), "teacher":teacher.cpu(),
                          "flat":flat.cpu(), "valid":valid.cpu(), "flows":flows, "reliable":reliable, "masks":masks,
                          "pairs":s["pairs"], "source":s["source"]})
            if (ci+1)%32 == 0: print("sequence cache",ci+1,"/",len(train),flush=True)
    torch.save(cache,out/"training_cache.pt"); return cache


def sequence_loss(model, row, exact, reset_h):
    teacher_features = [f.cuda().float() for f in row["features"]]
    state = model.refresh({"features":[f[:1] for f in teacher_features],"pooled":row["pooled"][:1].cuda(),"grid":row["grid"],"size":row["size"]})
    rgb = row["rgb"].cuda(); teacher = row["teacher"].cuda()
    valid = row["valid"].cuda(); flat_mask_value = row["flat"].cuda()
    terms = []; incoming = []
    for t in range(1,4):
        state, info = advance(model,state,rgb[t:t+1],row["masks"][t-1].cuda(),row["flows"][t-1],row["reliable"][t-1],reset_h)
        incoming.append(info["incoming_h_abs_mean"])
        pred = decode_with_grad(exact,state)
        feature = sum((f[:,1:]-target[t:t+1,1:]).abs().mean() for f,target in zip(state["features"],teacher_features))/4
        depth, edge = depth_retention(pred,teacher[t:t+1])
        flat = flat_excess_loss(pred,teacher[t:t+1],valid[t:t+1],flat_mask_value[t:t+1])
        terms.append(torch.stack((feature,depth,edge,flat)))
    means = torch.stack(terms).mean(0)
    return means @ means.new_tensor([1,5,10,20])+0*model.readout.weight.sum(), means, incoming


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=600); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    if args.steps < 1: raise ValueError("Positive steps required")
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cudnn.benchmark=False
    pp = json.loads(Path("work_dirs/qquality_feature_20260908/protocol.json").read_text())
    exact = load_exact(pp["checkpoint"])
    train_path = Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    train = torch.load(train_path,weights_only=False)
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"),Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev = {k:torch.load(p,weights_only=False) for k,p in zip(("L32","L256"),paths)}
    tf = {str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df = {str(Path(p).resolve()) for samples in dev.values() for s in samples for pair in s["pairs"] for p in pair}
    assert not tf & df and not any(set(Path(p).parts)&set(FINAL_TEST_SEQUENCES) for p in tf|df)
    models = {}
    for mode in ("carry","reset","mlp"):
        torch.manual_seed(739); models[mode] = AlignedAdapter("mlp" if mode=="mlp" else "ssm").cuda()
    for name in ("rgb","context","fuse","readout"):
        for a,b in zip(getattr(models["carry"],name).parameters(),getattr(models["mlp"],name).parameters()): assert torch.equal(a,b)
    assert all(torch.equal(a,b) for a,b in zip(models["carry"].parameters(),models["reset"].parameters()))
    args.out.mkdir(parents=True)
    protocol = {"role":"matched_K4_sequence_training_reused_development_no_final", "checkpoint":pp["checkpoint"],
                "training_data":file_record(train_path), "development":[file_record(p) for p in paths],
                "training_clips":len(train), "training_frames":sum(len(s["rgb"]) for s in train),
                "steps_each":args.steps,"unrolled_updates_per_step":3,"seed":739,"lr":.0003,"batch":1,
                "objective":"unchanged 1feature +5Q0logdepth +10Q0loggradient +20trainingGTflat; mean of3free_runningupdates; no teacher forcing after initial refresh",
                "controls":"carry/reset SSM identical initial weights; MLP same common initialization,89037 vs89040 parameters; same sequence order/loss/steps",
                "history":"full BPTT across3updates including transported features/h; reset control zeros only h, retains feature history; full Q0 resets all every4frames",
                "cache":"original teacherfeatures storedFP16; RGB/depth/pooled FP32; training flow precomputed, online dev flow fully timed",
                "gate":"unchanged quality_gate and >=5pct mean latency reduction", "timing":"synchronized GPU resident incl CPUflow/transfers/SSM/decoder/telemetry; excludes initialIO/H2D; RTX4090 not edge",
                "runtime":runtime_versions(),"opencv":cv2.__version__,
                "code":[file_record(p) for p in [__file__,"sokkanaem/qrecurrent_flow.py","sokkanaem/qaligned_adapter.py","sokkanaem/qflow.py","sokkanaem/qflow_features.py",
                          "sokkanaem/qdelta_ssm.py","sokkanaem/ssm.py","sokkanaem/qquality.py","sokkanaem/qflat_state.py","sokkanaem/qstream.py","sokkanaem/qmodel.py",
                          "sokkanaem/detector.py","scripts/qquality_study.py","scripts/qstream_study.py","scripts/quality_refinement.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol)
    cache = build_cache(exact,train,args.out)
    q.write(args.out/"cache_record.json",file_record(args.out/"training_cache.pt"))
    order = torch.randint(len(cache),(args.steps,),generator=torch.Generator().manual_seed(739)).tolist()
    q.write(args.out/"training_order.json",order)
    for name,model in models.items():
        opt = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001); history=[]
        for step,i in enumerate(order,1):
            loss,means,incoming = sequence_loss(model,cache[i],exact,reset_h=name=="reset")
            opt.zero_grad(set_to_none=True); loss.backward()
            a_grad = model.ssm.A_log.grad if name!="mlp" else None
            amax = float(a_grad.abs().max()) if a_grad is not None else None
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
            if step==1 or step%100==0 or step==args.steps:
                row={"step":step,"loss":float(loss.detach()),"components":means.detach().cpu().tolist(),"incoming_h":incoming,"A_data_gradient_max":amax}
                history.append(row); q.write(args.out/(name+"_history.json"),history); print(name,row,flush=True)
        model.eval().requires_grad_(False)
        torch.save({"adapter":model.state_dict(),"mode":name},args.out/(name+"_adapter.pt"))
    results={"promoted":False}; parity=[]
    for length,samples in dev.items():
        baseline,dense=evaluate(exact,samples); results[length]={"baseline":baseline}
        torch.save(dense,args.out/(length+"_dense_predictions.pt"))
        wrappers={"output_k2":RGBFlowStream(exact),"output_k4":K4OutputFlow(exact),
                  "features_no_update":RecurrentFlowStream(exact,models["carry"],no_delta=True),
                  "carry_carry":RecurrentFlowStream(exact,models["carry"]),
                  "carry_reset":RecurrentFlowStream(exact,models["carry"],reset_h=True),
                  "reset_reset":RecurrentFlowStream(exact,models["reset"],reset_h=True),
                  "reset_carry":RecurrentFlowStream(exact,models["reset"]),
                  "mlp":RecurrentFlowStream(exact,models["mlp"])}
        for name,wrapper in wrappers.items():
            measured,preds=evaluate(wrapper,samples)
            measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
            measured["speedup"]=baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["speedup"]>=1/.95
            count=0
            for ci,tele in enumerate(measured["telemetry"]):
                for t,info in enumerate(tele["frames"]):
                    if info["full_refresh"]: assert torch.equal(preds[ci][t],dense[ci][t]); count+=1
            parity.append({"length":length,"model":name,"comparisons":count,"max_abs":0.})
            results[length][name]=measured; torch.save(preds,args.out/(length+"_"+name+"_predictions.pt"))
            q.write(args.out/"results.json",results); q.write(args.out/"parity.json",parity)
            print(length,name,measured["metrics"]["balanced"],"ms",measured["mean_ms"],measured["quality_gate"],flush=True)
    for r in protocol["code"]+protocol["development"]+[protocol["checkpoint"],protocol["training_data"]]: assert file_record(r["path"])==r
    q.write(args.out/"integrity.json",{"input_hashes_unchanged":True,"full_refresh_parity_comparisons":sum(r["comparisons"] for r in parity)})


if __name__=="__main__":main()
