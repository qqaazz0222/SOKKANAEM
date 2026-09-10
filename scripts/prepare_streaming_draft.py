"""Freeze evidence, audit local real-data eligibility, render measured figures."""
import json
from pathlib import Path
import sys
import tomllib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record,FINAL_TEST_SEQUENCES
from scripts.quality_refinement import write

OUT=Path('paper/streaming_draft')
ROOT=Path('work_dirs/qpreflow_20260909')


def savefig(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    if OUT.exists():raise FileExistsError('New draft asset directory required')
    pp=json.loads((ROOT/'protocol.json').read_text())
    records=pp['code']+pp['development']+[pp[k] for k in ('checkpoint','parent_protocol','traces','order')]
    for r in records:assert file_record(r['path'])==r
    results=json.loads((ROOT/'results.json').read_text())
    timing=json.loads(Path('work_dirs/qpreflow_candidate_audit_20260909/audit.json').read_text())
    threshold=json.loads((ROOT/'thresholds.json').read_text())['mlp']['50']
    (OUT/'figures').mkdir(parents=True)
    write(OUT/'frozen_candidate.json',{'role':'development_candidate_frozen_for_draft_not_final_promotion',
        'policy':'MLP q50 RGB-only preflow; no SSM in selected policy','threshold':threshold,
        'maximum_gap':4,'mandatory_first_skip':True,'Q0_input':518,'score_size':256,'flow_gray_size':128,
        'weights':[file_record(ROOT/'mlp_risk.pt'),pp['checkpoint']],
        'evidence':[file_record(ROOT/n) for n in ('results.json','protocol.json','thresholds.json','integrity.json')]
            +[file_record('work_dirs/qpreflow_candidate_audit_20260909/'+n) for n in ('audit.json','protocol.json','integrity.json')],
        'code':records,'native_final_unchanged':True})
    cfg=tomllib.loads(Path('configs/main_v8.toml').read_text());inventory=[]
    for spec in cfg['data'][:2]:
        src,root=spec.split(':',1)
        for d in sorted(Path(root).iterdir()):
            if not (d/'rgb').is_dir() or not (d/'depth').is_dir():continue
            role='reserved_final' if d.name in FINAL_TEST_SEQUENCES else ('reused_development' if any(h in str(d) for h in cfg['holdout']) else 'training_eligible_not_independent')
            inventory.append({'source':src,'scene':str(d),'role':role})
    write(OUT/'split_audit.json',{'scope':'only configured local TUM/Bonn roots; not exhaustive audit of every dataset or historical sample',
        'configuration':file_record('configs/main_v8.toml'),'training_script':file_record('scripts/train_q.py'),
        'checkpoint_config':file_record('work_dirs/q0-calib-s0/config.toml'),'inventory':inventory,
        'new_certified_independent_real_scenes':0,
        'limitation':'Current config and recorded training command establish training eligibility, not exact historical sampled frames. Other synthetic/outdoor sources not certified untouched. No new generalization run claimed.'})
    table=[]
    for length in ('L32','L256'):
        base=results[length]['baseline']['metrics']['balanced']
        for name,row in results[length].items():
            m=row['metrics']['balanced']
            table.append({'length':length,'policy':name,'raw_absrel':m['absrel_raw'],'edge_absrel':m['absrel_edge'],
                'boundary_f1':m['boundary_f1'],'overshoot':m['overshoot'],'flat_tv':m['flat_tv'],
                'flat_change_pct':100*(m['flat_tv']/base['flat_tv']-1),'mean_ms_single_run':row['mean_ms'],
                'quality_pass':row.get('quality_gate',{}).get('pass',True),'skip_fraction':row['skip_fraction']})
    write(OUT/'evidence_table.json',table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#22354d','axes.labelcolor':'#22354d','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(12,4.1));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09,rounding_size=.12',facecolor=color,edgecolor='#b9c8d6'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
    def arrow(a,b,label=None):
        ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':'#526477','lw':1.6})
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=9)
    box(.2,1.4,1.3,1,'Current RGB\nframe','#e9f0f7')
    box(2,1.4,2.4,1,'RGB change + age\nMLP refresh decision','#dcebf4')
    box(5.2,2.6,2.5,.85,'Frozen Q0\nfull depth inference','#e0ece6')
    box(5.2,.35,2.5,.85,'DIS flow + nearest\nprevious-depth warp','#f2e9da')
    box(8.4,1.4,1.35,1,'Depth\noutput','#e9f0f7')
    box(10.25,1.4,1.5,1,'Cache\nupdate','#e9f0f7')
    arrow((1.5,1.9),(2,1.9));arrow((4.4,2.1),(5.2,2.95),'Refresh')
    arrow((4.4,1.6),(5.2,.85),'Reuse');arrow((7.7,2.95),(8.4,2.1));arrow((7.7,.85),(8.4,1.6));arrow((9.75,1.9),(10.25,1.9))
    ax.plot([11,11,3.2,3.2],[2.4,3.85,3.85,2.4],ls='--',color='#526477',lw=1.2)
    ax.text(8.8,3.63,'Previous cache / anchor / age',ha='center',fontsize=9)
    ax.text(2.7,.15,'Gap 1: reuse    |    Gaps 2–3: learned decision    |    Gap 4: refresh',fontsize=10)
    savefig(fig,'architecture')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for ax,length in zip(axes,('L32','L256')):
        keys=['dense','output_k2','mlp_q50'];means=[timing[length]['means_ms'][k] for k in keys]
        lo=[min(x['mean_ms'] for x in timing[length]['timings'][k]) for k in keys]
        hi=[max(x['mean_ms'] for x in timing[length]['timings'][k]) for k in keys]
        ax.barh(['Q0 dense','Flow K2','Pre-flow MLP'],means,color=['#aebdcd','#698eac','#438779'],xerr=[np.array(means)-lo,np.array(hi)-means],capsize=3)
        ax.invert_yaxis();ax.set_xlim(0,9);ax.set_xlabel('Synchronized latency (ms/frame)');ax.set_title(length)
        for i,v in enumerate(means):ax.text(v+.15,i,f'{v:.3f}',va='center')
    fig.tight_layout();savefig(fig,'latency')
    data_path=Path(pp['development'][1]['path']);data=torch.load(data_path,weights_only=False)
    names=['dense','output_k2','mlp_q50'];paths=[ROOT/('L256_'+n+'_predictions.pt') for n in names]
    preds=[torch.load(p,weights_only=False) for p in paths]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#d2d7dc');selection=[]
    def panel(rows,filename,crop=False):
        fig,axes=plt.subplots(len(rows),6,figsize=(14,2.7*len(rows)),squeeze=False)
        for j,title in enumerate(['RGB','GT','Q0 dense','Flow K2','Pre-flow MLP','MLP relative error']):axes[0,j].set_title(title,fontsize=10)
        for i,(ci,t,reason) in enumerate(rows):
            s=data[ci];gt=s['gt'][t,0].numpy();valid=s['valid'][t,0].numpy().astype(bool)
            ims=[s['rgb'][t].permute(1,2,0).numpy(),np.ma.array(gt,mask=~valid)]+[p[ci][t,0].numpy() for p in preds]
            ims.append(np.ma.array(np.abs(ims[-1]-gt)/np.maximum(gt,1e-4),mask=~valid))
            for j,im in enumerate(ims):
                if crop:im=im[64:192,64:192]
                ax=axes[i,j];ax.imshow(im,interpolation='nearest',**({} if j==0 else {'cmap':'magma' if j==5 else cmap,'vmin':0,'vmax':.3 if j==5 else 5}))
                ax.set_xticks([]);ax.set_yticks([])
            info=results['L256']['mlp_q50']['telemetry'][ci]['frames'][t]
            axes[i,0].set_ylabel(Path(s['scene']).name.replace('rgbd_dataset_freiburg3_','').replace('rgbd_bonn_','').replace('_','\n')+f'\nt={t}\nage={info["cache_age"]}',rotation=0,ha='right',va='center',fontsize=8)
            if not crop:selection.append({'clip':ci,'frame':t,'reason':reason,'pair':s['pairs'][t],'telemetry':info})
        fig.subplots_adjust(left=.13,right=.99,top=.93,bottom=.08,wspace=.03,hspace=.15)
        fig.text(.13,.015,'Depth: 0–5 m (saturated above 5 m). Relative error: 0–0.30. Gray: invalid GT. No GT scale fitting.',fontsize=9)
        savefig(fig,filename)
    rows=[]
    for ci,s in enumerate(data):
        frames=results['L256']['mlp_q50']['telemetry'][ci]['frames']
        skip=[t for t,f in enumerate(frames) if not f['full_refresh']]
        t=min(skip,key=lambda t:(abs(t-len(frames)//2),t));rows.append((ci,t,'skipped frame nearest temporal midpoint'))
    panel(rows,'qualitative');panel(rows,'boundary_crops',True)
    failures=[]
    for ci,s in enumerate(data):
        gt=s['gt'][:,0];valid=s['valid'][:,0].bool();base=preds[0][ci][:,0];cand=preds[2][ci][:,0]
        delta=(((cand-gt).abs()-(base-gt).abs())/gt.clamp_min(1e-4)*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
        skip=[t for t,f in enumerate(results['L256']['mlp_q50']['telemetry'][ci]['frames']) if not f['full_refresh']]
        t=max(skip,key=lambda t:float(delta[t]));failures.append((ci,t,'largest per-frame raw AbsRel increase vs Q0 among skipped frames; posthoc diagnostic'))
    panel(failures,'failure_cases')
    write(OUT/'visual_provenance.json',{'renderer':file_record(__file__),'data':file_record(data_path),
        'predictions':[file_record(p) for p in paths],'selection':selection,'crop':'fixed central [64:192,64:192], not edge-metric selection',
        'artifacts':[file_record(p) for p in sorted((OUT/'figures').iterdir())]})
    for r in records:assert file_record(r['path'])==r
    print('Prepared',OUT,'local scenes',len(inventory),'independent certified',0,flush=True)


if __name__=='__main__':main()
