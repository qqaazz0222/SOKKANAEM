"""Pre-register new TUM sequences, download official archives, freeze frames."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record,FINAL_TEST_SEQUENCES
from sokkanaem.data import _pair_by_timestamp
from scripts.quality_refinement import write

ROOT=Path('work_dirs/qindependent_20260909')
SCENES=['rgbd_dataset_freiburg1_room','rgbd_dataset_freiburg2_desk']


def main():
    if ROOT.exists():raise FileExistsError('New immutable experiment required')
    frozen=Path('paper/streaming_draft/frozen_candidate.json');candidate=json.loads(frozen.read_text())
    for r in candidate['weights']+candidate['code']+candidate['evidence']:assert file_record(r['path'])==r
    # Search history before creating the protocol; exclude this new preparation script.
    cmd=['rg','-l','|'.join(SCENES),'configs','manifests','paper','scripts','work_dirs',
         '-g','*.json','-g','*.toml','-g','*.md','-g','*.log','-g','*.py','-g','!qindependent_prepare.py']
    scan=subprocess.run(cmd,capture_output=True,text=True)
    assert scan.returncode==1 and not scan.stdout and not scan.stderr,scan.stdout+scan.stderr
    ROOT.mkdir();(ROOT/'archives').mkdir();(ROOT/'data').mkdir()
    protocol={'role':'new_local_sequence_holdout_after_candidate_freeze_not_pretraining_disjoint_certification',
        'candidate':file_record(frozen),'weights':candidate['weights'],'threshold':candidate['threshold'],
        'scenes':SCENES,'old_final_excluded':list(FINAL_TEST_SEQUENCES),
        'selection_before_predictions':'Per sequence L32 first32 and last32 associated frames; L256 two windows centered at1/3 and2/3 of associated timeline. Require disjoint RGB and depth paths across windows. No visual/GT/model-performance selection.',
        'expected_frames':{'L32':128,'L256':1024},'policies':['dense','output_k2','mlp_q50'],
        'scoring':'unchanged qstream quality_gate, report aggregate and every sequence; all failures retained, no retuning',
        'timing':'primary synchronized resident-input timing; no independent edge or energy claim',
        'limits':'Only two new TUM sequences; camera/environment shift, same benchmark family; DA2 pretraining overlap unknown. This test is consumed once evaluated.',
        'history_scan':{'command':cmd,'matches':[],'exit_code':scan.returncode},
        'training_config':file_record('work_dirs/q0-calib-s0/config.toml'),
        'configured_training_data':file_record('configs/main_v8.toml'),
        'source':file_record(__file__),
        'official_page':'https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download'}
    write(ROOT/'preregistered.json',protocol)
    def fetch(scene):
        group=scene.split('_')[2];url=f'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/{group}/{scene}.tgz'
        archive=ROOT/'archives'/(scene+'.tgz')
        print('download',scene,url,flush=True)
        urllib.request.urlretrieve(url,archive)
        count=0
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                parts=Path(member.name).parts
                if not member.isfile():continue
                if len(parts)!=3 or parts[0]!=scene or parts[1] not in ('rgb','depth') or '..' in parts:continue
                target=ROOT/'data'/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise FileExistsError(target)
                # Safe regular-file extraction only; never archive links or arbitrary paths.
                with tar.extractfile(member) as inp,target.open('xb') as out:
                    import shutil
                    shutil.copyfileobj(inp,out)
                count+=1
        print('downloaded/extracted',scene,count,flush=True)
        return {'scene':scene,'url':url,'archive':file_record(archive),'regular_rgb_depth_files':count}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:downloads=list(pool.map(fetch,SCENES))
    write(ROOT/'downloads.json',downloads)
    clips={32:[],256:[]};seen=set();hashes=[]
    for scene in SCENES:
        d=ROOT/'data'/scene;pairs=_pair_by_timestamp(str(d/'rgb'),str(d/'depth'))
        assert len(pairs)>=1000,(scene,len(pairs))
        for length,starts in ((32,[0,len(pairs)-32]),(256,[len(pairs)//3-128,2*len(pairs)//3-128])):
            for start in starts:
                selected=pairs[start:start+length];paths={str(Path(p).resolve()) for pair in selected for p in pair}
                assert not paths&seen,(scene,length,'overlap');seen.update(paths)
                clips[length].append({'source':'tum','scene':scene,'start_index':start,'pairs':selected})
                hashes.extend(file_record(p) for p in sorted(paths))
    for length,rows in clips.items():
        write(ROOT/f'manifest_L{length}.json',{'role':'new_local_holdout','clip_len':length,'size':256,
            'sources':{'tum':{'scale':5000.,'mode':'u16'}},'clips':rows})
    write(ROOT/'frame_hashes.json',hashes)
    write(ROOT/'prepared.json',{'preregistration':file_record(ROOT/'preregistered.json'),
        'manifests':[file_record(ROOT/f'manifest_L{n}.json') for n in (32,256)],
        'frames':file_record(ROOT/'frame_hashes.json'),'frame_paths_disjoint':True,
        'archives':downloads,'prepared_before_any_new_predictions':True})
    print('PREPARED',ROOT,flush=True)


if __name__=='__main__':main()
