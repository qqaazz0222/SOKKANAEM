"""Recover slow acquisition with HTTP ranges; preserve registered selection."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.qindependent_prepare import ROOT,SCENES
from scripts.quality_refinement import write
from sokkanaem.protocol import file_record
from sokkanaem.data import _pair_by_timestamp


def main():
    pre=json.loads((ROOT/'preregistered.json').read_text());assert file_record(pre['source']['path'])==pre['source']
    scene=SCENES[1];url=f'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg2/{scene}.tgz'
    prefix=ROOT/'archives'/(scene+'.tgz');start=prefix.stat().st_size
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:total=int(response.headers['Content-Length'])
    assert 0<start<total
    chunks=ROOT/'range_recovery';chunks.mkdir()
    metadata={'source':file_record(__file__),'preregistration':file_record(ROOT/'preregistered.json'),
        'reason':'Original acquisition connection became slow; terminated own downloader. Preserve prefix and resume missing bytes concurrently without changing data or sampling rules.',
        'prefix':file_record(prefix),'start':start,'total':total,'url':url}
    write(chunks/'protocol.json',metadata)
    def get(i):
        a=start+(total-start)*i//8;b=start+(total-start)*(i+1)//8-1
        path=chunks/f'{i}.part';request=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
        with urllib.request.urlopen(request,timeout=120) as inp:
            assert inp.status==206 and inp.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
            with path.open('xb') as out:shutil.copyfileobj(inp,out)
        assert path.stat().st_size==b-a+1
        print('range complete',i,path.stat().st_size,flush=True);return path
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(get,range(8)))
    complete=ROOT/'archives'/(scene+'.complete.tgz')
    with complete.open('xb') as out:
        for p in [prefix]+parts:
            with p.open('rb') as inp:shutil.copyfileobj(inp,out)
    assert complete.stat().st_size==total
    count=0
    with tarfile.open(complete,'r:gz') as tar:
        for member in tar:
            bits=Path(member.name).parts
            if not member.isfile() or len(bits)!=3 or bits[0]!=scene or bits[1] not in ('rgb','depth') or '..' in bits:continue
            target=ROOT/'data'/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as inp,target.open('xb') as out:shutil.copyfileobj(inp,out)
            count+=1
    downloads=[]
    for scene in SCENES:
        archive=complete if scene==SCENES[1] else ROOT/'archives'/(scene+'.tgz')
        downloads.append({'scene':scene,'url':f'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/{scene.split("_")[2]}/{scene}.tgz',
            'archive':file_record(archive)})
    write(ROOT/'downloads.json',downloads)
    clips={32:[],256:[]};seen=set();hashes=[]
    for scene in SCENES:
        d=ROOT/'data'/scene;pairs=_pair_by_timestamp(str(d/'rgb'),str(d/'depth'));assert len(pairs)>=1000
        for length,starts in ((32,[0,len(pairs)-32]),(256,[len(pairs)//3-128,2*len(pairs)//3-128])):
            for offset in starts:
                selected=pairs[offset:offset+length];paths={str(Path(p).resolve()) for pair in selected for p in pair}
                assert not paths&seen;seen.update(paths)
                clips[length].append({'source':'tum','scene':scene,'start_index':offset,'pairs':selected})
                hashes.extend(file_record(p) for p in sorted(paths))
    for length,rows in clips.items():
        write(ROOT/f'manifest_L{length}.json',{'role':'new_local_holdout','clip_len':length,'size':256,'sources':{'tum':{'scale':5000.,'mode':'u16'}},'clips':rows})
    write(ROOT/'frame_hashes.json',hashes)
    write(ROOT/'prepared.json',{'preregistration':file_record(ROOT/'preregistered.json'),
        'manifests':[file_record(ROOT/f'manifest_L{n}.json') for n in (32,256)],'frames':file_record(ROOT/'frame_hashes.json'),
        'frame_paths_disjoint':True,'archives':downloads,'recovery':file_record(chunks/'protocol.json'),'prepared_before_any_new_predictions':True})
    print('PREPARED',ROOT,flush=True)


if __name__=='__main__':main()
