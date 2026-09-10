"""Resume validated HTTP ranges in short requests; keep all partial data."""
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
    old=ROOT/'range_recovery';meta=json.loads((old/'protocol.json').read_text());start,total=meta['start'],meta['total']
    out=ROOT/'block_recovery';out.mkdir();jobs=[];pieces=[Path(meta['prefix']['path'])]
    assert file_record(pieces[0])==meta['prefix']
    for i in range(8):
        a=start+(total-start)*i//8;b=start+(total-start)*(i+1)//8
        part=old/f'{i}.part';pieces.append(part);a+=part.stat().st_size
        while a<b:
            end=min(a+4*1024*1024,b);path=out/f'{a}.part';jobs.append((a,end,path));pieces.append(path);a=end
    write(out/'protocol.json',{'source':file_record(__file__),'original_recovery':file_record(old/'protocol.json'),
        'reason':'Short cache-distinct range requests after slow long connections; unchanged official archive and selected sequences.',
        'existing_parts':[file_record(p) for p in pieces if p.exists()],
        'jobs':[{'start':a,'end_exclusive':b,'path':str(p)} for a,b,p in jobs]})
    def fetch(job):
        a,b,path=job;request=urllib.request.Request(meta['url']+f'?chunk={a}',headers={'Range':f'bytes={a}-{b-1}'})
        with urllib.request.urlopen(request,timeout=90) as inp:
            assert inp.status==206 and inp.headers['Content-Range'].startswith(f'bytes {a}-{b-1}/')
            with path.open('xb') as dest:shutil.copyfileobj(inp,dest)
        assert path.stat().st_size==b-a
        return path
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,p in enumerate(pool.map(fetch,jobs),1):
            if i%8==0:print('blocks',i,'/',len(jobs),flush=True)
    scene=SCENES[1];complete=ROOT/'archives'/(scene+'.complete.tgz')
    with complete.open('xb') as dest:
        for p in pieces:
            with p.open('rb') as inp:shutil.copyfileobj(inp,dest)
    assert complete.stat().st_size==total
    with tarfile.open(complete,'r:gz') as tar:
        for member in tar:
            bits=Path(member.name).parts
            if not member.isfile() or len(bits)!=3 or bits[0]!=scene or bits[1] not in ('rgb','depth') or '..' in bits:continue
            target=ROOT/'data'/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as inp,target.open('xb') as dest:shutil.copyfileobj(inp,dest)
    downloads=[]
    for scene in SCENES:
        archive=complete if scene==SCENES[1] else ROOT/'archives'/(scene+'.tgz')
        downloads.append({'scene':scene,'url':f'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/{scene.split("_")[2]}/{scene}.tgz','archive':file_record(archive)})
    write(ROOT/'downloads.json',downloads);clips={32:[],256:[]};seen=set();hashes=[]
    for scene in SCENES:
        d=ROOT/'data'/scene;pairs=_pair_by_timestamp(str(d/'rgb'),str(d/'depth'));assert len(pairs)>=1000
        for length,starts in ((32,[0,len(pairs)-32]),(256,[len(pairs)//3-128,2*len(pairs)//3-128])):
            for offset in starts:
                selected=pairs[offset:offset+length];paths={str(Path(p).resolve()) for pair in selected for p in pair}
                assert not paths&seen;seen.update(paths)
                clips[length].append({'source':'tum','scene':scene,'start_index':offset,'pairs':selected});hashes.extend(file_record(p) for p in sorted(paths))
    for length,rows in clips.items():
        write(ROOT/f'manifest_L{length}.json',{'role':'new_local_holdout','clip_len':length,'size':256,'sources':{'tum':{'scale':5000.,'mode':'u16'}},'clips':rows})
    write(ROOT/'frame_hashes.json',hashes)
    write(ROOT/'prepared.json',{'preregistration':file_record(ROOT/'preregistered.json'),
        'manifests':[file_record(ROOT/f'manifest_L{n}.json') for n in (32,256)],'frames':file_record(ROOT/'frame_hashes.json'),
        'frame_paths_disjoint':True,'archives':downloads,'recovery':file_record(out/'protocol.json'),'prepared_before_any_new_predictions':True})
    print('PREPARED',ROOT,flush=True)


if __name__=='__main__':main()
