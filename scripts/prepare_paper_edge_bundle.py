"""Generate a private local edge bundle with frozen native EMA and development RGB only."""
import json
from pathlib import Path
import shutil
import sys
import zipfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.paper_closeout_study import prior_check, BASE


def main():
    prior_check()
    root = Path('work_dirs/paper_edge_bundle')
    root.mkdir(parents=True, exist_ok=True)
    files = ['scripts/paper_edge_input_bench.py', 'paper/submission/EDGE_RUN.md',
             *['sokkanaem/'+s+'.py' for s in ('__init__', 'model', 'ssm', 'detector', 'gmc', 'scan_triton')]]
    for name in files:
        dest = root/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    (root/'weights').mkdir(exist_ok=True)
    state = torch.load(BASE, map_location='cpu', weights_only=True)
    ema = state['ema']
    torch.save(ema, root/'weights/latest.pt')
    loaded = torch.load(root/'weights/latest.pt', map_location='cpu', weights_only=True)
    assert set(loaded) == set(ema) and all(torch.equal(loaded[k], ema[k]) for k in ema)
    shutil.copy2(BASE.with_name('config.toml'), root/'weights/config.toml')
    doc = json.loads(Path('manifests/acc_real_L256.json').read_text())
    seen, clips, source_rgb = set(), [], []
    for c in doc['clips']:
        sequence = Path(c['pairs'][0][0]).parent.parent.name
        if sequence in seen:
            continue
        index = len(seen)
        seen.add(sequence)
        paths = []
        for t, pair in enumerate(c['pairs']):
            name = 'rgb/s{}/{:06d}.png'.format(index, t)
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pair[0], target)
            paths.append(name)
            source_rgb.append(file_record(pair[0]))
        clips.append(dict(id='development:{}'.format(index), sequence=sequence, rgb=paths))
    records = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name != 'edge_manifest.json':
            rec = file_record(p)
            rec['path'] = str(p.relative_to(root))
            records.append(rec)
    manifest = dict(role='development_edge_timing_preparation', final_test_used=False,
        original_checkpoint=file_record(BASE), ema_export_tensor_exact=True,
        source_rgb=source_rgb, files=records, clips=clips,
        local_private_bundle=True, redistribution_permission='author must confirm before any public release')
    (root/'edge_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = Path('work_dirs/paper_edge_bundle.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                z.write(p, 'paper_edge_bundle/'+str(p.relative_to(root)))
    Path('work_dirs/paper_edge_bundle_archive.json').write_text(json.dumps(file_record(archive), indent=2)+'\n')
    print(archive, archive.stat().st_size, 'bytes; PRIVATE LOCAL ONLY; device execution pending')


if __name__ == '__main__':
    main()
