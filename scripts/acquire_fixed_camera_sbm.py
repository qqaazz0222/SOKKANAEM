"""Acquire the documented SBM URFD ceiling-camera derivative; no inference."""
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile
from sokkanaem.protocol import file_record

ROOT = Path('work_dirs/qfixed_validation_20260910')
URL = 'https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/Shadows/Shadows.zip'
CORRECTION = Path('work_dirs/qfixed_public_probe_20260910/sbm_fall01cam1_update.zip')


def main():
    ROOT.mkdir(exist_ok=False)
    pre = {'source': file_record(__file__), 'url': URL, 'expected_archive_bytes': 630619899,
           'correction': file_record(CORRECTION), 'sequence': 'fall01cam1',
           'role': 'data admission before any model predictions', 'candidate': file_record('paper/streaming_draft/frozen_candidate.json'),
           'selection_reason': 'Previously selected first fall event, documented ceiling camera; not model quality',
           'independent_locations': 'not established; a single event is not broad validation',
           'no_model_training_or_predictions': True}
    (ROOT/'acquisition_protocol.json').write_text(json.dumps(pre, indent=2)+'\n')
    archive = ROOT/'Shadows.zip'
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                    '--connect-timeout', '15', '--max-time', '300', '--output', str(archive), URL], check=True)
    assert archive.stat().st_size == pre['expected_archive_bytes']
    records = []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        (ROOT/'archive_members.json').write_text(json.dumps(names, indent=2)+'\n')
        selected = [n for n in names if 'fall01cam1' in PurePosixPath(n).parts and not n.endswith('/')]
        assert selected, sorted(set(str(PurePosixPath(n).parent) for n in names))
        for name in selected:
            parts = PurePosixPath(name).parts
            rel = PurePosixPath(*parts[parts.index('fall01cam1')+1:])
            assert '..' not in rel.parts and not rel.is_absolute()
            dst = ROOT/'original_sbm'/Path(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open('xb') as f:
                f.write(z.read(name))  # zipfile verifies each selected member's CRC
            records.append(file_record(dst))
    with zipfile.ZipFile(CORRECTION) as z:
        for name in z.namelist():
            if name.endswith('/'): continue
            rel = PurePosixPath(name)
            assert '..' not in rel.parts and not rel.is_absolute()
            dst = ROOT/'corrected_sbm'/Path(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open('xb') as f: f.write(z.read(name))
            records.append(file_record(dst))
    result = {'archive': file_record(archive), 'selected_member_crc_verified': True,
              'frames': records, 'no_predictions': True, 'source': file_record(__file__)}
    (ROOT/'acquired.json').write_text(json.dumps(result, indent=2)+'\n')
    print('ACQUIRED', len(records), 'files', flush=True)


if __name__ == '__main__': main()
