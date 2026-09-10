"""Post-download integrity/decoding audit. No model inference or GT admission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from sokkanaem.behave_io import iter_depth_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('work_dirs/qbehave_pilot_acquisition_20260910'))
    args = ap.parse_args(); root = args.root
    target = root/'integrity_audit.json'
    if target.exists(): raise FileExistsError(target)
    acquired = json.loads((root/'downloaded.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    for r in acquired['videos']+[acquired['calibration'], acquired['protocol'], protocol['source'], protocol['candidate']]:
        assert file_record(r['path']) == r, r['path']
    audit_protocol = {'source': file_record(__file__), 'helper': file_record('sokkanaem/behave_io.py'),
                      'acquisition': file_record(root/'downloaded.json'),
                      'checks': 'Full-video ffprobe decoded frame counts vs timestamps; first three native depth frames',
                      'no_model_inference': True, 'gt_admission_permitted': False}
    (root/'integrity_protocol.json').write_text(json.dumps(audit_protocol, indent=2)+'\n')
    rows = []
    for record in acquired['videos']:
        p = Path(record['path']); is_depth = '.depth-reg.' in p.name
        prefix = p.name.removesuffix('.depth-reg.mp4' if is_depth else '.color.mp4')
        times = json.loads((root/'times'/(prefix+'.time.json')).read_text())
        key = 'depth' if is_depth else 'color'
        command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,pix_fmt,nb_read_frames,codec_name,r_frame_rate',
                   '-of', 'json', str(p)]
        probe = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
        if probe.stderr.strip(): raise ValueError(f'Video decoder errors: {probe.stderr}')
        stream = json.loads(probe.stdout)['streams'][0]
        count = int(stream['nb_read_frames'])
        row = {'file': record, 'stream': stream, 'timestamp_count': len(times[key]),
               'decoded_count_matches': count == len(times[key])}
        if is_depth:
            generator = iter_depth_video(p)
            samples = []
            try:
                for index in range(3):
                    a = next(generator); valid = a[a > 0]
                    samples.append({'index': index, 'dtype': str(a.dtype), 'shape': list(a.shape),
                                    'positive_fraction': float((a > 0).mean()),
                                    'positive_mm_min': int(valid.min()) if valid.size else None,
                                    'positive_mm_max': int(valid.max()) if valid.size else None})
            finally: generator.close()
            row['native_depth_samples'] = samples
        rows.append(row)
        print(p.name, count, row['decoded_count_matches'], flush=True)
    calibration = json.loads((root/'calibration_inventory.json').read_text())
    names = [r['name'] for r in calibration['members']]
    # Inventory only; no assumptions that Date02 equals an independent room.
    selected_calibration_names = [n for n in names if '/Date02/' in '/'+n or '/intrinsics/' in '/'+n]
    result = {'protocol': file_record(root/'integrity_protocol.json'), 'videos': rows,
              'all_decoded_counts_match': all(r['decoded_count_matches'] for r in rows),
              'calibration_zip_crc_verified': calibration['crc_verified'],
              'relevant_calibration_members': selected_calibration_names,
              'gt_admitted': False, 'physical_locations_verified': False,
              'cross_modal_registration_independently_verified': False}
    target.write_text(json.dumps(result, indent=2)+'\n')
    print('AUDIT', result['all_decoded_counts_match'], flush=True)


if __name__ == '__main__': main()
