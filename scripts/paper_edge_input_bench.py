"""Real-PNG streaming latency preparation for user-owned edge devices.

Python 3.6 syntax; actual ARM/JetPack execution remains to be verified.
No install, clock changes, network access, power inference or dataset download.
"""
from __future__ import print_function
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_optional(path):
    try:
        return Path(path).read_text().replace('\x00', '').strip()
    except (OSError, UnicodeError):
        return None


def inventory():
    return dict(platform=platform.platform(), python=sys.version,
        board=read_optional('/proc/device-tree/model'),
        jetson_release=read_optional('/etc/nv_tegra_release'),
        meminfo=read_optional('/proc/meminfo'),
        thermal={str(p): read_optional(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')},
        cpu_governors={str(p): read_optional(p) for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor')})


def verify_bundle(root, doc):
    for record in doc['files']:
        target = (root/record['path']).resolve()
        if root.resolve() not in target.parents:
            raise ValueError('bundle path escapes root')
        if sha(target) != record['sha256']:
            raise ValueError('bundle hash mismatch: '+record['path'])


def crop(image):
    w, h = image.size
    scale = 256/min(w, h)
    rw, rh = max(256, round(w*scale)), max(256, round(h*scale))
    left, top = (rw-256)//2, (rh-256)//2
    from PIL import Image
    resampling = getattr(Image, 'Resampling', Image)
    return image.resize((rw, rh), resampling.BILINEAR).crop((left, top, left+256, top+256))


class AllActive(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, frame, state=None):
        b, _, h, w = frame.shape
        return frame.new_ones(b, (h//self.p)*(w//self.p)), None

    def is_keyframe(self, state):
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', default='.')
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--reference-scan', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--output', default='edge_results.json')
    args = ap.parse_args()
    env = inventory()
    if args.inspect:
        print(json.dumps(env, indent=2))
        return
    if args.threads < 1:
        ap.error('threads must be positive')
    dest = Path(args.output)
    if dest.exists():
        ap.error('output already exists; choose a new path to preserve prior measurements')
    root = Path(args.bundle).resolve()
    doc = json.loads((root/'edge_manifest.json').read_text())
    verify_bundle(root, doc)
    import numpy as np
    import torch
    from PIL import Image, __version__ as pillow_version
    sys.path.insert(0, str(root))
    from sokkanaem import from_checkpoint, scan_triton
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA unavailable in the installed PyTorch')
    if args.reference_scan:
        scan_triton.HAVE_TRITON = False
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    env.update(torch=torch.__version__, numpy=np.__version__, pillow=pillow_version,
        cuda=torch.version.cuda, device=args.device, threads=args.threads,
        gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
        scan='reference_chunked' if args.reference_scan or args.device == 'cpu' or not scan_triton.HAVE_TRITON else 'triton',
        precision='FP32 eager, TF32 off')
    if args.device == 'cuda' and torch.cuda.get_device_capability()[0] < 7 and not args.reference_scan:
        ap.error('older GPU: use --reference-scan; do not label this as the desktop fused implementation')
    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()
    rows = []
    clips = doc['clips'][:1] if args.smoke else doc['clips']
    expected_predictions = {}
    with torch.no_grad():
        for clip in clips:
            paths = clip['rgb'][:8] if args.smoke else clip['rgb']
            for repeat in (range(1) if args.smoke else range(-1, 5)):
                order = ['sparse_k30', 'dense_carry'] if repeat % 2 == 0 else ['dense_carry', 'sparse_k30']
                for name in order:
                    kwargs = dict(gmc=False, keyframe_every=30, tau_on=.05, tau_off=.025)
                    if name == 'dense_carry':
                        kwargs.update(spatial_cache=False, temporal_cache=False)
                    model = from_checkpoint(root/'weights/latest.pt', args.device, **kwargs).eval()
                    if name == 'dense_carry':
                        model.detector = AllActive(model.p)
                    state, times, telemetry = None, [], []
                    digest = hashlib.sha256()
                    if args.device == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    before = inventory()
                    for path in paths:
                        sync()
                        start = time.perf_counter()
                        with Image.open(str(root/path)) as im:
                            view = crop(im.convert('RGB'))
                            array = np.asarray(view).copy()
                        x = torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0).to(args.device)
                        keyframe = model.detector.is_keyframe(state['det'] if state else None)
                        prediction, state, info = model.step(x, state)
                        output = prediction.cpu().contiguous()
                        sync()
                        times.append((time.perf_counter()-start)*1000)
                        if not bool(torch.isfinite(output).all()):
                            raise ValueError('nonfinite output')
                        digest.update(output.numpy().tobytes())
                        active = float(info['mask'].mean())
                        raw = float(state['det']['prev_mask'].mean()) if state['det'] else 1.
                        telemetry.append(dict(keyframe=keyframe, active=active,
                                              dense_fallback=active == 1. and raw < 1.))
                    prediction_sha = digest.hexdigest()
                    identity = (clip['id'], name)
                    if identity in expected_predictions and expected_predictions[identity] != prediction_sha:
                        raise ValueError('prediction changed across identical repeats')
                    expected_predictions[identity] = prediction_sha
                    if repeat >= 0:
                        rows.append(dict(clip=clip['id'], sequence=clip['sequence'], model=name,
                            repeat=repeat, frame_ms=times, mean_ms=float(np.mean(times)),
                            p50_ms=float(np.quantile(times, .5)), p95_ms=float(np.quantile(times, .95)),
                            prediction_sha256=prediction_sha, telemetry=telemetry,
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                            before=before, after=inventory()))
                    print(clip['id'], name, repeat, 'mean ms', float(np.mean(times)), flush=True)
                    del model, state, prediction, output, x
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
    verify_bundle(root, doc)
    result = dict(environment=env, manifest_sha256=sha(root/'edge_manifest.json'), smoke=args.smoke,
        boundary='warm-cache PNG file read/decode/crop/normalize/transfer/model/CPU output; excludes load/hash/telemetry',
        warmup=0 if args.smoke else 1, repeats=1 if args.smoke else 5, results=rows,
        energy_measured=False, energy_j_per_frame=None, accuracy_evaluated=False,
        caveat='No GT here. Compare quality separately on this runtime before any accuracy-preserving speed claim.')
    with dest.open('x') as f:
        json.dump(result, f, indent=2)
    print('Saved', dest)


if __name__ == '__main__':
    main()
