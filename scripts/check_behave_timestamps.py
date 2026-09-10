"""Audit a licensed local BEHAVE time.json without selecting or scoring a model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.behave_io import associate_timestamps
from sokkanaem.protocol import file_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('timestamps', type=Path)
    ap.add_argument('--tolerance-us', type=float, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    data = json.loads(args.timestamps.read_text())
    result = associate_timestamps(data['color'], data['depth'], args.tolerance_us)
    result.update(input=file_record(args.timestamps), source=file_record(__file__),
                  helper=file_record('sokkanaem/behave_io.py'), no_model_inference=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print({k: v for k, v in result.items() if k != 'frames'}, flush=True)


if __name__ == '__main__': main()
