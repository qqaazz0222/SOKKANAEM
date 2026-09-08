import json
from pathlib import Path
import sys

import pytest

from scripts import paper_closeout_study as closeout


def test_final_requires_explicit_flag_before_any_inference(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['paper_closeout_study.py', '--phase', 'final'])
    with pytest.raises(SystemExit) as exc:
        closeout.main()
    assert exc.value.code == 2


def test_final_flag_cannot_be_used_for_development(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['paper_closeout_study.py', '--phase', 'smoke', '--final-test'])
    with pytest.raises(SystemExit) as exc:
        closeout.main()
    assert exc.value.code == 2


def test_completion_rejects_partial_model_set_and_duplicates(tmp_path):
    path = tmp_path/'rows.jsonl'
    row = dict(id='a', no_gt=False, models={n: {} for n in closeout.FINAL_MODELS})
    path.write_text(json.dumps(row)+'\n')
    assert closeout.load_done(path, ['a']) == {'a'}
    path.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError):
        closeout.load_done(path, ['a'])
    row['models'].pop('sparse_k30')
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):
        closeout.load_done(path, ['a'])


def test_final_input_hash_verification(tmp_path):
    path = tmp_path/'data'
    path.write_text('frozen')
    record = closeout.file_record(path)
    closeout.verify_files([record])
    path.write_text('changed')
    with pytest.raises(ValueError):
        closeout.verify_files([record])
