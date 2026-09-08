import copy
import pytest
from scripts.paper_revision_report import aggregate_scale, FINAL_MODELS, FIELDS


def records():
    rows = []
    for s in range(4):
        for i in range(s+1):
            val = dict(scores={k: float(s+1) for k in FIELDS}, failed=False)
            rows.append(dict(id=f'{s}:{i}', sequence=str(s), no_gt=False,
                models={m: dict(space='depth', gauges={'none': copy.deepcopy(val)}) for m in FINAL_MODELS}))
    return rows


def test_equal_sequence_not_clip_weighted():
    clips, seq, summary = aggregate_scale(records(), 8)
    assert len(clips) == 80 and len(seq) == 32
    assert all(s['equal_sequence_mean'] == 2.5 for s in summary)


def test_nonfinite_not_silently_dropped():
    rows = records()
    rows[0]['models']['sparse_k30']['gauges']['none']['scores']['scale_step'] = float('nan')
    with pytest.raises(ValueError):
        aggregate_scale(rows, 8)


def test_all_sequences_required():
    with pytest.raises(ValueError):
        aggregate_scale([r for r in records() if r['sequence'] != '3'], 8)


def test_failure_retained():
    rows = records()
    rows[0]['models']['sparse_k30']['gauges']['none']['failed'] = True
    _, _, summary = aggregate_scale(rows, 8)
    assert all(s['failed'] == 1 for s in summary if s['model'] == 'sparse_k30')
