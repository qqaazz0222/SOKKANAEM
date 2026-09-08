import copy
import pytest

from scripts.paper_closeout_report import validate_coverage, tex_table, FINAL_MODELS


def sample():
    pairs = [['/data/seq/rgb/1.png', '/data/seq/depth/1.png']]
    doc = dict(sources=['tum'], clip_len=1, clips=[dict(source='tum', pairs=pairs)])
    row = dict(id='tum:0', source='tum', sequence='seq', clip_len=1, pairs=pairs,
               no_gt=False, models={m: {} for m in FINAL_MODELS})
    return doc, row


def test_full_coverage_and_identity_required():
    doc, row = sample()
    validate_coverage([row], doc)
    with pytest.raises(ValueError):
        validate_coverage([row, row], doc)
    wrong = copy.deepcopy(row)
    wrong['pairs'][0][0] = '/data/seq/rgb/2.png'
    with pytest.raises(ValueError):
        validate_coverage([wrong], doc)
    wrong = copy.deepcopy(row)
    wrong['models'].pop('sparse_k30')
    with pytest.raises(ValueError):
        validate_coverage([wrong], doc)


def test_no_gt_requires_empty_model_results():
    doc, row = sample()
    row['no_gt'] = True
    with pytest.raises(ValueError):
        validate_coverage([row], doc)
    row['models'] = {}
    validate_coverage([row], doc)


def test_tex_rows_have_two_backslashes():
    result = tex_table('Caption', 'tab:x', 'A & B', [['a', 'b']], 'lr')
    assert 'a & b \\\\\n' in result
