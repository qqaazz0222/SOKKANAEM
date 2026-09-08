"""Lightweight provenance tests; no inference or raw-tensor loading."""
import json
from pathlib import Path


def test_qualitative_replay_matches_original_predictions():
    audit = json.loads(Path('work_dirs/paper_qualitative_20260908/inference.json').read_text())
    original = {r['id']: r for r in map(json.loads,
        Path('work_dirs/paper_closeout/final_L256.jsonl').read_text().splitlines())}
    assert [r['id'] for r in audit['clips']] == ['tum:0', 'tum:4', 'tum:7', 'tum:10']
    assert audit['post_hoc'] and not audit['training_performed'] and not audit['frozen_results_modified']
    for row in audit['clips']:
        assert set(row['models']) == {'sparse_k30', 'dense_carry', 'midas_256', 'output_hold'}
        for name, model in row['models'].items():
            assert model['matches_frozen']
            assert model['prediction_sha256'] == original[row['id']]['models'][name]['prediction_sha256']
            assert model['fit']['mode'] == 'scaleshift'
            assert len(model['frame_absrel']) == len(model['display_clipped_fraction']) == 256


def test_qualitative_display_discloses_fixed_gauge_and_active_window():
    data = json.loads(Path('paper/submission/qualitative/provenance.json').read_text())
    assert data['depth_range_m'] == [0, 5] and data['per_frame_alignment'] is False
    assert data['playback_fps'] == 10 and data['post_hoc']
    row = next(r for r in data['clips'] if r['id'] == 'tum:7')
    assert all(row['models']['sparse_k30']['telemetry'][t]['active'] == 1 for t in range(28, 33))
    tex = Path('paper/submission/manuscript.tex').read_text()
    assert 'does not isolate a sparse-reuse benefit' in tex
