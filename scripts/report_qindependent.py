"""Generate an unfiltered report of the frozen new-sequence evaluation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write


def main():
    root=Path('work_dirs/qindependent_20260909');ev=root/'evaluation'
    result=json.loads((ev/'results.json').read_text());integrity=json.loads((ev/'integrity.json').read_text())
    protocol=json.loads((ev/'protocol.json').read_text())
    for rec in protocol['inputs']+[protocol['source']]:assert file_record(rec['path'])==rec
    lines=['# Frozen-candidate new-sequence validation — 2026-09-09','',
        'The MLP q50 weights and threshold were fixed before acquiring these sequences. No retraining, retuning, scene replacement or use of the old reserved final occurred.',
        '', '## Scope and pre-registration','',
        '- Newly acquired official TUM Freiburg1 room and Freiburg2 desk. These sequence identifiers were absent from searched local configs, manifests, paper, scripts and text experiment history before preparation.',
        '- This supports separation from the recorded local Q0 calibration and policy-selection pipeline, not certified disjointness from Depth Anything V2 pretraining or every possible historical external experiment.',
        '- Two new sequences are not a broad statistical generalization benchmark. Camera motion and acquisition environment differ from the reused development clips.',
        '- Before prediction: L32 first/last32 associated frames per sequence; L256 windows centered at one-third/two-thirds. RGB/depth paths are disjoint across selected windows. Same256px loader/scorer, Q0 internal518px, DIS128px.',
        '- Source: [official TUM download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Archive URLs and SHA256, per-frame hashes, manifests and pre-registration are preserved in the experiment directory.',
        '- Depth conversion uses PNG value/5000. Camera-specific scale corrections are already applied by the dataset; no extra1.035/1.031 factor is applied. [Official format and calibration documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).',
        '', '## Aggregate results','',
        '| Length | Policy | Raw AbsRel | Edge AbsRel | Boundary F1 | Flat-TV change vs Q0 | Screen | Mean ms | Skip |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|']
    failures={}
    for length,policies in result.items():
        base=policies['dense']['metrics']['balanced']
        for name,row in policies.items():
            m=row['metrics']['balanced'];flat=100*(m['flat_tv']/base['flat_tv']-1)
            bad=[k for k,v in row['quality_gate']['checks'].items() if not v]
            failures[length+'_'+name]=bad
            lines.append(f'| {length} | {name} | {m["absrel_raw"]:.6f} | {m["absrel_edge"]:.6f} | {m["boundary_f1"]:.6f} | {flat:+.3f}% | {"pass" if not bad else "FAIL"} | {row["mean_ms"]:.3f} | {100*row["skip_fraction"]:.2f}% |')
    lines += ['', 'Times are single-run synchronized shared-RTX4090 resident-input measurements, including internal policy/flow/transfers but excluding initial I/O/H2D. They are not repeated timing estimates or edge-device evidence.',
        '', '## Sequence-specific diagnostics','',
        '| Length | Sequence | Policy | Raw AbsRel | Boundary F1 | Flat-TV change | Failed checks |',
        '|---|---|---|---:|---:|---:|---|']
    for length,policies in result.items():
        for name in ('output_k2','mlp_q50'):
            for scene,row in policies[name]['sequence_gates'].items():
                b,m=row['baseline'],row['candidate'];bad=[k for k,v in row['gate']['checks'].items() if not v]
                lines.append(f'| {length} | {Path(scene).name} | {name} | {m["absrel_raw"]:.6f} | {m["boundary_f1"]:.6f} | {100*(m["flat_tv"]/b["flat_tv"]-1):+.3f}% | {", ".join(bad) if bad else "none"} |')
    passed=all(result[l]['mlp_q50']['quality_gate']['pass'] for l in result)
    seqpassed=all(v['gate']['pass'] for p in result.values() for v in p['mlp_q50']['sequence_gates'].values())
    lines += ['', '## Interpretation','',
        f'- Aggregate MLP screen on both lengths: **{"PASS" if passed else "FAIL"}**.',
        f'- All sequence-specific MLP diagnostic screens: **{"PASS" if seqpassed else "FAIL"}**.',
        '- A quality-screen failure is retained even when latency improves. No acceptance threshold is revised after this test.',
        '- Passing tolerances would not establish statistical equivalence, temporal stability or all-scene safety. Failure means the earlier development claim cannot be generalized to these new sequences without qualification.',
        f'- Exact Q0 refresh comparisons: {integrity["exact_refresh_comparisons"]}; all zero difference. Input/code/checkpoint hashes unchanged.',
        '- This holdout is now consumed. If its results inform future training or threshold selection, it becomes development data for that future model.',
        '', '## Reproduction','',
        '`scripts/qindependent_prepare.py`, `scripts/qindependent_eval.py`, `scripts/report_qindependent.py`.',
        '`work_dirs/qindependent_20260909/preregistered.json`, `prepared.json`, `evaluation/results.json`, `evaluation/integrity.json`.',
        'No native-model final or original submission source was changed.']
    path=Path('paper/Q0_INDEPENDENT_VALIDATION_20260909.md')
    if path.exists():raise FileExistsError('New report only')
    path.write_text('\n'.join(lines)+'\n')
    write(ev/'report_provenance.json',{'source':file_record(__file__),'results':file_record(ev/'results.json'),
        'report':file_record(path),'aggregate_candidate_pass':passed,'every_sequence_candidate_pass':seqpassed,'failures':failures})
    print(path,'aggregate pass',passed,'every sequence pass',seqpassed,flush=True)


if __name__=='__main__':main()
