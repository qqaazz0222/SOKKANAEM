"""Read-only evidence checks, then optional new-directory delivery copy."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write


def reference_status(urfd, robust, reg2):
    return {'complete_reference_acquired': urfd['reference_frames'] == 160 and urfd['archive_crc_verified'],
            'complete_reference_value_checks_passed': bool(urfd['C1_pass'] and urfd['C3_pass'] and urfd['C4_pass'] and robust['c4_would_pass_unmasked']),
            'C1_pass': urfd['C1_pass'], 'C3_mask_retention_pass': urfd['C3_pass'],
            'C4_masked_pass': urfd['C4_pass'], 'C4_unmasked_pass': robust['c4_would_pass_unmasked'],
            'unit_error_excluded': None,
            'unit_statement': 'Bulk values compatible with millimetres; all scale/processing errors not excluded',
            'registration_corroborated': reg2['registration_corroborated'],
            'gt_admitted': reg2['admitted_as_masked_gt']}


def check_markdown_links(paths, boundary=None):
    checked = 0
    for path in paths:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if link.startswith(('http://', 'https://', '#', 'mailto:')): continue
            target = (path.parent/link.split('#')[0]).resolve()
            if boundary is not None:
                assert target.is_relative_to(boundary.resolve()), (path, link, 'outside bundle')
            assert target.exists(), (path, link)
            checked += 1
    return checked


def snapshot_manifest(paths):
    return {str(p): file_record(p)['sha256'] for root in paths
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deliver',type=Path)
    ap.add_argument('--resume',action='store_true',help='Only retain hash-identical existing files and copy missing files')
    ap.add_argument('--repository-layout', action='store_true', help='Bundle paper/ and referenced work_dirs/ metadata, preserving relative links')
    ap.add_argument('--report',type=Path,default=Path('paper/STREAMING_DRAFT_AUDIT_20260910.json'))
    args=ap.parse_args()
    root=Path('paper/streaming_draft');draft=Path('paper/draft.md');text=draft.read_text()
    frozen=json.loads((root/'frozen_candidate.json').read_text());visual=json.loads((root/'visual_provenance.json').read_text())
    records=frozen['weights']+frozen['code']+frozen['evidence']+visual['predictions']+[visual['data'],visual['renderer'],visual['layout_revision']]+visual['artifacts']
    for r in records:assert file_record(r['path'])==r
    links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
    local=[p for p in links if not p.startswith(('https://','http://','#'))]
    for p in local:assert (draft.parent/p).exists(),p
    evidence=json.loads((root/'evidence_table.json').read_text());table_checks=0
    for row in evidence:
        if row['policy'] not in ('baseline','output_k2','mlp_q50'):continue
        for key in ('raw_absrel','edge_absrel','boundary_f1'):
            formatted=f'{row[key]:.6f}'.removeprefix('0')
            assert formatted in text,(row['policy'],key,formatted);table_checks+=1
    timing=json.loads(Path('work_dirs/qpreflow_candidate_audit_20260909/audit.json').read_text())
    for l in ('L32','L256'):
        for k,v in timing[l]['means_ms'].items():assert f'{v:.3f}' in text
    independent_root=Path('work_dirs/qindependent_20260909/evaluation')
    independent=json.loads((independent_root/'results.json').read_text())
    provenance=json.loads((independent_root/'report_provenance.json').read_text())
    for key in ('source','results','report'):
        r=provenance[key];assert file_record(r['path'])==r
    independent_cells=0
    for policies in independent.values():
        for row in policies.values():
            for key in ('absrel_raw','boundary_f1'):
                assert f'{row["metrics"]["balanced"][key]:.6f}'.removeprefix('0') in text
                independent_cells+=1
    assert not independent['L32']['mlp_q50']['quality_gate']['pass']
    assert '+2.285%' in text and 'FAIL' in text
    pose_root=Path('work_dirs/qfixed_scope_20260910')
    pose_protocol=json.loads((pose_root/'protocol.json').read_text())
    pose=json.loads((pose_root/'audit.json').read_text())
    for r in [pose_protocol['scope'],pose_protocol['source']]+pose['pose_files']:
        assert file_record(r['path'])==r
    assert sum(c['label']=='camera_motion_present' for c in pose['clips'])==11
    assert sum(c['label']=='unclassified_incomplete_pose' for c in pose['clips'])==1
    assert not any(c['fixed_mount_verified'] for c in pose['clips'])
    assert (root/'CAMERA_POSE_AUDIT.md').read_bytes()==(pose_root/'report.md').read_bytes()
    assert 'No fixed-camera GT-accuracy result is reported yet.' in text
    fixed_root=Path('work_dirs/qfixed_validation_20260910')
    fixed_admission=json.loads((fixed_root/'admission.json').read_text())
    fixed_bench=json.loads((fixed_root/'rgb_only_benchmark/results.json').read_text())
    fixed_report=json.loads((fixed_root/'report_provenance.json').read_text())
    for r in fixed_report['inputs']+[fixed_report['source'],fixed_report['report']]+fixed_report['figures']:
        assert file_record(r['path'])==r
    assert not fixed_admission['admitted_for_single_event_pilot']
    assert not fixed_bench['gt_quality_gate_evaluated']
    for row in fixed_bench['results'].values():
        assert f'{row["mean_of_run_means_ms"]:.3f}' in text
    assert fixed_bench['exact_refresh_comparisons']==624
    assert fixed_bench['exact_repeated_frames']==1280
    urfd_root=Path('work_dirs/qfixed_urfd_20260910')
    urfd=json.loads((urfd_root/'urfd_admission.json').read_text())
    reg2=json.loads((urfd_root/'registration_audit2.json').read_text())
    for r in (urfd['protocol'],urfd['source'],reg2['protocol'],reg2['source'],reg2['prior_admission'],
              reg2['prior_registration_attempt']):
        assert file_record(r['path'])==r
    assert urfd['reference_frames']==160 and urfd['archive_crc_verified']
    assert not urfd['C1_pass'] and urfd['C3_pass'] and urfd['C4_pass']
    assert sum(f['C2_above_original_max'] for f in urfd['frames'])==0
    assert sum(f['C2_below_original_min'] for f in urfd['frames'])==564275
    assert reg2['attempt1_edge_test_failed'] and reg2['attempt2_mutual_information_failed']
    assert reg2['void'] and not reg2['registration_corroborated']
    assert not reg2['admitted_as_masked_gt']
    historical=(root/'FIXED_CAMERA_REFERENCE_ADMISSION.md').read_text()
    for value in (f'{urfd["C1_min_fraction"]:.4f}',f'{urfd["C1_max_fraction"]:.4f}'):
        assert value in historical,value
    for value in (urfd['C3_median_retention'],urfd['C3_min_retention']):
        assert f'{value:.4f}' in historical,value
    scene=json.loads((urfd_root/'scene_structure.json').read_text())
    assert file_record(scene['source']['path'])==scene['source']
    assert f'{scene["largest_median_8px_response_in_bins"]:.2f}' in historical
    assert f'{scene["max_foreground_fraction"]*100:.2f}%' in historical
    robust=json.loads((urfd_root/'scale_robustness.json').read_text())
    assert file_record(robust['source']['path'])==robust['source']
    assert not robust['c4_would_pass_unmasked']
    assert f'{robust["unmasked_median_worst_quantile_mm"]:.1f}mm' in historical
    for key in ('p50','p75','p95','p25','p05'):
        assert f'{robust["unmasked_max_abs_diff_mm"][key]:.1f}mm' in historical,key
    screen=json.loads((urfd_root/'sbm_relief_screen.json').read_text())
    assert file_record(screen['source']['path'])==screen['source']
    assert len(screen['sequences'])==5 and screen['no_admission_claim']
    assert max(r['median_relief_in_bins'] for r in screen['sequences'])<1.
    boot=json.loads((urfd_root/'bootstrapping_relief_screen.json').read_text())
    assert file_record(boot['source']['path'])==boot['source']==screen['source']
    assert len(boot['sequences'])==5
    assert max(r['median_relief_in_bins'] for r in screen['sequences']+boot['sequences'])<1.
    for r in screen['sequences']+boot['sequences']:
        assert f'{r["median_relief_in_bins"]:.2f}' in historical,r['sequence']
    rgbid=json.loads((urfd_root/'rgb_identity.json').read_text())
    assert file_record(rgbid['source']['path'])==rgbid['source']
    assert rgbid['all_pixel_exact'] and rgbid['pixel_exact_frames']==160
    assert 'all RGB frames are pixel-identical' in text
    assert 'researcher-defined' in text and 'does not exclude all scale or processing errors' in text
    assert 'smooth indoor Kinect depth cannot support' not in text
    instrument_root=Path('work_dirs/qregistration_instrument_20260910')
    instrument=json.loads((instrument_root/'results.json').read_text())
    ip=json.loads((instrument_root/'protocol.json').read_text())
    for r in [instrument['protocol'], ip['source'], ip['fixed_candidate']]+ip['inputs']:
        assert file_record(r['path']) == r
    assert instrument['recovered'] == instrument['total'] == 27
    assert instrument['flat_control_ambiguous'] and not instrument['gt_admitted']
    assert '27 injected shifts' in text
    manuscript_links=check_markdown_links([draft]+list(root.glob('*.md')))
    tex=(root/'manuscript.tex').read_text();log=(root/'manuscript.log').read_text()
    assert tex.count('\\begin{longtable}')==4
    tables=re.findall(r'\\begin\{longtable\}(.*?)\\end\{longtable\}',tex,re.S)
    assert all('\\linewidth -' in t for t in tables)
    for bad in ('Overfull \\hbox','Missing character:','Undefined control sequence'):assert bad not in log,bad
    result={'scope':'draft consistency, not independent scientific approval','verified_record_count':len(records),
        'checked_quality_cells':table_checks,'independent_quality_cells':independent_cells,
        'moving_camera_failure_preserved':True,'pose_audit_verified':True,
        'fixed_camera_validation_complete':False,
        'fixed_camera_gt_admission_failed':True,'rgb_only_pilot_checked':True,
        'reference_admission':reference_status(urfd, robust, reg2),
        'instrument_recovered_shifts':instrument['recovered'],
        'instrument_is_cross_modal_approval':False,
        'registration_attempts_failed':3,'registration_corroborated':False,
        'local_markdown_links':len(local),'all_current_markdown_links':manuscript_links,'textwidth_tables':4,
        'pdf_compiled':True,'overfull_or_missing_character_errors':False,
        'draft':file_record(draft),'tex':file_record(root/'manuscript.tex'),'pdf':file_record(root/'manuscript.pdf')}
    if args.deliver:
        dest=args.deliver
        if dest.exists() and not args.resume:raise FileExistsError('New delivery folder or explicit hash-checked resume only')
        prior=[dest.parent/n for n in ('streaming_refresh_20260909','fixed_camera_scope_20260910',
               'fixed_camera_validation_20260910','fixed_camera_reference_20260910',
               'fixed_camera_review_20260910') if (dest.parent/n).is_dir()]
        before=snapshot_manifest(prior)
        dest.mkdir(parents=True,exist_ok=args.resume)
        sources=[p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.aux','.log','.out')]
        sources += [draft,Path('paper/draft_native_20260909.md')]
        if args.repository_layout:
            # Evidence metadata only: no dataset pixels, videos or model weights.
            sources += sorted(urfd_root.glob('*.json')) + sorted(instrument_root.glob('*.json'))
            readiness=Path('work_dirs/qbehave_readiness_20260910')
            if readiness.exists():
                rp=json.loads((readiness/'results.json').read_text())
                for r in [rp['protocol'], rp['page']]+[s['file'] for s in rp['sources']]:
                    assert file_record(r['path']) == r
                assert rp['dataset_bytes_downloaded'] == 0 and not rp['gt_admitted']
                sources += [readiness/'protocol.json', readiness/'results.json']
        mapping={}
        for src in sources:
            target=dest/(src if args.repository_layout else src.relative_to('paper'))
            mapping[str(target.relative_to(dest))]=src
            if target.exists():
                assert file_record(src)['sha256']==file_record(target)['sha256'],f'Refusing to overwrite changed destination: {target}'
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                # Mounted drive supports content writes but can reject copy2 metadata.
                shutil.copyfile(src,target)
        copied=[]
        for p in dest.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(dest);src=mapping[str(rel)]
            assert file_record(p)['sha256']==file_record(src)['sha256']
            copied.append(str(rel))
        after=snapshot_manifest(prior)
        assert before == after, 'Prior delivery snapshots changed during this delivery'
        bundle_links=None
        if args.repository_layout:
            bundle_links=check_markdown_links([dest/'paper/draft.md']+list((dest/'paper/streaming_draft').glob('*.md')), dest)
        result['delivery']={'path':str(dest.resolve()),'verified_copied_files':copied,
            'repository_layout':args.repository_layout,'verified_bundle_markdown_links':bundle_links,
            'prior_snapshot_paths':[str(p) for p in prior], 'prior_snapshot_file_count':len(before),
            'prior_snapshot_hashes_before':before,'prior_snapshot_hashes_after':after,
            'prior_snapshots_unchanged':before == after}
    write(args.report,result);print(result,flush=True)


if __name__=='__main__':main()
