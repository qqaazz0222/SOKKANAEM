import pytest
from scripts.audit_streaming_draft import reference_status, check_markdown_links, snapshot_manifest


def test_reference_failure_is_not_reported_as_pass():
    status = reference_status({'reference_frames': 160, 'archive_crc_verified': True,
                              'C1_pass': False, 'C3_pass': True, 'C4_pass': True},
                             {'c4_would_pass_unmasked': False},
                             {'registration_corroborated': False, 'admitted_as_masked_gt': False})
    assert status['complete_reference_acquired']
    assert not status['complete_reference_value_checks_passed']
    assert status['unit_error_excluded'] is None
    assert not status['gt_admitted']


def test_bundle_links_cannot_escape(tmp_path):
    doc = tmp_path/'draft.md'; doc.write_text('[bad](../outside.md)')
    with pytest.raises(AssertionError, match='outside bundle'):
        check_markdown_links([doc], tmp_path)


def test_bundle_links_require_existing_targets(tmp_path):
    doc = tmp_path/'draft.md'; doc.write_text('[missing](not_here.json)')
    with pytest.raises(AssertionError):
        check_markdown_links([doc], tmp_path)
    (tmp_path/'not_here.json').write_text('{}')
    assert check_markdown_links([doc], tmp_path) == 1


def test_snapshot_records_content_changes(tmp_path):
    doc = tmp_path/'draft.md'; doc.write_text('before')
    before = snapshot_manifest([tmp_path])
    doc.write_text('after')
    assert before != snapshot_manifest([tmp_path])
