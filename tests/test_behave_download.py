import io
import tarfile
import pytest
import scripts.download_behave_pilot as download


def test_tar_index_offsets_match_payloads(monkeypatch):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as tar:
        for name, payload in [('skip.bin', b'x'*1027), ('seq.1.color.mp4', b'correct payload')]:
            m = tarfile.TarInfo(name); m.size = len(payload)
            tar.addfile(m, io.BytesIO(payload))
    blob = out.getvalue()
    monkeypatch.setattr(download, 'request_range', lambda meta, start, end: io.BytesIO(blob[start:end+1]))
    r = download.index_selected({'bytes': len(blob)}, ['seq.1.color.mp4'])
    m = r['selected']['seq.1.color.mp4']
    assert blob[m['offset']:m['offset']+m['bytes']] == b'correct payload'
    assert not r['whole_archive_acquired']


def test_range_download_completeness_and_no_overwrite(tmp_path, monkeypatch):
    blob = b'prefixpayloadsuffix'
    monkeypatch.setattr(download, 'request_range', lambda meta, start, end: io.BytesIO(blob[start:end+1]))
    path = tmp_path/'member.mp4'
    r = download.download_range({}, 6, 7, path)
    assert path.read_bytes() == b'payload' and r['bytes'] == 7
    with pytest.raises(FileExistsError): download.download_range({}, 6, 7, path)


def test_truncated_member_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(download, 'request_range', lambda *args: io.BytesIO(b''))
    path = tmp_path/'member.mp4'
    with pytest.raises(IOError): download.download_range({}, 0, 7, path)
    assert not path.exists() and path.with_suffix('.mp4.part').exists()


def test_ignored_range_is_rejected(monkeypatch):
    class Reply(io.BytesIO):
        status = 200
        headers = {'Content-Range': 'bytes 0-3/4', 'ETag': 'v1'}
    monkeypatch.setattr(download, 'urlopen', lambda *args, **kwargs: Reply(b'abcd'))
    with pytest.raises(ValueError, match='exact immutable'):
        download.request_range({'url': 'https://example.org/a', 'etag': 'v1', 'bytes': 4}, 0, 3)
