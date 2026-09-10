import io
import json
import tarfile
import pytest
from scripts.acquire_behave_timestamps import inspect_archive


def archive(name='times/sequence.0.time.json', data=None, kind=None):
    b = io.BytesIO()
    raw = json.dumps(data if data is not None else {'color': [0, 33333], 'depth': [5, 33338]}).encode()
    with tarfile.open(fileobj=b, mode='w') as tar:
        m = tarfile.TarInfo(name); m.size = len(raw)
        if kind is not None: m.type = kind
        tar.addfile(m, io.BytesIO(raw))
    return b.getvalue()


def test_timestamp_archive_inventory():
    rows = inspect_archive(archive())
    assert len(rows) == 1 and rows[0]['ordered_timestamps']
    assert rows[0]['color']['interval_us']['p50'] == 33333


@pytest.mark.parametrize('name', ['../../escape.json', '/absolute.json'])
def test_archive_path_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        inspect_archive(archive(name))


def test_archive_symlink_rejected():
    with pytest.raises(ValueError, match='Unsafe'):
        inspect_archive(archive(kind=tarfile.SYMTYPE))


def test_duplicate_timestamps_are_reported_not_accepted():
    r = inspect_archive(archive(data={'color': [0, 0], 'depth': [0, 1]}))
    assert not r[0]['ordered_timestamps']
