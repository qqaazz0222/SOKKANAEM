"""Fetch the official pinned portable Tectonic binary into a project-local folder."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

DEST = Path('work_dirs/paper_tools/tectonic-0.15.0')
URL = ('https://github.com/tectonic-typesetting/tectonic/releases/download/'
       'tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    binary, record = DEST/'tectonic', DEST/'source.json'
    if binary.exists():
        old = json.loads(record.read_text())
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == old['binary_sha256']
        print(binary)
        return
    request = urllib.request.Request(URL, headers={'User-Agent': 'SOKKANAEM-paper-build'})
    with urllib.request.urlopen(request, timeout=45) as stream:
        data = stream.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        member = archive.getmember('tectonic')
        assert member.isfile()
        content = archive.extractfile(member).read()
    binary.write_bytes(content)
    binary.chmod(0o755)
    record.write_text(json.dumps(dict(url=URL, version='0.15.0',
        archive_sha256=hashlib.sha256(data).hexdigest(), binary_sha256=hashlib.sha256(content).hexdigest(),
        provenance='official release download; locally computed hashes, not a detached signature'), indent=2)+'\n')
    print(binary)


if __name__ == '__main__':
    main()
