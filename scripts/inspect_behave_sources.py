"""Public docs/code provenance plus archive HEAD responses; no dataset GET."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record

PAGE = 'https://virtualhumans.mpi-inf.mpg.de/behave/license.html'


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            url = dict(attrs).get('href', '')
            self.links.append(urljoin(PAGE, url))


def fetch_public(url):
    with urlopen(Request(url, headers={'User-Agent': 'SOKKANAEM research source audit'}), timeout=25) as response:
        data = response.read(2_000_001)
        if len(data) > 2_000_000: raise ValueError('Public-document size cap')
        return data


def head(url):
    try:
        with urlopen(Request(url, method='HEAD'), timeout=25) as r:
            return {'url': url, 'method': 'HEAD', 'status': r.status,
                    'content_length': int(r.headers.get('Content-Length', 0)),
                    'accept_ranges': r.headers.get('Accept-Ranges'),
                    'etag': r.headers.get('ETag'), 'last_modified': r.headers.get('Last-Modified'),
                    'body_bytes_downloaded': 0}
    except Exception as e:
        return {'url': url, 'method': 'HEAD', 'error': str(e), 'body_bytes_downloaded': 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('work_dirs/qbehave_readiness_20260910'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    protocol = {'source': file_record(__file__), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'public documentation and HTTP metadata only',
                'dataset_get_allowed': False, 'no_model_inference': True,
                'candidate': file_record('paper/streaming_draft/frozen_candidate.json')}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    page = fetch_public(PAGE)
    (args.out/'license_page.html').write_bytes(page)
    parser = Links(); parser.feed(page.decode())
    urls = sorted({u for u in parser.links if urlparse(u).hostname == 'datasets.d2.mpi-inf.mpg.de'
                   and ('date02' in u.lower() or u.endswith('/calibs.zip'))})
    with ThreadPoolExecutor(max_workers=3) as pool:
        headers = list(pool.map(head, urls))
    sources = []
    for repo, branch, files in [
        ('xiexh20/behave-dataset', 'main', ['README.md', 'LICENSE', 'data/video_reader.py', 'data/kinect_calib.py']),
        ('vguzov/videoio', 'master', ['LICENSE', 'videoio/video_uint16.py'])]:
        commit = json.loads(fetch_public(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
        for name in files:
            url = f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
            p = args.out/'sources'/repo/name
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(fetch_public(url))
            sources.append({'repository': repo, 'commit': commit, 'url': url, 'file': file_record(p)})
    result = {'protocol': file_record(args.out/'protocol.json'), 'page': file_record(args.out/'license_page.html'),
              'headers': headers, 'sources': sources, 'dataset_bytes_downloaded': 0,
              'data_acquired': False, 'gt_admitted': False, 'license_confirmation': 'pending explicit non-commercial research confirmation'}
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'headers': headers, 'source_files': len(sources), 'data_acquired': False}), flush=True)


if __name__ == '__main__': main()
