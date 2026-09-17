"""Pinned, size/SHA256-verified Wikimedia Chinese article acquisition."""
from pathlib import Path
import hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent / 'modern_corpus_v1' / 'wiki_sources'
ROOT.mkdir(parents=True, exist_ok=True)
BASE = 'https://huggingface.co'

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()

def main():
    meta = json.loads(get(BASE + '/api/datasets/wikimedia/wikipedia'))
    revision = meta['sha']
    files = json.loads(get(BASE + '/api/datasets/wikimedia/wikipedia/tree/' + revision + '/20231101.zh'))
    (ROOT / 'dataset_card.md').write_bytes(get(BASE + '/datasets/wikimedia/wikipedia/resolve/' + revision + '/README.md'))
    manifest = dict(dataset='wikimedia/wikipedia', revision=revision,
                    source='https://huggingface.co/datasets/wikimedia/wikipedia',
                    license=['CC-BY-SA-3.0','GFDL'], files=files)
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    def download(info):
        dest = ROOT / Path(info['path']).name
        sha = info['lfs']['oid']
        if dest.exists() and dest.stat().st_size == info['size'] and hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() == sha:
            print(json.dumps(dict(file=dest.name, status='already_verified')), flush=True)
            return
        url = BASE + '/datasets/wikimedia/wikipedia/resolve/' + revision + '/' + info['path']
        for attempt in range(4):
            try:
                digest = hashlib.sha256(); total = 0; start = time.perf_counter(); tick = start
                with urllib.request.urlopen(url + '?download=true&attempt=' + str(attempt), timeout=90) as r, dest.with_suffix('.partial').open('wb') as out:
                    while chunk := r.read(1024 * 1024):
                        out.write(chunk); digest.update(chunk); total += len(chunk)
                        if time.perf_counter() - tick > 20:
                            print(json.dumps(dict(file=dest.name, downloaded=total, expected=info['size'])), flush=True); tick=time.perf_counter()
                assert total == info['size'], (total, info['size'])
                assert digest.hexdigest() == sha
                dest.with_suffix('.partial').replace(dest)
                print(json.dumps(dict(file=dest.name, status='verified', seconds=time.perf_counter()-start, bytes=total)), flush=True)
                return
            except Exception as exc:
                print(json.dumps(dict(file=dest.name, attempt=attempt, error=repr(exc))), flush=True)
        raise RuntimeError('Download failed: ' + dest.name)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))

if __name__ == '__main__': main()
