"""Resumable download of domestic Chinese text, for a larger character-network run.

Why this exists: huggingface.co small requests work through the local proxy, but
every large file redirects to the Xet CDN (us.aws.cdn.hf.co, cas-bridge.xethub.hf.co)
which transfers at 0.01 MB/s or resets. ModelScope serves comparable non-synthetic
Chinese text at about 0.6 MB/s, measured 2026-09-13.

Nothing here touches modern_corpus_v1, which stays frozen. Output goes to
raw_modelscope/ and a manifest records bytes and sha256 for each finished file.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'raw_modelscope'
TABLE = 'https://www.modelscope.cn/api/v1/datasets/{ns}/{name}/repo/tree?Revision={rev}'
BLOB = 'https://www.modelscope.cn/api/v1/datasets/{ns}/{name}/repo?Revision={rev}&FilePath={path}'
PROXY = 'http://127.0.0.1:10808'
MIN_BYTES = 100_000_000

SOURCES = [
    ('caoaolong', 'zhwiki', 'zhwiki_dataset.jsonl'),
    ('Illusionna', 'wiki-zh-521MB', 'local_train.txt'),
    ('AI-ModelScope', 'wikipedia-cn-20230720-filtered', 'wikipedia-cn-20230720-filtered.jsonl'),
]


def opener():
    # ModelScope is domestic: a direct connection measured 10.5 MB/s on 2026-09-14,
    # while routing it through the local proxy measured 0.6 MB/s and held a long-lived
    # tunnel open. Direct is both faster and does not touch the proxy client.
    build = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    build.addheaders = [('User-Agent', 'python-urllib/3.13')]
    return build


def tree_size(build, ns, name, path, rev='master'):
    with build.open(TABLE.format(ns=ns, name=name, rev=rev), timeout=30) as r:
        data = json.loads(r.read().decode('utf-8'))
    for entry in walk_files(data.get('Data', {}).get('Files', [])):
        if entry.get('Path') == path:
            return int(entry['Size'])
    raise KeyError(f'{ns}/{name}:{path} not listed')


def walk_files(entries):
    for entry in entries:
        if entry.get('Type') == 'tree':
            continue
        yield entry


def digest_of(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(build, ns, name, path, size, rev='master'):
    dest = OUT / f'{ns}__{name}__{Path(path).name}'
    part = dest.with_suffix(dest.suffix + '.part')
    if dest.exists() and dest.stat().st_size == size:
        print(json.dumps(dict(file=dest.name, status='already_present')), flush=True)
        return dest, digest_of(dest)
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    url = BLOB.format(ns=ns, name=name, rev=rev, path=urllib.parse.quote(path))
    request = urllib.request.Request(url)
    if have:
        request.add_header('Range', f'bytes={have}-')
    started = time.perf_counter()
    with build.open(request, timeout=60) as r, part.open('ab') as out:
        if have and r.status != 206:
            print(json.dumps(dict(file=dest.name, status='range_unsupported')), flush=True)
            return None, None
        done = have
        tick = started
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if time.perf_counter() - tick > 30:
                rate = (done - have) / max(time.perf_counter() - started, 1e-9)
                print(json.dumps(dict(file=dest.name, bytes=done, of=size,
                                      MBps=round(rate / 1e6, 2))), flush=True)
                tick = time.perf_counter()
    got = part.stat().st_size
    if got != size:
        print(json.dumps(dict(file=dest.name, status='incomplete', bytes=got, expected=size)), flush=True)
        return None, None
    part.replace(dest)
    seconds = time.perf_counter() - started
    digest = digest_of(dest)
    print(json.dumps(dict(file=dest.name, status='done', bytes=size, seconds=round(seconds, 1),
                          mbps=round(size / 1e6 / max(seconds, 1e-9), 2), sha256=digest)), flush=True)
    return dest, digest


def main():
    OUT.mkdir(exist_ok=True)
    build = opener()
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8')) if manifest_path.exists() else dict(files=[])
    known = {entry['file']: entry for entry in manifest['files']}
    for ns, name, path in SOURCES:
        try:
            size = tree_size(build, ns, name, path)
        except Exception as exc:
            print(json.dumps(dict(source=f'{ns}/{name}', error=repr(exc)[:160])), flush=True)
            continue
        if size < MIN_BYTES:
            print(json.dumps(dict(source=f'{ns}/{name}', skipped='smaller than threshold', bytes=size)), flush=True)
            continue
        try:
            dest, digest = download(build, ns, name, path, size)
        except Exception as exc:
            print(json.dumps(dict(source=f'{ns}/{name}', error=repr(exc)[:160])), flush=True)
            continue
        if dest is None:
            continue
        known[dest.name] = dict(file=dest.name, source=f'{ns}/{name}', path=path,
                                bytes=size, sha256=digest)
        manifest['files'] = list(known.values())
        manifest['proxy'] = 'direct (no proxy)'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()