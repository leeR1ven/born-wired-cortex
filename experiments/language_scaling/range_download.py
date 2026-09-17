"""Resumable bounded parallel HTTP ranges with full source digest validation."""
from pathlib import Path
import hashlib, urllib.request, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

def download(url, destination, size, sha256, workers=12, chunk_size=8*1024*1024):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.stat().st_size==size:
        with destination.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual==sha256: return dict(file=str(destination),size=size,sha256=actual,cached=True)
    parts=destination.parent/(destination.name+'.parts'); parts.mkdir(exist_ok=True)
    def part(start):
        end=min(start+chunk_size,size)-1
        path=parts/str(start)
        if path.exists() and path.stat().st_size==end-start+1: return path
        for attempt in range(5):
            try:
                sep='&' if '?' in url else '?'
                request=urllib.request.Request(url+sep+f'part={start}&attempt={attempt}',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=90) as r, path.with_suffix('.partial').open('wb') as f:
                    assert r.status==206, r.status
                    assert r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}',dict(r.headers)
                    count=0
                    while chunk:=r.read(1024*1024): f.write(chunk); count+=len(chunk)
                assert count==end-start+1, (count,end-start+1)
                path.with_suffix('.partial').replace(path)
                return path
            except Exception as exc:
                print(json.dumps(dict(file=destination.name,range_start=start,attempt=attempt,error=repr(exc))),flush=True)
        raise RuntimeError(f'Could not download range {start}')
    start_time=time.perf_counter(); tick=start_time; done_bytes=0
    starts=list(range(0,size,chunk_size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(part,s):s for s in starts}
        for future in as_completed(futures):
            path=future.result(); done_bytes+=path.stat().st_size
            if time.perf_counter()-tick>20:
                print(json.dumps(dict(file=destination.name,complete_bytes=done_bytes,total=size)),flush=True); tick=time.perf_counter()
    h=hashlib.sha256()
    with destination.with_suffix('.assembled').open('wb') as out:
        for s in starts:
            with (parts/str(s)).open('rb') as f:
                while chunk:=f.read(1024*1024): out.write(chunk); h.update(chunk)
    assert h.hexdigest()==sha256, (h.hexdigest(),sha256)
    destination.with_suffix('.assembled').replace(destination)
    result=dict(file=str(destination),size=size,sha256=h.hexdigest(),seconds=time.perf_counter()-start_time)
    destination.with_suffix(destination.suffix+'.verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('destination');p.add_argument('size',type=int);p.add_argument('sha256');p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();download(a.url,a.destination,a.size,a.sha256,a.workers)
