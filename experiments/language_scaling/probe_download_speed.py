import urllib.request,time,json,concurrent.futures,pathlib
m=json.loads(pathlib.Path('modern_corpus_v1/wiki_sources/manifest.json').read_text('utf-8'))
path='/datasets/wikimedia/wikipedia/resolve/'+m['revision']+'/20231101.zh/train-00001-of-00006.parquet?speed=2'
def probe(host):
    t=time.perf_counter()
    try:
        req=urllib.request.Request('https://'+host+path,headers={'Range':'bytes=0-1048575'})
        with urllib.request.urlopen(req,timeout=25) as r:
            b=r.read();return dict(host=host,size=len(b),seconds=time.perf_counter()-t,status=r.status)
    except Exception as e:return dict(host=host,error=repr(e),seconds=time.perf_counter()-t)
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for x in pool.map(probe,['huggingface.co','hf-mirror.com']):print(json.dumps(x),flush=True)
