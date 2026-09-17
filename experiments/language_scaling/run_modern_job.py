"""Local foreground experiment coordinator; no external services or scheduling."""
from pathlib import Path
import datetime,json,subprocess,sys,time

HERE=Path(__file__).resolve().parent
ROOT=HERE/'modern_corpus_v1'
def status(phase,**kw):
    data=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    (ROOT/'job_status.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data),flush=True)

def main():
    required=[ROOT/'wiki_sources/train-00001-of-00006.parquet',
              *[ROOT/'dialogue_sources/lccc_base'/f'lccc_base_{split}.jsonl.gz' for split in ('train','valid','test')]]
    status('waiting_for_verified_download_files');start=time.monotonic()
    while not all(p.exists() for p in required):
        if time.monotonic()-start>7200:raise TimeoutError('Acquisition did not finish in two hours')
        time.sleep(3)
    for phase,script in [('preparing_corpus','prepare_modern_corpus.py'),('training_and_evaluating','run_modern_scale.py'),('writing_report','report_modern_scale.py')]:
        status(phase)
        process=subprocess.Popen([sys.executable,'-X','utf8',script],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        with (ROOT/'job.log').open('a',encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
        if process.wait()!=0:raise RuntimeError(script+' failed; evidence retained')
    status('completed')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=repr(exc));raise
