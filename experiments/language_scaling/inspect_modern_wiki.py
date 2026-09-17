from pathlib import Path
from collections import Counter
import json,random,time
import pyarrow.parquet as pq
from prepare_modern_corpus import clean

def main():
    root=Path(__file__).resolve().parent/'modern_corpus_v1/wiki_sources'
    path=root/'train-00001-of-00006.parquet';rng=random.Random(19462)
    chars=Counter();samples=[];documents=0;start=time.perf_counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['id','title','text']):
        for d in batch.to_pylist():
            text=clean(d['text']);chars.update(text);documents+=1
            example=dict(id=d['id'],title=d['title'],characters=len(text),excerpt=d['text'][:180])
            if len(samples)<12:samples.append(example)
            else:
                j=rng.randrange(documents)
                if j<12:samples[j]=example
    result=dict(documents=documents,normalized_characters=sum(chars.values()),vocabulary=len(chars),
                han_characters=sum(n for c,n in chars.items() if '\u3400'<=c<='\u9fff' or 0x20000<=ord(c)<=0x323af),
                sampled_articles=samples,seconds=time.perf_counter()-start)
    (root/'inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sampled_articles'},ensure_ascii=False),flush=True)
    print(json.dumps([x['title'] for x in samples],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
