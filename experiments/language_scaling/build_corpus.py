"""Reproducible public-domain Chinese corpus, whole-paragraph block splits."""
from __future__ import annotations
import hashlib
import gzip
import json
from pathlib import Path
import random
import re
import urllib.request

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
SOURCES=[(24264,'紅樓夢','曹雪芹'),(27166,'吶喊','鲁迅')]


def sha(blob):return hashlib.sha256(blob).hexdigest()


def main():
    raw_dir=DATA/'verified_raw';raw_dir.mkdir(parents=True,exist_ok=True)
    sources=[];splits={k:[] for k in ('train','validation','test')};blocks=[]
    seen_paragraphs=set();duplicates=0
    for number,title,author in SOURCES:
        url=f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt'
        request=urllib.request.Request(url,headers={'User-Agent':'CharacterHebbResearch/1.0',
                                                   'Accept-Encoding':'gzip'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request,timeout=25) as response:
                    transfer=response.read()
                    blob=gzip.decompress(transfer) if response.headers.get('Content-Encoding')=='gzip' else transfer
                break
            except Exception:
                if attempt==2:raise
        text=blob.decode('utf-8-sig')  # strict: incomplete/corrupt downloads fail
        (raw_dir/f'pg{number}.txt').write_bytes(blob)
        start=re.search(r'\*\*\* START OF .*?\*\*\*',text)
        end=re.search(r'\*\*\* END OF .*?\*\*\*',text)
        if not start or not end or start.end()>=end.start():raise ValueError('Missing Gutenberg boundaries')
        body=text[start.end():end.start()].replace('\r\n','\n').replace('\r','\n').strip()
        paragraphs=[]
        for chunk in re.split(r'\n\s*\n',body):
            # Remove print line wraps, retaining the complete paragraph order.
            paragraph=''.join(line.strip() for line in chunk.splitlines()).strip()
            if not paragraph:continue
            if len(paragraph)>=30:
                key=re.sub(r'\s+','',paragraph)
                if key in seen_paragraphs:duplicates+=1;continue
                seen_paragraphs.add(key)
            paragraphs.append(paragraph)
        # Adjacent whole paragraphs form ~5000-char blocks; never split characters
        # randomly or allocate individual sentences to different partitions.
        grouped=[];current=[];size=0
        for paragraph in paragraphs:
            if current and size>=5000:
                grouped.append('\n\n'.join(current));current=[];size=0
            current.append(paragraph);size+=len(paragraph)
        if current:grouped.append('\n\n'.join(current))
        indices=list(range(len(grouped)));random.Random(19313+number).shuffle(indices)
        count=max(1,round(len(indices)*.1))
        val=set(indices[:count]);test=set(indices[count:2*count])
        per_source=[]
        for i,chunk in enumerate(grouped):
            split='validation' if i in val else 'test' if i in test else 'train'
            record=dict(source=number,block=i,split=split,characters=len(chunk),sha256=sha(chunk.encode('utf-8')))
            blocks.append(record);per_source.append(record)
            splits[split].append((number,i,chunk))
        sources.append(dict(id=number,title=title,author=author,url=url,
                            landing_page=f'https://www.gutenberg.org/ebooks/{number}',
                            license='Project Gutenberg: public domain in the USA; original license retained in raw file',
                            raw_file=f'verified_raw/pg{number}.txt',raw_sha256=sha(blob),raw_bytes=len(blob),
                            body_characters=len(body),blocks=len(grouped)))
    # Interleave works, retaining source order within each work. This prevents
    # the smaller work from appearing only after the 400k training cap.
    output={}
    for split,records in splits.items():
        records.sort(key=lambda x:(x[1],x[0]))
        text='\n\n'.join(r[2] for r in records)+'\n'
        path=DATA/(split+'.txt');path.write_text(text,encoding='utf-8')
        output[split]=dict(file=path.name,characters=len(text),unique_characters=len(set(text)),
                           blocks=len(records),sha256=sha(path.read_bytes()))
    # Independent cross-partition exact paragraph overlap audit, including short ones.
    paragraph_sets={k:{p for _,_,s in records for p in s.split('\n\n') if len(p)>=30}
                    for k,records in splits.items()}
    overlap={a+'_'+b:len(paragraph_sets[a]&paragraph_sets[b])
             for a,b in [('train','validation'),('train','test'),('validation','test')]}
    if any(overlap.values()):raise ValueError('Long paragraph leakage')
    manifest=dict(version=1,sources=sources,splits=output,blocks=blocks,
                  split_unit='consecutive whole paragraphs grouped into approximately 5000-character blocks',
                  paragraph_duplicates_removed=duplicates,cross_split_exact_paragraph_overlap_ge30=overlap,
                  near_duplicate_deduplication=False,script_sha256=sha(Path(__file__).read_bytes()))
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'data_sources.md').write_text('# 本轮真实文本来源\n\n'
        +'\n'.join(f'- [{s["title"]}]({s["landing_page"]})，{s["author"]}；[下载]({s["url"]})。Project Gutenberg标注美国公版，完整许可保留在原文。' for s in sources)
        +'\n\n严格UTF-8解码验证下载；剔除Gutenberg头尾；去掉印刷换行，以完整段落组成约5000字的连续块，按固定种子约80/10/10分区。原稿的繁体和标点保留，没有自动翻译或简繁转换。跨分区30字以上完全相同段落为0；未声称排除了改写、共同人物和情节。各分区来自同两部作品，属于同来源未见段落测试，不是跨书、跨领域泛化。\n\n'
        +'语料包含叙事与对话，但不是现代百科知识或通用任务集合。旧data/raw的临时下载不进入实验；正式数据仅来自verified_raw。\n',encoding='utf-8')
    print(json.dumps({'splits':output,'overlap':overlap,'sources':[{k:s[k] for k in ('title','raw_bytes','blocks')} for s in sources]},ensure_ascii=False))


if __name__=='__main__':main()
