"""Modern articles + real conversations, disjoint complete-document holdouts.

Only typographic punctuation, whitespace and control/format codes are removed.
No simplified/traditional conversion, vocabulary-frequency filtering, responses,
synthetic text, repeated epochs or language-model generation are introduced.
"""
from pathlib import Path
from collections import Counter
import gzip, hashlib, json, time, unicodedata
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'modern_corpus_v1'
OUT=ROOT/'prepared'
PROBES=['你好','早上好','晚安','谢谢','不客气','对不起','没关系','为什么','什么意思',
        '怎么回事','你在干什么','你在干嘛','吃饭了吗','你是谁','不知道','我不知道',
        '看不懂','听不懂','没听懂','你说什么','能再说一遍吗','今天天气','我喜欢你',
        '是一种','是一个','位于','主要用于','通常被称为','中华人民共和国','计算机','地球','因为','所以','科学研究','人工智能']
TABLE={i:None for i in range(0x110000) if unicodedata.category(chr(i)).startswith('P') or unicodedata.category(chr(i)) in ('Cc','Cf') or chr(i).isspace()}

def clean(text): return text.translate(TABLE)
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def digest_file(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    import pyarrow.parquet as pq
    if OUT.exists(): raise SystemExit('Refuse to overwrite a prepared corpus')
    OUT.mkdir(parents=True)
    started=time.perf_counter(); seen=set(); counts=Counter(); stat={}; utterances=Counter()
    cue_counts=Counter(); wiki_cues=Counter(); exact_counts=Counter(); replies={p:[] for p in PROBES}
    output={kind:(OUT/(kind+'.jsonl')).open('w',encoding='utf-8',newline='\n') for kind in ('dialogue_train','wiki_train','validation','test')}
    hashes={kind:hashlib.sha256() for kind in output}
    train_stats=Counter(); inputs=[]
    def emit(turns,source,original_id,forced=None):
        turns=[clean(t) for t in turns]
        turns=[t for t in turns if t]
        text=''.join(turns)
        if not text: return
        digest=hashlib.sha256(text.encode('utf-8')).digest()
        st=stat.setdefault(source,Counter())
        st['raw_documents']+=1;st['normalized_characters_before_dedup']+=len(text)
        if digest in seen: st['duplicate_documents_removed']+=1;return
        seen.add(digest)
        bucket=int.from_bytes(digest[:8],'big')%1000
        split=forced or ('validation' if bucket<5 else 'test' if bucket<10 else 'train')
        key=('wiki_train' if source=='wikipedia' else 'dialogue_train') if split=='train' else split
        doc=dict(source=source,id=original_id,sha256=digest.hex(),turns=turns)
        line=json.dumps(doc,ensure_ascii=False,separators=(',',':'))+'\n'
        output[key].write(line);hashes[key].update(line.encode('utf-8'))
        st[split+'_documents']+=1; st[split+'_characters']+=len(text)
        st[split+'_han_characters']+=sum(1 for c in text if '\u3400'<=c<='\u9fff' or 0x20000<=ord(c)<=0x323af)
        if split=='train':
            counts.update(text);train_stats[key]+=len(text)
            if source=='lccc_base':
                for i,t in enumerate(turns):
                    if 2<=len(t)<=8: utterances[t]+=1
                    for p in PROBES:
                        cue_counts[p]+=t.count(p)
                        if t==p:
                            exact_counts[p]+=1
                            if i+1<len(turns) and len(replies[p])<5: replies[p].append(turns[i+1])
            else:
                for p in PROBES:wiki_cues[p]+=text.count(p)
    # Preserve official dialogue holdouts and remove their exact normalized
    # duplicates from training before applying additional hash-based holdouts.
    for split,forced in [('valid','validation'),('test','test'),('train',None)]:
        path=ROOT/'dialogue_sources/lccc_base'/f'lccc_base_{split}.jsonl.gz'
        expected={'valid':'5cc27e7ac3447c5a31386178f82ff01cab56e27827445ef8d429809301491759',
                  'test':'cf8757587bdb8f360cc94fc38baadf9e185bad65a26155527a8430c048676016',
                  'train':'2162e0ed923fba62329cabf7e1493fbe59248afc94a62508e4abdea61e624627'}[split]
        actual=digest_file(path);assert actual==expected
        inputs.append(dict(path=str(path.relative_to(HERE)),sha256=actual,bytes=path.stat().st_size))
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for index,line in enumerate(f):
                turns=json.loads(line)
                assert isinstance(turns,list) and all(isinstance(t,str) for t in turns)
                emit(turns,'lccc_base',split+':'+str(index),forced)
                if index and index%250000==0: print(json.dumps(dict(source='lccc_base',split=split,records=index,training_characters=sum(train_stats.values()))),flush=True)
    wiki=ROOT/'wiki_sources/train-00001-of-00006.parquet'
    meta=json.loads((ROOT/'wiki_sources/manifest.json').read_text('utf-8'))
    sha=digest_file(wiki);assert sha==meta['files'][1]['lfs']['oid']
    inputs.append(dict(path=str(wiki.relative_to(HERE)),sha256=sha,bytes=wiki.stat().st_size))
    index=0
    for batch in pq.ParquetFile(wiki).iter_batches(batch_size=256,columns=['id','text']):
        for doc in batch.to_pylist():
            emit([doc['text']],'wikipedia',doc['id']);index+=1
            if train_stats['wiki_train']>=200_000_000:break
        if train_stats['wiki_train']>=200_000_000:break
        if index%2560==0: print(json.dumps(dict(source='wikipedia',records=index,training_characters=sum(train_stats.values()))),flush=True)
    for f in output.values():f.close()
    vocabulary=''.join(sorted(counts))
    dump(OUT/'vocabulary.json',dict(vocabulary=vocabulary,counts={c:counts[c] for c in vocabulary}))
    dump(OUT/'common_cues.json',dict(fixed_cues=[dict(prompt=p,training_substring_occurrences=cue_counts[p]+wiki_cues[p],training_dialogue_substring_occurrences=cue_counts[p],training_wikipedia_substring_occurrences=wiki_cues[p],training_exact_utterances=exact_counts[p],first_training_replies=replies[p]) for p in PROBES],
                                  most_common_short_utterances=utterances.most_common(40)))
    manifest=dict(completed=True,inputs=inputs,source_stats=stat,training_characters=sum(train_stats.values()),
                  training_parts=train_stats,vocabulary_size=len(vocabulary),dense_matrix_bytes=4*len(vocabulary)**2,
                  clean_rule='Remove Unicode P punctuation, Cc/Cf control/format codes and whitespace; preserve other identities, case and simplified/traditional forms',
                  split_rule='Official LCCC valid/test reserved first; remove exact normalized complete-document duplicates globally; other documents SHA256 modulo 1000: 0..4 validation, 5..9 test, else train',
                  duplicate_rule='SHA256 of concatenated normalized turns; exact complete documents only, no near-duplicate or short-phrase removal',
                  source_order='Official dialogue holdouts, dialogue train, Wikipedia shard 00001 until at least 200M training characters or shard exhausted; shard chosen for download size before viewing results',
                  files={k:dict(path=k+'.jsonl',sha256=v.hexdigest()) for k,v in hashes.items()},seconds=time.perf_counter()-started)
    dump(OUT/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
