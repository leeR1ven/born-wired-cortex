"""One chronological Hebbian graph, hundreds of millions of modern characters.

Evaluation records and phrase-frequency probes are external measurements only.
Neither training nor recall receives record IDs, split labels or answer tables.
"""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import hashlib,json,random,time,platform
import numpy as np
from hebb_text import CharacterHebb,Config
from fast_hebb import train_ids,make_lookup,encode
from persistent_candidates import PersistentRecall,RecallConfig
from run_candidate_budget import trajectory

HERE=Path(__file__).resolve().parent
DATA=HERE/'modern_corpus_v1/prepared'
OUT=HERE/'modern_scale_v1'
CONFIGS={'recent_only':RecallConfig(candidate_count=0,candidate_budget=0.),
         'persistent8_weak':RecallConfig(candidate_count=8,candidate_budget=.25),
         'persistent4_strong':RecallConfig(candidate_count=4,candidate_budget=1.)}

def read(path):return json.loads(path.read_text('utf-8'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def file_hash(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def weight_hash(graph):
    h=hashlib.sha256(graph.vocabulary.encode('utf-8'))
    for start in range(0,len(graph.vocabulary),64):h.update(graph.weights[start:start+64].tobytes())
    return h.hexdigest()

def save(graph,path,counts):
    path.mkdir()
    np.save(path/'weights.npy',graph.weights,allow_pickle=False)
    np.save(path/'character_counts.npy',counts,allow_pickle=False)
    meta=dict(vocabulary=graph.vocabulary,config=asdict(graph.config),clock=graph.clock,
              learned_characters=graph.learned_characters,update_events=graph.update_events,
              activity=list(graph.activity.items()),weight_sha256=weight_hash(graph))
    dump(path/'metadata.json',meta)
    return meta

def load(path):
    path=Path(path);m=read(path/'metadata.json')
    graph=CharacterHebb.__new__(CharacterHebb)
    graph.vocabulary=m['vocabulary'];graph.ids={c:i for i,c in enumerate(graph.vocabulary)}
    graph.config=Config(**m['config']);graph.weights=np.load(path/'weights.npy',mmap_mode='r',allow_pickle=False)
    for key in ('clock','learned_characters','update_events'):setattr(graph,key,m[key])
    graph.activity={int(i):float(a) for i,a in m['activity']}
    return graph

def stats(graph,counts):
    connected=0;maximum=0.;total=0.;degrees=[]
    for start in range(0,len(graph.vocabulary),128):
        rows=graph.weights[start:start+128]
        degrees.extend(np.count_nonzero(rows,axis=1).tolist())
        positive=rows[rows>0];connected+=len(positive)
        if positive.size: maximum=max(maximum,float(positive.max()));total+=float(positive.sum(dtype=np.float64))
    bins=Counter();tokens=Counter()
    for c,n in zip(graph.vocabulary,counts):
        if not ('\u3400'<=c<='\u9fff' or 0x20000<=ord(c)<=0x323af):continue
        key='0' if n==0 else '1-9' if n<10 else '10-99' if n<100 else '100-999' if n<1000 else '1000+'
        bins[key]+=1;tokens[key]+=int(n)
    return dict(connections=connected,weight_max=maximum,weight_mean=total/connected if connected else 0.,
                matrix_bytes=graph.weights.nbytes,han_frequency_types=bins,han_frequency_tokens=tokens,
                top_characters=[dict(character=graph.vocabulary[i],count=int(counts[i]),out_degree=degrees[i]) for i in np.argsort(-counts,kind='stable')[:30]])

def text_chunks(path,characters=1_000_000):
    pieces=[];size=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            text=''.join(json.loads(line)['turns'])
            pieces.append(text);size+=len(text)
            if size>=characters:
                yield ''.join(pieces);pieces=[];size=0
    if pieces:yield ''.join(pieces)

def training_stream():
    # Keep source proportions similar at all scales, with one continuous state.
    parts=read(DATA/'manifest.json')['training_parts']
    dialogue_block=100_000
    wiki_block=max(1,round(dialogue_block*parts['wiki_train']/parts['dialogue_train']))
    streams=[iter(text_chunks(DATA/(kind+'.jsonl'),size)) for kind,size in
             (('dialogue_train',dialogue_block),('wiki_train',wiki_block))]
    alive=[True,True]
    while any(alive):
        for i,stream in enumerate(streams):
            if alive[i]:
                text=next(stream,None)
                if text is None:alive[i]=False
                else:yield ('lccc_base' if i==0 else 'wikipedia'),text

def prepare_probes():
    rng=random.Random(19460);reservoirs={k:[] for k in ('lccc_base','wikipedia','reply_first')};seen=Counter()
    for line in (DATA/'test.jsonl').open(encoding='utf-8'):
        doc=json.loads(line);text=''.join(doc['turns'])
        choices=[]
        if len(text)>=8:
            p=rng.randrange(1,len(text))
            choices.append((doc['source'],text[max(0,p-64):p],text[p]))
        if doc['source']=='lccc_base' and len(doc['turns'])>=2:
            p=rng.randrange(1,len(doc['turns']))
            choices.append(('reply_first',''.join(doc['turns'][:p])[-64:],doc['turns'][p][0]))
        for kind,prefix,target in choices:
            seen[kind]+=1
            row=dict(kind=kind,source=doc['source'],id=doc['id'],document_sha256=doc['sha256'],prefix=prefix,target=target)
            if len(reservoirs[kind])<128:reservoirs[kind].append(row)
            else:
                j=rng.randrange(seen[kind])
                if j<128:reservoirs[kind][j]=row
    common=read(DATA/'common_cues.json')
    prompts=[x['prompt'] for x in common['fixed_cues']]
    top=[]
    for p,n in common['most_common_short_utterances']:
        if p not in prompts and len(set(p))>1:top.append(p)
        if len(top)==8:break
    prompts+=top
    novel=['请解释一下咕噜帕索是什么意思','蓝色的星期八正在给月亮充电','我刚说的哒喵兹噜是什么意思']
    prompts+=novel
    return dict(seed=19460,records=[r for rows in reservoirs.values() for r in rows],
                sampling='Reservoir of 128 complete documents per source with one uniform noninitial character per document, plus 128 dialogue response-start probes',
                eligible_records=seen,prompts=prompts,novel_prompts=novel,common_cues=common)

def evaluate(graph,probes,counts):
    results={};snapshot=(graph.clock,graph.activity.copy(),graph.update_events,graph.learned_characters)
    for name,cfg in CONFIGS.items():
        recall=PersistentRecall(graph,cfg);rows=[]
        for source_row in probes['records']:
            recall.reset_probe();recall.warm(source_row['prefix']);raw=recall.currents();prob=recall.probabilities(raw)
            order=np.argsort(-prob,kind='stable');target=graph.ids.get(source_row['target'])
            rank=None if target is None else int(np.flatnonzero(order==target)[0])+1
            rows.append(dict(**source_row,predicted=graph.vocabulary[order[0]],rank=rank,
                             target_probability=None if target is None else float(prob[target]),
                             stage_training_target_count=0 if target is None else int(counts[target]),
                             top5=[graph.vocabulary[i] for i in order[:5]]))
        metrics={}
        for kind in ('lccc_base','wikipedia','reply_first'):
            group=[r for r in rows if r['kind']==kind];known=[r for r in group if r['target_probability'] is not None]
            metrics[kind]=dict(n=len(group),known=len(known),top1=sum(r['rank']==1 for r in group),
                top5=sum(r['rank'] is not None and r['rank']<=5 for r in group),
                bits_per_known_character=float(np.mean([-np.log2(r['target_probability']) for r in known])) if known else None)
        # Preserve every raw output. Active-group traces for fixed illustrative
        # prompts; these do not affect training or choose output examples.
        spoken=[];silent=[]
        for prompt in probes['prompts']:
            t=trajectory(graph,prompt,cfg)
            spoken.append(dict(prompt=prompt,emitted=t['emitted'],tail_period=t['tail_period'],
                               clarification_keyword_present=any(s in t['emitted'] for s in ('什么意思','为什么','不知道','不懂','再说','说清楚'))))
        if cfg.candidate_count:
            for prompt in ('你好','为什么','你在干什么',probes['novel_prompts'][0]):silent.append(trajectory(graph,prompt,cfg,silent=True))
        results[name]=dict(metrics=metrics,rows=rows,spoken=spoken,silent=silent)
    assert snapshot==(graph.clock,graph.activity,graph.update_events,graph.learned_characters)
    return results

def main(milestones=(3_000_000,30_000_000,100_000_000,300_000_000)):
    if OUT.exists():raise SystemExit('Refuse to overwrite run evidence')
    started=time.perf_counter();manifest=read(DATA/'manifest.json');v=read(DATA/'vocabulary.json')['vocabulary']
    for info in manifest['files'].values():assert file_hash(DATA/info['path'])==info['sha256']
    assert read(HERE/'fast_hebb_verification.json')['completed']
    final=manifest['training_characters'];stages=sorted(set([n for n in (*milestones,final) if n<=final]))
    probes=prepare_probes();OUT.mkdir()
    protocol=dict(config=asdict(Config(trace_decay=.25)),recall_configs={k:asdict(c) for k,c in CONFIGS.items()},
        stages=stages,main_condition='persistent8_weak',data_manifest=manifest,
        interpretation='Fixed-rule scale experiment; simplified character graph, not full PFC/PetBrain. No parameter selection from current test.',
        training='One pass; alternate approximately 100k-character dialogue blocks and article blocks scaled by full training source character ratio; preserve complete documents inside blocks; one continuous trace/clock, no synthetic boundaries or role labels',
        vocabulary='All identities found in final training split only, allocated at every stage; stage counts distinguish allocated from experienced characters',
        preprocessing_change='Remove spaces and control characters in addition to prior Unicode P deletion because LCCC contains tokenization spaces; this differs from old novel experiment',
        probes=probes,source_sha256={p:file_hash(HERE/p) for p in ('hebb_text.py','fast_hebb.py','persistent_candidates.py','run_modern_scale.py','prepare_modern_corpus.py',
            'run_candidate_budget.py','run_persistent_candidates.py','run_experiment.py','fast_hebb_verification.json')},
        prepared_metadata_sha256={p:file_hash(DATA/p) for p in ('manifest.json','vocabulary.json','common_cues.json')},
        python=platform.python_version(),numpy=np.__version__)
    dump(OUT/'frozen_protocol.json',protocol)
    graph=CharacterHebb(v,Config(trace_decay=.25));lookup=make_lookup(v);counts=np.zeros(len(v),np.int64)
    stage_index=0;consumed=Counter();stream_hash=hashlib.sha256();summaries={};last_print=time.perf_counter()
    for source,text in training_stream():
        ids=encode(text,lookup);offset=0
        while offset<len(ids):
            assert stage_index<len(stages), 'Training stream contains more characters than its manifest'
            end=min(len(ids),offset+stages[stage_index]-graph.learned_characters)
            train_ids(graph,ids[offset:end]);counts+=np.bincount(ids[offset:end],minlength=len(v));consumed[source]+=end-offset
            stream_hash.update(text[offset:end].encode('utf-8'));offset=end
            if graph.learned_characters==stages[stage_index]:
                stage=graph.learned_characters;meta=save(graph,OUT/f'model_{stage}',counts)
                measured=evaluate(graph,probes,counts)
                assert weight_hash(graph)==meta['weight_sha256']
                dump(OUT/f'evaluation_{stage}.json',measured)
                summary=dict(characters=stage,source_characters=dict(consumed),stream_sha256=stream_hash.hexdigest(),
                    weight_sha256=meta['weight_sha256'],stats=stats(graph,counts),
                    metrics={k:r['metrics'] for k,r in measured.items()},
                    outputs={k:r['spoken'] for k,r in measured.items()},seconds_since_start=time.perf_counter()-started)
                summaries[str(stage)]=summary;dump(OUT/f'stage_{stage}.json',summary)
                print(json.dumps(dict(stage=stage,source_characters=consumed,main_metrics=summary['metrics']['persistent8_weak'])),flush=True)
                stage_index+=1
            if time.perf_counter()-last_print>20:
                print(json.dumps(dict(training_characters=graph.learned_characters,elapsed=time.perf_counter()-started)),flush=True);last_print=time.perf_counter()
    assert graph.learned_characters==final and stage_index==len(stages)
    assert dict(consumed)=={'lccc_base':manifest['training_parts']['dialogue_train'],'wikipedia':manifest['training_parts']['wiki_train']}
    dump(OUT/'summary.json',dict(completed=True,training_characters=final,stages=summaries,
        protocol_sha256=file_hash(OUT/'frozen_protocol.json'),elapsed_seconds=time.perf_counter()-started,
        final_model=f'model_{final}',no_test_learning=True))
    print('COMPLETE',final,flush=True)

if __name__=='__main__':main()
