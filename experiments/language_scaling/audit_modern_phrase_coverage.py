"""Training-only phrase/reply coverage; never fed into the model controller."""
from collections import Counter
from run_modern_scale import DATA,OUT,read,dump,file_hash
import json,time

def main():
    start=time.perf_counter();protocol=read(OUT/'frozen_protocol.json')
    common=read(DATA/'common_cues.json');wanted={x['prompt'] for x in common['fixed_cues']}
    exact=Counter();followed=Counter();first={p:Counter() for p in wanted}
    novel={p:Counter() for p in protocol['probes']['novel_prompts']}
    for kind in ('dialogue_train','wiki_train'):
        with (DATA/(kind+'.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                doc=json.loads(line);turns=doc['turns'];text=''.join(turns)
                for p in novel:novel[p][kind]+=text.count(p)
                if kind=='dialogue_train':
                    for i,t in enumerate(turns):
                        if t in wanted:
                            exact[t]+=1
                            if i+1<len(turns):
                                followed[t]+=1;first[t][turns[i+1][0]]+=1
    for row in common['fixed_cues']:assert exact[row['prompt']]==row['training_exact_utterances']
    result=dict(completed=True,scope='Exact normalized utterances; a following reply must be inside the same retained training conversation. Novel occurrence search is inside complete normalized training documents.',
        phrases={p:dict(exact_utterances=exact[p],with_following_reply=followed[p],reply_first_characters=first[p].most_common(12)) for p in sorted(wanted)},
        novel_occurrences=novel,seconds=time.perf_counter()-start,
        prepared_manifest_sha256=file_hash(DATA/'manifest.json'))
    dump(OUT/'phrase_coverage_audit.json',result)
    print(json.dumps({p:result['phrases'][p] for p in ('你好','谢谢','为什么','什么意思','你在干什么')},ensure_ascii=False),flush=True)
    print(json.dumps(novel,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
