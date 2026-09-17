"""Short-context-only validation selection, followed by frozen new test positions.

The character-Hebb implementation and training material are unchanged. Only
the activity-decay parameter is varied; there are no word/answer-specific links.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from hebb_text import CharacterHebb, Config
from run_experiment import evaluate, context_interventions, generate

HERE=Path(__file__).resolve().parent
OUT=HERE/'short_context_v1'
DECAYS=(.1,.25,.5,.65,.8)


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def positions(text,count,seed,excluded=()):
    blocked=set(excluded)
    choices=[i for i in range(64,len(text)) if i not in blocked and '\u3400'<=text[i]<='\u9fff']
    rng=np.random.default_rng(seed)
    return sorted(map(int,rng.choice(choices,min(count,len(choices)),replace=False)))


def effective_span(decay,floor):
    amplitude=1.;span=0
    while amplitude>=floor:
        span+=1;amplitude*=decay
    return span


def paired_effect(iv):
    full=iv['full']['rows'];last=iv['last_only']['rows'];shuffled=iv['shuffled_earlier']['rows']
    assert [r['position'] for r in full]==[r['position'] for r in last]==[r['position'] for r in shuffled]
    return dict(n=len(full),full_correct=sum(r['rank']==1 for r in full),last_only_correct=sum(r['rank']==1 for r in last),
                earlier_context_helped=sum(a['rank']==1 and b['rank']!=1 for a,b in zip(full,last)),
                earlier_context_harmed=sum(a['rank']!=1 and b['rank']==1 for a,b in zip(full,last)),
                prediction_changed_with_last_only=sum(a['predicted']!=b['predicted'] for a,b in zip(full,last)),
                prediction_changed_with_shuffled_earlier=sum(a['predicted']!=b['predicted'] for a,b in zip(full,shuffled)),
                full_most_predicted=Counter(r['predicted'] for r in full).most_common(8))


def main():
    if OUT.exists():raise SystemExit('Refuse to overwrite an existing run')
    OUT.mkdir()
    start=time.perf_counter()
    data=HERE/'data_no_punct_v1'
    train=(data/'train.txt').read_text(encoding='utf-8')
    validation=(data/'validation.txt').read_text(encoding='utf-8')
    test=(data/'test.txt').read_text(encoding='utf-8')
    vocabulary=''.join(sorted(set(train)))
    old_protocol=read(HERE/'results_no_punct_v1/frozen_protocol.json')
    old_matched=read(data/'matched_positions.json')
    excluded=set(old_protocol['test_positions'])|set(old_matched['new_positions'])
    validation_positions=positions(validation,1024,19351)
    test_positions=positions(test,1024,19352,excluded)
    configs={f'decay_{str(d).replace(".","p")}':Config(trace_decay=d,trace_floor=.02,growth='diminishing') for d in DECAYS}
    protocol=dict(version=1,corpus='data_no_punct_v1',train_characters=len(train),vocabulary_characters=len(vocabulary),
                  candidates={name:asdict(c)|{'effective_character_span':effective_span(c.trace_decay,c.trace_floor)} for name,c in configs.items()},
                  selection='minimum validation mean_nll_known; ties follow candidate order',
                  validation_positions=validation_positions,test_positions=test_positions,excluded_test_positions=sorted(excluded),
                  test_scope='previously unscored positions in the same held-out text, not new independent books',
                  source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__),HERE/'hebb_text.py',HERE/'run_experiment.py',data/'train.txt',data/'validation.txt',data/'test.txt',
                       HERE/'results_no_punct_v1/frozen_protocol.json',data/'matched_positions.json']})
    dump(OUT/'frozen_protocol.json',protocol)
    candidates={}
    # No test scoring occurs before the validation selection is saved.
    for name,cfg in configs.items():
        b=CharacterHebb(vocabulary,cfg)
        training_start=time.perf_counter();b.train(train);training_seconds=time.perf_counter()-training_start
        assert b.clock==len(train)==b.learned_characters
        evaluated=evaluate(b,validation,set(validation_positions))
        directory=OUT/name;directory.mkdir()
        dump(directory/'validation.json',evaluated)
        b.save(directory/'model.npz')
        restored=CharacterHebb.load(directory/'model.npz')
        assert np.array_equal(b.weights,restored.weights) and b.activity==restored.activity and b.clock==restored.clock
        candidates[name]=dict(config=asdict(cfg),span=effective_span(cfg.trace_decay,cfg.trace_floor),
                              validation=evaluated['metrics'],weight_sha256=b.weight_hash(),stats=b.stats(),training_seconds=training_seconds)
        print(json.dumps({'candidate':name,**candidates[name]},ensure_ascii=False),flush=True)
        del b,restored
    selected=min(candidates,key=lambda name:candidates[name]['validation']['mean_nll_known'])
    dump(OUT/'selection.json',dict(selected=selected,candidates=candidates,test_scored_yet=False))
    selection_hash=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    names=list(dict.fromkeys([selected,'decay_0p8']))
    test_results={}
    for name in names:
        b=CharacterHebb.load(OUT/name/'model.npz')
        initial_hash=b.weight_hash()
        scored=evaluate(b,test,set(test_positions))
        interventions=context_interventions(b,test,set(test_positions))
        recurrent=evaluate(b,test,set(test_positions),recurrent_steps=2)
        # Prompts fixed by the frozen test-position sequence, no example selection.
        prompts=[test[max(0,p-32):p] for p in test_positions[::256]][:4]
        examples=[dict(prompt=p,greedy=generate(b,p),greedy_two_steps=generate(b,p,recurrent_steps=2)) for p in prompts]
        assert b.weight_hash()==initial_hash and b.clock==len(train)
        result=dict(test=scored,context_interventions=interventions,context_effect=paired_effect(interventions),
                    two_recurrent_steps=recurrent,continuations=examples,probe_state_restored=True)
        # <=18-character traces must agree with evaluation using a 64-character prefix.
        for a,c in zip(scored['rows'],interventions['full']['rows']):
            assert a['position']==c['position'] and a['rank']==c['rank'] and a['probability']==c['probability']
        dump(OUT/name/'test_and_context.json',result)
        test_results[name]=dict(metrics=scored['metrics'],two_recurrent_steps=recurrent['metrics'],context_effect=result['context_effect'])
        print(json.dumps({'test_candidate':name,**test_results[name]},ensure_ascii=False),flush=True)
        del b
    # A control only; it cannot replace the selected multi-character candidate.
    baseline_path=HERE/'results_no_punct_v1/adjacent_diminishing/model.npz'
    baseline=CharacterHebb.load(baseline_path)
    baseline_scored=evaluate(baseline,test,set(test_positions))
    dump(OUT/'one_character_control.json',baseline_scored)
    assert selection_hash==hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    summary=dict(completed=True,protocol=protocol,selected=selected,candidates=candidates,
                 test_results=test_results,one_character_control=baseline_scored['metrics'],
                 selection_file_sha256=selection_hash,elapsed_seconds=time.perf_counter()-start)
    dump(OUT/'summary.json',summary)
    print(json.dumps({'complete':True,'selected':selected,'one_character_control':baseline_scored['metrics'],
                      'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
