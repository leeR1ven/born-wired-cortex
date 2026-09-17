"""Paired mechanism tests; no retraining or linguistic-success selection."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import json,time
import numpy as np
from run_modern_scale import load,read,dump,file_hash,weight_hash,HERE,OUT as MODELS
from persistent_candidates import RecallConfig
from population_propagation import BinaryPopulation,MaskedRecall,state_key

OUT=HERE/'population_probe_v1'
SPECS={
 'weighted8_spoken':dict(kind='weighted',k=8,budget=.25,spoken=True,no_self=False),
 'weighted8_silent':dict(kind='weighted',k=8,budget=.25,spoken=False,no_self=False),
 'weighted8_silent_no_self':dict(kind='weighted',k=8,budget=.25,spoken=False,no_self=True),
 'weighted32_same_budget':dict(kind='weighted',k=32,budget=.25,spoken=False,no_self=False),
 'weighted32_scaled_budget':dict(kind='weighted',k=32,budget=1.,spoken=False,no_self=False),
 **{f'binary{k}'+('_no_self' if ns else ''):dict(kind='binary',k=k,no_self=ns,refractory=0)
    for k,ns in [(8,False),(8,True),(32,False),(32,True),(128,True)]},
 'binary32_no_self_rest1':dict(kind='binary',k=32,no_self=True,refractory=1),
 'binary32_no_self_rest3':dict(kind='binary',k=32,no_self=True,refractory=3),
}
TRACE_PROMPTS=['你好','为什么','地球','请解释一下咕噜帕索是什么意思']

def make_engine(graph,spec):
    if spec['kind']=='binary':return BinaryPopulation(graph,spec['k'],spec['no_self'],spec['refractory'])
    return MaskedRecall(graph,RecallConfig(candidate_count=spec['k'],candidate_budget=spec['budget']),spec['no_self'])

def run(graph,prompt,spec,steps=160):
    e=make_engine(graph,spec);e.warm(prompt)
    seen={};repeat=None;union=set(e.activity);rows=[];sets=[];emitted=''
    for tick in range(steps):
        key=state_key(e)
        if key in seen and repeat is None:repeat=dict(first=seen[key],again=tick,period=tick-seen[key])
        seen.setdefault(key,tick)
        old=set(e.activity);sets.append(frozenset(old))
        a=np.array(list(e.activity.values()),np.float64);total=float(a.sum())
        raw=e.currents();order=np.argsort(-raw,kind='stable')[:8]
        winner=int(order[0]) if raw[order[0]]>0 else None
        row=dict(step=tick,active_count=len(old),effective_count=float(total*total/np.square(a).sum()) if total else 0.,
                 top_current=None if winner is None else graph.vocabulary[winner],
                 positive_candidates=int(np.count_nonzero(raw>0)))
        if prompt in TRACE_PROMPTS:
            row['activity']=[[graph.vocabulary[i],v] for i,v in e.activity.items()]
            row['top8_currents']=[[graph.vocabulary[i],float(raw[i])] for i in order if raw[i]>0]
            if isinstance(e,BinaryPopulation):row['cooldown']=[[graph.vocabulary[i],int(e.cooldown[i])] for i in np.flatnonzero(e.cooldown)]
            # Full learned connections contributing to the top target at selected
            # frames; includes self-edge omission explicitly.
            if winner is not None and (tick<6 or tick>=steps-4):
                row['contributors']=[dict(source=graph.vocabulary[i],activation=v,weight=float(graph.weights[i,winner]),
                   used_current=0. if spec['no_self'] and i==winner else float(np.float32(v)*graph.weights[i,winner]),
                   self_edge=i==winner) for i,v in e.activity.items()]
        if isinstance(e,BinaryPopulation):
            chosen=e.advance(raw)
            if np.count_nonzero(raw>0)>=spec['k'] and not spec['refractory']:
                assert len(chosen)==spec['k']
            assert all(v==1. for v in e.activity.values())
        else:
            c=graph.vocabulary[winner] if spec['spoken'] and winner is not None else None
            e.advance(c,raw)
            if c is not None:emitted+=c
        new=set(e.activity);row['new_vs_previous']=len(new-old);row['new_ever']=len(new-union)
        row['next_active_count']=len(new);union.update(new);rows.append(row)
    late=rows[-32:]
    return dict(prompt=prompt,emitted=emitted if spec.get('spoken') else None,first_exact_repeat=repeat,
        distinct_units=len(union),final_active_count=len(e.activity),
        last32_distinct_member_sets=len(set(sets[-32:])),
        mean_late_turnover=float(np.mean([r['new_vs_previous'] for r in late])),
        late_first_time_units=sum(r['new_ever'] for r in late),
        late_effective_count=float(np.mean([r['effective_count'] for r in late])),
        final_members=[graph.vocabulary[i] for i in sorted(e.activity)],rows=rows)

def main():
    if OUT.exists():raise SystemExit('Refuse to overwrite evidence')
    start=time.perf_counter();saved=read(MODELS/'summary.json');model=MODELS/saved['final_model'];graph=load(model)
    original=read(MODELS/f"stage_{saved['training_characters']}.json")
    prompts=read(MODELS/'frozen_protocol.json')['probes']['prompts']
    before=weight_hash(graph);assert before==read(model/'metadata.json')['weight_sha256']
    OUT.mkdir()
    protocol=dict(model=str(model.relative_to(HERE)),weight_sha256=before,prompts=prompts,specs=SPECS,steps=160,
       trace_prompts=TRACE_PROMPTS,
       weighted_rule='Original residual-first-floor and post-injection floor kept exactly; spoken clamps one winner to one; silent has no winner feedback. K32 budget1 changes total energy as well as K.',
       binary_rule='NEW isolated recall dynamics: original .25/.02 sensory trace seeds input, then positive top K ALL fire at one; no residual activity or spoken winner. Sources summed in canonical ID order. Local cooldown optional.',
       intervention='No self: set only diagonal contributions to zero on every recall step, including warm-up for weighted family; original disk matrix remains unchanged.',
       inference_limits='Member turnover, exact state cycles and amplitude/effective population are dynamics measures, not linguistic or AGI scores. Binary groups are simultaneous sets, never decoded as ordered sentences.',
       source_sha256={n:file_hash(HERE/n) for n in ('run_population_probe.py','population_propagation.py','persistent_candidates.py','run_modern_scale.py','hebb_text.py')})
    dump(OUT/'frozen_protocol.json',protocol);summary={}
    baseline={r['prompt']:r['emitted'] for r in original['outputs']['persistent8_weak']}
    for name,spec in SPECS.items():
        results=[]
        for prompt in prompts:
            result=run(graph,prompt,spec)
            if name=='weighted8_spoken':assert result['emitted'][:64]==baseline[prompt]
            results.append(result)
        dump(OUT/(name+'.json'),dict(spec=spec,results=results))
        summary[name]=dict(prompts=len(results),exact_state_repeats=sum(r['first_exact_repeat'] is not None for r in results),
            repeat_periods=dict(Counter(str(r['first_exact_repeat']['period']) if r['first_exact_repeat'] else 'none' for r in results)),
            final_extinctions=sum(r['final_active_count']==0 for r in results),
            mean_final_active=float(np.mean([r['final_active_count'] for r in results])),
            mean_distinct_lifetime_units=float(np.mean([r['distinct_units'] for r in results])),
            mean_last32_member_sets=float(np.mean([r['last32_distinct_member_sets'] for r in results])),
            mean_late_turnover=float(np.mean([r['mean_late_turnover'] for r in results])),
            mean_late_effective_count=float(np.mean([r['late_effective_count'] for r in results])),
            mean_late_first_time_units=float(np.mean([r['late_first_time_units'] for r in results])))
        print(json.dumps(dict(condition=name,**summary[name]),ensure_ascii=False),flush=True)
    assert weight_hash(graph)==before
    dump(OUT/'summary.json',dict(completed=True,conditions=summary,weight_sha256=before,
         protocol_sha256=file_hash(OUT/'frozen_protocol.json'),seconds=time.perf_counter()-start,
         original_core_and_weights_unchanged=True))
    graph.weights._mmap.close()

if __name__=='__main__':main()
