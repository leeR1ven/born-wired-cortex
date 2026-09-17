"""Explicit stratified diagnostic, because natural text samples underweight rare units."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from hebb_text import CharacterHebb
from persistent_candidates import RecallConfig
from run_candidate_budget import evaluate
from run_expanded_scale import frequency_bin

HERE=Path(__file__).resolve().parent
OUT=HERE/'rare_character_probe_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite diagnostic')
    main_summary=read(HERE/'expanded_scale_v1/summary.json')
    assert main_summary['completed']
    train_path=HERE/'data_expanded_v1/train.txt'
    test_path=HERE/'data_expanded_v1/new_test.txt'
    train=train_path.read_text(encoding='utf-8')
    test=test_path.read_text(encoding='utf-8')
    counts=Counter(train)
    excluded=set(main_summary['protocol']['new_test_positions'])
    groups={name:[] for name in ('0','1-9','10-99','100-999','1000+')}
    for position in range(64,len(test)):
        if position not in excluded and '\u3400'<=test[position]<='\u9fff':
            groups[frequency_bin(counts[test[position]])].append(position)
    selected={name:sorted(map(int,np.random.default_rng(19441+i).choice(values,min(128,len(values)),replace=False)))
              for i,(name,values) in enumerate(groups.items())}
    stages=[main_summary['protocol']['stages'][0],main_summary['protocol']['stages'][-1]]
    configs=dict(recent_only=RecallConfig(candidate_count=0,candidate_budget=0.),
                 persistent8_weak=RecallConfig(candidate_count=8,candidate_budget=.25))
    paths=[Path(__file__),train_path,test_path,HERE/'expanded_scale_v1/summary.json',
           HERE/'run_candidate_budget.py',HERE/'persistent_candidates.py',HERE/'run_expanded_scale.py']
    protocol=dict(reason='Added after natural-frequency evaluation contained no final-frequency 1-9 targets; explicit diagnostic, no tuning',
        groups_defined_by='final full training character count, held fixed across both checkpoints',
        available_per_group={k:len(v) for k,v in groups.items()},positions=selected,stages=stages,
        configs={k:asdict(v) for k,v in configs.items()},excluded_test_positions=sorted(excluded),
        overall_accuracy='not reported; deliberately non-natural frequency distribution',
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    OUT.mkdir()
    dump(OUT/'frozen_protocol.json',protocol)
    results={}
    locations=sorted(p for group in selected.values() for p in group)
    for stage in stages:
        graph=CharacterHebb.load(HERE/f'expanded_scale_v1/model_{stage}.npz')
        before=graph.weight_hash()
        for name,cfg in configs.items():
            score=evaluate(graph,test,locations,cfg)
            by_position={r['position']:r for r in score['rows']}
            from run_experiment import summarize
            group_results={group:dict(metrics=summarize([by_position[p] for p in values]),
                rows=[by_position[p]|{'final_training_frequency':counts[test[p]]} for p in values])
                for group,values in selected.items() if values}
            key=f'{stage}_{name}'
            dump(OUT/(key+'.json'),dict(groups=group_results,weight_sha256=before,weights_unchanged=True))
            results[key]={g:r['metrics'] for g,r in group_results.items()}
            print(json.dumps(dict(condition=key,groups=results[key]),ensure_ascii=False),flush=True)
        assert graph.weight_hash()==before
    dump(OUT/'summary.json',dict(completed=True,protocol=protocol,results=results))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
