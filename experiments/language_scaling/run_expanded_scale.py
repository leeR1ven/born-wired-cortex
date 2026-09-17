"""More real experience, fixed growth and recall rules, paired stages and prompts."""
from dataclasses import asdict
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hebb_text import CharacterHebb, Config
from persistent_candidates import RecallConfig
from run_persistent_candidates import positions
from run_candidate_budget import evaluate, trajectory
from run_experiment import summarize

HERE = Path(__file__).resolve().parent
OUT = HERE/'expanded_scale_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def frequency_bin(count):
    return '0' if count==0 else '1-9' if count<10 else '10-99' if count<100 else '100-999' if count<1000 else '1000+'


def context_diagnostics(training,frequencies,text,locations):
    # Post-hoc measurements ONLY. None of these lookups enter model activity.
    result = {}
    for position in locations:
        prefix=text[position-64:position]
        target=text[position]
        count=frequencies[target]
        result[position]=dict(training_frequency=count,frequency_bin=frequency_bin(count),
            seen_prefix3=prefix[-3:] in training,
            seen_prefix16=prefix[-16:] in training,
            seen_prefix16_plus_target=(prefix[-16:]+target) in training,
            seen_prefix64=prefix in training,
            seen_prefix64_plus_target=(prefix+target) in training)
    return result


def annotate(score,diagnostics):
    for row in score['rows']:
        row.update(diagnostics[row['position']])
    score['by_training_frequency']={name:summarize(rows) for name in ('0','1-9','10-99','100-999','1000+')
        if (rows:=[row for row in score['rows'] if row['frequency_bin']==name])}
    score['prefix_exposure_counts']={name:sum(row[name] for row in score['rows'])
        for name in ('seen_prefix3','seen_prefix16','seen_prefix16_plus_target','seen_prefix64','seen_prefix64_plus_target')}


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite experiment')
    start = time.perf_counter()
    data = HERE/'data_expanded_v1'
    manifest = read(data/'manifest.json')
    train = (data/'train.txt').read_text(encoding='utf-8')
    old_test = (HERE/'data_no_punct_v1/test.txt').read_text(encoding='utf-8')
    new_test = (data/'new_test.txt').read_text(encoding='utf-8')
    previous = read(HERE/'candidate_budget_v1/summary.json')
    excluded = set(previous['protocol']['excluded_test_positions']) | set(previous['protocol']['test_positions'])
    old_positions = positions(old_test,256,19431,excluded)
    new_positions = positions(new_test,256,19432)
    vocabulary = ''.join(sorted(set(train)))
    origin = manifest['old_training_prefix_characters']
    assert origin < len(train)
    stages = sorted(set([origin]+[n for n in (1000000,2000000,3000000) if n<len(train)]+[len(train)]))
    configs = dict(recent_only=RecallConfig(candidate_count=0,candidate_budget=0.),
        persistent8_weak=RecallConfig(candidate_count=8,candidate_budget=.25),
        persistent4_strong=RecallConfig(candidate_count=4,candidate_budget=1.))
    prompts = previous['protocol']['generation_prompts'][:4] + [new_test[p-32:p] for p in new_positions[::64]]
    cfg = Config(trace_decay=.25,growth='diminishing')
    sources = [Path(__file__),HERE/'prepare_expanded_corpus.py',data/'manifest.json',data/'train.txt',data/'new_test.txt',
        HERE/'hebb_text.py',HERE/'persistent_candidates.py',HERE/'run_candidate_budget.py',HERE/'run_persistent_candidates.py',
        HERE/'run_experiment.py',HERE/'candidate_budget_v1/summary.json',HERE/'data_no_punct_v1/test.txt',
        HERE/'short_context_v2/decay_0p25/model.npz']
    protocol = dict(stages=stages,train_characters=len(train),vocabulary_characters=len(vocabulary),
        graph_config=asdict(cfg),recall_configs={k:asdict(v) for k,v in configs.items()},
        old_test_positions=old_positions,new_test_positions=new_positions,old_excluded_positions=sorted(excluded),
        generation_prompts=prompts,evaluation='same true 64-character prefix and same positions at every stage; no test learning',
        selection='none; all stages and recall settings fixed before first scoring',
        corpus_scope='same-book held-out blocks; appended public-domain novels, not a broad modern-knowledge benchmark',
        vocabulary_preallocated_from_full_training_only=True,
        frequency_and_prefix_diagnostics='Scoring annotations only; no influence on learning or recall; exact substrings do not rule out shorter memorized fragments',
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.mkdir()
    dump(OUT/'frozen_protocol.json',protocol)
    graph = CharacterHebb(vocabulary,cfg)
    stage_results = []
    consumed = 0
    frequencies = Counter()
    for stage in stages:
        train_start = time.perf_counter()
        graph.train(train[consumed:stage])
        frequencies.update(train[consumed:stage])
        train_seconds = time.perf_counter()-train_start
        assert graph.clock==graph.learned_characters==stage
        before = (graph.weight_hash(),graph.activity.copy(),graph.clock)
        projected_old_exact = None
        if stage==origin:
            old = CharacterHebb.load(HERE/'short_context_v2/decay_0p25/model.npz')
            indices = np.array([graph.ids[ch] for ch in old.vocabulary],dtype=np.int64)
            assert np.array_equal(graph.weights[np.ix_(indices,indices)],old.weights)
            projected_old_exact = True
            del old
        record = dict(stage=stage,increment=stage-consumed,training_seconds=train_seconds,
            weight_sha256=graph.weight_hash(),stats=graph.stats(),old_weight_projection_exact=projected_old_exact,conditions={})
        degrees=np.count_nonzero(graph.weights,axis=1)
        per_character=[dict(character=ch,count=frequencies[ch],frequency_bin=frequency_bin(frequencies[ch]),
            outgoing_connections=int(degrees[i])) for i,ch in enumerate(vocabulary)]
        coverage=dict(character_counts=dict(Counter(row['frequency_bin'] for row in per_character)),
            han_character_counts=dict(Counter(row['frequency_bin'] for row in per_character if '\u3400'<=row['character']<='\u9fff')),
            token_counts={name:sum(row['count'] for row in per_character if row['frequency_bin']==name)
                          for name in ('0','1-9','10-99','100-999','1000+')},
            most_common=frequencies.most_common(20),per_character=per_character)
        dump(OUT/f'{stage}_coverage.json',coverage)
        record['coverage']={k:v for k,v in coverage.items() if k!='per_character'}
        diagnostic_old=context_diagnostics(train[:stage],frequencies,old_test,old_positions)
        diagnostic_new=context_diagnostics(train[:stage],frequencies,new_test,new_positions)
        for name,rcfg in configs.items():
            old_score = evaluate(graph,old_test,old_positions,rcfg)
            new_score = evaluate(graph,new_test,new_positions,rcfg)
            annotate(old_score,diagnostic_old)
            annotate(new_score,diagnostic_new)
            spoken = [trajectory(graph,p,rcfg) for p in prompts]
            silent = [trajectory(graph,p,rcfg,silent=True) for p in prompts] if rcfg.candidate_count else []
            result = dict(old_test=old_score,new_test=new_score,trajectories=spoken,silent_trajectories=silent)
            dump(OUT/f'{stage}_{name}.json',result)
            summary = dict(old_test=old_score['metrics'],new_test=new_score['metrics'],
                old_by_frequency=old_score['by_training_frequency'],new_by_frequency=new_score['by_training_frequency'],
                old_prefix_exposure_counts=old_score['prefix_exposure_counts'],new_prefix_exposure_counts=new_score['prefix_exposure_counts'],
                suffix_periods=[t['tail_period'] for t in spoken],
                old_mean_active_count=old_score['mean_active_count'],new_mean_active_count=new_score['mean_active_count'])
            record['conditions'][name] = summary
            print(json.dumps(dict(stage=stage,condition=name,old_test=summary['old_test'],new_test=summary['new_test'],suffix_periods=summary['suffix_periods']),ensure_ascii=False),flush=True)
        assert (graph.weight_hash(),graph.activity,graph.clock)==before
        graph.save(OUT/f'model_{stage}.npz')
        stage_results.append(record)
        dump(OUT/'summary.json',dict(completed=False,protocol=protocol,stages=stage_results,elapsed_seconds=time.perf_counter()-start))
        consumed=stage
    dump(OUT/'summary.json',dict(completed=True,protocol=protocol,stages=stage_results,elapsed_seconds=time.perf_counter()-start))
    print('COMPLETE',len(train),flush=True)


if __name__=='__main__':
    main()
