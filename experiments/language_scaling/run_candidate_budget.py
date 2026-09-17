"""Separate frozen validation sweep of candidate feedback strength, same weights."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hebb_text import CharacterHebb
from persistent_candidates import PersistentRecall, RecallConfig
from run_experiment import prediction_record, summarize
from run_persistent_candidates import positions, suffix_period

HERE = Path(__file__).resolve().parent
OUT = HERE / 'candidate_budget_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def evaluate(graph, text, locations, cfg):
    recall = PersistentRecall(graph, cfg)
    rows = []
    for position in locations:
        recall.reset_probe()
        recall.warm(text[position-64:position])
        raw = recall.currents()
        row = prediction_record(graph, text, position, recall.probabilities(raw))
        row['activity'] = recall.activity_summary()
        rows.append(row)
    return dict(metrics=summarize(rows), rows=rows,
        mean_active_count=float(np.mean([r['activity']['count'] for r in rows])),
        mean_effective_count=float(np.mean([r['activity']['effective_count'] for r in rows])))


def trajectory(graph, prompt, cfg, silent=False):
    recall = PersistentRecall(graph, cfg)
    recall.warm(prompt)
    rows = []
    emitted = ''
    for step in range(64):
        raw = recall.currents()
        chosen = graph.vocabulary[int(np.argmax(raw))] if np.any(raw > 0) else None
        row = dict(step=step, persistent=recall.activity_summary(), strongest_candidate=chosen)
        row['injected_candidates'] = recall.advance(None if silent else chosen, raw)
        row['next_persistent'] = recall.activity_summary()
        rows.append(row)
        if not silent and chosen is not None:
            emitted += chosen
    return dict(prompt=prompt, silent=silent, emitted=None if silent else emitted,
        tail_period=None if silent else suffix_period(emitted), rows=rows)


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite evidence')
    start = time.perf_counter()
    prior = read(HERE/'persistent_candidates_v1/summary.json')
    growth = prior['protocol']['configs'][prior['selected']]['growth']
    stage = prior['protocol']['train_characters']
    model_path = HERE/f'persistent_candidates_v1/{growth}_{stage}.npz'
    graph = CharacterHebb.load(model_path)
    before = (graph.weight_hash(), graph.activity.copy(), graph.clock)
    validation_path = HERE/'data_no_punct_v1/validation.txt'
    test_path = HERE/'data_no_punct_v1/test.txt'
    validation = validation_path.read_text(encoding='utf-8')
    test = test_path.read_text(encoding='utf-8')
    excluded = set(prior['protocol']['excluded_test_positions']) | set(prior['protocol']['test_positions'])
    val_positions = positions(validation, 256, 19363)
    test_positions = positions(test, 512, 19364, excluded)
    configs = {f'k{k}_b{str(budget).replace(".","p")}':RecallConfig(candidate_count=k, candidate_budget=budget)
               for k in (4,8) for budget in (.25,.5,1.)}
    configs['baseline'] = RecallConfig(candidate_count=0, candidate_budget=0.)
    prompts = prior['protocol']['generation_prompts'][:4] + [test[p-32:p] for p in test_positions[::128]]
    sources = [Path(__file__), HERE/'persistent_candidates.py', HERE/'run_persistent_candidates.py',
               HERE/'hebb_text.py', HERE/'run_experiment.py', model_path,
               HERE/'persistent_candidates_v1/summary.json', validation_path, test_path]
    protocol = dict(configs={k:asdict(v) for k,v in configs.items()}, model=str(model_path.relative_to(HERE)),
        model_weight_sha256=before[0], validation_positions=val_positions, test_positions=test_positions,
        excluded_test_positions=sorted(excluded), generation_prompts=prompts,
        selection='minimum validation NLL among six persistent configurations; baseline excluded; freeze before new test',
        evaluation='True preceding 64-character prefix; no target input before prediction; no weight learning',
        reason='Follow-up to strong feedback in persistent_candidates_v1; tune recall budget only; no new training',
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.mkdir()
    dump(OUT/'frozen_protocol.json', protocol)
    validation_results = {}
    for name,cfg in configs.items():
        scored = evaluate(graph, validation, val_positions, cfg)
        dump(OUT/f'{name}_validation.json', scored)
        validation_results[name] = {k:v for k,v in scored.items() if k != 'rows'}
        print(json.dumps(dict(validation=name, **validation_results[name]), ensure_ascii=False), flush=True)
    eligible = [name for name in configs if name != 'baseline']
    selected = min(eligible, key=lambda name: validation_results[name]['metrics']['mean_nll_known'])
    strong = f'k{configs[selected].candidate_count}_b1p0'
    test_names = list(dict.fromkeys(['baseline', selected, strong]))
    dump(OUT/'selection.json', dict(selected=selected, test_names=test_names,
        validation_results=validation_results, test_scored_yet=False))
    selection_hash = hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    results = {}
    for name in test_names:
        cfg = configs[name]
        scored = evaluate(graph, test, test_positions, cfg)
        spoken = [trajectory(graph,p,cfg) for p in prompts]
        silent = [trajectory(graph,p,cfg,silent=True) for p in prompts] if cfg.candidate_count else []
        dump(OUT/f'{name}_test.json', dict(score=scored, trajectories=spoken, silent_trajectories=silent))
        results[name] = dict(metrics=scored['metrics'], mean_active_count=scored['mean_active_count'],
            mean_effective_count=scored['mean_effective_count'], suffix_periods=[t['tail_period'] for t in spoken])
        print(json.dumps(dict(test=name, **results[name]), ensure_ascii=False), flush=True)
    assert (graph.weight_hash(), graph.activity, graph.clock) == before
    assert selection_hash == hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    dump(OUT/'summary.json', dict(completed=True, protocol=protocol, selected=selected,
        validation_results=validation_results, test_results=results, selection_sha256=selection_hash,
        weights_unchanged=True, elapsed_seconds=time.perf_counter()-start))
    print('COMPLETE', selected, flush=True)


if __name__ == '__main__':
    main()
