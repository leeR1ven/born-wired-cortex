"""Frozen factorial probe: data volume, weight growth, persistent top-K activity."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hebb_text import CharacterHebb, Config
from persistent_candidates import PersistentRecall, RecallConfig
from run_experiment import prediction_record, summarize, generate

HERE = Path(__file__).resolve().parent
OUT = HERE / 'persistent_candidates_v1'
DATA = HERE / 'data_no_punct_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def positions(text, n, seed, excluded=()):
    blocked = set(excluded)
    candidates = [i for i in range(64, len(text)) if i not in blocked and '\u3400' <= text[i] <= '\u9fff']
    return sorted(map(int, np.random.default_rng(seed).choice(candidates, n, replace=False)))


def evaluate(graph, text, locations, count):
    # Fixed 64-character input prefixes, reset ONLY transient probe activity.
    before = graph.weight_hash()
    recall = PersistentRecall(graph, RecallConfig(candidate_count=count))
    rows = []
    for position in locations:
        recall.reset_probe()
        recall.warm(text[position-64:position])
        raw = recall.currents()
        row = prediction_record(graph, text, position, recall.probabilities(raw))
        activity = recall.activity_summary()
        row['persistent_count'] = activity['count']
        row['effective_count'] = activity['effective_count']
        row['total_activity'] = activity['total']
        rows.append(row)
    assert graph.weight_hash() == before
    return dict(metrics=summarize(rows), rows=rows, weights_unchanged=True,
        mean_persistent_count=float(np.mean([r['persistent_count'] for r in rows])),
        mean_effective_count=float(np.mean([r['effective_count'] for r in rows])))


def suffix_period(text, minimum_span=32):
    for period in range(1, 17):
        tail = text[-minimum_span:]
        if len(tail) == minimum_span and all(tail[i] == tail[i-period] for i in range(period, len(tail))):
            return period
    return None


def trajectory(graph, prompt, count, silent=False, steps=64):
    recall = PersistentRecall(graph, RecallConfig(candidate_count=count))
    recall.warm(prompt)
    rows = []
    emitted = ''
    for step in range(steps):
        raw = recall.currents()
        identity = int(np.argmax(raw))
        chosen = graph.vocabulary[identity] if np.any(raw > 0) else None
        activity = recall.activity_summary()
        current_top = np.argsort(-raw, kind='stable')[:8]
        row = dict(step=step, persistent=activity,
            candidate_currents=[dict(character=graph.vocabulary[int(i)], current=float(raw[i]))
                                for i in current_top if raw[i] > 0],
            emitted=chosen if not silent else None,
            strongest_candidate=chosen)
        # Silent mode deliberately does not feed the strongest candidate back.
        row['injected_candidates'] = recall.advance(None if silent else chosen, raw)
        row['next_persistent'] = recall.activity_summary()
        rows.append(row)
        if not silent and chosen is not None:
            emitted += chosen
    pairs = zip(rows, rows[1:])
    similarities = []
    for a, b in pairs:
        sa = {x['character'] for x in a['persistent']['top'][:8]}
        sb = {x['character'] for x in b['persistent']['top'][:8]}
        similarities.append(len(sa & sb)/max(1, len(sa | sb)))
    return dict(prompt=prompt, silent=silent, emitted=emitted if not silent else None,
        tail_period=suffix_period(emitted) if not silent else None,
        mean_active_count=float(np.mean([r['persistent']['count'] for r in rows])),
        mean_effective_count=float(np.mean([r['persistent']['effective_count'] for r in rows])),
        mean_top8_adjacent_jaccard=float(np.mean(similarities)), rows=rows)


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite existing evidence')
    start = time.perf_counter()
    train = (DATA/'train.txt').read_text(encoding='utf-8')
    validation = (DATA/'validation.txt').read_text(encoding='utf-8')
    test = (DATA/'test.txt').read_text(encoding='utf-8')
    vocabulary = ''.join(sorted(set(train)))
    excluded = set(read(HERE/'results_no_punct_v1/frozen_protocol.json')['test_positions'])
    excluded.update(read(DATA/'matched_positions.json')['new_positions'])
    excluded.update(read(HERE/'short_context_v2/frozen_protocol.json')['test_positions'])
    val_positions = positions(validation, 256, 19361)
    test_positions = positions(test, 512, 19362, excluded)
    old_examples = read(HERE/'short_context_v2/decay_0p25/test_and_context.json')['continuations']
    prompts = [e['prompt'] for e in old_examples] + [test[p-32:p] for p in test_positions[::128]]
    growths = dict(diminishing=Config(trace_decay=.25, growth='diminishing'),
                   cap4=Config(trace_decay=.25, growth='saturating', ceiling=4.))
    stages = (100000, len(train))
    configs = {f'{growth}_{stage}_k{k}':dict(growth=growth, stage=stage, k=k)
               for growth in growths for stage in stages for k in (0,4,8)}
    sources = [Path(__file__), HERE/'persistent_candidates.py', HERE/'hebb_text.py', HERE/'run_experiment.py',
               DATA/'train.txt', DATA/'validation.txt', DATA/'test.txt',
               HERE/'short_context_v2/frozen_protocol.json', HERE/'results_no_punct_v1/frozen_protocol.json',
               DATA/'matched_positions.json', HERE/'short_context_v2/decay_0p25/test_and_context.json']
    protocol = dict(version=1, configs=configs, growths={k:asdict(v) for k,v in growths.items()},
        recall_defaults=asdict(RecallConfig()), selection='minimum validation NLL among full-data k4/k8; no test-based reselection',
        vocabulary_from_training_only=True, train_characters=len(train), vocabulary_characters=len(vocabulary),
        validation_positions=val_positions, test_positions=test_positions, excluded_test_positions=sorted(excluded),
        evaluation='Fixed true preceding 64-character prefix, transient reset per probe, no learning; score target before presenting it',
        training='Real characters only, one continuous stream per growth rule; candidate activity changes recall only',
        generation_prompts=prompts, generation_steps=64, silent='No winning character reinjected after prompt; inspect activity groups, not sentences',
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.mkdir()
    dump(OUT/'frozen_protocol.json', protocol)
    checkpoints = {}
    validation_results = {}
    for growth, cfg in growths.items():
        graph = CharacterHebb(vocabulary, cfg)
        previous = 0
        for stage in stages:
            graph.train(train[previous:stage])
            assert graph.clock == graph.learned_characters == stage
            checkpoint = OUT/f'{growth}_{stage}.npz'
            graph.save(checkpoint)
            if growth == 'diminishing' and stage == len(train):
                old = CharacterHebb.load(HERE/'short_context_v2/decay_0p25/model.npz')
                assert graph.weight_hash() == old.weight_hash()
                del old
            positive = graph.weights[graph.weights > 0]
            checkpoints[f'{growth}_{stage}'] = dict(weight_sha256=graph.weight_hash(), stats=graph.stats(),
                quantiles={str(q):float(np.quantile(positive,q)) for q in (.5,.9,.99,.999,1.)})
            for count in (0,4,8):
                name = f'{growth}_{stage}_k{count}'
                scored = evaluate(graph, validation, val_positions, count)
                dump(OUT/f'{name}_validation.json', scored)
                validation_results[name] = {k:v for k,v in scored.items() if k != 'rows'}
                print(json.dumps(dict(validation=name, **validation_results[name]), ensure_ascii=False), flush=True)
            previous = stage
        del graph
    eligible = [name for name,cfg in configs.items() if cfg['stage'] == len(train) and cfg['k'] > 0]
    selected = min(eligible, key=lambda name: validation_results[name]['metrics']['mean_nll_known'])
    choice = configs[selected]
    test_names = list(dict.fromkeys([
        f'{g}_{len(train)}_k{k}' for g in growths for k in (0,choice['k'])] +
        [f'{choice["growth"]}_100000_k{k}' for k in (0,choice['k'])]))
    selection = dict(selected=selected, test_names=test_names, validation_results=validation_results, test_scored_yet=False)
    dump(OUT/'selection.json', selection)
    selection_hash = hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    test_results = {}
    for name in test_names:
        cfg = configs[name]
        graph = CharacterHebb.load(OUT/f'{cfg["growth"]}_{cfg["stage"]}.npz')
        before = graph.weight_hash()
        scored = evaluate(graph, test, test_positions, cfg['k'])
        paths = [trajectory(graph, prompt, cfg['k']) for prompt in prompts]
        silent_paths = [trajectory(graph, prompt, cfg['k'], silent=True) for prompt in prompts] if cfg['k'] else []
        if not cfg['k']:
            for prompt, path in zip(prompts, paths):
                assert generate(graph, prompt, length=64) == path['emitted']
        assert graph.weight_hash() == before
        result = dict(score=scored, trajectories=paths, silent_trajectories=silent_paths, weights_unchanged=True)
        dump(OUT/f'{name}_test.json', result)
        test_results[name] = dict(metrics=scored['metrics'], mean_persistent_count=scored['mean_persistent_count'],
            mean_effective_count=scored['mean_effective_count'],
            suffix_periods=[p['tail_period'] for p in paths],
            distinct_final_silent_top8=len({''.join(x['character'] for x in p['rows'][-1]['persistent']['top'][:8]) for p in silent_paths}) if silent_paths else None)
        print(json.dumps(dict(test=name, **test_results[name]), ensure_ascii=False), flush=True)
        del graph
    assert selection_hash == hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    dump(OUT/'summary.json', dict(completed=True, protocol=protocol, checkpoints=checkpoints,
        selected=selected, validation_results=validation_results, test_results=test_results,
        selection_sha256=selection_hash, elapsed_seconds=time.perf_counter()-start))
    print('COMPLETE', selected, flush=True)


if __name__ == '__main__':
    main()
