"""Two controls for the weight-rule dose-response.

1. Scale control. The readout takes the largest sum of activity times weight, so a
   global rescaling of the whole matrix should leave every prediction unchanged.
   That is asserted numerically rather than argued, because it decides whether the
   gain from a stronger growth rule is structural or only a change of weight scale.

2. One further dose. power2 improved on the published rule; power3 asks whether
   that continues, so the comparison is a curve and not a single lucky setting.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'instinct_bias_20260913'))

from hebb_text import CharacterHebb, Config
from run_modern_scale import DATA, read, file_hash, weight_hash
from run_power_audit import build_probes, hits_for, edge_stats, wilson, mcnemar, KINDS
from run_weight_rule_dose import train_variant, frequency_table

OUT = HERE / 'weight_rule_dose_v1'
SCALE = 3.7


def main():
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['completed']
    say = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    records, probe_counts = build_probes()
    assert probe_counts == summary['probe_counts'], 'probe set changed'
    kinds = np.array([r['kind'] for r in records])
    out = dict(completed=True, scale_factor=SCALE, probe_counts=probe_counts,
               source_sha256={p: file_hash(HERE / p) for p in
                              ('run_weight_rule_dose_addendum.py', 'run_weight_rule_dose.py',
                               'fast_hebb_power.py')})

    # 1. Scale control on the published rule, on a spread subset, both readouts.
    subset = records[::7]
    graph = CharacterHebb(vocabulary, Config(trace_decay=.25))
    graph.weights = np.load(OUT / 'weights_diminishing.npy', allow_pickle=False)
    rowsum = graph.weights.sum(axis=1, dtype=np.float64)
    before = {label: hits_for(graph, subset, rowsum, label) for label in ('plain', 'row05')}
    digest_before = weight_hash(graph)
    graph.weights = np.ascontiguousarray(graph.weights.astype(np.float32) * np.float32(SCALE))
    rowsum_scaled = graph.weights.sum(axis=1, dtype=np.float64)
    after = {label: hits_for(graph, subset, rowsum_scaled, label) for label in ('plain', 'row05')}
    out['scale_control'] = dict(
        records=len(subset),
        identical={label: bool(np.array_equal(before[label], after[label])) for label in before},
        hits_before={label: int(before[label].sum()) for label in before},
        hits_after={label: int(after[label].sum()) for label in after},
        note='Predictions are invariant to a global rescaling of the learned matrix.')
    say(dict(scale_control=out['scale_control']))
    assert all(out['scale_control']['identical'].values()), 'readout is not scale-invariant'
    del graph.weights, rowsum, rowsum_scaled, graph

    # 2. One further dose, trained on the identical stream.
    graph, counts, seconds = train_variant('power3', dict(power=3), say)
    path = OUT / 'weights_power3.npy'
    np.save(path, graph.weights, allow_pickle=False)
    digest = weight_hash(graph)
    say(dict(variant='power3', trained=True, seconds=round(seconds, 1), weight_sha256=digest))
    rowsum = graph.weights.sum(axis=1, dtype=np.float64)
    entry = dict(spec=dict(power=3), weight_sha256=digest, training_seconds=round(seconds, 1))
    vectors = {}
    for label in ('plain', 'row05'):
        t0 = time.perf_counter()
        vector = hits_for(graph, records, rowsum, label)
        vectors[label] = vector
        per = {k: int(vector[kinds == k].sum()) for k in KINDS}
        entry[label] = dict(top1=per, total=int(vector.sum()),
                            wilson={k: [round(x, 4) for x in wilson(per[k], int((kinds == k).sum()))]
                                    for k in KINDS},
                            by_frequency=frequency_table(records, counts, vector, graph.ids),
                            seconds=round(time.perf_counter() - t0, 1))
        say(dict(variant='power3', readout=label, top1=per, total=int(vector.sum())))
    entry['edges'] = edge_stats(path)
    say(dict(variant='power3', edges=dict(edges=entry['edges']['edges'],
                                          share_edges_ge_1=round(entry['edges']['share_edges_ge_1'], 4),
                                          top1pct_mass=round(entry['edges']['top1pct_mass_share'], 4))))
    assert weight_hash(graph) == digest, 'recall changed the weights'
    count_check = np.load(OUT / 'character_counts.npy')
    assert np.array_equal(counts, count_check), 'stream differs from the dose run'
    del graph.weights, rowsum, graph

    def load_hits(name, label):
        return np.array(summary['variants'][name][label]['hits'], dtype=np.int8) if name != 'power3' \
            else vectors[label]

    entry['paired'] = {}
    for other in ('diminishing', 'power2'):
        for label in ('plain', 'row05'):
            a, b = load_hits(other, label), vectors[label]
            block = {}
            for kind in (*KINDS, 'all'):
                mask = np.ones(len(a), bool) if kind == 'all' else (kinds == kind)
                only_a = int(((a == 1) & (b == 0) & mask).sum())
                only_b = int(((a == 0) & (b == 1) & mask).sum())
                block[kind] = dict(only_other=only_a, only_power3=only_b,
                                   p=round(mcnemar(only_a, only_b), 6), n=int(mask.sum()))
            entry['paired'][f'{other}:{label}'] = block
            say(dict(comparison=f'power3 vs {other}', readout=label, all=block['all']))
    for label in ('plain', 'row05'):
        entry[label]['hits'] = [int(x) for x in vectors[label]]
    out['power3'] = entry
    out['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    (OUT / 'addendum.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    say(dict(COMPLETE=True, elapsed_seconds=out['elapsed_seconds']))


if __name__ == '__main__':
    main()