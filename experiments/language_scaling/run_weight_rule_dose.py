"""Dose-response of the weight-growth rule.

The published rule already makes a connection that is already strong grow more
slowly than a weak one: delta = learning_rate * amplitude / (1 + old). The open
question is whether going further helps, or whether the rule as published is
already past the useful point. Four rules are trained on the identical stream,
for the identical 344,787,049 characters, and measured on the identical 9,000
held-out positions and the same two readouts.

    linear        delta = lr * amplitude                     (no suppression)
    diminishing   delta = lr * amplitude / (1 + w)            (published)
    power2        delta = lr * amplitude / (1 + w)**2         (much stronger)
    ceiling1      delta = lr * amplitude * (1 - w/1.0)        (hard cap at 1)

Every number is measured; the published condition is asserted against its
published weight digest rather than assumed to reproduce.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'instinct_bias_20260913'))

from hebb_text import CharacterHebb, Config
from fast_hebb import train_ids, make_lookup, encode
from fast_hebb_power import train_ids_power
from run_modern_scale import DATA, read, file_hash, training_stream, weight_hash
from run_power_audit import build_probes, hits_for, edge_stats, wilson, mcnemar, KINDS

TARGET = 344_787_049
OUT = HERE / 'weight_rule_dose_v1'
PUBLISHED = HERE / 'modern_scale_v1' / 'model_344787049' / 'metadata.json'

VARIANTS = (
    ('linear', dict(growth='linear')),
    ('diminishing', dict(growth='diminishing')),
    ('power2', dict(power=2)),
    ('ceiling1', dict(growth='saturating', ceiling=1.0)),
)

def bin_label(count):
    if count == 0:
        return '0'
    if count < 10:
        return '1-9'
    if count < 100:
        return '10-99'
    if count < 1000:
        return '100-999'
    if count < 10000:
        return '1000-9999'
    return '10000+'


def train_variant(name, spec, say):
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    settings = dict(trace_decay=.25)
    settings.update({k: v for k, v in spec.items() if k != 'power'})
    graph = CharacterHebb(vocabulary, Config(**settings))
    lookup = make_lookup(vocabulary)
    counts = np.zeros(len(vocabulary), np.int64)
    started = time.perf_counter()
    last = started
    for source, text in training_stream():
        ids = encode(text, lookup)
        offset = 0
        while offset < len(ids) and graph.learned_characters < TARGET:
            end = min(len(ids), offset + TARGET - graph.learned_characters)
            chunk = ids[offset:end]
            if 'power' in spec:
                train_ids_power(graph, chunk, spec['power'])
            else:
                train_ids(graph, chunk)
            counts += np.bincount(chunk, minlength=len(vocabulary))
            offset = end
        if graph.learned_characters >= TARGET:
            break
        if time.perf_counter() - last > 30:
            say(dict(variant=name, training_characters=graph.learned_characters,
                     elapsed=round(time.perf_counter() - started, 1)))
            last = time.perf_counter()
    assert graph.learned_characters == TARGET, graph.learned_characters
    return graph, counts, time.perf_counter() - started


def frequency_table(records, counts, hits, ids):
    table = {}
    for record, hit in zip(records, hits):
        target = ids.get(record['target'])
        label = 'unallocated' if target is None else bin_label(int(counts[target]))
        entry = table.setdefault(label, dict(n=0, hits=0))
        entry['n'] += 1
        entry['hits'] += int(hit)
    for entry in table.values():
        entry['rate'] = entry['hits'] / entry['n']
    return table


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite evidence')
    published = json.loads(PUBLISHED.read_text(encoding='utf-8'))
    assert published['learned_characters'] == TARGET
    published_sha = published['weight_sha256']

    say = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    say(dict(building_probes=True))
    records, probe_counts = build_probes()
    kinds = np.array([r['kind'] for r in records])
    say(dict(probes=probe_counts))
    OUT.mkdir()
    out = dict(completed=True, target_characters=TARGET, published_weight_sha256=published_sha,
               published_model=str(PUBLISHED), probe_counts=probe_counts,
               variants={name: dict(spec=spec) for name, spec in VARIANTS},
               source_sha256={p: file_hash(HERE / p) for p in
                              ('fast_hebb_power.py', 'run_weight_rule_dose.py', 'hebb_text.py',
                               'fast_hebb.py', 'run_modern_scale.py', 'instinct_bias_20260913/run_power_audit.py',
                               'instinct_bias_20260913/run_bias_v2.py')})
    counts_reference = None
    hits = {}
    for name, spec in VARIANTS:
        graph, counts, seconds = train_variant(name, spec, say)
        digest = weight_hash(graph)
        if counts_reference is None:
            counts_reference = counts.copy()
            np.save(OUT / 'character_counts.npy', counts)
        else:
            assert np.array_equal(counts, counts_reference), 'stream differs between variants'
        path = OUT / f'weights_{name}.npy'
        np.save(path, graph.weights, allow_pickle=False)
        say(dict(variant=name, trained=True, seconds=round(seconds, 1), weight_sha256=digest,
                 reproduces_published=(digest == published_sha) if name == 'diminishing' else None))
        assert weight_hash(graph) == digest
        if name == 'diminishing':
            assert digest == published_sha, (digest, published_sha)

        rowsum = graph.weights.sum(axis=1, dtype=np.float64)
        entry = dict(spec=spec, weight_sha256=digest, training_seconds=round(seconds, 1))
        vectors = {}
        for label in ('plain', 'row05'):
            t0 = time.perf_counter()
            vector = hits_for(graph, records, rowsum, label)
            vectors[label] = vector
            per = {k: int(vector[kinds == k].sum()) for k in KINDS}
            entry[label] = dict(top1=per, total=int(vector.sum()),
                                wilson={k: [round(x, 4) for x in wilson(per[k], int((kinds == k).sum()))]
                                        for k in KINDS},
                                overall_wilson=[round(x, 4) for x in wilson(int(vector.sum()), len(vector))],
                                by_frequency=frequency_table(records, counts, vector, graph.ids),
                                seconds=round(time.perf_counter() - t0, 1))
            say(dict(variant=name, readout=label, top1=per, total=int(vector.sum())))
        hits[name] = vectors
        assert weight_hash(graph) == digest, 'recall changed the weights'
        del graph.weights, rowsum, graph
        entry['edges'] = edge_stats(path)
        say(dict(variant=name, edges=dict(edges=entry['edges']['edges'],
                                          share_edges_ge_1=round(entry['edges']['share_edges_ge_1'], 4),
                                          top1pct_mass=round(entry['edges']['top1pct_mass_share'], 4),
                                          max_weight=round(entry['edges'].get('max_weight', float('nan')), 3)
                                          if 'max_weight' in entry['edges'] else None)))
        out['variants'][name] = entry

    out['paired_vs_diminishing'] = {}
    for name, _ in VARIANTS:
        if name == 'diminishing':
            continue
        for label in ('plain', 'row05'):
            base = hits['diminishing'][label]
            other = hits[name][label]
            block = {}
            for kind in (*KINDS, 'all'):
                mask = np.ones(len(base), bool) if kind == 'all' else (kinds == kind)
                only_base = int(((base == 1) & (other == 0) & mask).sum())
                only_other = int(((base == 0) & (other == 1) & mask).sum())
                block[kind] = dict(only_diminishing=only_base, only_other=only_other,
                                   p=round(mcnemar(only_base, only_other), 6), n=int(mask.sum()))
            out['paired_vs_diminishing'][f'{name}:{label}'] = block
            say(dict(comparison=f'{name}:{label}', all=block['all']))

    for name in hits:
        out['variants'][name]['plain']['hits'] = [int(x) for x in hits[name]['plain']]
        out['variants'][name]['row05']['hits'] = [int(x) for x in hits[name]['row05']]
    out['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    (OUT / 'summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    say(dict(COMPLETE=True, elapsed_seconds=out['elapsed_seconds']))


if __name__ == '__main__':
    main()