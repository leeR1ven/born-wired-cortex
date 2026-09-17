"""Stratified by target frequency: are rare continuations worse, and does the
growth rule change that?

The natural probe set is 97 percent targets from the most frequent bin, so it
cannot answer the question "does the network forget what it saw rarely". This
draws held-out positions whose target character belongs to a chosen frequency
band under the frozen training counts, and scores two frozen checkpoints from the
dose run on exactly those positions.

Frozen counts come from the dose run, so the bands are defined by how often a
character occurred in the training stream, not by anything measured on the probes.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'instinct_bias_20260913'))

from hebb_text import CharacterHebb, Config
from run_modern_scale import DATA, read, file_hash
from run_power_audit import hits_for, wilson, KINDS

DOSE = HERE / 'weight_rule_dose_v1'
OUT = HERE / 'rare_target_probe_v1'
BANDS = (('1-9', 1, 9), ('10-99', 10, 99), ('100-999', 100, 999),
         ('1000-9999', 1000, 9999), ('10000+', 10000, 10 ** 9))
PER_BAND = 128
SEED = 20260917
MODELS = (('diminishing', DOSE / 'weights_diminishing.npy'),
          ('power3', DOSE / 'weights_power3.npy'))


def band_of(count):
    for name, low, high in BANDS:
        if low <= count <= high:
            return name
    return None


def build_positions(counts, vocabulary):
    rng = random.Random(SEED)
    ids = {c: i for i, c in enumerate(vocabulary)}
    pools = {name: [] for name, _, _ in BANDS}
    seen = Counter()
    for line in (DATA / 'test.jsonl').open(encoding='utf-8'):
        doc = json.loads(line)
        text = ''.join(doc['turns'])
        for position in range(64, len(text)):
            index = ids.get(text[position])
            if index is None or not ('\u3400' <= text[position] <= '\u9fff'):
                continue
            name = band_of(int(counts[index]))
            if name is None:
                continue
            row = dict(kind=doc['source'], id=doc['id'], prefix=text[position - 64:position],
                       target=text[position], final_training_frequency=int(counts[index]))
            pool = pools[name]
            seen[name] += 1
            if len(pool) < PER_BAND:
                pool.append(row)
            else:
                j = rng.randrange(seen[name])
                if j < PER_BAND:
                    pool[j] = row
    return pools, seen


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite evidence')
    say = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    counts = np.load(DOSE / 'character_counts.npy')
    pools, seen = build_positions(counts, vocabulary)
    pools = {k: v for k, v in pools.items() if v}
    total = sum(len(v) for v in pools.values())
    say(dict(pools={k: len(v) for k, v in pools.items()}, eligible=dict(seen), total=total))
    out = dict(completed=True, seed=SEED, per_band=PER_BAND, pools={k: len(v) for k, v in pools.items()},
               eligible=dict(seen),
               band_definition='final training character count of the target character, from the dose run',
               source_sha256={p: file_hash(HERE / p) for p in ('run_rare_target_probe.py',)},
               models={})
    records = [r for name, _, _ in BANDS for r in pools.get(name, [])]
    bands = np.array([band_of(r['final_training_frequency']) for r in records])
    for name, path in MODELS:
        graph = CharacterHebb(vocabulary, Config(trace_decay=.25))
        graph.weights = np.load(path, allow_pickle=False)
        rowsum = graph.weights.sum(axis=1, dtype=np.float64)
        entry = {}
        for label in ('plain', 'row05'):
            vector = hits_for(graph, records, rowsum, label)
            per = {}
            for band, _, _ in BANDS:
                mask = bands == band
                if not mask.any():
                    continue
                hits = int(vector[mask].sum())
                low, high = wilson(hits, int(mask.sum()))
                per[band] = dict(hits=hits, n=int(mask.sum()), rate=round(hits / int(mask.sum()), 4),
                                 wilson=[round(low, 4), round(high, 4)])
            entry[label] = dict(by_band=per, total=int(vector.sum()),
                                hits=[int(x) for x in vector])
            say(dict(model=name, readout=label, by_band={k: v['hits'] for k, v in per.items()},
                     total=int(vector.sum())))
        out['models'][name] = entry
        del graph.weights, rowsum, graph
    # Paired comparison between the two rules, band by band.
    for label in ('plain', 'row05'):
        a = np.array(out['models']['diminishing'][label]['hits'], dtype=np.int8)
        b = np.array(out['models']['power3'][label]['hits'], dtype=np.int8)
        block = {}
        for band, _, _ in BANDS:
            mask = bands == band
            if not mask.any():
                continue
            only_a = int(((a == 1) & (b == 0) & mask).sum())
            only_b = int(((a == 0) & (b == 1) & mask).sum())
            block[band] = dict(only_diminishing=only_a, only_power3=only_b, n=int(mask.sum()))
        out.setdefault('paired', {})[label] = block
        say(dict(paired=label, **block))
    out['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    OUT.mkdir()
    (OUT / 'summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    say(dict(COMPLETE=True, elapsed_seconds=out['elapsed_seconds']))


if __name__ == '__main__':
    main()