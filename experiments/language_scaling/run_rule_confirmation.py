"""Confirmation draw for the dose-response headline comparison.

The growth exponent was chosen by looking at the primary 9,000 positions, so the
two rules that carry the headline claim are scored again on an independent draw
of 9,000 positions from the same held-out split.
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
from run_modern_scale import DATA, read, file_hash
from run_power_audit import build_probes, hits_for, wilson, KINDS

DOSE = HERE / 'weight_rule_dose_v1'
OUT = DOSE / 'confirmation.json'
SEED = 20260915
MODELS = (('diminishing', 'weights_diminishing.npy'),
          ('power2', 'weights_power2.npy'),
          ('power3', 'weights_power3.npy'))


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite evidence')
    say = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    records, counts = build_probes(seed=SEED)
    kinds = np.array([r['kind'] for r in records])
    out = dict(completed=True, seed=SEED, probe_counts=counts,
               note='Independent draw over the same held-out split as the primary 9,000 positions.',
               source_sha256={p: file_hash(HERE / p) for p in ('run_rule_confirmation.py',)},
               models={})
    for name, filename in MODELS:
        graph = CharacterHebb(vocabulary, Config(trace_decay=.25))
        graph.weights = np.load(DOSE / filename, allow_pickle=False)
        rowsum = graph.weights.sum(axis=1, dtype=np.float64)
        entry = {}
        for label in ('plain', 'row05'):
            vector = hits_for(graph, records, rowsum, label)
            per = {k: int(vector[kinds == k].sum()) for k in KINDS}
            entry[label] = dict(top1=per, total=int(vector.sum()), hits=[int(x) for x in vector],
                                wilson={k: [round(x, 4) for x in wilson(per[k], int((kinds == k).sum()))]
                                        for k in KINDS})
            say(dict(model=name, readout=label, top1=per, total=int(vector.sum())))
        out['models'][name] = entry
        del graph.weights, rowsum, graph
    a = np.array(out['models']['diminishing']['plain']['hits'], dtype=np.int8)
    b = np.array(out['models']['power3']['plain']['hits'], dtype=np.int8)
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    out['paired_power3_vs_diminishing_plain'] = dict(only_diminishing=only_a, only_power3=only_b)
    say(out['paired_power3_vs_diminishing_plain'])
    out['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    say(dict(COMPLETE=True))


if __name__ == '__main__':
    main()