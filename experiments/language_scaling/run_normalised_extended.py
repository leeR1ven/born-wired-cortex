"""Both readouts, and the edge audit, on the extended checkpoints.

Section 4.8 showed that dividing each source's contribution by the square root of
its total outgoing weight raises held-out accuracy without retraining, and
Section 4.11 measured the same correction at four exposures. The extended series
shows the plain readout reallocating accuracy towards encyclopedia text as more
of it is added. This measures both readouts, and the connection statistics, on
the checkpoints beyond the published range, on the same 9,000 held-out positions.
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
from run_modern_scale import read, DATA
from run_power_audit import build_probes, hits_for, edge_stats, wilson, KINDS

EXTENDED = HERE / 'extended_scale_v1'
FINAL = 1_435_795_207
STAGES = (600_000_000, 1_000_000_000, FINAL)


def main():
    started = time.perf_counter()
    target = HERE / 'normalised_extended_results.json'
    if target.exists():
        raise SystemExit('Refuse to overwrite evidence')
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    records, counts = build_probes()
    kinds = np.array([r['kind'] for r in records])
    out = dict(completed=True, stages=list(STAGES), probe_counts=counts, records={}, edges={})
    for stage in STAGES:
        path = EXTENDED / f'model_{stage}' / 'weights.npy'
        out['edges'][str(stage)] = edge_stats(path)
        graph = CharacterHebb(vocabulary, Config(trace_decay=.25))
        graph.weights = np.load(path, allow_pickle=False)
        rowsum = graph.weights.sum(axis=1, dtype=np.float64)
        entry = dict(stage=stage)
        plain = None
        for label, variant in (('plain', 'plain'), ('row05', 'row05')):
            hits = hits_for(graph, records, rowsum, variant)
            per = {k: int(hits[kinds == k].sum()) for k in KINDS}
            entry[label] = dict(top1=per, total=int(hits.sum()),
                                wilson={k: [round(x, 4) for x in wilson(per[k], int((kinds == k).sum()))]
                                        for k in KINDS},
                                overall_wilson=[round(x, 4) for x in wilson(int(hits.sum()), len(hits))])
            if label == 'plain':
                plain = hits
            else:
                from math import comb
                entry['row05']['vs_plain'] = {}
                for k in KINDS:
                    b = int(((plain == 1) & (hits == 0) & (kinds == k)).sum())
                    c = int(((plain == 0) & (hits == 1) & (kinds == k)).sum())
                    n = b + c
                    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
                    entry['row05']['vs_plain'][k] = dict(only_plain=b, only_row05=c, p=p)
        out['records'][str(stage)] = entry
        print(json.dumps(dict(stage=stage, plain=entry['plain']['top1'], row05=entry['row05']['top1'],
                              edges=out['edges'][str(stage)]['edges'],
                              share_edges_ge_1=round(out['edges'][str(stage)]['share_edges_ge_1'], 4),
                              top1pct_mass=round(out['edges'][str(stage)]['top1pct_mass_share'], 4)),
                         ensure_ascii=False), flush=True)
        del graph.weights, graph
    out['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()