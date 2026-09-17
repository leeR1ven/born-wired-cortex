"""Does a better growth rule convert the extra text into accuracy?

The published rule peaked at 600 million characters and then fell, over a stream
extended to 1,435,795,207 characters. That measurement was made with the rule
fixed. Here the same extension is trained with the stronger growth rule that won
the dose-response comparison, so the question 'is more text useful' is asked
again after the learning side has been improved.

Two probe sets are used. The first is the identical 9,000 positions used by the
published series, so the numbers are directly comparable. The second is an
independent draw from the same held-out split with a different seed, because the
growth exponent was chosen by looking at the first set, and a choice made on a
measurement should not be confirmed only by that measurement.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'instinct_bias_20260913'))

from hebb_text import CharacterHebb, Config
from fast_hebb import make_lookup, encode
from fast_hebb_power import train_ids_power
from run_modern_scale import DATA, read, file_hash, weight_hash
from run_power_audit import build_probes, hits_for, edge_stats, wilson, mcnemar, KINDS
from run_extended_scale import extended_stream
from run_weight_rule_dose import frequency_table

OUT = HERE / 'weight_rule_scale_v1'
POWER = 3
EXTENDED = HERE / 'extended_scale_v1' / 'summary.json'
DOSE = HERE / 'weight_rule_dose_v1' / 'summary.json'
STAGES = (600_000_000, 1_000_000_000, 1_435_795_207)
CONFIRM_SEED = 20260915


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite evidence')
    say = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    extended = json.loads(EXTENDED.read_text(encoding='utf-8'))
    dose = json.loads(DOSE.read_text(encoding='utf-8'))
    assert extended['completed'] and dose['completed']
    started = time.perf_counter()
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    records, probe_counts = build_probes()
    confirm, confirm_counts = build_probes(seed=CONFIRM_SEED)
    say(dict(probes=probe_counts, confirmation_probes=confirm_counts))
    OUT.mkdir()
    out = dict(completed=True, power=POWER, stages=list(STAGES), probe_counts=probe_counts,
               confirmation=dict(seed=CONFIRM_SEED, counts=confirm_counts,
                                 note='Independent draw over the same held-out split; overlap with '
                                      'the primary draw is possible because both sample the same documents.'),
               published=dict(diminishing_plain={s: extended['stages'][str(s)]['metrics']['all']['hits'] for s in STAGES},
                              row05_source='normalised_extended_results.json'),
               source_sha256={p: file_hash(HERE / p) for p in
                              ('run_weight_rule_scale.py', 'fast_hebb_power.py',
                               'run_extended_scale.py', 'run_weight_rule_dose.py')},
               variants={})
    published_row05 = json.loads((HERE / 'normalised_extended_results.json').read_text(encoding='utf-8'))
    out['published']['diminishing_row05'] = {s: published_row05['records'][str(s)]['row05']['total'] for s in STAGES}

    graph = CharacterHebb(vocabulary, Config(trace_decay=.25))
    lookup = make_lookup(vocabulary)
    counts = np.zeros(len(vocabulary), np.int64)
    kinds = np.array([r['kind'] for r in records])
    confirm_kinds = np.array([r['kind'] for r in confirm])
    pending = list(STAGES)
    stage_index = 0
    consumed = Counter()
    last_print = time.perf_counter()
    for source, text in extended_stream():
        ids = encode(text, lookup)
        offset = 0
        while offset < len(ids):
            take = len(ids) - offset
            if stage_index < len(pending):
                take = min(take, pending[stage_index] - graph.learned_characters)
            train_ids_power(graph, ids[offset:offset + take], POWER)
            counts += np.bincount(ids[offset:offset + take], minlength=len(vocabulary))
            consumed[source] += take
            offset += take
            if stage_index < len(pending) and graph.learned_characters == pending[stage_index]:
                stage = pending[stage_index]
                path = OUT / f'weights_{stage}.npy'
                np.save(path, graph.weights, allow_pickle=False)
                digest = weight_hash(graph)
                entry = dict(characters=stage, weight_sha256=digest,
                             source_characters=dict(consumed),
                             seconds_since_start=round(time.perf_counter() - started, 1))
                rowsum = graph.weights.sum(axis=1, dtype=np.float64)
                for label in ('plain', 'row05'):
                    vector = hits_for(graph, records, rowsum, label)
                    per = {k: int(vector[kinds == k].sum()) for k in KINDS}
                    entry[label] = dict(top1=per, total=int(vector.sum()),
                                        wilson={k: [round(x, 4) for x in wilson(per[k], int((kinds == k).sum()))]
                                                for k in KINDS},
                                        overall_wilson=[round(x, 4) for x in wilson(int(vector.sum()), len(vector))],
                                        by_frequency=frequency_table(records, counts, vector, graph.ids),
                                        hits=[int(x) for x in vector])
                    say(dict(stage=stage, readout=label, top1=per, total=int(vector.sum())))
                vector = hits_for(graph, confirm, rowsum, 'plain')
                per = {k: int(vector[confirm_kinds == k].sum()) for k in KINDS}
                entry['confirmation_plain'] = dict(top1=per, total=int(vector.sum()),
                                                   n=len(confirm),
                                                   wilson={k: [round(x, 4) for x in wilson(per[k], int((confirm_kinds == k).sum()))]
                                                           for k in KINDS},
                                                   hits=[int(x) for x in vector])
                say(dict(stage=stage, readout='confirmation_plain', top1=per, total=int(vector.sum())))
                assert weight_hash(graph) == digest, 'recall changed the weights'
                del rowsum
                entry['edges'] = edge_stats(path)
                say(dict(stage=stage, edges=dict(edges=entry['edges']['edges'],
                                                 share_edges_ge_1=round(entry['edges']['share_edges_ge_1'], 4),
                                                 top1pct_mass=round(entry['edges']['top1pct_mass_share'], 4))))
                out['variants'][str(stage)] = entry
                stage_index += 1
            if time.perf_counter() - last_print > 30:
                say(dict(training_characters=graph.learned_characters,
                         elapsed=round(time.perf_counter() - started, 1)))
                last_print = time.perf_counter()
    assert graph.learned_characters == STAGES[-1], graph.learned_characters
    # Counts are rule-independent, so this proves the training stream is identical
    # to the published extended run rather than merely similar.
    reference_counts = np.load(HERE / 'extended_scale_v1' / f'model_{STAGES[-1]}' / 'character_counts.npy')
    assert np.array_equal(counts, reference_counts), 'training stream differs from the published extension'
    say(dict(stream_identical_to_published_extension=True))
    for stage in STAGES:
        entry = out['variants'][str(stage)]
        base_plain = extended['stages'][str(stage)]['metrics']['all']['hits']
        base_row05 = out['published']['diminishing_row05'][stage]
        entry['versus_published_diminishing'] = dict(
            plain_hits=entry['plain']['total'], plain_published=base_plain,
            plain_gain=entry['plain']['total'] - base_plain,
            row05_hits=entry['row05']['total'], row05_published=base_row05,
            row05_gain=entry['row05']['total'] - base_row05,
            edges_published=published_row05['edges'][str(stage)]['edges'],
            top1pct_mass_published=round(published_row05['edges'][str(stage)]['top1pct_mass_share'], 4))
        say(dict(stage=stage, versus_published=entry['versus_published_diminishing']))
    out['elapsed_seconds'] = round(time.perf_counter() - started, 1)
    (OUT / 'summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    say(dict(COMPLETE=True, elapsed_seconds=out['elapsed_seconds']))


if __name__ == '__main__':
    main()