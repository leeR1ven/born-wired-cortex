"""The published single-pass character stream, continued with 4.3 times more text.

Held fixed to modern_scale_v1: the frozen 19,168-unit vocabulary, the learning
rule, the alternation of approximately 100k-character dialogue blocks with
article blocks scaled by the source character ratio, one continuous trace and
clock, a single pass, no epochs, no synthetic boundaries, no test learning.

The only change is that after the two v1 sources are exhausted the stream
continues with additional non-synthetic encyclopedia text. The first 344,787,049
characters are therefore character-for-character the published stream, and the
five checkpoints up to that point must reproduce the published weight digests
exactly. That equality is asserted here rather than assumed.
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
from fast_hebb import train_ids, make_lookup, encode
from persistent_candidates import PersistentRecall, RecallConfig
from run_modern_scale import (read, dump, file_hash, weight_hash, save, stats,
                              text_chunks, DATA, OUT as V1_OUT)
from run_power_audit import build_probes, wilson

OUT = HERE / 'extended_scale_v1'
EXTRA = HERE / 'modern_corpus_v2' / 'extra'
MAIN = RecallConfig(candidate_count=8, candidate_budget=.25)
VERIFY_STAGES = (3_000_000, 30_000_000, 100_000_000, 300_000_000, 344_787_049)
SAVE_STAGES = (600_000_000, 1_000_000_000)
EXTRA_SOURCES = ('zhwiki', 'wikicn', 'wikitxt')


def extended_stream():
    """The v1 alternation, then the additional encyclopedia text, round-robin."""
    parts = read(DATA / 'manifest.json')['training_parts']
    dialogue_block = 100_000
    wiki_block = max(1, round(dialogue_block * parts['wiki_train'] / parts['dialogue_train']))
    streams = [iter(text_chunks(DATA / (kind + '.jsonl'), size)) for kind, size in
               (('dialogue_train', dialogue_block), ('wiki_train', wiki_block))]
    alive = [True, True]
    while any(alive):
        for index, stream in enumerate(streams):
            if alive[index]:
                text = next(stream, None)
                if text is None:
                    alive[index] = False
                else:
                    yield ('lccc_base' if index == 0 else 'wikipedia'), text
    streams = [iter(text_chunks(EXTRA / f'{name}.jsonl', wiki_block)) for name in EXTRA_SOURCES]
    alive = [True] * len(streams)
    while any(alive):
        for index, stream in enumerate(streams):
            if alive[index]:
                text = next(stream, None)
                if text is None:
                    alive[index] = False
                else:
                    yield EXTRA_SOURCES[index], text


def score(graph, records):
    recall = PersistentRecall(graph, MAIN)
    hits = Counter()
    total = Counter()
    for row in records:
        total[row['kind']] += 1
        recall.reset_probe()
        recall.warm(row['prefix'])
        raw = recall.currents()
        if graph.vocabulary[int(np.argmax(raw))] == row['target']:
            hits[row['kind']] += 1
    return hits, total


def measurement(graph, records):
    hits, total = score(graph, records)
    entry = {}
    for kind in ('lccc_base', 'wikipedia', 'reply_first'):
        low, high = wilson(hits[kind], total[kind])
        entry[kind] = dict(hits=hits[kind], n=total[kind],
                           rate=round(hits[kind] / total[kind], 5),
                           wilson=[round(low, 5), round(high, 5)])
    entry['all'] = dict(hits=sum(hits.values()), n=sum(total.values()))
    return entry


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite run evidence')
    started = time.perf_counter()
    manifest = read(DATA / 'manifest.json')
    extra = read(HERE / 'modern_corpus_v2' / 'manifest.json')
    vocabulary = read(DATA / 'vocabulary.json')['vocabulary']
    final = manifest['training_characters'] + extra['extra_characters']
    records, probe_counts = build_probes()
    OUT.mkdir()
    protocol = dict(
        stages=list(VERIFY_STAGES) + list(SAVE_STAGES) + [final],
        recall_config=dict(candidate_count=MAIN.candidate_count, candidate_budget=MAIN.candidate_budget),
        vocabulary=len(vocabulary), v1_training_characters=manifest['training_characters'],
        extra_characters=extra['extra_characters'], total_characters=final,
        ratio=round(final / manifest['training_characters'], 3), probes=probe_counts,
        training='One pass; the v1 alternation is reproduced exactly, then the additional '
                 'encyclopedia sources are consumed round-robin in the same block size',
        interpretation='Fixed-rule scale experiment on a simplified character graph, not the full PFC/PetBrain',
        extra_manifest_sha256=file_hash(HERE / 'modern_corpus_v2' / 'manifest.json'))
    dump(OUT / 'protocol.json', protocol)
    graph = CharacterHebb(vocabulary, Config(trace_decay=.25))
    lookup = make_lookup(vocabulary)
    counts = np.zeros(len(vocabulary), np.int64)
    pending = list(VERIFY_STAGES) + list(SAVE_STAGES)
    stage_index = 0
    consumed = Counter()
    stream_hash = hashlib.sha256()
    summaries = {}
    last_print = time.perf_counter()
    for source, text in extended_stream():
        ids = encode(text, lookup)
        offset = 0
        while offset < len(ids):
            if stage_index < len(pending):
                take = min(len(ids) - offset, pending[stage_index] - graph.learned_characters)
            else:
                take = len(ids) - offset
            train_ids(graph, ids[offset:offset + take])
            counts += np.bincount(ids[offset:offset + take], minlength=len(vocabulary))
            consumed[source] += take
            stream_hash.update(text[offset:offset + take].encode('utf-8'))
            offset += take
            if stage_index < len(pending) and graph.learned_characters == pending[stage_index]:
                stage = pending[stage_index]
                if stage in VERIFY_STAGES:
                    digest = weight_hash(graph)
                    reference = read(V1_OUT / f'model_{stage}' / 'metadata.json')['weight_sha256']
                    summaries[str(stage)] = dict(characters=stage, weight_sha256=digest,
                                                 matches_published=digest == reference)
                    print(json.dumps(dict(stage=stage, reproduced=digest == reference,
                                          stream_sha256=stream_hash.hexdigest()), ensure_ascii=False), flush=True)
                else:
                    meta = save(graph, OUT / f'model_{stage}', counts)
                    entry = dict(characters=stage, source_characters=dict(consumed),
                                 stream_sha256=stream_hash.hexdigest(),
                                 weight_sha256=meta['weight_sha256'], stats=stats(graph, counts),
                                 metrics=measurement(graph, records),
                                 seconds_since_start=round(time.perf_counter() - started, 1))
                    dump(OUT / f'stage_{stage}.json', entry)
                    summaries[str(stage)] = dict(characters=stage, weight_sha256=meta['weight_sha256'],
                                                 metrics=entry['metrics'])
                    print(json.dumps(dict(stage=stage, metrics=entry['metrics'],
                                          seconds=entry['seconds_since_start']), ensure_ascii=False), flush=True)
                stage_index += 1
            if time.perf_counter() - last_print > 30:
                print(json.dumps(dict(training_characters=graph.learned_characters,
                                      elapsed=round(time.perf_counter() - started, 1))), flush=True)
                last_print = time.perf_counter()
    assert graph.learned_characters == final, (graph.learned_characters, final)
    assert all(summaries[str(s)]['matches_published'] for s in VERIFY_STAGES), 'v1 prefix differs'
    meta = save(graph, OUT / f'model_{final}', counts)
    entry = dict(characters=final, source_characters=dict(consumed),
                 stream_sha256=stream_hash.hexdigest(), weight_sha256=meta['weight_sha256'],
                 stats=stats(graph, counts), metrics=measurement(graph, records),
                 seconds_since_start=round(time.perf_counter() - started, 1))
    dump(OUT / f'stage_{final}.json', entry)
    summaries[str(final)] = dict(characters=final, weight_sha256=meta['weight_sha256'],
                                 metrics=entry['metrics'])
    print(json.dumps(dict(stage=final, metrics=entry['metrics']), ensure_ascii=False), flush=True)
    dump(OUT / 'summary.json', dict(completed=True, training_characters=final, stages=summaries,
                                    reproduction='Every checkpoint up to 344,787,049 characters '
                                                'reproduces the published weight digest exactly',
                                    elapsed_seconds=round(time.perf_counter() - started, 1),
                                    no_test_learning=True))
    print('COMPLETE', final, flush=True)


if __name__ == '__main__':
    main()