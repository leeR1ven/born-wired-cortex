"""Compiled sequential updates, identical float32 arithmetic to CharacterHebb.

No count approximation, batching of weight changes, hidden phrases, or fastmath.
Only Python loop overhead is removed. Transient trace order survives chunks.
"""
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def update(weights, observations, sources, amplitudes, active, decay, floor,
           learning_rate, growth, ceiling):
    events = 0
    for target in observations:
        for k in range(active):
            old = weights[sources[k], target]
            delta = np.float32(np.float32(learning_rate) * np.float32(amplitudes[k]))
            if growth == 1:
                delta = np.float32(delta / np.float32(np.float32(1.) + old))
            elif growth == 2:
                fraction = np.float32(old / np.float32(ceiling))
                delta = np.float32(delta * max(np.float32(0.), np.float32(np.float32(1.) - fraction)))
            weights[sources[k], target] = np.float32(old + delta)
        events += active
        kept = 0
        found = -1
        for k in range(active):
            value = amplitudes[k] * decay
            if value >= floor:
                sources[kept] = sources[k]
                amplitudes[kept] = value
                if sources[kept] == target:
                    found = kept
                kept += 1
        if found >= 0:
            amplitudes[found] = 1.
        else:
            sources[kept] = target
            amplitudes[kept] = 1.
            kept += 1
        active = kept
    return active, events

def train_ids(graph, ids):
    cfg = graph.config
    capacity = 2
    a = 1.
    while a * cfg.trace_decay >= cfg.trace_floor:
        capacity += 1; a *= cfg.trace_decay
    capacity = max(capacity, len(graph.activity)+1)
    sources = np.zeros(capacity, np.int64)
    amplitudes = np.zeros(capacity, np.float64)
    for k,(i,a) in enumerate(graph.activity.items()):
        sources[k] = i; amplitudes[k] = a
    ids = np.asarray(ids, dtype=np.int32)
    if ids.size and (ids.min() < 0 or ids.max() >= len(graph.vocabulary)):
        raise ValueError('Unknown training character')
    active, events = update(graph.weights, ids, sources, amplitudes, len(graph.activity),
                            cfg.trace_decay, cfg.trace_floor, cfg.learning_rate,
                            {'linear':0,'diminishing':1,'saturating':2}[cfg.growth], cfg.ceiling)
    graph.activity = {int(sources[k]):float(amplitudes[k]) for k in range(active)}
    graph.clock += len(ids)
    graph.learned_characters += len(ids)
    graph.update_events += events

def make_lookup(vocabulary):
    lookup = np.full(0x110000, -1, np.int32)
    for i,c in enumerate(vocabulary): lookup[ord(c)] = i
    return lookup

def encode(text, lookup):
    return lookup[np.frombuffer(text.encode('utf-32-le'), dtype='<u4')]

def verify():
    from hebb_text import CharacterHebb, Config
    import time, json
    from pathlib import Path
    rng = np.random.default_rng(19450)
    checks = []
    for growth in ('linear','diminishing','saturating'):
        for decay in (0., .25, .8):
            cfg = Config(trace_decay=decay, growth=growth)
            vocab = '的一是人在这里你好世界为什么不知道甲乙丙丁0123\n'
            a,b = CharacterHebb(vocab,cfg),CharacterHebb(vocab,cfg)
            ids = rng.integers(0,len(vocab),10000,dtype=np.int32)
            # Include repeated-unit overwrites and large-weight rounding.
            ids[300:400] = 0
            initial = (rng.random(a.weights.shape)*200).astype(np.float32)
            a.weights[:] = initial; b.weights[:] = initial
            for chunk in np.array_split(ids,7):
                a.train(''.join(vocab[i] for i in chunk))
                train_ids(b,chunk)
                assert np.array_equal(a.weights,b.weights), (growth,decay,float(np.max(abs(a.weights-b.weights))))
                assert list(a.activity.items()) == list(b.activity.items())
                assert (a.clock,a.learned_characters,a.update_events)==(b.clock,b.learned_characters,b.update_events)
            checks.append(dict(growth=growth,decay=decay,characters=len(ids),exact=True))
    g=CharacterHebb(''.join(chr(0x4e00+i) for i in range(8000)),Config(trace_decay=.25))
    ids=rng.integers(0,8000,1000000,dtype=np.int32)
    t=time.perf_counter(); train_ids(g,ids); elapsed=time.perf_counter()-t
    result=dict(completed=True, checks=checks, numpy=np.__version__, benchmark_characters=len(ids),
                benchmark_seconds=elapsed, characters_per_second=len(ids)/elapsed)
    Path('fast_hebb_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__ == '__main__': verify()
