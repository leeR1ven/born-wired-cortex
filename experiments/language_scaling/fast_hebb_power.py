"""Weight growth with a tunable exponent, otherwise identical to fast_hebb.

fast_hebb.update implements three growth modes. This module keeps the same trace
bookkeeping and the same float32 arithmetic, and changes only the growth term:

    delta = learning_rate * amplitude / (1 + old_weight) ** power

power = 1 reproduces the published 'diminishing' rule bit for bit, which the
verification below asserts rather than assumes.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def update_power(weights, observations, sources, amplitudes, active, decay, floor,
                 learning_rate, power):
    events = 0
    for target in observations:
        for k in range(active):
            old = weights[sources[k], target]
            delta = np.float32(np.float32(learning_rate) * np.float32(amplitudes[k]))
            denominator = np.float32(np.float32(1.) + old)
            if power == 1:
                delta = np.float32(delta / denominator)
            elif power == 2:
                delta = np.float32(delta / np.float32(denominator * denominator))
            else:
                delta = np.float32(delta / np.float32(denominator ** np.float32(power)))
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


def train_ids_power(graph, ids, power):
    cfg = graph.config
    capacity = 2
    a = 1.
    while a * cfg.trace_decay >= cfg.trace_floor:
        capacity += 1
        a *= cfg.trace_decay
    capacity = max(capacity, len(graph.activity) + 1)
    sources = np.zeros(capacity, np.int64)
    amplitudes = np.zeros(capacity, np.float64)
    for k, (i, a) in enumerate(graph.activity.items()):
        sources[k] = i
        amplitudes[k] = a
    ids = np.asarray(ids, dtype=np.int32)
    if ids.size and (ids.min() < 0 or ids.max() >= len(graph.vocabulary)):
        raise ValueError('Unknown training character')
    active, events = update_power(graph.weights, ids, sources, amplitudes, len(graph.activity),
                                  cfg.trace_decay, cfg.trace_floor, cfg.learning_rate, power)
    graph.activity = {int(sources[k]): float(amplitudes[k]) for k in range(active)}
    graph.clock += len(ids)
    graph.learned_characters += len(ids)
    graph.update_events += events


def verify():
    """power=1 must equal the published diminishing rule bit for bit."""
    import json
    from pathlib import Path
    from hebb_text import CharacterHebb, Config
    from fast_hebb import train_ids
    rng = np.random.default_rng(20260914)
    checks = []
    for decay in (0., .25, .8):
        vocabulary = '的一是人在这里你好世界为什么不知道甲乙丙丁0123\n'
        ids = rng.integers(0, len(vocabulary), 20000, dtype=np.int32)
        ids[300:400] = 0
        initial = (rng.random((len(vocabulary), len(vocabulary))) * 200).astype(np.float32)
        a = CharacterHebb(vocabulary, Config(trace_decay=decay, growth='diminishing'))
        b = CharacterHebb(vocabulary, Config(trace_decay=decay, growth='diminishing'))
        a.weights[:] = initial
        b.weights[:] = initial
        for chunk in np.array_split(ids, 9):
            train_ids(a, chunk)
            train_ids_power(b, chunk, 1)
            assert np.array_equal(a.weights, b.weights), ('power1 mismatch', decay,
                                                          float(np.max(abs(a.weights - b.weights))))
            assert list(a.activity.items()) == list(b.activity.items())
            assert (a.clock, a.learned_characters, a.update_events) == (b.clock, b.learned_characters, b.update_events)
        checks.append(dict(decay=decay, characters=int(len(ids)), identical=True, max_weight=float(a.weights.max())))
    result = dict(completed=True, power_1_reproduces_published_diminishing_rule=True, checks=checks)
    Path(__file__).with_name('fast_hebb_power_verification.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    verify()