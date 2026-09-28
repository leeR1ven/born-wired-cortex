"""每一步的耗时到底是跟细胞数走，还是跟连接数走？"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.adaptive import AdaptiveNetwork

def build(n, out_degree):
    rng = np.random.default_rng(0)
    e = n*out_degree
    src = np.repeat(np.arange(n, dtype=np.int64), out_degree)
    shift = rng.integers(0, max(1, n - out_degree), (n, 1))
    offsets = np.arange(1, out_degree + 1, dtype=np.int64)
    dst = ((np.arange(n, dtype=np.int64)[:, None] + shift + offsets) % n).ravel()
    signs = np.where(rng.random(n) < .8, 1., -1.)
    syn = BoundedSynapses(src, dst, np.full(e, .05), signs, n, w_max=1., budgets=40., plasticity=0.)
    net = AdaptiveNetwork(syn, tau=.02, adaptation_gain=0., bias=np.zeros(n))
    return syn, net, e

print('%-10s %-10s %-9s %12s %12s %10s' % ('细胞', '连接', '出度', '数组占用MB', '每步ms', '可跑Hz'))
for n, d in ((49000, 4), (49000, 8), (200000, 4), (200000, 10), (500000, 4), (1000000, 2)):
    syn, net, e = build(n, d)
    ext = np.zeros(n); ext[:max(1, n//100)] = 1.
    for _ in range(3): net.step(ext, dt=.01, learn=False)
    t0 = time.perf_counter()
    for _ in range(10): net.step(ext, dt=.01, learn=False)
    ms = (time.perf_counter()-t0)/10*1000
    parts = []
    for obj in (syn, net):
        for name in dir(obj):
            value = getattr(obj, name, None)
            if isinstance(value, np.ndarray) and value.size > 1000:
                parts.append(value.nbytes)
        parts.append(0)
    mb = sum(dict.fromkeys(parts) if False else []) or sum(
        a.nbytes for obj in (syn, net) for name in dir(obj)
        if isinstance((a := getattr(obj, name, None)), np.ndarray) and a.size > 1000)/1e6
    print('%-10d %-10d %-9d %12.1f %12.2f %10.1f' % (n, e, d, mb, ms, 1000./ms))
    del syn, net
