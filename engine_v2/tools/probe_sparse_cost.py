"""点亮多少细胞，影响每一步耗时吗？同一张图，只改外部驱动覆盖多少细胞。"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.adaptive import AdaptiveNetwork

n, out_degree = 100000, 4
rng = np.random.default_rng(0)
src = np.repeat(np.arange(n, dtype=np.int64), out_degree)
shift = rng.integers(0, n - out_degree, (n, 1))
dst = ((np.arange(n, dtype=np.int64)[:, None] + shift
        + np.arange(1, out_degree + 1, dtype=np.int64)) % n).ravel()
syn = BoundedSynapses(src, dst, np.full(n*out_degree, .05),
                      np.where(rng.random(n) < .8, 1., -1.), n,
                      w_max=1., budgets=40., plasticity=0.)
net = AdaptiveNetwork(syn, tau=.02, adaptation_gain=0., bias=np.zeros(n))
ext = np.zeros(n)
print('%-14s %10s %14s %12s %12s' % ('外部驱动比例', '驱动细胞数', '点亮细胞数', '每步ms', '相对基准'))
base = None
for fraction in (0., .0001, .001, .01, .10, .50):
    ext[:] = 0.
    count = int(n*fraction)
    if count:
        ext[rng.choice(n, count, replace=False)] = 1.
    for _ in range(3): net.step(ext, dt=.01, learn=False)
    t0 = time.perf_counter()
    for _ in range(8): net.step(ext, dt=.01, learn=False)
    ms = (time.perf_counter() - t0)/8*1000
    lit = int((np.asarray(net.activity) > .01).sum())
    base = ms if base is None else base
    print('%-14s %10d %14d %12.2f %11.2fx' % ('%.4g%%' % (100*fraction), count, lit, ms, ms/base))
