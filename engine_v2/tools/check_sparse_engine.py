"""新的稀疏执行引擎：算得和原来一样吗？快了没有？

第一部分并排跑两个引擎，比每一步的活动、以及收尾对齐后的整套状态。
第二部分量每步耗时，看它是跟"亮着多少细胞"走，还是跟"有多少细胞"走。
"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.adaptive import AdaptiveNetwork
from born_wired.sparse import SparseNetwork


def graph(n, degree, seed=0, positive_bias=.0):
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n, dtype=np.int64), degree)
    shift = rng.integers(0, max(1, n - degree), (n, 1))
    dst = ((np.arange(n, dtype=np.int64)[:, None] + shift
            + np.arange(1, degree + 1, dtype=np.int64)) % n).ravel()
    weights = rng.uniform(.02, .20, n*degree)
    signs = np.where(rng.random(n) < .8, 1., -1.)
    syn = BoundedSynapses(src, dst, weights, signs, n, w_max=1., budgets=40., plasticity=0.)
    tau = rng.uniform(.02, .12, n)
    gain = rng.uniform(1., 3., n)
    bias = rng.uniform(-.3, positive_bias, n)
    return syn, tau, gain, bias


print('=== 1. 两个引擎算的一样吗（400步；驱动前120个细胞60步）===')
print('%-8s %-8s %-9s %14s %14s %14s %10s' %
      ('细胞', '出度', '阈值', '每步活动最大差', '电压最大差', '适应最大差', '平均亮着'))
for n, degree in ((400, 6), (3000, 8)):
    syn, tau, gain, bias = graph(n, degree, seed=4, positive_bias=.02)
    drive = np.zeros(n); drive[:120] = 1.
    for cutoff in (0., 1e-3):
        exact = AdaptiveNetwork(syn, tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=bias)
        sparse = SparseNetwork(syn, tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=bias,
                               cutoff=cutoff)
        biggest, speaking = 0., []
        for step in range(400):
            ext = drive if step < 60 else np.zeros(n)
            a = np.asarray(exact.step(ext, dt=.01, learn=False)).copy()
            b = np.asarray(sparse.step(ext, dt=.01, learn=False)).copy()
            biggest = max(biggest, float(np.abs(a - b).max()))
            speaking.append(sparse.speaking)
        sparse.catch_up()
        print('%-8d %-8d %-9g %14.2e %14.2e %14.2e %10.0f'
              % (n, degree, cutoff, biggest,
                 float(np.abs(np.asarray(exact.voltage) - np.asarray(sparse.voltage)).max()),
                 float(np.abs(np.asarray(exact.adaptation) - np.asarray(sparse.adaptation)).max()),
                 np.mean(speaking)))

print()
print('=== 2. 每步耗时跟谁走（10万细胞，出度4）===')
n, degree = 100000, 4
syn, tau, gain, bias = graph(n, degree, seed=1, positive_bias=-.01)
dense = AdaptiveNetwork(syn, tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=bias)
sparse = SparseNetwork(syn, tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=bias,
                       cutoff=1e-3)
rng = np.random.default_rng(0)
print('%-10s %10s %12s %12s %14s' % ('驱动细胞', '亮着总数', '稀疏引擎ms', '原引擎ms', '外部驱动ms'))
base = None
for count in (0, 10, 100, 1000, 10000):
    cells = np.sort(rng.choice(n, count, replace=False)) if count else np.empty(0, dtype=np.int64)
    amounts = np.ones(count)
    full = np.zeros(n)
    if count:
        full[cells] = 1.
    for _ in range(30):
        sparse.step_sparse((cells, amounts), dt=.01, learn=False)
        dense.step(full, dt=.01, learn=False)
    t0 = time.perf_counter()
    for _ in range(40):
        sparse.step_sparse((cells, amounts), dt=.01, learn=False)
    sparse_ms = (time.perf_counter() - t0)/40*1000
    t0 = time.perf_counter()
    for _ in range(10):
        dense.step(full, dt=.01, learn=False)
    dense_ms = (time.perf_counter() - t0)/10*1000
    t0 = time.perf_counter()
    for _ in range(40):
        sparse.step(full, dt=.01, learn=False)
    dense_drive_ms = (time.perf_counter() - t0)/40*1000
    print('%-10d %10d %12.2f %12.2f %14.2f' % (count, sparse.speaking, sparse_ms, dense_ms,
                                              dense_drive_ms))
