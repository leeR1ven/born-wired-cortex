"""同一个活跃量，把细胞数放大10倍，每步耗时会不会跟着涨？

先固定细胞数、只改驱动多少个细胞（耗时应该跟着活跃量涨）；
再固定驱动细胞数、只改总细胞数（耗时应该基本不动）。
"""
import sys, time, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.sparse import SparseNetwork

DT, DEGREE, DRIVEN = .01, 4, 1000


def build(n, seed=0):
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n, dtype=np.int64), DEGREE)
    shift = rng.integers(0, max(1, n - DEGREE), (n, 1))
    dst = ((np.arange(n, dtype=np.int64)[:, None] + shift
            + np.arange(1, DEGREE + 1, dtype=np.int64)) % n).ravel()
    syn = BoundedSynapses(src, dst, rng.uniform(.02, .20, src.size),
                          np.where(rng.random(n) < .8, 1., -1.), n,
                          w_max=1., budgets=40., plasticity=0.)
    net = SparseNetwork(syn, tau=rng.uniform(.02, .12, n), adaptation_tau=.4,
                        adaptation_gain=rng.uniform(1., 3., n),
                        bias=rng.uniform(-.3, 0., n), cutoff=0.)
    return net


def time_it(net, cells, amounts, repeats=40, warmup=20):
    for _ in range(warmup):
        net.step_sparse((cells, amounts), dt=DT, learn=False)
    t0 = time.perf_counter()
    for _ in range(repeats):
        net.step_sparse((cells, amounts), dt=DT, learn=False)
    return (time.perf_counter() - t0)/repeats*1000, net.speaking


report = {}
print('%-10s %10s %12s %12s %14s' % ('细胞', '驱动细胞', '亮着总数', '每步ms', '每千个亮着/ms'))
for n in (100000, 200000, 400000, 800000):
    net = build(n)
    rng = np.random.default_rng(1)
    for count in (0, 100, 1000, 5000):
        cells = np.sort(rng.choice(n, count, replace=False)) if count else np.empty(0, dtype=np.int64)
        ms, lit = time_it(net, cells, np.ones(count))
        print('%-10d %10d %12d %12.2f %14.2f'
              % (n, count, lit, ms, 1000.*ms/max(lit, 1)))
        report['%d/%d' % (n, count)] = dict(ms=ms, lit=lit)
    del net

print()
print('=== 只放大细胞数，驱动细胞数固定 %d ===' % DRIVEN)
print('%-10s %12s %12s %16s' % ('细胞', '亮着总数', '每步ms', '相对10万细胞'))
reference = None
for n in (100000, 200000, 400000, 800000):
    net = build(n)
    rng = np.random.default_rng(1)
    cells = np.sort(rng.choice(n, DRIVEN, replace=False))
    ms, lit = time_it(net, cells, np.ones(DRIVEN))
    reference = ms if reference is None else reference
    print('%-10d %12d %12.2f %15.2fx' % (n, lit, ms, ms/reference))
    report['fixed/%d' % n] = dict(ms=ms, lit=lit)
    del net

artifact = ROOT/'artifacts'/'probe_sparse_scaling.json'
artifact.write_text(json.dumps(dict(dt=DT, degree=DEGREE, driven=DRIVEN, runs=report),
                               ensure_ascii=False, indent=2), encoding='utf-8')
print()
print('写到 %s' % artifact)
