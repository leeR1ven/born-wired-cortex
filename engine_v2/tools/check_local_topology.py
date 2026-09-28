import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.topology import make_edges, neighbour_edges

print('=== 1. 和 make_edges 逐条对齐吗 ===')
rng = np.random.default_rng(3)
for n, d, local, random in ((200, 2, 4, 2), (500, 3, 6, 2), (300, 3, 3, 3), (150, 1, 5, 1)):
    positions = rng.random((n, d))
    a_src, a_dst = make_edges(positions, local, random, seed=11)
    b_src, b_dst = neighbour_edges(positions, local, random, seed=11)
    same_src = np.array_equal(a_src, b_src)
    same_local = all(set(a_dst[a_src == s][:local]) == set(b_dst[b_src == s][:local]) for s in range(n))
    same_random = all(set(a_dst[a_src == s][local:]) == set(b_dst[b_src == s][local:]) for s in range(n))
    exact = np.array_equal(np.sort(a_dst), np.sort(b_dst))
    print('n=%-4d d=%d local=%d random=%d  src一致=%s 近邻集合一致=%s 随机集合一致=%s 整体一致=%s'
          % (n, d, local, random, same_src, same_local, same_random, exact))

print()
print('=== 2. 同样的点，格子法快多少、能到多大 ===')
for n, d in ((20000, 2), (20000, 3), (100000, 3), (400000, 3)):
    positions = rng.random((n, d))
    t0 = time.perf_counter()
    neighbour_edges(positions, 6, 2, seed=1)
    grid_time = time.perf_counter() - t0
    line = 'n=%-7d d=%d  格子法 %7.2f s' % (n, d, grid_time)
    if n <= 20000:
        t0 = time.perf_counter()
        make_edges(positions, 6, 2, seed=1)
        line += '   暴力法 %7.2f s' % (time.perf_counter() - t0)
    else:
        estimated = grid_time*(n/20000.)**2
        line += '   暴力法估计要 {:,.0f} 秒（约 {:.0f} 小时）'.format(estimated, estimated/3600)
    print(line)
