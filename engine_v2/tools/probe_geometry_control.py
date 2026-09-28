"""控制变量：点亮多少细胞，到底跟什么有关？跟"立体"有没有关系？

上一次那组对比没有控制变量。同样是 1000 个细胞，平面格子每边约 32 格，
立体格子每边只有 10 格：立体那组的"最近邻"在物理上远了将近 10 倍，一步兴奋
就扩散得更远、点亮得更多。那组差别是"邻居更远"造成的，不是"立体"造成的。

这次把能固定的都固定住：
  格子间距一样（都是 1/46 格）、驱动细胞数一样（离物体中心最近的 64 个）、
  步数、阈值、时间常数一样。每次只改一个因素，看点亮的细胞数怎么动。
格子用周期边界，所以每个细胞的接线完全相同，没有边缘差别。
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.adaptive import AdaptiveNetwork

SPACING = 1./46
STEPS, DT, ACTIVE = 30, .01, .05
DRIVEN, WEIGHT, RADIUS = 64, .10, 2**.5
PLACES = (0.25, 0.50, 0.75)
DEPTHS = (0.50, 0.75)


def lattice(shape):
    grid = np.indices(shape).reshape(len(shape), -1).T.astype(np.int64)
    return grid, grid*SPACING


def offsets_within(dimension, radius):
    reach = int(np.ceil(radius))
    found = [o for o in itertools.product(range(-reach, reach+1), repeat=dimension)
             if any(o) and float(np.linalg.norm(o)) <= radius + 1e-9]
    found.sort(key=lambda o: (float(np.linalg.norm(o)), o))
    return found


def build(shape, radius, degree, weight):
    grid, points = lattice(shape)
    cells = len(grid)
    strides = np.array([int(np.prod(shape[k+1:])) for k in range(len(shape))], dtype=np.int64)
    every = np.arange(cells, dtype=np.int64)
    offsets = offsets_within(len(shape), radius)[:degree]
    src = np.tile(every, len(offsets))
    dst = np.concatenate([((grid + np.asarray(o)) % np.asarray(shape)) @ strides
                          for o in offsets])
    synapses = BoundedSynapses(src, dst, np.full(src.size, weight), np.ones(cells), cells,
                               w_max=1., budgets=1e6, plasticity=0.)
    return synapses, points, cells


def present(synapses, points, centre, count, cells):
    external = np.zeros(cells)
    gap = np.linalg.norm(points - np.asarray(centre), axis=1)
    external[np.argsort(gap, kind='stable')[:count]] = 1.
    network = AdaptiveNetwork(synapses, tau=.02, adaptation_gain=0., bias=np.zeros(cells))
    rates = None
    for _ in range(STEPS):
        rates = np.asarray(network.step(external, dt=DT, learn=False))
    return set(np.flatnonzero(rates > ACTIVE).tolist())


def overlap(a, b):
    return 0. if not a or not b else len(a & b)/min(len(a), len(b))


def run(label, shape, radius, degree, weight, driven):
    started = time.time()
    synapses, points, cells = build(shape, radius, degree, weight)
    if len(shape) == 2:
        centres = [(x, y) for x in PLACES for y in PLACES]
    else:
        centres = [(x, y, z) for z in DEPTHS for x in PLACES for y in PLACES]
    patterns = [present(synapses, points, centre, driven, cells) for centre in centres]
    apart, same_place = [], []
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if centres[i][:2] == centres[j][:2]:
                same_place.append(overlap(patterns[i], patterns[j]))
            elif abs(centres[i][0] - centres[j][0]) > .2 or abs(centres[i][1] - centres[j][1]) > .2:
                apart.append(overlap(patterns[i], patterns[j]))
    lit = float(np.mean([len(p) for p in patterns]))
    return dict(label=label, cells=cells, degree=degree, weight=weight, driven=driven,
                lit=lit, fraction=lit/cells, apart=float(np.mean(apart)),
                same_place=float(np.mean(same_place)) if same_place else None,
                seconds=time.time() - started)


SHEET, BLOCK = (46, 46), (46, 46, 46)
runs = [
    ('平面2D 基准（间距1/46，最近6个，权重.10，驱动64）', SHEET, RADIUS, 6, WEIGHT, DRIVEN),
    ('立体3D 基准（同上，只改几何）', BLOCK, RADIUS, 6, WEIGHT, DRIVEN),
    ('平面2D 同半径内全部邻居（8个）', SHEET, RADIUS, 8, WEIGHT, DRIVEN),
    ('立体3D 同半径内全部邻居（18个）', BLOCK, RADIUS, 18, WEIGHT, DRIVEN),
    ('平面2D 只改权重 .10→.20', SHEET, RADIUS, 6, .20, DRIVEN),
    ('立体3D 只改权重 .10→.20', BLOCK, RADIUS, 6, .20, DRIVEN),
    ('平面2D 只改驱动 64→32', SHEET, RADIUS, 6, WEIGHT, 32),
    ('立体3D 只改驱动 64→32', BLOCK, RADIUS, 6, WEIGHT, 32),
]

print('%-40s %8s %5s %6s %6s %10s %9s %9s' %
      ('接线与驱动', '细胞', '出度', '权重', '驱动', '点亮细胞', '点亮比例', '不同物体'))
report = []
for label, shape, radius, degree, weight, driven in runs:
    row = run(label, shape, radius, degree, weight, driven)
    report.append(row)
    print('%-40s %8d %5d %6.2f %6d %10.0f %8.2f%% %9.3f'
          % (label, row['cells'], row['degree'], row['weight'], row['driven'],
             row['lit'], 100.*row['fraction'], row['apart']))

print()
print('=== 同一个位置、不同深度的一对物体 ===')
block = [row for row in report if row['same_place'] is not None]
for row in block:
    print('%-40s 重叠 %.3f（平面里这种输入完全相同，重叠必然是 1.000）'
          % (row['label'], row['same_place']))

print()
base = report[0]
print('=== 读数 ===')
print('只把平面换成同间距的立体：点亮 %.0f → %.0f 个（%.2f倍），比例 %.2f%% → %.2f%%'
      % (base['lit'], report[1]['lit'], report[1]['lit']/base['lit'],
         100.*base['fraction'], 100.*report[1]['fraction']))
print('只把出度 6 换成同半径内的 18 个：点亮 %.0f → %.0f 个（%.2f倍）'
      % (base['lit'], report[3]['lit'], report[3]['lit']/base['lit']))
print('只把权重 .10 换成 .20：点亮 %.0f → %.0f 个（%.2f倍）'
      % (base['lit'], report[4]['lit'], report[4]['lit']/base['lit']))
print('只把驱动 64 换成 32：点亮 %.0f → %.0f 个（%.2f倍）'
      % (base['lit'], report[6]['lit'], report[6]['lit']/base['lit']))
print('细胞数 %.0f → %.0f（%.0f倍），所以同样的驱动在立体里占的比例小得多'
      % (base['cells'], report[1]['cells'], report[1]['cells']/base['cells']))

artifact = ROOT/'artifacts'/'probe_geometry_control.json'
artifact.write_text(json.dumps(dict(spacing=SPACING, steps=STEPS, dt=DT, active=ACTIVE,
                                    rows=report), ensure_ascii=False, indent=2),
                    encoding='utf-8')
print()
print('写到 %s' % artifact)
