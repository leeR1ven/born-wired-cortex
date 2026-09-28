"""模型放大之后，分辨得更细了吗？需要点亮的比例变小了吗？

三种规模 x 三种接线。同一个物体（同样大小、同样位置），只改网络有多少细胞、
坐标是二维还是三维、以及输入接到哪儿。每个物体单独呈现，记下最后亮着哪些细胞。
"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.adaptive import AdaptiveNetwork
from born_wired.topology import neighbour_edges

LOCAL, RANDOM, RADIUS, SHIFT = 6, 2, .18, .06
STEPS, DT, ACTIVE, OBJECTS = 30, .01, .05, 16
RNG = np.random.default_rng(7)


def factors(total, dimension):
    """A box of whole cells as close to `total` as a whole grid allows."""
    per = int(round(total ** (1./dimension)))
    shape = [per]*dimension
    shape[0] = max(2, int(round(total/np.prod(shape[1:]))))
    return shape


def sheet_positions(total, dimension):
    shape = factors(total, dimension)
    grid = np.indices(shape).reshape(dimension, -1).T
    return grid/ (np.asarray(shape, dtype=float) - 1.)


def build(total, dimension, mode, receptors_per_axis):
    positions = sheet_positions(total, dimension)
    cells = len(positions)
    src, dst = neighbour_edges(positions, LOCAL, RANDOM, seed=0)
    receptors = (np.indices((receptors_per_axis,)*3).reshape(3, -1).T + .5)/receptors_per_axis
    if mode == 'random':
        target = RNG.integers(0, cells, len(receptors))
    else:
        # In blocks: the full table of receptor-to-cell distances is billions of
        # entries at the top size and does not fit in this machine's memory.
        view = receptors[:, :dimension]
        target = np.empty(len(receptors), dtype=np.int64)
        block = max(1, int(2e7/max(1, cells)))
        for start in range(0, len(receptors), block):
            piece = view[start:start + block]
            target[start:start + block] = np.argmin(
                ((piece[:, None, :] - positions[None, :, :])**2).sum(-1), axis=1)
    src = np.r_[src, np.arange(len(receptors)) + cells]
    dst = np.r_[dst, target]
    weights = np.r_[np.full(len(positions)*(LOCAL + RANDOM), .10), np.ones(len(receptors))]
    synapses = BoundedSynapses(src, dst, weights, np.ones(cells + len(receptors)),
                               cells + len(receptors), w_max=10., budgets=40., plasticity=0.)
    network = AdaptiveNetwork(synapses, tau=.02, adaptation_gain=0.,
                              bias=np.zeros(cells + len(receptors)))
    return network, cells, receptors


def present(network, cells, receptors, centre):
    external = np.zeros(network.n_neurons)
    external[cells:][np.linalg.norm(receptors - centre, axis=1) < RADIUS] = 1.
    rates = None
    for _ in range(STEPS):
        rates = np.asarray(network.step(external, dt=DT, learn=False))
    return set(np.flatnonzero(rates[:cells] > ACTIVE).tolist())


def overlap(a, b):
    return 0. if not a or not b else len(a & b)/min(len(a), len(b))


def run(total, dimension, mode, receptors_per_axis):
    started = time.time()
    network, cells, receptors = build(total, dimension, mode, receptors_per_axis)
    places = RNG.uniform(RADIUS, 1. - RADIUS, (OBJECTS//2, 2))
    centres = np.array([[x, y, z] for x, y in places for z in (.20, .80)])
    patterns = [present(network, cells, receptors, centre) for centre in centres]
    different, depth = [], []
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if abs(centres[i][2] - centres[j][2]) > 3.*RADIUS:
                depth.append(overlap(patterns[i], patterns[j]))
            else:
                different.append(overlap(patterns[i], patterns[j]))
    shifted = []
    for i, centre in enumerate(centres):
        for delta in ((SHIFT, 0., 0.), (0., SHIFT, 0.)):
            moved = np.clip(centre + delta, RADIUS, 1. - RADIUS)
            shifted.append(overlap(patterns[i], present(network, cells, receptors, moved)))
    return dict(cells=cells, lit=float(np.mean([len(p) for p in patterns])),
                different=float(np.mean(different)), shifted=float(np.mean(shifted)),
                depth=float(np.mean(depth)), seconds=time.time() - started)


print('%-24s %9s %10s %9s %11s %9s %9s %8s' %
      ('接线', '细胞', '平均点亮', '点亮比例', '不同物体', '挪一点', '同位置异深', '用时s'))
sizes = ((1000, 8), (10000, 16), (100000, 32))
if len(sys.argv) > 1:
    sizes = tuple((int(value), 32) for value in sys.argv[1:])
for total, receptors_per_axis in sizes:
    for label, dimension, mode in (('随机接', 2, 'random'),
                                   ('平面 2D', 2, 'map'),
                                   ('立体 3D', 3, 'map')):
        r = run(total, dimension, mode, receptors_per_axis)
        print('%-24s %9d %10.0f %8.2f%% %11.3f %9.3f %9.3f %8.1f' %
              ('%s / %d' % (label, total), r['cells'], r['lit'],
               100.*r['lit']/r['cells'], r['different'], r['shifted'], r['depth'], r['seconds']))
