"""Does a three-dimensional sheet tell apart things a flat one cannot?

Same number of cells (1000), same number of outgoing connections per cell
(6 local + 2 random), same input volume, same seeds. The only difference is
whether the sheet's coordinates are 2D or 3D, and where its local neighbours
therefore are. A third arm wires the input to random cells: that is the
control that says how much of any result is just the wiring being local.

An object is a blob of receptors. Twenty of them are placed in the volume and
each one is presented on its own. Printed is the overlap (Jaccard) between the
sets of cells that ended up active:
  different  - two unrelated objects; lower is better, it means the sheet keeps
               them apart
  shifted    - the same object nudged a little; higher is better, it means the
               sheet does not treat a small move as a new object
  same column- two objects sharing x and y and differing only in depth
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.synapses import BoundedSynapses
from born_wired.adaptive import AdaptiveNetwork
from born_wired.topology import make_edges

N_SHEET, LOCAL, RANDOM = 1000, 6, 2
RECEPTORS = 8
RADIUS, SHIFT, STEPS, DT, ACTIVE = .18, .06, 40, .01, .05
RNG = np.random.default_rng(7)


def sheet_positions(dimension):
    if dimension == 2:
        grid = np.indices((40, 25)).reshape(2, -1).T
        return grid / np.array([39., 24.])
    grid = np.indices((10, 10, 10)).reshape(3, -1).T
    return grid / 9.


def build(dimension, mode):
    positions = sheet_positions(dimension)
    cells = len(positions)
    src, dst = make_edges(positions, LOCAL, RANDOM, seed=0)
    receptors = (np.indices((RECEPTORS,)*3).reshape(3, -1).T + .5) / RECEPTORS
    if mode == 'random':
        target = RNG.integers(0, cells, len(receptors))
    else:
        view = receptors[:, :dimension]
        target = np.argmin(((view[:, None, :] - positions[None, :, :])**2).sum(-1), axis=1)
    src = np.r_[src, np.arange(len(receptors)) + cells]
    dst = np.r_[dst, target]
    weights = np.r_[np.full(len(positions)* (LOCAL+RANDOM), .10), np.ones(len(receptors))]
    signs = np.ones(cells + len(receptors))
    synapses = BoundedSynapses(src, dst, weights, signs, cells + len(receptors),
                               w_max=10., budgets=40., plasticity=0.)
    network = AdaptiveNetwork(synapses, tau=.02, adaptation_gain=0, bias=np.zeros(cells + len(receptors)))
    return network, cells, receptors


def present(network, cells, receptors, centre):
    which = np.linalg.norm(receptors - centre, axis=1) < RADIUS
    external = np.zeros(network.n_neurons)
    external[cells:][which] = 1.
    rates = None
    for _ in range(STEPS):
        rates = np.asarray(network.step(external, dt=DT, learn=False))
    return set(np.flatnonzero(rates[:cells] > ACTIVE).tolist())


def jaccard(a, b):
    union = len(a | b)
    return 0. if not union else len(a & b)/union


def overlap(a, b):
    return 0. if not a or not b else len(a & b)/min(len(a), len(b))


def run(dimension, mode):
    network, cells, receptors = build(dimension, mode)
    # Twenty objects: ten places across the picture, two depths at each place.
    places = RNG.uniform(RADIUS, 1. - RADIUS, (10, 2))
    centres = np.array([[x, y, z] for x, y in places for z in (.20, .80)])
    patterns, sizes = [], []
    for centre in centres:
        pattern = present(network, cells, receptors, centre)
        patterns.append(pattern); sizes.append(len(pattern))
    different, shifted, column = [], [], []
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if np.linalg.norm(centres[i] - centres[j]) < 3.*RADIUS:
                continue
            if abs(centres[i][2] - centres[j][2]) > 3.*RADIUS:
                column.append(overlap(patterns[i], patterns[j]))   # same place, other depth
            else:
                different.append(overlap(patterns[i], patterns[j]))
    for i, centre in enumerate(centres):
        for dx, dy, dz in ((SHIFT, 0., 0.), (0., SHIFT, 0.)):
            moved = np.clip(centre + [dx, dy, dz], RADIUS, 1. - RADIUS)
            shifted.append(overlap(patterns[i], present(network, cells, receptors, moved)))
    return (np.mean(different) if different else float('nan'),
            np.mean(shifted) if shifted else float('nan'),
            np.mean(column) if column else float('nan'),
            np.mean(sizes), len(column))


print('%-24s %10s %10s %12s %9s' % ('wiring', 'different', 'shifted', 'same column', 'cells lit'))
for label, dimension, mode in (('input at random cells', 2, 'random'),
                               ('flat sheet, 2D coordinates', 2, 'map'),
                               ('solid block, 3D coordinates', 3, 'map')):
    d, s, c, n, pairs = run(dimension, mode)
    print('%-24s %10.3f %10.3f %12.3f %9.0f   (%d depth pairs)' % (label, d, s, c, n, pairs))
