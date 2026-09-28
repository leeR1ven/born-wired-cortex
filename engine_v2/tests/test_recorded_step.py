"""The recorded step must land on the numbers the eager step lands on.

`torch_execution` can record one step of a sheet of cells once per dt and
replay it as a single launch. The recording adds no arithmetic of its own, so
the strongest check available is the direct one: two sheets built alike, fed
the same inputs, one through each path, must agree to the last bit - and the
narrow rate read must be the whole sheet's numbers for those cells.
"""
import os
import unittest

import numpy as np

from born_wired import torch_execution
from born_wired.adaptive import AdaptiveNetwork
from born_wired.regulation import RegulatedSynapses


def build(cells, seed=5):
    """A small regulated graph with every kind of cell and edge in it."""
    rng = np.random.default_rng(seed)
    count = 8 * cells
    src = rng.integers(0, cells, 2*count)
    dst = rng.integers(0, cells, 2*count)
    apart = src != dst
    src, dst = src[apart], dst[apart]
    # One row per directed pair: the graph holds at most one edge between two
    # cells, so the test draws pairs and keeps the first of each.
    _, first = np.unique(src*cells + dst, return_index=True)
    first = np.sort(first)[:count]
    src, dst = src[first], dst[first]
    return RegulatedSynapses(src, dst, rng.uniform(0, .5, count),
                             np.where(rng.random(cells) < .8, 1, -1), cells,
                             lower=0, upper=.8, budgets=40., plasticity=.4, tether=.05,
                             learning_rate=.12, target_activity=.15)


class RecordedStepTests(unittest.TestCase):
    cells = 96

    def setUp(self):
        device = torch_execution.resolve()
        if device is None or device.type != 'cuda':
            self.skipTest('recording needs a device that can record a step')
        self.previous = os.environ.get(torch_execution.RECORDING)
        self.addCleanup(self.restore)

    def restore(self):
        if self.previous is None:
            os.environ.pop(torch_execution.RECORDING, None)
        else:
            os.environ[torch_execution.RECORDING] = self.previous

    def network(self):
        return AdaptiveNetwork(build(self.cells), tau=.035, adaptation_tau=.4,
                               adaptation_gain=2.5, bias=0., learning_interval=.05)

    def test_the_two_paths_agree_bit_for_bit_over_a_run(self):
        traces = {}
        for recorded in (True, False):
            os.environ[torch_execution.RECORDING] = 'on' if recorded else 'off'
            network = self.network()
            self.assertIsNotNone(network._device)
            self.assertEqual(network._device.recorded_available(), recorded)
            rng = np.random.default_rng(7)
            trace = []
            for step in range(60):
                excitation = rng.uniform(0, 1., self.cells)
                inhibition = rng.uniform(0, .3, self.cells) if step % 3 == 0 else None
                rates = network.step(excitation, inhibition, dt=.01)
                trace.append((np.asarray(rates), np.asarray(network.voltage),
                              network.synapses.weights.copy()))
            traces[recorded] = trace
        for recorded, eager in zip(traces[True], traces[False]):
            for mine, theirs in zip(recorded, eager):
                np.testing.assert_array_equal(mine, theirs)

    def test_asking_for_named_cells_returns_the_whole_sheet_there(self):
        os.environ[torch_execution.RECORDING] = 'on'
        network = self.network()
        rng = np.random.default_rng(11)
        wanted = np.array([3, 5, 5, 40, 91])
        for _ in range(20):
            part = network.step(rng.uniform(0, 1., self.cells), dt=.01, wanted=wanted)
            np.testing.assert_array_equal(part, np.asarray(network.activity)[wanted])


if __name__ == '__main__':
    unittest.main()
