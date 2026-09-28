import unittest

import numpy as np

from born_wired.adaptive import AdaptiveNetwork
from born_wired.sparse import SparseNetwork
from born_wired.synapses import BoundedSynapses


def random_graph(cells, degree, seed):
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(cells, dtype=np.int64), degree)
    shift = rng.integers(0, cells - degree, (cells, 1))
    dst = ((np.arange(cells, dtype=np.int64)[:, None] + shift
            + np.arange(1, degree + 1, dtype=np.int64)) % cells).ravel()
    synapses = BoundedSynapses(src, dst, rng.uniform(.02, .2, src.size),
                               np.where(rng.random(cells) < .8, 1., -1.), cells,
                               w_max=1., budgets=40., plasticity=0.)
    return synapses, rng.uniform(.02, .12, cells), rng.uniform(1., 3., cells)


def compare(cells, synapses, tau, gain, bias, opening, steps=400, drive_steps=0, dt=.01,
            cutoff=0.):
    """Run both engines together and return the biggest disagreement in activity."""
    parameters = dict(tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=bias,
                      initial_voltage=opening)
    exact = AdaptiveNetwork(synapses, **parameters)
    sparse = SparseNetwork(synapses, cutoff=cutoff, **parameters)
    drive = np.zeros(cells); drive[:max(1, cells//8)] = 1.
    biggest, speaking = 0., []
    for step in range(steps):
        external = drive if step < drive_steps else np.zeros(cells)
        reference = np.asarray(exact.step(external, dt=dt, learn=False)).copy()
        instead = np.asarray(sparse.step(external, dt=dt, learn=False)).copy()
        biggest = max(biggest, float(np.abs(instead - reference).max()))
        speaking.append(sparse.speaking)
    return exact, sparse, biggest, speaking


class SparseEngineTests(unittest.TestCase):
    def test_quiet_cell_that_wakes_itself_is_not_missed(self):
        # Cell zero excites one, one inhibits zero. Zero falls silent, then its
        # own positive bias lifts it back over the cutoff; an engine that only
        # touches speech, its targets and outside drive never sees that happen
        # and the two engines part ways.
        synapses = BoundedSynapses([0, 1], [1, 0], [.5, .5], [1, -1], 2,
                                   w_max=1., budgets=40., plasticity=0.)
        exact, sparse, biggest, activity = compare(
            2, synapses, np.full(2, .05), np.full(2, 2.), np.array([.05, -.1]),
            np.array([.3, 0.]))
        self.assertLess(biggest, 1e-12)
        quiet = any(value == 0. for value in activity)
        woke = any(activity[index] > 0 and activity[index - 1] == 0
                   for index in range(1, len(activity)))
        self.assertTrue(quiet and woke, 'the fixture never left and re-entered speech')
        sparse.catch_up()
        np.testing.assert_allclose(sparse.voltage, exact.voltage, atol=1e-12)

    def test_random_graph_matches_the_dense_engine_exactly(self):
        synapses, tau, gain = random_graph(60, 3, seed=11)
        bias = np.random.default_rng(12).uniform(-.3, .05, 60)
        exact, sparse, biggest, _ = compare(60, synapses, tau, gain, bias, 0.,
                                            steps=300, drive_steps=40)
        self.assertLess(biggest, 1e-12)
        sparse.catch_up()
        np.testing.assert_allclose(sparse.voltage, exact.voltage, atol=1e-12)
        np.testing.assert_allclose(sparse.adaptation, exact.adaptation, atol=1e-12)

    def test_long_lag_is_reconstructed_exactly(self):
        for tau in (.02, .5):
            synapses = BoundedSynapses([0], [1], [.4], [1, -1], 2,
                                       w_max=1., budgets=40., plasticity=0.)
            exact, sparse, biggest, _ = compare(
                2, synapses, np.full(2, tau), np.full(2, 2.), np.array([-.05, -.2]), 0.,
                steps=500, drive_steps=5)
            self.assertLess(biggest, 1e-12)
            self.assertGreater(sparse._ticks - int(sparse._last[1]), 100)

    def test_sparse_drive_matches_dense_drive(self):
        synapses, tau, gain = random_graph(40, 3, seed=5)
        bias = np.random.default_rng(6).uniform(-.3, .05, 40)
        parameters = dict(tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=bias,
                          cutoff=0.)
        dense = np.zeros(40); dense[[3, 7, 20]] = [.5, .5, 1.]
        cells, amounts = np.array([3, 7, 7, 20]), np.array([.5, .25, .25, 1.])
        first = SparseNetwork(synapses, **parameters)
        second = SparseNetwork(synapses, **parameters)
        for _ in range(60):
            one = np.asarray(first.step(dense, dt=.01, learn=False)).copy()
            other = np.asarray(second.step_sparse((cells, amounts), dt=.01, learn=False)).copy()
            np.testing.assert_array_equal(one, other)
        self.assertGreater(float(np.asarray(first.activity).max()), 0.)

    def test_lower_cutoff_stays_closer_to_the_reference(self):
        synapses, tau, gain = random_graph(60, 3, seed=21)
        bias = np.random.default_rng(22).uniform(-.3, .05, 60)
        gap = {}
        for cutoff in (1e-2, 1e-3):
            _, _, gap[cutoff], _ = compare(60, synapses, tau, gain, bias, 0.,
                                           steps=200, drive_steps=40, cutoff=cutoff)
        self.assertLess(gap[1e-3], gap[1e-2])

    def test_reads_are_read_only(self):
        synapses, tau, gain = random_graph(20, 2, seed=3)
        sparse = SparseNetwork(synapses, tau=tau, adaptation_tau=.4, adaptation_gain=gain,
                               bias=-.1, cutoff=0.)
        for name in ('activity', 'voltage', 'adaptation'):
            view = getattr(sparse, name)
            self.assertFalse(view.flags.writeable)
            with self.assertRaises(ValueError):
                view[0] = 99.
            self.assertFalse(getattr(sparse, name).flags.writeable)

    def test_invalid_inputs(self):
        synapses, tau, gain = random_graph(6, 2, seed=4)
        settings = dict(tau=tau, adaptation_tau=.4, adaptation_gain=gain, bias=0.)
        sparse = SparseNetwork(synapses, cutoff=0., **settings)
        with self.assertRaises(NotImplementedError):
            sparse.step(np.zeros(6), learn=True)
        with self.assertRaises(NotImplementedError):
            sparse.step_sparse((np.array([0]), np.array([1.])), learn=True)
        with self.assertRaises(ValueError):
            sparse.catch_up()
        for kwargs in ({'external_exc': np.zeros(5)}, {'external_exc': -1},
                       {'external_exc': np.nan}, {'external_exc': np.zeros(6), 'dt': 0},
                       {'external_exc': np.zeros(6), 'external_inh': [1.]}):
            with self.assertRaises(ValueError):
                sparse.step(**kwargs)
        for kwargs in ({'external_exc': ([0], [1.], [2.])}, {'external_exc': ([9], [1.])},
                       {'external_exc': ([0], [-1.])}, {'external_exc': 0}):
            with self.assertRaises(ValueError):
                sparse.step_sparse(**kwargs)
        for kwargs in ({'cutoff': -.1}, {'tau': 0}, {'initial_adaptation': 2}):
            with self.assertRaises(ValueError):
                SparseNetwork(synapses, **dict(settings, **kwargs))


if __name__ == '__main__':
    unittest.main()
