import unittest

import numpy as np

from born_wired.adaptive import AdaptiveNetwork
from born_wired.synapses import BoundedSynapses


def two_cells(**parameters):
    return BoundedSynapses([0, 1], [1, 0], [.4, .4], [1, -1], 2,
                          w_max=3, budgets=3, **parameters)


class AdaptiveTests(unittest.TestCase):
    def test_negative_voltage_and_exponential_update(self):
        synapses = BoundedSynapses([], [], [], [1], 1)
        network = AdaptiveNetwork(synapses, tau=.1, adaptation_tau=.2,
                                  adaptation_gain=2, bias=-1, initial_voltage=.5, initial_adaptation=.25)
        network.step(0, dt=.02, learn=False)
        expected_v = np.exp(-.2) * .5 + (1 - np.exp(-.2)) * -1.5
        expected_a = np.exp(-.1) * .25 + (1 - np.exp(-.1)) * .5
        self.assertAlmostEqual(network.voltage[0], expected_v)
        self.assertAlmostEqual(network.adaptation[0], expected_a)
        network.step(0, dt=10, learn=False)
        self.assertLess(network.voltage[0], 0)
        self.assertEqual(network.activity[0], 0)

    def test_bounded_states_without_learning(self):
        synapses = two_cells()
        initial_weights = synapses.weights.copy()
        network = AdaptiveNetwork(synapses, tau=[.01, .1], adaptation_tau=[.2, .5], bias=[-.5, .8])
        rng = np.random.default_rng(19)
        for step in range(1000):
            network.step(rng.uniform(0, 2, 2), rng.uniform(0, 3, 2),
                         dt=1000 if step == 500 else .002, learn=False)
            self.assertTrue(np.all(network.voltage >= -9))
            self.assertTrue(np.all(network.voltage <= 6))
            self.assertTrue(np.all((network.activity >= 0) & (network.activity <= 1)))
            self.assertTrue(np.all((network.adaptation >= 0) & (network.adaptation <= 1)))
        np.testing.assert_array_equal(synapses.weights, initial_weights)

    def test_parameters_and_readonly_snapshots_are_independent(self):
        bias, initial_voltage, initial_adaptation = np.ones(2), np.full(2, .4), np.full(2, .2)
        network = AdaptiveNetwork(two_cells(), bias=bias, initial_voltage=initial_voltage,
                                  initial_adaptation=initial_adaptation)
        bias[:] = initial_voltage[:] = initial_adaptation[:] = 99
        np.testing.assert_array_equal(network.voltage, [.4, .4])
        for name in ("activity", "voltage", "adaptation"):
            snapshot = getattr(network, name)
            self.assertFalse(snapshot.flags.writeable)
            snapshot.flags.writeable = True
            snapshot[:] = 99
            self.assertTrue(np.all(getattr(network, name) < 99))
        network.step(0, learn=False)
        self.assertTrue(np.all(network.voltage < 1))

    def test_invalid_inputs_and_failed_learning_rollback(self):
        network = AdaptiveNetwork(two_cells(), initial_voltage=[.5, .3], initial_adaptation=.1)
        voltage, adaptation, weights = network.voltage, network.adaptation, network.synapses.weights
        for kwargs in ({"external_exc": -1}, {"external_exc": np.nan}, {"external_exc": [1]},
                       {"external_exc": 0, "external_inh": np.inf}, {"external_exc": 0, "dt": 0},
                       {"external_exc": 0, "modulator": 2, "learn": False}, {"external_exc": 0, "learn": 1}):
            with self.assertRaises(ValueError):
                network.step(**kwargs)
            np.testing.assert_array_equal(network.voltage, voltage)
            np.testing.assert_array_equal(network.adaptation, adaptation)
            np.testing.assert_array_equal(network.synapses.weights, weights)
        for kwargs in ({"tau": 0}, {"adaptation_tau": -1}, {"adaptation_gain": -1},
                       {"initial_adaptation": 2}, {"initial_voltage": np.inf}, {"bias": True}):
            with self.assertRaises(ValueError):
                AdaptiveNetwork(two_cells(), **kwargs)
        failing = AdaptiveNetwork(two_cells(learning_rate=1e308), bias=1, initial_voltage=.5)
        voltage, adaptation, weights = failing.voltage, failing.adaptation, failing.synapses.weights
        with self.assertRaises(FloatingPointError):
            failing.step(1, dt=1e308)
        np.testing.assert_array_equal(failing.voltage, voltage)
        np.testing.assert_array_equal(failing.adaptation, adaptation)
        np.testing.assert_array_equal(failing.synapses.weights, weights)

    def test_autonomous_antiphase_oscillation_with_learning(self):
        synapses = BoundedSynapses([0, 1], [1, 0], [2.5, 2.5], [-1, -1], 2,
                                  w_max=3, budgets=3, learning_rate=.01, decay=.001)
        network = AdaptiveNetwork(synapses, bias=.8, initial_voltage=[.001, 0])
        activity = np.asarray([network.step(0) for _ in range(5000)])[2500:]
        self.assertTrue(np.all(np.ptp(activity, axis=0) > .2))
        self.assertLess(np.corrcoef(activity.T)[0, 1], -.7)
        difference = activity[:, 0] - activity[:, 1]
        crossings = np.count_nonzero((difference[:-1] < 0) & (difference[1:] >= 0))
        self.assertGreaterEqual(crossings, 3)
        self.assertFalse(np.array_equal(synapses.weights, [2.5, 2.5]))


if __name__ == "__main__":
    unittest.main()
