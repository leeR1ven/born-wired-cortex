import unittest

import numpy as np

from born_wired.dynamics import ConductanceNetwork
from born_wired.synapses import BoundedSynapses
from born_wired.topology import make_edges


class DynamicsTests(unittest.TestCase):
    def test_single_direction_response_and_inhibition(self):
        synapses = BoundedSynapses(
            [0], [1], [1.0], [1, 1], 2, plasticity=0
        )
        network = ConductanceNetwork(synapses, threshold=0.1)
        inhibited = ConductanceNetwork(synapses, threshold=0.1)
        for _ in range(20):
            network.step([1, 0], learn=False)
            inhibited.step([1, 0], external_inh=[0, 0.8], learn=False)
        self.assertGreater(network.activity[1], 0)
        self.assertLess(inhibited.activity[1], network.activity[1])

    def test_no_learning_half_steps_match_full_step(self):
        synapses = BoundedSynapses([], [], [], [1, 1], 2)
        full = ConductanceNetwork(synapses)
        half = ConductanceNetwork(synapses)
        full.step([0.3, 0.4], learn=False, dt=0.002)
        half.step([0.3, 0.4], learn=False, dt=0.001)
        half.step([0.3, 0.4], learn=False, dt=0.001)
        np.testing.assert_allclose(full.voltage, half.voltage, rtol=0, atol=1e-14)
        np.testing.assert_allclose(full.activity, half.activity, rtol=0, atol=1e-14)

    def test_constant_drive_equilibrium_and_leak_decay(self):
        driven = ConductanceNetwork(
            BoundedSynapses([], [], [], [1], 1),
            tau=0.02,
            leak=0.5,
            threshold=0.2,
            initial_voltage=0,
        )
        for _ in range(1000):
            driven.step([1], learn=False, dt=0.002)
        np.testing.assert_allclose(driven.voltage, [1 / 1.5], rtol=0, atol=1e-12)

        leaking = ConductanceNetwork(
            BoundedSynapses([], [], [], [1], 1),
            tau=0.02,
            leak=1.0,
            initial_voltage=0.8,
        )
        for _ in range(100):
            leaking.step([0], learn=False, dt=0.002)
        expected = 0.8 * np.exp(-10)
        np.testing.assert_allclose(leaking.voltage, [expected], rtol=0, atol=1e-14)

    def test_learning_changes_weights_while_frozen_stays_fixed(self):
        mutable = BoundedSynapses(
            [0],
            [1],
            [0.1],
            [1, 1],
            2,
            learning_rate=0.1,
            decay=0,
            budgets=2,
        )
        mutable_network = ConductanceNetwork(mutable, threshold=0.1)
        for _ in range(40):
            mutable_network.step([1, 1], learn=True)
        self.assertGreater(mutable.weights[0], 0.1)

        frozen = BoundedSynapses(
            [0],
            [1],
            [0.1],
            [1, 1],
            2,
            plasticity=0,
        )
        frozen_network = ConductanceNetwork(frozen, threshold=0.1)
        before = frozen.weights.copy()
        for _ in range(40):
            frozen_network.step([1, 1], learn=True)
        np.testing.assert_array_equal(frozen.weights, before)

    def test_invalid_or_overflowing_step_is_atomic(self):
        synapses = BoundedSynapses([0], [1], [0.5], [1, 1], 2)
        network = ConductanceNetwork(
            synapses, threshold=0.1, initial_voltage=[0.3, 0.4]
        )
        voltage = network.voltage.copy()
        activity = network.activity.copy()
        weights = synapses.weights.copy()
        bad_calls = (
            lambda: network.step([np.nan, 0], learn=True),
            lambda: network.step([-0.1, 0], learn=True),
            lambda: network.step([1.0], learn=True),
            lambda: network.step([1, 0], external_inh=[np.inf, 0], learn=True),
            lambda: network.step([1, 0], dt=0, learn=True),
            lambda: network.step([1, 0], dt=np.nan, learn=True),
            lambda: network.step([1, 0], learn=1),
            lambda: network.step([1, 0], learn=False, modulator=2),
            lambda: network.step([1e308, 0], dt=1e308, learn=False),
        )
        for bad_call in bad_calls:
            with self.subTest(bad_call=bad_call), self.assertRaises(
                (ValueError, FloatingPointError)
            ):
                bad_call()
            np.testing.assert_array_equal(network.voltage, voltage)
            np.testing.assert_array_equal(network.activity, activity)
            np.testing.assert_array_equal(synapses.weights, weights)

    def test_five_seed_sixty_four_cell_long_run(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            positions = np.column_stack(
                (np.arange(64) % 8, np.arange(64) // 8)
            )
            src, dst = make_edges(positions, 3, 2, seed=seed)
            signs = np.where(rng.random(64) < 0.75, 1.0, -1.0)
            synapses = BoundedSynapses(
                src,
                dst,
                np.full(src.size, 0.02),
                signs,
                64,
                learning_rate=0.01,
                decay=0.001,
                budgets=2.0,
            )
            network = ConductanceNetwork(synapses, threshold=0.1)
            activity_min = 1.0
            activity_max = 0.0
            weight_min = float(np.min(synapses.weights))
            weight_max = float(np.max(synapses.weights))
            all_dead_steps = 0
            near_all_bright_steps = 0
            for step in range(10000):
                phase = (step // 500) % 4
                external = np.zeros(64)
                if phase == 0:
                    external[:16] = 0.8
                elif phase == 1:
                    external[16:32] = 0.8
                elif phase == 2:
                    external[rng.random(64) < 0.10] = 0.8
                activity = network.step(external, learn=True)
                activity_min = min(activity_min, float(np.min(activity)))
                activity_max = max(activity_max, float(np.max(activity)))
                current_weights = synapses.weights
                weight_min = min(weight_min, float(np.min(current_weights)))
                weight_max = max(weight_max, float(np.max(current_weights)))
                if np.all(activity <= 1e-12):
                    all_dead_steps += 1
                if np.mean(activity >= 0.9) >= 0.9:
                    near_all_bright_steps += 1
            with self.subTest(seed=seed):
                self.assertLess(all_dead_steps, 10000)
                self.assertLess(near_all_bright_steps, 10000)
                self.assertGreaterEqual(activity_min, -1e-14)
                self.assertLessEqual(activity_max, 1 + 1e-14)
                self.assertGreaterEqual(weight_min, -1e-14)
                self.assertLessEqual(weight_max, 1 + 1e-10)


if __name__ == "__main__":
    unittest.main()
