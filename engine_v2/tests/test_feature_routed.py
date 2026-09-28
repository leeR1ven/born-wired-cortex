import unittest

import numpy as np

from born_wired.feature_routed import FeatureInputNetwork, FeatureRoutedController
from born_wired.go2_body import Go2Body
from born_wired.innate import InnateController


class FeatureRoutedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()

    def make(self, kind, **kwargs):
        return kind(self.body.home_angles, self.body.lower_limits, self.body.upper_limits,
                    motor_units=20, seed=3, **kwargs)

    def test_extension_preserves_existing_graph_and_cell_parameters(self):
        original = self.make(InnateController)
        extended = self.make(FeatureRoutedController)
        n, edges = original.network.n_neurons, len(original.synapses.src)
        self.assertIsInstance(extended.network, FeatureInputNetwork)
        self.assertIs(extended.network.synapses, extended.synapses)
        self.assertEqual(extended.network.n_neurons, n + 13)
        for field in ("src", "dst", "weights", "lower", "w_max", "plasticity", "tether", "anchor"):
            np.testing.assert_array_equal(getattr(extended.synapses, field)[:edges], getattr(original.synapses, field))
        for field in ("signs", "budgets", "target_activity"):
            np.testing.assert_array_equal(getattr(extended.synapses, field)[:n], getattr(original.synapses, field))
        for field in ("_tau", "_adaptation_tau", "_gain", "_bias", "voltage", "adaptation"):
            np.testing.assert_array_equal(getattr(extended.network, field)[:n], getattr(original.network, field))

    def test_no_feature_input_retains_parent_outputs_with_learning(self):
        original = self.make(InnateController)
        extended = self.make(FeatureRoutedController)
        observation = self.body.observe()
        for _ in range(20):
            expected = original.step(observation, dt=.01)
            actual = extended.step(observation, dt=.01)
            for left, right in zip(expected, actual):
                np.testing.assert_allclose(left, right, atol=1e-14, rtol=0)

    def test_features_are_sensory_currents_and_errors_are_transactional(self):
        controller = self.make(FeatureRoutedController)
        observation = self.body.observe()
        before_v, before_w = controller.network.voltage, controller.synapses.weights
        with self.assertRaises(ValueError):
            controller.step(observation, features=[0, 0, np.nan, 0])
        broken = dict(observation)
        broken["joint_position"] = np.zeros(2)
        with self.assertRaises(ValueError):
            controller.step(broken, features=[1, 0, 0, 0])
        np.testing.assert_array_equal(controller.network.voltage, before_v)
        np.testing.assert_array_equal(controller.synapses.weights, before_w)
        np.testing.assert_array_equal(controller.network._features, np.zeros(4))
        controller.step(observation, features=[1, 0, 0, 0], dt=.01, learn=False)
        rates = controller.network.activity[controller.groups["feature_receptors"]]
        self.assertGreater(rates[0], 0)
        np.testing.assert_array_equal(rates[1:], np.zeros(3))
        np.testing.assert_array_equal(controller.network._features, np.zeros(4))

    def test_compressed_only_ablation_removes_exactly_hidden_output_routes(self):
        hidden = self.make(FeatureRoutedController)
        compressed = self.make(FeatureRoutedController, hidden_routes=False)
        self.assertEqual(len(hidden.synapses.src) - len(compressed.synapses.src), 8)
        for controller, count in ((hidden, 8), (compressed, 0)):
            src, dst = controller.synapses.src, controller.synapses.dst
            routes = np.isin(src, controller.groups["feature_hidden"]) & (dst == controller.groups["flexion"][0])
            self.assertEqual(np.count_nonzero(routes), count)


if __name__ == "__main__":
    unittest.main()
