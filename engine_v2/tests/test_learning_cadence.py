import unittest
from unittest.mock import patch
import numpy as np
from born_wired.adaptive import AdaptiveNetwork
from born_wired.synapses import BoundedSynapses


class LearningCadenceTests(unittest.TestCase):
    def make(self):
        syn = BoundedSynapses([0], [1], [.2], [1, 1], 2, w_max=1., budgets=2., learning_rate=.1)
        return AdaptiveNetwork(syn, initial_voltage=.5, adaptation_gain=0., learning_interval=.05)

    def test_learning_uses_elapsed_time_while_activity_keeps_updating(self):
        net = self.make()
        initial = net.voltage.copy()
        weights = net.synapses.weights.copy()
        with patch.object(net.synapses, 'update', wraps=net.synapses.update) as update:
            for _ in range(4):
                net.step([1.,1.], dt=.01)
            self.assertEqual(update.call_count, 0)
            np.testing.assert_array_equal(weights, net.synapses.weights)
            self.assertGreater(np.max(abs(net.voltage-initial)), .01)
            net.step([1.,1.], dt=.01)
            self.assertEqual(update.call_count, 1)
            self.assertAlmostEqual(update.call_args.kwargs['dt'], .05)
            self.assertGreater(np.max(abs(net.synapses.weights-weights)), 0.)

    def test_failed_learning_does_not_commit_elapsed_or_neural_state(self):
        net = self.make()
        for _ in range(4):
            net.step([1.,1.], dt=.01)
        state, adaptation = net.voltage, net.adaptation
        with patch.object(net.synapses, 'update', side_effect=ValueError('invalid update')):
            with self.assertRaises(ValueError):
                net.step([1.,1.], dt=.01)
        np.testing.assert_array_equal(state, net.voltage)
        np.testing.assert_array_equal(adaptation, net.adaptation)
        self.assertAlmostEqual(net._learning_elapsed, .04)
        net.step([1.,1.], dt=.01)
        self.assertEqual(net._learning_elapsed, 0.)

    def test_learning_off_does_not_accumulate_extra_learning_time(self):
        net = self.make()
        for _ in range(9):
            net.step([1.,1.], dt=.1, learn=False)
        self.assertEqual(net._learning_elapsed, 0.)
        self.assertEqual(net.synapses.weights[0], .2)


if __name__ == '__main__':
    unittest.main()
