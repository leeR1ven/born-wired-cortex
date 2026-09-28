"""Independent integration checks by the coordinating reviewer."""
import unittest
import numpy as np

from born_wired.dynamics import ConductanceNetwork
from born_wired.synapses import BoundedSynapses


class DynamicsReviewTests(unittest.TestCase):
    def test_synaptic_overflow_does_not_commit_neural_state(self):
        syn = BoundedSynapses([0], [1], [.1], [1, 1], 2,
                              learning_rate=1e308, decay=0)
        brain = ConductanceNetwork(syn, initial_voltage=[.7, .7], threshold=.1)
        before = (brain.voltage, brain.activity, syn.weights)
        # Neural integration is finite, but the following plasticity update overflows.
        with self.assertRaises(FloatingPointError):
            brain.step([1, 1], dt=1000)
        for old, new in zip(before, (brain.voltage, brain.activity, syn.weights)):
            np.testing.assert_array_equal(old, new)

    def test_initial_input_and_exported_states_are_independent(self):
        syn = BoundedSynapses([], [], [], [1, -1], 2)
        initial = np.array([.4, .5])
        brain = ConductanceNetwork(syn, initial_voltage=initial)
        initial[:] = 0
        for exported in (brain.voltage, brain.activity):
            exported.flags.writeable = True
            exported[:] = 99
        np.testing.assert_array_equal(brain.voltage, [.4, .5])
        stimulus = np.array([.3, .2])
        before = stimulus.copy()
        brain.step(stimulus, learn=False)
        np.testing.assert_array_equal(stimulus, before)

    def test_paired_stimulation_changes_later_signal_transfer(self):
        def create():
            syn = BoundedSynapses([0], [1], [.05], [1, 1], 2,
                                  learning_rate=1, decay=0)
            return ConductanceNetwork(syn, threshold=.1)
        learning, control = create(), create()
        for _ in range(400):
            learning.step([1, 1], dt=.02, learn=True)
            control.step([1, 1], dt=.02, learn=False)
        # Remove residual voltage so the difference must persist in the weights.
        for _ in range(100):
            learning.step([0, 0], dt=.02, learn=False)
            control.step([0, 0], dt=.02, learn=False)
        for _ in range(50):
            learning.step([1, 0], dt=.02, learn=False)
            control.step([1, 0], dt=.02, learn=False)
        self.assertGreater(learning.activity[1], control.activity[1] + .02)
        self.assertEqual(control.synapses.weights[0], .05)

    def test_constructor_rejects_unphysical_parameters(self):
        syn = BoundedSynapses([], [], [], [1], 1)
        for kwargs in ({'tau': 0}, {'leak': -1}, {'threshold': 1},
                       {'threshold': -.1}, {'initial_voltage': 2},
                       {'initial_voltage': float('nan')}, {'tau': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConductanceNetwork(syn, **kwargs)


if __name__ == '__main__':
    unittest.main()
