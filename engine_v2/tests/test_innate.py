import unittest
import numpy as np
from born_wired.go2_body import Go2Body
from born_wired.innate import InnateController


class InnateTests(unittest.TestCase):
    def setUp(self):
        self.body = Go2Body()
        self.brain = InnateController(self.body.home_angles, self.body.lower_limits, self.body.upper_limits)

    def test_silencing_removes_active_force(self):
        target, activation = self.brain.step(self.body.observe(), silence_motor=True)
        self.body.step(target, activation=activation)
        np.testing.assert_array_equal(self.body.data.ctrl, 0)

    def test_invalid_input_does_not_change_brain(self):
        before = self.brain.network.voltage
        weights = self.brain.synapses.weights
        with self.assertRaises(ValueError):
            self.brain.step(self.body.observe(), cues=[1, 0, np.nan, 0])
        np.testing.assert_array_equal(before, self.brain.network.voltage)
        np.testing.assert_array_equal(weights, self.brain.synapses.weights)

    def test_intermediate_proprioception_reaches_withdrawal(self):
        observation = self.body.observe()
        observation['joint_position'][0] = self.body.upper_limits[0]
        for _ in range(100):
            self.brain.step(observation, dt=.01, learn=False)
        ids = self.brain.groups['limit_exc']
        self.assertGreater(self.brain.network.activity[ids[1]], .95)
        self.assertLess(self.brain.network.activity[ids[0]], .01)
        # This route is anatomical; no compressed association output is read.
        s = self.brain.synapses
        routes = np.isin(s.src, self.brain.groups['proprioception']) & np.isin(s.dst, ids)
        self.assertTrue(routes.any())

    def test_learning_off_and_learning_on_are_distinct(self):
        observation = self.body.observe()
        for _ in range(40):
            self.brain.step(observation, dt=.01, learn=False)
        np.testing.assert_array_equal(self.brain.synapses.weights, self.brain.initial_weights)
        for _ in range(40):
            self.brain.step(observation, flexion=.8, cues=[1,0,0,0], dt=.01, learn=True)
        self.assertGreater(self.brain.diagnostics()['memory_weights'][0], .01)
        self.assertTrue(np.all(self.brain.synapses.plasticity > 0))


if __name__ == '__main__':
    unittest.main()
