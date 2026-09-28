"""A remembered leg contact must raise that leg's next swing, and only then."""

import unittest

import numpy as np

from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body

STEPS = 400
NEURAL_DT = .01
# Joint 1 of leg 0 is the thigh; the swing lift raises its commanded angle.
THIGH_FL0 = 1


class ObstacleClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()
        cls.observation = cls.body.observe()

    def controller(self):
        return EmbodiedController(self.body.home_angles, self.body.lower_limits, self.body.upper_limits,
                                  motor_units=20, proprio_units=8, association_units=8)

    def walk(self, obstacle):
        """Walk in place with one leg touching an obstacle; return what it caused."""
        controller = self.controller()
        environment = {name: np.zeros(4) for name in
                       ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
        environment['foot_support'] = np.zeros(4, dtype=bool)
        clearance = np.zeros(4)
        thigh = -np.inf
        for _ in range(STEPS):
            environment['foot_obstacle'] = np.asarray(obstacle, dtype=float)
            target, _ = controller.step(self.observation, environment=environment, locomotion=1.,
                                        autonomy=False, dt=NEURAL_DT, learn=False)
            clearance = np.maximum(clearance, controller.network.activity[controller.groups['clearance']])
            thigh = max(thigh, float(target[THIGH_FL0]))
        return controller, clearance, thigh

    def test_contact_with_the_swing_raises_the_clearance_state(self):
        controller, clearance, _ = self.walk([1., 0., 0., 0.])
        self.assertGreater(clearance[0], .05)
        self.assertEqual(controller.groups['clearance'].shape, (4,))

    def test_no_contact_leaves_the_clearance_state_silent(self):
        _, clearance, _ = self.walk([0., 0., 0., 0.])
        self.assertLess(float(clearance.max()), 1e-9)

    def test_a_remembered_contact_raises_the_commanded_swing_angle(self):
        _, _, touched = self.walk([1., 0., 0., 0.])
        _, _, clean = self.walk([0., 0., 0., 0.])
        self.assertGreater(touched, clean + .05)

    def test_the_state_is_restricted_to_the_leg_that_was_touched(self):
        _, clearance, _ = self.walk([0., 0., 1., 0.])
        self.assertGreater(clearance[2], .05)
        self.assertLess(float(np.delete(clearance, 2).max()), 1e-9)
