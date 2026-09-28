"""A body on the ground instead of on its feet drives the righting burst."""

import unittest

import numpy as np

from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body

STEPS = 300
NEURAL_DT = .01
UP = (0., 0., -1.)
ON_BACK = (0., 0., 1.)


class RightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()
        cls.observation = cls.body.observe()

    def controller(self, **over):
        return EmbodiedController(self.body.home_angles, self.body.lower_limits, self.body.upper_limits,
                                  motor_units=20, proprio_units=8, association_units=8, **over)

    def walk(self, gravity, touch=None, **over):
        """Hold one body orientation; report what the fall pathway did."""
        controller = self.controller(**over)
        observation = dict(self.observation, gravity_direction=np.asarray(gravity, dtype=float),
                           body_up=np.asarray(gravity, dtype=float) * -1.)
        environment = {name: np.zeros(4) for name in
                       ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
        environment['foot_support'] = np.zeros(4, dtype=bool)
        if touch is not None:
            environment['body_touch'] = np.asarray(touch, dtype=float)
        state = np.zeros(1)
        push = np.zeros(4)
        target = np.zeros(12)
        for _ in range(STEPS):
            target, _ = controller.step(observation, environment=environment, locomotion=1.,
                                        autonomy=False, dt=NEURAL_DT, learn=False)
            state = np.maximum(state, controller.network.activity[controller.groups['righting']])
            push = np.maximum(push, controller.network.activity[controller.groups['righting_push']])
        return controller, state, push, target

    def test_on_its_feet_the_fall_state_is_silent(self):
        _, state, push, _ = self.walk(UP)
        self.assertEqual(float(state.max()), 0.)
        self.assertEqual(float(push.max()), 0.)

    def test_losing_the_up_axis_drives_the_state_and_the_burst(self):
        _, state, push, _ = self.walk(ON_BACK)
        self.assertGreater(float(state[0]), .5)
        self.assertGreater(float(push.min()), .05)

    def test_body_contact_adds_to_the_same_state(self):
        # One body sector touching the ground is a small drive on its own; the
        # loss of the up axis is the main one and the contact adds to it.
        _, touched, _, _ = self.walk(UP, touch=[1., 0., 0., 0.])
        _, clean, _, _ = self.walk(UP)
        self.assertGreater(float(touched[0]), float(clean[0]))

    def test_the_burst_moves_the_shared_motor_units(self):
        _, _, _, lying = self.walk(ON_BACK)
        _, _, _, standing = self.walk(UP)
        self.assertGreater(float(np.abs(lying - standing).max()), .5)

    def test_the_burst_is_what_moves_the_joint(self):
        _, _, _, with_burst = self.walk(ON_BACK)
        _, _, _, without = self.walk(ON_BACK, righting_gain=0.)
        self.assertGreater(float(np.abs(with_burst - without).max()), .5)
