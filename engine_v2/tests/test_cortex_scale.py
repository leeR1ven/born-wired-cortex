"""A bigger cortex is the same animal: the picture size is one animal's own.

The eye is the only resolution knob, and it belongs to the instance, so one
process can hold animals of different sizes without one resizing the other.
What the rest of the graph reads must not change with the pixel count: the
three summary cells of a sector carry the same total incoming weight at every
size, which is why walking is unchanged when the sheet grows.
"""
import unittest

import numpy as np

from born_wired.embodied import (EmbodiedController, EYE_HEIGHT, EYE_WIDTH, RETINA_SUMMARY_GAIN,
                                 REFERENCE_SUMMARY_PIXELS, LEARNING_INTERVAL)
from born_wired.go2_body import Go2Body


class CortexScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()

    def build(self, width, height, **extra):
        return EmbodiedController(self.body.home_angles, self.body.lower_limits, self.body.upper_limits,
                                  motor_units=20, proprio_units=8, association_units=8,
                                  eye_width=width, eye_height=height, **extra)

    @staticmethod
    def incoming(brain, ids):
        return np.bincount(brain.synapses.dst, weights=brain.synapses.weights,
                           minlength=brain.network.n_neurons)[ids]

    def test_the_module_default_is_still_the_reference_animal(self):
        brain = self.build(EYE_WIDTH, EYE_HEIGHT)
        self.assertEqual((brain.eye_width, brain.eye_height), (EYE_WIDTH, EYE_HEIGHT))
        self.assertEqual(brain.eye_shape, (2, EYE_HEIGHT, EYE_WIDTH, 3))

    def test_a_bigger_eye_is_the_same_animal_at_a_finer_grain(self):
        small = self.build(48, 36)
        big = self.build(96, 72)
        self.assertEqual(small.eye_shape, (2, 36, 48, 3))
        self.assertEqual(big.eye_shape, (2, 72, 96, 3))
        self.assertEqual(big.groups['photoreceptors'].size, 4*small.groups['photoreceptors'].size)
        self.assertEqual(big.groups['retinal_opponent'].size, 4*small.groups['retinal_opponent'].size)

    def test_summary_drive_does_not_depend_on_the_pixel_count(self):
        share = RETINA_SUMMARY_GAIN*REFERENCE_SUMMARY_PIXELS
        for width, height in ((48, 36), (96, 72), (144, 108)):
            with self.subTest(width=width, height=height):
                brain = self.build(width, height)
                totals = self.incoming(brain, brain.groups['retina'])
                np.testing.assert_allclose(totals, share, rtol=2e-3)

    def test_two_sizes_coexist_in_one_process(self):
        small = self.build(48, 36)
        edges, cells = len(small.synapses.src), small.network.n_neurons
        self.build(120, 90)
        self.assertEqual((len(small.synapses.src), small.network.n_neurons), (edges, cells))
        self.assertEqual(small.eye_shape, (2, 36, 48, 3))

    def test_the_animal_learns_on_its_own_cadence_not_every_step(self):
        brain = self.build(48, 36)
        self.assertEqual(brain.network.learning_interval, LEARNING_INTERVAL)
        weights = brain.synapses.weights.copy()
        pixels = np.zeros(brain.eye_shape, dtype=np.uint8)
        environment = {name: np.zeros(4) for name in
                       ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
        observation = self.body.reset('stand')
        for _ in range(4):
            brain.step(observation, environment=environment, eye_pixels=pixels, dt=.01)
        np.testing.assert_array_equal(weights, brain.synapses.weights)
        brain.step(observation, environment=environment, eye_pixels=pixels, dt=.01)
        self.assertGreater(np.max(np.abs(brain.synapses.weights-weights)), 0.)

    def test_an_eye_too_small_to_hold_the_offset_bank_is_refused(self):
        with self.assertRaises(ValueError):
            self.build(31, 36)
        with self.assertRaises(ValueError):
            self.build(48, 23)

    def test_learning_interval_must_be_a_nonnegative_number(self):
        with self.assertRaises(ValueError):
            self.build(48, 36, learning_interval=-1)


if __name__ == '__main__':
    unittest.main()
