"""Raw-pixel neural response tests, without interpreting activity as distance."""

import unittest
import numpy as np

from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body


class EmbodiedVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()
        cls.observation = cls.body.observe()
        cls.environment = {name: np.zeros(4) for name in
                           ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}

    def controller(self):
        return EmbodiedController(self.body.home_angles, self.body.lower_limits, self.body.upper_limits,
                                  motor_units=20, proprio_units=8, association_units=8)

    def run_pixels(self, pixels, **options):
        controller = self.controller()
        targets = []
        for _ in range(60):
            q, _ = controller.step(self.observation, environment=self.environment,
                                   eye_pixels=pixels, dt=.01, learn=False, autonomy=False, **options)
            targets.append(q)
        return controller, np.asarray(targets)

    @staticmethod
    def colored_patch(channel, side):
        pixels = np.zeros(EmbodiedController.eye_shape, dtype=np.uint8)
        start = 0 if side == 0 else EmbodiedController.eye_shape[2] - 8
        pixels[:, :, start:start+8, channel] = 255
        return pixels

    def test_shifted_bright_edges_recruit_different_binocular_cells(self):
        for offset in (2, 5):
            with self.subTest(offset=offset):
                pixels = np.zeros(EmbodiedController.eye_shape, dtype=np.uint8)
                pixels[0, :, 18, :] = 255
                pixels[1, :, 18-offset, :] = 255
                controller, _ = self.run_pixels(pixels)
                rates = np.asarray(controller.diagnostics()['binocular_population_activity'])
                self.assertGreater(rates[offset-1], .02)
                self.assertGreater(rates[offset-1], np.max(np.delete(rates, offset-1)) + .01)
                # Removing one eye removes the coincidence input, not just a
                # nominal depth output. No physical distance is assigned here.
                pixels[1] = 0
                covered, _ = self.run_pixels(pixels)
                mono = np.asarray(covered.diagnostics()['binocular_population_activity'])
                self.assertLess(np.max(mono), rates[offset-1] * .1)

    def test_irregular_texture_preserves_offset_preference(self):
        texture = np.random.default_rng(391).integers(
            0, 2, EmbodiedController.eye_shape[1:3], dtype=np.uint8) * 255
        for offset in (2, 5):
            with self.subTest(offset=offset):
                pixels = np.zeros(EmbodiedController.eye_shape, dtype=np.uint8)
                pixels[0] = texture[..., None]
                pixels[1, :, :EmbodiedController.eye_shape[2]-offset] = pixels[0, :, offset:]
                controller, _ = self.run_pixels(pixels)
                rates = np.asarray(controller.diagnostics()['binocular_population_activity'])
                self.assertGreater(rates[offset-1], np.max(np.delete(rates, offset-1)) + .05)

    def test_a_picture_as_weak_as_a_rendered_room_still_reaches_the_bank(self):
        """A rendered edge moves a contrast cell a fifth of the way, a fully
        saturated synthetic one moves it all the way, and the bank asks a
        *pair* of cells for more than either can give alone. That is why a
        saturated copy of the contrast cells sits in front of it: with the copy
        silent, a real room left the whole disparity population at zero. Both
        statements are checked here, on one weak texture with the right eye
        shifted by three columns.
        """
        def run(pixels, **overrides):
            controller = EmbodiedController(self.body.home_angles, self.body.lower_limits,
                                            self.body.upper_limits, motor_units=20,
                                            proprio_units=8, association_units=8, **overrides)
            for _ in range(60):
                controller.step(self.observation, environment=self.environment,
                                eye_pixels=pixels, dt=.01, learn=False, autonomy=False)
            return controller

        texture = 100 + 10*np.random.default_rng(77).integers(
            0, 2, EmbodiedController.eye_shape[1:3], dtype=np.uint8)
        pixels = np.zeros(EmbodiedController.eye_shape, dtype=np.uint8)
        pixels[0] = texture[..., None]
        pixels[1, :, :EmbodiedController.eye_shape[2]-3] = pixels[0, :, 3:]
        controller = run(pixels)
        rates = np.asarray(controller.diagnostics()['binocular_population_activity'])
        self.assertGreater(rates[2], .02)
        self.assertGreater(rates[2], np.max(np.delete(rates, 2)) + .01)
        silent = run(pixels, eye_relay_gain=0.)
        self.assertLess(np.max(np.asarray(silent.diagnostics()['binocular_population_activity'])), 1e-9)
        pixels[1] = 0
        one_eye = run(pixels)
        self.assertLess(np.max(np.asarray(one_eye.diagnostics()['binocular_population_activity'])),
                        rates[2] * .1)

    def test_raw_color_and_retinal_side_reach_distinct_downstream_cells(self):
        for side in (0, 1):
            green, _ = self.run_pixels(self.colored_patch(1, side))
            diagnostic = green.diagnostics()
            retinal = np.asarray(diagnostic['retinal_activity'])
            sector = 0 if side == 0 else 2
            self.assertGreater(retinal[:, sector, 1].min(), .5)
            self.assertLess(retinal[:, sector, 0].max(), .05)
            orient = diagnostic['reflex_activity']['orienting']
            self.assertGreater(orient[side], orient[1-side] + .5)
            self.assertLess(max(diagnostic['reflex_activity']['aversive']), .05)
            red, _ = self.run_pixels(self.colored_patch(0, side))
            aversive = red.diagnostics()['reflex_activity']['aversive']
            self.assertGreater(aversive[sector], .5)
            self.assertLess(max(red.diagnostics()['reflex_activity']['orienting']), .05)

    def test_invalid_raw_pixels_leave_neural_and_synaptic_state_unchanged(self):
        controller = self.controller()
        controller.step(self.observation, environment=self.environment,
                        eye_pixels=self.colored_patch(1, 0), dt=.01, learn=False)
        before = (controller.network.voltage, controller.network.adaptation,
                  controller.synapses.weights, controller.network.pixel_current.copy())
        invalid = (np.zeros((2, 24, 32, 3), dtype=np.float32),
                   np.zeros((2, 24, 32, 3), dtype=np.uint16),
                   np.zeros((24, 32, 3), dtype=np.uint8),
                   np.zeros((2, 24, 32, 3), dtype=bool))
        for pixels in invalid:
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype):
                with self.assertRaises(ValueError):
                    controller.step(self.observation, environment=self.environment,
                                    eye_pixels=pixels, dt=.01, learn=True)
                after = (controller.network.voltage, controller.network.adaptation,
                         controller.synapses.weights, controller.network.pixel_current)
                for previous, current in zip(before, after):
                    np.testing.assert_array_equal(current, previous)

    def test_visual_ablation_removes_raw_pixel_effect_on_motor_trajectory(self):
        red = self.colored_patch(0, 0)
        green = self.colored_patch(1, 0)
        for switch in ({'reflexes': False}, {'feedback': False}):
            with self.subTest(switch=switch):
                first, first_targets = self.run_pixels(red, **switch)
                second, second_targets = self.run_pixels(green, **switch)
                np.testing.assert_array_equal(first_targets, second_targets)
                np.testing.assert_array_equal(first.network.voltage, second.network.voltage)


if __name__ == '__main__':
    unittest.main()
