"""Independent behavioural checks for the reflex controller candidate."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from born_wired.go2_body import Go2Body
from born_wired.reflex_controller import ReflexController
from born_wired.reflex_senses import ReflexSenses


NEURAL_DT = 0.01
SMALL_POPULATION = {
    "motor_units": 20,
    "proprio_units": 8,
    "association_units": 8,
}


def group_activity(controller: ReflexController, name: str) -> np.ndarray:
    """Read a named cell population without depending on its wiring."""
    return controller.network.activity[np.asarray(controller.groups[name], dtype=int)]


class ReflexControllerChecks(unittest.TestCase):
    def setUp(self):
        self.body = Go2Body()
        self.observation = self.body.observe()
        self.senses = ReflexSenses(self.body)
        self.environment = self.senses.observe()

    def controller(self, *, seed: int = 0, **overrides) -> ReflexController:
        parameters = dict(SMALL_POPULATION)
        parameters.update(overrides)
        return ReflexController(
            self.body.home_angles,
            self.body.lower_limits,
            self.body.upper_limits,
            seed=seed,
            balanced_gait=True,
            **parameters,
        )

    def environment_with(self, **updates) -> dict:
        environment = copy.deepcopy(self.environment)
        environment.update(updates)
        return environment

    def step(self, controller: ReflexController, environment: dict, **kwargs):
        kwargs.setdefault("dt", NEURAL_DT)
        kwargs.setdefault("learn", False)
        return controller.step(self.observation, environment=environment, **kwargs)

    def test_malformed_inputs_do_not_mutate_neural_state_or_weights(self):
        controller = self.controller()
        malformed = (
            ("auditory_activity", {}, {"auditory_activity": np.full((2,3), np.nan)}),
            ("body_touch", {"body_touch": np.ones(3)}, {}),
            ("foot_obstacle", {"foot_obstacle": [0, 0, 0, np.inf]}, {}),
            ("foot_load", {"foot_load": [0, 0, 0, 1.1]}, {}),
            ("foot_slip", {"foot_slip": [0, 0, -0.1, 0]}, {}),
            ("foot_support", {"foot_support": [2, 0, 0, 0]}, {}),
            ("motor_effort", {"motor_effort": -0.1}, {}),
            ("startle", {}, {"startle": 1.1}),
            ("reflexes", {}, {"reflexes": 1}),
            ("autonomy", {}, {"autonomy": 0}),
            ("feedback", {}, {"feedback": 1}),
        )
        for name, environment_updates, step_updates in malformed:
            with self.subTest(name=name):
                environment = self.environment_with(**environment_updates)
                before = (
                    controller.network.voltage.copy(),
                    controller.network.adaptation.copy(),
                    controller.network.activity.copy(),
                    controller.synapses.weights.copy(),
                )
                with self.assertRaises((ValueError, FloatingPointError)):
                    self.step(controller, environment, learn=True, **step_updates)
                np.testing.assert_array_equal(controller.network.voltage, before[0])
                np.testing.assert_array_equal(controller.network.adaptation, before[1])
                np.testing.assert_array_equal(controller.network.activity, before[2])
                np.testing.assert_array_equal(controller.synapses.weights, before[3])

        observation = dict(self.observation)
        observation["joint_position"] = np.zeros(11)
        before = (
            controller.network.voltage.copy(),
            controller.network.adaptation.copy(),
            controller.synapses.weights.copy(),
        )
        with self.assertRaises(ValueError):
            controller.step(observation, environment=self.environment, dt=NEURAL_DT, learn=True)
        np.testing.assert_array_equal(controller.network.voltage, before[0])
        np.testing.assert_array_equal(controller.network.adaptation, before[1])
        np.testing.assert_array_equal(controller.synapses.weights, before[2])

    def test_repeated_startle_input_adapts_transient(self):
        controller = self.controller()
        period = 5
        pulses = 8
        samples = period * pulses + 5
        startle_activity = []
        for index in range(samples):
            startle = 1.0 if index < period * pulses and index % period == 0 else 0.0
            self.step(
                controller,
                self.environment,
                autonomy=False,
                locomotion=0.0,
                startle=startle,
                learn=False,
            )
            startle_activity.append(float(group_activity(controller, "startle")[0]))

        windows = [
            startle_activity[index * period:(index + 1) * period]
            for index in range(pulses)
        ]
        early_peak = max(max(window) for window in windows[:3])
        late_peak = max(max(window) for window in windows[-3:])
        self.assertGreater(early_peak, 0.1)
        self.assertLess(late_peak, early_peak)

    def test_frontal_touch_inhibits_locomotion(self):
        controller_clear = self.controller(seed=7)
        controller_obstacle = self.controller(seed=7)
        clear = self.environment_with(body_touch=[0., 0., 0., 0.])
        obstacle = self.environment_with(body_touch=[.8, 0., 0., 0.])
        clear_activity = []
        obstacle_activity = []
        for _ in range(80):
            self.step(
                controller_clear,
                clear,
                autonomy=False,
                locomotion=0.65,
                learn=False,
            )
            self.step(
                controller_obstacle,
                obstacle,
                autonomy=False,
                locomotion=0.65,
                learn=False,
            )
            clear_activity.append(float(group_activity(controller_clear, "locomotion")[0]))
            obstacle_activity.append(float(group_activity(controller_obstacle, "locomotion")[0]))

        clear_mean = float(np.mean(clear_activity[10:]))
        obstacle_mean = float(np.mean(obstacle_activity[10:]))
        self.assertGreater(clear_mean, 0.1)
        self.assertLess(obstacle_mean, clear_mean)

    def test_effort_increases_fatigue_and_lowers_initiation(self):
        controller_low = self.controller(seed=11)
        controller_high = self.controller(seed=11)
        low = self.environment_with(motor_effort=0.0)
        high = self.environment_with(motor_effort=1.0)
        low_fatigue = []
        high_fatigue = []
        low_initiation = []
        high_initiation = []
        for _ in range(500):
            self.step(controller_low, low, autonomy=True, locomotion=0.0, learn=False)
            self.step(controller_high, high, autonomy=True, locomotion=0.0, learn=False)
            low_fatigue.append(float(group_activity(controller_low, "fatigue")[0]))
            high_fatigue.append(float(group_activity(controller_high, "fatigue")[0]))
            low_initiation.append(float(group_activity(controller_low, "initiation")[0]))
            high_initiation.append(float(group_activity(controller_high, "initiation")[0]))

        self.assertGreater(float(np.mean(high_fatigue[-100:])), float(np.mean(low_fatigue[-100:])))
        self.assertLess(float(np.mean(high_initiation[-100:])), float(np.mean(low_initiation[-100:])))

    def test_sensor_effects_absent_when_reflexes_disabled(self):
        controller_clear = self.controller(seed=13)
        controller_rich = self.controller(seed=13)
        clear = self.environment_with()
        rich = self.environment_with(
            ray_distance=[2.0, 0.25, 2.0],
            body_touch=[1.0, 1.0, 1.0, 1.0],
            foot_obstacle=[1.0, 1.0, 1.0, 1.0],
            foot_load=[1.0, 1.0, 1.0, 1.0],
            foot_slip=[1.0, 1.0, 1.0, 1.0],
            motor_effort=1.0,
        )
        for _ in range(20):
            target_clear, activation_clear = self.step(
                controller_clear,
                clear,
                autonomy=False,
                locomotion=0.65,
                reflexes=False,
                learn=False,
            )
            target_rich, activation_rich = self.step(
                controller_rich,
                rich,
                autonomy=False,
                locomotion=0.65,
                reflexes=False,
                startle=1.0,
                learn=False,
            )
            np.testing.assert_array_equal(target_clear, target_rich)
            np.testing.assert_array_equal(activation_clear, activation_rich)

    def test_a_sound_from_behind_recruits_orienting_on_that_side(self):
        """The outer ear's behind cell turns the animal toward the ear it fired
        on, and its ahead cell is deliberately left with no output."""
        def settled(pinna, gain=1.2, steps=40):
            controller = self.controller(pinna_orient_gain=gain)
            for _ in range(steps):
                self.step(controller, self.environment, autonomy=False, locomotion=0.,
                          auditory_pinna_activity=pinna)
            return group_activity(controller, "orienting")

        quiet = settled(np.zeros(4))
        behind_left = settled((0., 1., 0., 0.))     # ear 0 is the left ear
        behind_right = settled((0., 0., 0., 1.))
        ahead = settled((1., 0., 0., 0.))
        silent = settled((0., 1., 0., 0.), gain=0.)
        self.assertGreater(behind_left[0]-behind_left[1], quiet[0]-quiet[1] + .05)
        self.assertGreater(behind_right[1]-behind_right[0], quiet[1]-quiet[0] + .05)
        np.testing.assert_allclose(ahead, quiet, atol=1e-12)
        np.testing.assert_allclose(silent, quiet, atol=1e-12)

    def test_a_light_lasting_wall_contact_turns_the_body_aside(self):
        """A wall is not a knock.

        The contact a body makes with a wall is light - a tenth of the touch
        scale - and it lasts, so a reflex that only answers the instant of the
        hit does nothing with it. One slow cell per sector holds that contact
        and reaches the hips on a route of its own: the shared avoidance gain
        cannot be raised far enough to turn the body without the walking body
        falling over, which is measured in the implementation report.
        """
        def settled(gain=12., hip=.8, touch=(.06, 0., 0., 0.), steps=250):
            controller = self.controller(wall_gain=gain, wall_hip_gain=hip)
            targets = None
            for _ in range(steps):
                targets, _ = self.step(controller,
                                       self.environment_with(body_touch=list(touch)),
                                       autonomy=False, locomotion=0., flexion=0.)
            return controller, targets

        touched, targets = settled()
        quiet, quiet_targets = settled(touch=(0., 0., 0., 0.))
        silenced, silent_targets = settled(gain=0.)
        # The contact state answers the light push and outlasts it...
        self.assertGreater(float(group_activity(touched, "wall_contact")[0]), .35)
        self.assertLess(float(np.max(group_activity(quiet, "wall_contact"))), .01)
        # ...it turns the hips on its own...
        self.assertGreater(float(np.mean(group_activity(touched, "wall_turn"))), .15)
        hips = np.array([0, 3, 6, 9])
        self.assertGreater(float(np.abs(targets[hips] - quiet_targets[hips]).max()), .01)
        # ...and silencing it leaves the contact cell at zero, so the hips do
        # not move for the same light push.
        self.assertLess(float(np.max(group_activity(silenced, "wall_contact"))), 1e-12)
        hips = np.array([0, 3, 6, 9])
        # Without it, the same light push reaches the hips only through the
        # avoidance tilt, which measured is a hundredth of a radian.
        self.assertLess(float(np.abs(silent_targets[hips] - quiet_targets[hips]).max()), .02)
        self.assertGreater(float(np.abs(targets[hips] - silent_targets[hips]).max()), .01)

    def test_a_lean_lights_its_direction_and_that_side_steadies_it(self):
        """Gravity in body axes says which of four ways the body is going over.

        The leg on that side is the one that reaches out: its knee extends and
        its hip turns outward, which widens the base. Both are the mirror of
        the withdrawal reflex on the same motor units, and both are silent
        while the body is upright.
        """
        def settled(gravity, **overrides):
            controller = self.controller(**overrides)
            observation = dict(self.observation)
            observation["gravity_direction"] = np.asarray(gravity, dtype=float)
            targets = None
            for _ in range(120):
                targets, _ = controller.step(observation, environment=self.environment,
                                             dt=NEURAL_DT, learn=False, autonomy=False,
                                             locomotion=0.)
            return controller, targets

        # A lean small enough that the protective circuit has not taken over.
        upright, upright_targets = settled((0., 0., -1.))
        left, left_targets = settled((0., .4, -.9165))
        right, right_targets = settled((0., -.4, -.9165))
        forward, _ = settled((.4, 0., -.9165))
        np.testing.assert_allclose(group_activity(upright, "lean"), 0., atol=1e-12)
        self.assertGreater(float(left.network.activity[left.groups["lean"][2]]), .2)
        self.assertLess(float(left.network.activity[left.groups["lean"][3]]), .01)
        self.assertGreater(float(right.network.activity[right.groups["lean"][3]]), .2)
        self.assertGreater(float(forward.network.activity[forward.groups["lean"][0]]), .2)
        # The left legs answer a lean to the left, the right ones a lean right.
        # They also reach out with the hip: measured on this body, turning a
        # left hip outward moves the body right, so that is the side that
        # widens the base under a lean to the left.
        left_steady = group_activity(left, "steady")
        self.assertGreater(float(left_steady[0]), float(left_steady[1]) + .1)
        self.assertGreater(float(left_steady[2]), float(left_steady[3]) + .1)
        right_steady = group_activity(right, "steady")
        self.assertGreater(float(right_steady[1]), float(right_steady[0]) + .1)
        self.assertGreater(float(np.abs(left_targets - upright_targets).max()), .02)
        # Isolating the reflex: the same lean with it silenced moves the body
        # less, and the legs on the other side are left alone.
        _, off_targets = settled((0., .4, -.9165), steady_gain=0., steady_hip_gain=0.)
        self.assertGreater(float(np.abs(left_targets[[0, 6]] - off_targets[[0, 6]]).max()), .02)
        np.testing.assert_allclose(left_targets[[3, 9]], off_targets[[3, 9]], atol=1e-12)
        np.testing.assert_array_equal(off_targets[[0, 3, 6, 9]], upright_targets[[0, 3, 6, 9]])

    def test_outputs_are_finite_bounded_and_plastic(self):
        controller = self.controller(seed=17)
        initial_weights = controller.synapses.weights.copy()
        initial_body_time = self.body.data.time
        environment = self.environment_with(
            ray_distance=[1.4, 0.8, 1.8],
            body_touch=[0.1, 0.2, 0.05, 0.0],
            foot_obstacle=[0.0, 0.3, 0.0, 0.1],
            foot_load=[0.2, 0.1, 0.3, 0.0],
            foot_slip=[0.1, 0.2, 0.0, 0.05],
            motor_effort=0.5,
        )
        for _ in range(50):
            target, activation = self.step(
                controller,
                environment,
                autonomy=False,
                locomotion=0.65,
                learn=True,
            )
            self.assertTrue(np.isfinite(target).all())
            self.assertTrue(np.isfinite(activation).all())
            self.assertTrue(np.all(target >= controller.lower - 1e-12))
            self.assertTrue(np.all(target <= controller.upper + 1e-12))
            self.assertTrue(np.all((activation >= 0) & (activation <= 1)))
            self.assertTrue(np.isfinite(controller.network.voltage).all())
            self.assertTrue(np.isfinite(controller.network.adaptation).all())
            self.assertTrue(np.all((controller.network.adaptation >= 0) &
                                   (controller.network.adaptation <= 1)))
            self.assertTrue(np.isfinite(controller.network.activity).all())
            self.assertTrue(np.all((controller.network.activity >= 0) &
                                   (controller.network.activity <= 1)))
            weights = controller.synapses.weights
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue(np.all(weights >= controller.synapses.lower - 1e-12))
            self.assertTrue(np.all(weights <= controller.synapses.w_max + 1e-12))
            diagnostics = controller.diagnostics()
            self.assertTrue(np.isfinite(diagnostics["max_weight_change"]))
            self.assertTrue(np.isfinite(diagnostics["scaffold_max_relative_change"]))
            self.assertTrue(np.isfinite(diagnostics["mean_activity"]))
            self.assertTrue(np.isfinite(diagnostics["active_fraction"]))
            self.assertTrue(all(np.isfinite(values).all()
                                for values in diagnostics["reflex_activity"].values()))

        self.assertTrue(np.any(controller.synapses.plasticity > 0))
        self.assertGreater(float(np.max(np.abs(controller.synapses.weights - initial_weights))), 0.0)
        self.assertEqual(self.body.data.time, initial_body_time)


if __name__ == "__main__":
    unittest.main()
