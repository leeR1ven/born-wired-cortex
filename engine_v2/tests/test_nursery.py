"""The nursery: a backup route, a gate that says "this moment counts", a lesson."""

from __future__ import annotations

import unittest

import numpy as np

from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body
from born_wired.reflex_controller import ReflexController
from born_wired.reflex_senses import ReflexSenses

NEURAL_DT = .01
SMALL_POPULATION = {"motor_units": 20, "proprio_units": 8, "association_units": 8}
TOUCHED = np.ones(4)
TONE = np.array([1., 0., 0., 1., 0., 0.])     # both ears, bottom band
QUIET = np.zeros(6)


def base_environment():
    result = {name: np.zeros(4) for name in
              ("body_touch", "foot_obstacle", "foot_load", "foot_slip")}
    result["foot_support"] = np.zeros(4, dtype=bool)
    return result


class NurseryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()
        cls.observation = cls.body.observe()

    def embodied(self, **over):
        return EmbodiedController(self.body.home_angles, self.body.lower_limits,
                                  self.body.upper_limits, seed=0, **SMALL_POPULATION, **over)

    def reflex(self, **over):
        return ReflexController(self.body.home_angles, self.body.lower_limits,
                                self.body.upper_limits, seed=0, balanced_gait=True,
                                **SMALL_POPULATION, **over)

    def test_the_gate_switched_off_is_the_animal_it_was(self):
        """dopamine_gain zero leaves the graph it was, hand wired in or not."""
        plain = self.embodied()
        wired = self.embodied(dopamine_gain=0., touch_dopamine_gain=4.)
        environment = base_environment()
        environment["body_touch"] = TOUCHED
        for _ in range(60):
            plain.step(self.observation, environment=environment, dt=NEURAL_DT,
                       learn=True, locomotion=.5)
            wired.step(self.observation, environment=environment, dt=NEURAL_DT,
                       learn=True, locomotion=.5)
        # Every cell is where it was: the dopamine cell projects nowhere, so
        # wiring a hand into it cannot reach any of the others.
        self.assertEqual(plain.network.n_neurons, wired.network.n_neurons)
        elsewhere = np.ones(plain.network.n_neurons, dtype=bool)
        elsewhere[wired.groups["dopamine"][0]] = False
        np.testing.assert_array_equal(plain.network.activity[elsewhere],
                                      wired.network.activity[elsewhere])
        self.assertEqual(wired.nursery_state()["edges"], 0)

    def test_a_backup_route_carries_nothing_until_a_lesson_writes_it(self):
        brain = self.reflex(backup_routes=(("cochlea", "retreat"),))
        state = brain.nursery_state()
        self.assertEqual(state["edges"], 6)
        self.assertEqual(state["weights"], [1e-4]*6)
        environment = base_environment()
        # A loud tone with nobody there: learning off, the route may not move.
        for _ in range(20):
            brain.step(self.observation, environment=environment, auditory_activity=TONE,
                       dt=NEURAL_DT, learn=False)
        self.assertEqual(brain.nursery_state()["weights"], [1e-4]*6)
        quiet = float(np.mean(brain.network.rates_at(brain.groups["retreat"])))
        self.assertLess(quiet, .05)

    def test_a_lesson_writes_the_route_and_then_the_tone_alone_drives_it(self):
        brain = self.reflex(backup_routes=(("cochlea", "retreat"),),
                            dopamine_gain=.9, touch_dopamine_gain=4.)
        environment = base_environment()

        def lesson(steps):
            environment["body_touch"] = TOUCHED
            for _ in range(steps):
                brain.step(self.observation, environment=environment, auditory_activity=TONE,
                           dt=NEURAL_DT, learn=True, locomotion=.5)

        def only_the_tone(steps):
            environment["body_touch"] = np.zeros(4)
            for _ in range(steps):
                brain.step(self.observation, environment=environment, auditory_activity=TONE,
                           dt=NEURAL_DT, learn=False)
            return float(np.mean(brain.network.rates_at(brain.groups["retreat"])))

        def in_silence(steps):
            environment["body_touch"] = np.zeros(4)
            for _ in range(steps):
                brain.step(self.observation, environment=environment, auditory_activity=QUIET,
                           dt=NEURAL_DT, learn=False)
            return float(np.mean(brain.network.rates_at(brain.groups["retreat"])))

        lesson(300)
        grown = brain.nursery_state()["weights"]
        # The two cochlear cells the tone drives are the two that grew; the
        # four bands it does not drive cannot have moved.
        self.assertGreater(grown[0], .2)
        self.assertGreater(grown[3], .2)
        self.assertEqual(sorted(set(grown))[0], 1e-4)
        heard = only_the_tone(50)
        silent = in_silence(50)
        self.assertGreater(heard, silent + .1)
        self.assertGreater(heard, .2)

    def test_the_hand_is_what_releases_dopamine(self):
        brain = self.reflex(backup_routes=(("cochlea", "retreat"),),
                            dopamine_gain=.9, touch_dopamine_gain=4.)
        environment = base_environment()
        cell = brain.groups["dopamine"][0]
        quiet = 0.
        for _ in range(20):
            brain.step(self.observation, environment=environment, dt=NEURAL_DT, learn=False)
            quiet = max(quiet, float(brain.network.activity[cell]))
        environment["body_touch"] = TOUCHED
        held = 0.
        for _ in range(20):
            brain.step(self.observation, environment=environment, dt=NEURAL_DT, learn=False)
            held = max(held, float(brain.network.activity[cell]))
        self.assertLess(quiet, .05)
        self.assertGreater(held, .5)

    def test_the_nursery_refuses_what_it_cannot_mean(self):
        for over in (dict(dopamine_gain=1.5), dict(dopamine_gain=-.1),
                     dict(backup_routes=(("cochlea", "no_such_group"),)),
                     dict(backup_routes=("cochlea",))):
            with self.assertRaises(ValueError):
                self.reflex(**over)


if __name__ == "__main__":
    unittest.main()
