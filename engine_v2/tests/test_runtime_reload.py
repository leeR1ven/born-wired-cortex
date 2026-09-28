"""The live window must rebuild its brain from one single module generation."""

import unittest
from unittest import mock

import numpy as np

from born_wired import runtime
from born_wired.auditory_neurons import OUTPUT_KEYS, AuditoryNeurons
from born_wired.embodied import AUDITORY_KEYS_USED, EmbodiedController
from born_wired.go2_body import Go2Body


class ModuleListTests(unittest.TestCase):
    def test_every_package_module_is_reloaded_or_fixed(self):
        self.assertEqual(runtime.unlisted_modules(), [])

    def test_reload_order_respects_module_level_imports(self):
        order = list(runtime.RELOAD_ORDER)
        for name in order:
            for dependency in runtime.module_dependencies(name):
                if dependency in order:
                    self.assertLess(order.index(dependency), order.index(name),
                                    f'{name} must be reloaded after {dependency}')

    def test_a_newly_added_module_cannot_be_left_out(self):
        added = runtime.package_modules() | {'module_added_later'}
        with mock.patch.object(runtime, 'package_modules', lambda: added):
            self.assertEqual(runtime.unlisted_modules(), ['module_added_later'])
            with self.assertRaises(RuntimeError):
                runtime.reload_runtime_modules()


class AuditoryInterfaceTests(unittest.TestCase):
    def test_step_returns_exactly_the_declared_keys(self):
        output = AuditoryNeurons().step(np.zeros((2, 160)))
        self.assertEqual(set(output), set(OUTPUT_KEYS))

    def test_the_controller_reads_only_declared_keys(self):
        self.assertLessEqual(set(AUDITORY_KEYS_USED), set(OUTPUT_KEYS))


class LiveRebuildTests(unittest.TestCase):
    """Covers the rebuild path that stopped the running window on a missing key."""

    @classmethod
    def setUpClass(cls):
        cls.body = Go2Body()
        cls.observation = cls.body.observe()
        cls.environment = {name: np.zeros(4) for name in
                           ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
        cls.pixels = np.zeros(EmbodiedController.eye_shape, dtype=np.uint8)
        cls.ear_waveform = np.zeros((2, 160))

    def build(self):
        from tools.live_dog import build_brain
        return build_brain(self.body, dict(controller='embodied', parameters=dict(
            motor_units=20, proprio_units=8, association_units=8)))

    def step_once(self, brain):
        brain.step(self.observation, environment=self.environment, eye_pixels=self.pixels,
                   ear_waveform=self.ear_waveform, dt=.01, learn=False, autonomy=False)

    def test_rebuild_consumes_the_inputs_the_window_supplies(self):
        self.step_once(self.build())

    def test_rebuild_replaces_the_module_the_process_was_holding(self):
        import born_wired.auditory_neurons as auditory
        import born_wired.embodied as embodied

        class OlderCircuit:
            def __init__(self, *args, **kwargs):
                pass

            def step(self, raw):
                return dict(rates=np.zeros((2, 3)), orientation=np.zeros(2))

        # A rebuild reloads the modules, so the older definitions cannot
        # survive in the process that keeps running.
        self.addCleanup(runtime.reload_runtime_modules)
        auditory.AuditoryNeurons = OlderCircuit
        auditory.OUTPUT_KEYS = ('rates', 'orientation')
        brain = self.build()
        self.assertNotIsInstance(brain.auditory, OlderCircuit)
        self.assertIs(type(brain.auditory), auditory.AuditoryNeurons)
        self.assertIs(embodied.AuditoryNeurons, auditory.AuditoryNeurons)
        self.step_once(brain)
