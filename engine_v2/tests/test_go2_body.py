import unittest

import mujoco
import numpy as np

from born_wired.go2_body import Go2Body


class Go2BodyTests(unittest.TestCase):
    def setUp(self):
        self.body = Go2Body()

    def test_named_mapping_and_reset_limits(self):
        body = self.body
        for i, name in enumerate(body.joint_names):
            joint = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.removesuffix("_joint"))
            self.assertEqual(body.model.actuator_trnid[actuator, 0], joint)
            self.assertEqual(body.home_angles[i], body.model.key_qpos[0, body.model.jnt_qposadr[joint]])
        np.testing.assert_array_equal(body.observe()["joint_position"], body.home_angles)
        for pose in ("stand", "crouch"):
            first = body.reset(pose, seed=17, joint_noise=0.01, tilt=(0.05, -0.03))
            second = body.reset(pose, seed=17, joint_noise=0.01, tilt=(0.05, -0.03))
            np.testing.assert_array_equal(first["joint_position"], second["joint_position"])
            self.assertTrue(np.all(first["joint_position"] >= body.lower_limits))
            self.assertTrue(np.all(first["joint_position"] <= body.upper_limits))

    def test_physics_and_real_foot_contacts(self):
        body = self.body
        target = body.home_angles.copy()
        for _ in range(50):
            observation = body.step(target)
        self.assertAlmostEqual(observation["time"], 1.0)
        self.assertTrue(all(np.isfinite(v).all() for v in observation.values()))
        np.testing.assert_array_equal(target, body.home_angles)
        self.assertTrue(observation["foot_contact"].any())
        body.data.qpos[body._base_qpos + 2] += 1.0
        mujoco.mj_forward(body.model, body.data)
        self.assertFalse(body.observe()["foot_contact"].any())

    def test_invalid_commands_never_advance_physics(self):
        body = self.body
        bad_targets = (np.zeros(11), np.full(12, np.nan), np.full(12, np.inf), body.upper_limits + 1)
        for target in bad_targets:
            with self.assertRaises(ValueError):
                body.step(target)
            self.assertEqual(body.data.time, 0)
        for duration in (0, -1, np.nan, np.inf, 0.003, True):
            with self.assertRaises(ValueError):
                body.step(body.home_angles, duration)
            self.assertEqual(body.data.time, 0)
        with self.assertRaises(ValueError):
            body.apply_force([0, np.nan, 0], 0.1)
        for activation in (np.zeros(11), np.full(12, np.nan), np.full(12, -0.1), np.full(12, 1.1)):
            with self.assertRaises(ValueError):
                body.step(body.home_angles, activation=activation)
        self.assertEqual(body.data.time, 0)

    def test_zero_activation_removes_active_torque(self):
        body = self.body
        body.reset(joint_noise=0.05, seed=2)
        body.step(body.home_angles, activation=np.zeros(12))
        np.testing.assert_array_equal(body.data.ctrl[body._actuators], np.zeros(12))
        body.step(body.home_angles, activation=np.ones(12))
        self.assertGreater(np.max(np.abs(body.data.ctrl[body._actuators])), 0)

    def test_zero_and_small_force_with_expiry(self):
        body = self.body
        body.apply_force([0, 0, 0], 0.02)
        self.assertEqual(body.data.time, 0)
        zero = body.step(body.home_angles)
        body.reset()
        control = body.step(body.home_angles)
        np.testing.assert_array_equal(zero["joint_position"], control["joint_position"])
        body.apply_force([2, 0, 0], 0.04)
        initial_time = body.data.time
        body.step(body.home_angles)
        self.assertEqual(body.data.xfrc_applied[body._base, 0], 2)
        body.step(body.home_angles)
        self.assertAlmostEqual(body.data.time - initial_time, 0.04)
        np.testing.assert_array_equal(body.data.xfrc_applied[body._base], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
