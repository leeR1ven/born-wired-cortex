from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from born_wired.go2_body import Go2Body, DEFAULT_MODEL
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes, StereoSenses


@contextmanager
def stereo_body(distance=.8, textured=True, eye_screen=False):
    source = Path(DEFAULT_MODEL).parent
    robot = ET.parse(source / "go2.xml").getroot()
    robot.find("compiler").set("meshdir", str(source / "assets"))
    scene = ET.parse(source / "scene.xml").getroot()
    for section in scene:
        if section.tag == "include":
            continue
        existing = robot.find(section.tag)
        if existing is not None and section.tag in ("asset", "worldbody"):
            existing.extend(list(section))
        else:
            robot.append(section)
    world = robot.find("worldbody")
    base = world.find("body[@name='base']")
    for name, lateral in (("left", .055), ("right", -.055)):
        ET.SubElement(base, "camera", name=f"eye_{name}", pos=f".30 {lateral} .05",
                      xyaxes="0 -1 0 0 0 1", fovy="60")
    # A real textured geometry rendered by MuJoCo, not synthesized disparities.
    asset = robot.find("asset")
    texture = ET.SubElement(asset, "texture", name="range_checker", type="2d")
    ET.SubElement(asset, "material", name="range_checker", texture="range_checker", texrepeat="1 1")
    material = dict(material="range_checker") if textured else dict(rgba=".5 .5 .5 1")
    ET.SubElement(world, "geom", name="range_target", type="box",
                  pos=f"{.30 + distance + .04} 0 .32", size=".04 .8 .6", **material)
    if eye_screen:
        # An actual opaque body-mounted cover, leaving the other camera exposed.
        ET.SubElement(base, "geom", name="eye_cover", type="box", pos=".36 .055 .05",
                      size=".004 .05 .04", rgba=".5 .5 .5 1", contype="0", conaffinity="0")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.xml"
        texture_path = Path(directory) / "checker.png"
        checker = np.indices((16, 16)).sum(axis=0) % 2
        shades = .1 + .6 * checker + np.random.default_rng(172).uniform(0, .25, checker.shape)
        pixels = np.repeat(np.repeat(shades, 16, axis=0), 16, axis=1)
        Image.fromarray(np.repeat((255 * pixels).astype(np.uint8)[..., None], 3, axis=2)).save(texture_path)
        texture.set("file", str(texture_path))
        path.write_text(ET.tostring(robot, encoding="unicode"), encoding="utf-8")
        yield Go2Body(model_path=path)


class StereoSensesTests(unittest.TestCase):
    def test_raw_eyes_are_independent_of_stereo_initialization_and_processing(self):
        with stereo_body() as body:
            # Raw cameras also work without the offline rectified geometry.
            body.model.cam_pos[body.model.camera("eye_right").id, 2] += .01
            mujoco.mj_forward(body.model, body.data)
            qpos, before = body.data.qpos.copy(), body.data.time
            with (patch.object(StereoSenses, "__init__", side_effect=AssertionError("offline calibration called")),
                  patch.object(StereoSenses, "_match", side_effect=AssertionError("offline matching called")),
                  patch.object(StereoSenses, "_retinal_channels", side_effect=AssertionError("offline features called")),
                  RawEyes(body) as eyes):
                rgb = eyes.observe_raw()
                self.assertEqual(rgb.shape, EmbodiedController.eye_shape)
                self.assertEqual(rgb.dtype, np.uint8)
                self.assertGreater(float(np.std(rgb)), 10)
                self.assertFalse(hasattr(eyes, "baseline"))
                self.assertFalse(hasattr(eyes, "focal_px"))
                self.assertEqual(body.data.time, before)
                np.testing.assert_array_equal(body.data.qpos, qpos)
            eyes.close()
            with self.assertRaises(RuntimeError):
                eyes.observe_raw()

    def test_real_rendered_stereo_range_and_no_physics_mutation(self):
        measured = []
        disparities = []
        for truth in (.8, 1.4):
            with stereo_body(truth) as body, StereoSenses(body) as eyes:
                qpos, qvel, before = body.data.qpos.copy(), body.data.qvel.copy(), body.data.time
                sample = eyes.observe()
                central = sample["valid"][1:3, 2:6]
                self.assertGreaterEqual(np.sum(central), 3)
                estimate = np.median(sample["distance"][1:3, 2:6][central])
                measured.append(float(estimate))
                disparities.append(float(np.median(sample["disparity"][1:3, 2:6][central])))
                self.assertLess(abs(estimate / truth - 1), .20)
                self.assertEqual(sample["rgb"].shape, (2, 64, 96, 3))
                self.assertEqual(sample["rgb"].dtype, np.uint8)
                np.testing.assert_array_equal(body.data.qpos, qpos)
                np.testing.assert_array_equal(body.data.qvel, qvel)
                self.assertEqual(body.data.time, before)
        self.assertGreater(disparities[0], disparities[1])
        self.assertGreater(measured[1], measured[0])

    def test_blank_and_covered_eye_do_not_report_near_depth(self):
        with stereo_body(textured=False) as body, StereoSenses(body) as eyes:
            sample = eyes.observe()
            self.assertFalse(sample["valid"][1:3, 2:6].any())
            np.testing.assert_array_equal(sample["distance"][1:3, 2:6], 3)
            np.testing.assert_array_equal(sample["confidence"][1:3, 2:6], 0)
        with stereo_body(eye_screen=True) as body, StereoSenses(body) as eyes:
            sample = eyes.observe()
            self.assertFalse(sample["valid"][1:3, 2:6].any())

    def test_opponent_channels_respond_to_actual_eye_cover_color(self):
        with stereo_body(eye_screen=True) as body, StereoSenses(body) as eyes:
            cover = body.model.geom("eye_cover").id
            body.model.geom_rgba[cover] = [1, 0, 0, 1]
            sample = eyes.observe()
            colors = sample["color_opponent"]
            self.assertEqual(colors.shape, (2, 3, 3))
            self.assertTrue(np.all((colors >= 0) & (colors <= 1)))
            self.assertGreater(colors[0, 1, 0], .25)
            self.assertGreater(colors[0, 1, 0], colors[1, 1, 0] + .15)
            self.assertLess(colors[0, 1, 1], .02)

    def test_missing_camera_and_close_validation(self):
        with self.assertRaises(ValueError):
            StereoSenses(Go2Body())
        with stereo_body() as body:
            eyes = StereoSenses(body)
            eyes.close()
            eyes.close()
            with self.assertRaises(RuntimeError):
                eyes.observe()
            with self.assertRaises(RuntimeError):
                eyes.observe_raw()
            with self.assertRaises(ValueError):
                StereoSenses(body, width=3)
            with self.assertRaises(ValueError):
                StereoSenses(body, max_range=np.inf)
            body.model.cam_pos[body.model.camera("eye_right").id, 2] += .01
            with self.assertRaises(ValueError):
                StereoSenses(body)

    def test_raw_control_path_does_no_computed_feature_extraction(self):
        with stereo_body() as body, StereoSenses(body) as eyes:
            def forbidden(*_):
                self.fail("computed diagnostic features entered the raw path")
            eyes._match = forbidden
            eyes._retinal_channels = forbidden
            before, qpos = body.data.time, body.data.qpos.copy()
            rgb = eyes.observe_raw()
            self.assertEqual(rgb.shape, (2, 64, 96, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertGreater(float(np.std(rgb)), 10)
            self.assertEqual(body.data.time, before)
            np.testing.assert_array_equal(body.data.qpos, qpos)

    def test_periodic_matching_ambiguity_is_missing_not_close(self):
        with stereo_body() as body, StereoSenses(body) as eyes:
            # An isolated matcher regression supplements the physical render
            # tests: multiple equally good correspondences must be rejected.
            stripe = ((np.arange(96) // 4) % 2).astype(np.float32)
            left = np.repeat(np.repeat(stripe[None, :, None], 64, axis=0), 3, axis=2)
            right = np.roll(left, -3, axis=1)
            _, confidence, distance, valid = eyes._match(left, right)
            self.assertFalse(valid[:, 2:6].any())
            np.testing.assert_array_equal(distance[:, 2:6], 3)
            np.testing.assert_array_equal(confidence[:, 2:6], 0)


if __name__ == "__main__":
    unittest.main()
