from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from born_wired.go2_body import Go2Body, DEFAULT_MODEL
from born_wired.reflex_senses import ReflexSenses


@contextmanager
def body_with_obstacles(obstacles):
    source_dir = Path(DEFAULT_MODEL).parent
    robot = ET.parse(source_dir/"go2.xml").getroot()
    robot.find("compiler").set("meshdir", str(source_dir/"assets"))
    scene = ET.parse(source_dir/"scene.xml").getroot()
    for section in scene:
        if section.tag == "include":
            continue
        existing = robot.find(section.tag)
        if existing is not None and section.tag in ("asset", "worldbody"):
            existing.extend(list(section))
        else:
            robot.append(section)
    world = robot.find("worldbody")
    for element in obstacles:
        world.append(ET.fromstring(element))
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as handle:
        path = Path(handle.name)
        handle.write(ET.tostring(robot, encoding="unicode"))
    try:
        yield Go2Body(model_path=path)
    finally:
        path.unlink(missing_ok=True)


class ReflexSensesTests(unittest.TestCase):
    def test_no_obstacle_no_step_and_independent_output(self):
        body = Go2Body()
        senses = ReflexSenses(body, include_ranges=True)
        qpos, qvel, groups, before = body.data.qpos.copy(), body.data.qvel.copy(), body.model.geom_group.copy(), body.data.time
        observed = senses.observe()
        np.testing.assert_array_equal(observed["ray_distance"], [2, 2, 2])
        self.assertFalse(observed["ray_hit"].any())
        observed["ray_distance"][:] = 99
        observed["foot_support"][:] = False
        np.testing.assert_array_equal(senses.observe()["ray_distance"], [2, 2, 2])
        np.testing.assert_array_equal(body.data.qpos, qpos)
        np.testing.assert_array_equal(body.data.qvel, qvel)
        np.testing.assert_array_equal(body.model.geom_group, groups)
        self.assertEqual(body.data.time, before)

    def test_rays_follow_body_rotation_and_real_geometry(self):
        obstacles = ['<geom name="front_wall" type="box" pos=".9 0 .3" size=".08 .18 .2"/>',
                     '<geom name="back_wall" type="box" pos="-1.1 0 .3" size=".08 .18 .2"/>']
        with body_with_obstacles(obstacles) as body:
            senses = ReflexSenses(body, include_ranges=True)
            forward = senses.observe()
            np.testing.assert_allclose(forward["ray_distance"], [2, .47, 2], atol=1e-9)
            self.assertEqual(body.model.geom(int(forward["ray_geom_id"][1])).name, "front_wall")
            body.reset(tilt=(0, 0, np.pi))
            backward = senses.observe()
            np.testing.assert_allclose(backward["ray_distance"], [2, .67, 2], atol=1e-9)
            self.assertEqual(body.model.geom(int(backward["ray_geom_id"][1])).name, "back_wall")

    def test_support_on_nonfloor_terrain_and_slip(self):
        obstacles = ['<body name="terrain" pos="0 0 .025"><geom name="raised_ground" type="box" size=".5 .5 .025"/></body>']
        with body_with_obstacles(obstacles) as body:
            joint = int(body.model.body_jntadr[body.model.body("base").id])
            body.data.qpos[int(body.model.jnt_qposadr[joint])+2] += .1
            mujoco.mj_forward(body.model, body.data)
            for _ in range(100):
                body.step(body.home_angles)
            senses = ReflexSenses(body)
            observed = senses.observe()
            self.assertTrue(observed["foot_support"].all())
            self.assertTrue(np.all(observed["foot_load_force_n"] > 1))
            self.assertFalse(body.observe()["foot_contact"].any())
            self.assertLess(observed["foot_obstacle_force_n"].max(), 1e-9)
            dof = int(body.model.jnt_dofadr[joint])
            body.data.qvel[dof:dof+3] = [.2, 0, 0]
            mujoco.mj_forward(body.model, body.data)
            current_time = body.data.time
            slipping = senses.observe()
            self.assertGreater(np.max(slipping["foot_slip_mps"]), .1)
            self.assertEqual(body.data.time, current_time)

    def test_leg_obstacle_uses_real_nonsupport_contact(self):
        reference = Go2Body()
        location = reference.data.geom_xpos[reference.model.geom("FL").id] + [.027, 0, 0]
        position = " ".join(map(str, location))
        with body_with_obstacles([f'<geom name="foot_wall" type="box" pos="{position}" size=".015 .015 .05"/>']) as body:
            observed = ReflexSenses(body).observe()
            self.assertGreater(observed["foot_obstacle_force_n"][0], 0)
            np.testing.assert_allclose(observed["foot_obstacle_force_n"][1:], 0, atol=1e-9)

    def test_real_body_collision_and_invalid_configuration(self):
        with body_with_obstacles(['<geom name="touch_wall" type="box" pos=".345 0 .3" size=".035 .10 .15"/>']) as body:
            observed = ReflexSenses(body).observe()
            self.assertGreater(observed["body_touch_force_n"][0], 0)
        body = Go2Body()
        for settings in ({"max_range": 0}, {"max_range": np.nan}, {"max_range": True},
                         {"sensor_offset": [0, 0, np.inf]}, {"support_cosine": 1.1}):
            with self.assertRaises(ValueError):
                ReflexSenses(body, **settings)


if __name__ == "__main__":
    unittest.main()
