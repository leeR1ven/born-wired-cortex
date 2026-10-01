"""Two eyes, two coordinate frames.

The position-field table records where the ball stood in *head* coordinates. What one eye
has to answer is where the ball sits in *its own* picture, so the wiring converts the table
through the numbers in tools/eye_geometry.py: where the two eyeballs stand, how high the ball
was held during the sweep, and how far away it was. Those numbers used to be written out by
hand in each tool and had drifted (the sweep used base height + .03 = .30 m, the eye tools
used .32 m). The model is stood up here and every one of them is checked against it.
"""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from born_wired.go2_body import Go2Body
from tools import eye_geometry as G

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "models" / "reflex_arena.xml"


class EyeGeometry(unittest.TestCase):
    def setUp(self):
        self.body = Go2Body(model_path=ARENA)
        self.model, self.data = self.body.model, self.body.data

    def body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def test_the_two_eyeballs_stand_where_the_geometry_says(self):
        for side, want in G.EYE_POS.items():
            i = self.body_id("eye_" + side)
            self.assertGreaterEqual(i, 0, "模型里没有 eye_" + side)
            np.testing.assert_allclose(self.data.xpos[i], want, atol=1e-9)

    def test_the_ball_was_held_at_the_height_the_sweep_used(self):
        base = self.body_id("base")
        self.assertAlmostEqual(float(self.data.xpos[base][2]) + G.BALL_OVER_BASE, G.BALL_Z,
                               places=9)

    def test_placing_the_ball_lands_it_where_the_sweep_put_it(self):
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
        self.assertGreaterEqual(target, 0, "模型里没有 green_target")
        self.model.geom_size[target] = [.06]*3
        base = self.body_id("base")
        for bearing, elevation in ((0., 0.), (.30, -.20), (-.45, .40)):
            G.place_ball(self.model, self.data, target, bearing, elevation)
            np.testing.assert_allclose(self.data.geom_xpos[target],
                                       G.ball_world(bearing, elevation), atol=1e-9)
            offset = self.data.geom_xpos[target] - self.data.xpos[base]
            rotation = self.data.xmat[base].reshape(3, 3)
            self.assertAlmostEqual(float(np.arctan2(offset @ rotation[:, 1],
                                                    offset @ rotation[:, 0])), bearing, places=9)

    def test_a_ball_straight_ahead_sits_to_opposite_sides_of_the_two_eyes(self):
        left = G.seen_by([0., 0.], "left")
        right = G.seen_by([0., 0.], "right")
        self.assertLess(left[0], 0., "左眼该看到球偏在自己右边")
        self.assertGreater(right[0], 0., "右眼该看到球偏在自己左边")
        self.assertAlmostEqual(left[0], -right[0], places=9)
        self.assertAlmostEqual(left[0], -0.18131977440, places=8)

    def test_a_nearer_ball_is_seen_further_to_the_side(self):
        near = G.seen_by([0., 0.], "left", distance=.50)
        far = G.seen_by([0., 0.], "left", distance=3.00)
        self.assertLess(near[0], far[0])
        self.assertGreater(abs(near[0]), abs(far[0]))


if __name__ == "__main__":
    unittest.main()