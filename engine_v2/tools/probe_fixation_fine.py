# -*- coding: utf-8 -*-
"""用高分辨率画面量「红球偏了这只眼多少度」——低分辨率下红球只有几个像素，量不准。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco
import wire_red_gaze as W
import eye_geometry as G
from tools import taskbank as tb

DT = .01
WIDE, HIGH = 480, 360


def main():
    spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    model, data = body.model, body.data
    renderer = [mujoco.Renderer(model, height=HIGH, width=WIDE) for _ in range(2)]
    cameras = [mujoco.MjvCamera() for _ in range(2)]
    for i, side in enumerate(("left", "right")):
        cameras[i].type = mujoco.mjtCamera.mjCAMERA_FIXED
        cameras[i].fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_" + side)
    half_h = np.radians(model.cam_fovy[0])/2
    per_pixel = np.degrees(np.arctan(np.tan(half_h)*WIDE/HIGH))/WIDE

    print("球位置            左眼：几何该到 / 实际 / 高分辨率画面偏心   右眼：几何该到 / 实际 / 偏心")
    for bearing, elevation in ((0., 0.), (.30, 0.), (-.30, 0.) , (0., .30), (0., -.30), (.45, 0.)):
        G.place_ball(model, data, target, bearing, elevation, .90)
        for _ in range(600):
            command = brain.eye_command()
            body.command_eyes(command)
            activation = brain.step(body.observe(), environment=env,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        ask = W.required(body, target)
        command = brain.eye_command()
        cells = []
        for i in range(2):
            renderer[i].update_scene(data, camera=cameras[i])
            image = np.asarray(renderer[i].render()).astype(int)
            mask = (image[:, :, 0] > 150) & (image[:, :, 1] < 90) & (image[:, :, 2] < 90)
            if mask.sum() < 20:
                cells.append((None, mask.sum())); continue
            rows, columns = np.nonzero(mask)
            cells.append(((columns.mean() - WIDE/2.)*per_pixel,
                          (rows.mean() - HIGH/2.)*per_pixel))
        left = cells[0][0]; right = cells[1][0]
        print(" %+.2f/%+.2f  %8.1f %8.1f %14s   %8.1f %8.1f %14s" %
              (bearing, elevation,
               np.degrees(ask[0][0]), np.degrees(command[0]),
               "看不见" if left is None else "%+.2f度" % left,
               np.degrees(ask[1][0]), np.degrees(command[2]),
               "看不见" if right is None else "%+.2f度" % right))
    for r in renderer:
        r.close()
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())