# -*- coding: utf-8 -*-
"""「该到」到底该按哪个原点算：眼球那个 body 的原点，还是相机自己的位置？"""
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


def main():
    spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    model, data = body.model, body.data
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    for which, side in ((0, "left"), (1, "right")):
        eye = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eye_" + side)
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_" + side)
        print("%s: 眼球体原点 %s   相机 local pos %s   cam_fovy %.1f" %
              (side, np.round(data.xpos[eye], 4).tolist(),
               np.round(model.cam_pos[cam], 4).tolist(), model.cam_fovy[cam]))
        print("        相机 local 朝向(矩阵)\n%s" % np.round(model.cam_mat0.reshape(-1, 3, 3)[cam], 3))
    print("\n球摆在正前方后（眼睛已经稳定）三个原点算出来的角度：")
    for bearing, elevation in ((0., 0.), (.30, 0.)):
        G.place_ball(model, data, target, bearing, elevation, .90)
        for _ in range(600):
            command = brain.eye_command()
            body.command_eyes(command)
            activation = brain.step(body.observe(), environment=env,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        rotation = np.asarray(data.xmat[base], dtype=float).reshape(3, 3)
        ball = np.asarray(data.geom_xpos[target], dtype=float)
        ask = W.required(body, target)
        command = brain.eye_command()
        print("\n球 %+.2f/%+.2f  左眼实际 %+.1f 度  右眼实际 %+.1f 度" %
              (bearing, elevation, np.degrees(command[0]), np.degrees(command[2])))
        for which, side in ((0, "left"), (1, "right")):
            eye = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eye_" + side)
            cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_" + side)
            for label, origin in (("眼球体", np.asarray(data.xpos[eye], dtype=float)),
                                  ("相机", np.asarray(data.cam_xpos[cam], dtype=float))):
                v = rotation.T @ (ball - origin)
                yaw = np.degrees(np.arctan2(v[1], v[0]))
                print("   %s  按%s算该到 %+6.1f 度  （和 W.required 的 %+6.1f 差 %+.1f）"
                      % (side, label, yaw, np.degrees(ask[which][0]),
                         yaw - np.degrees(ask[which][0])))
        print("   相机在头坐标系里的位置：左 %s  右 %s"
              % (np.round(rotation.T @ (np.asarray(data.cam_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'eye_left')]) - np.asarray(data.xpos[base])), 4).tolist(),
                 np.round(rotation.T @ (np.asarray(data.cam_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'eye_right')]) - np.asarray(data.xpos[base])), 4).tolist()))
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())