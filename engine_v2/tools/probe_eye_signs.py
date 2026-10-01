# -*- coding: utf-8 -*-
"""量三个东西：视角里的方位/高低怎么对应眼睛的角度、从正中出发多久稳住、先天那一套追得多准。

红球摆在若干方位/高低上，每摆一个位置让眼睛从正中出发跑一段，逐步记下
brain.eye_command()（神经给出的角度）和 observation['eye_position']（仿真里真实的角度）。
接线前必须先把这三个符号问题定下来，否则左右/上下会接反：

1. 视角里偏左的红细胞，应该驱动 +yaw 还是 -yaw？
2. 视角里偏高的红细胞，应该驱动 +pitch 还是 -pitch？
3. 从正中出发，多少步之后稳得住（后面每一题的步数按这个定）。

    python tools/probe_eye_signs.py --steps 120
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank as tb                                      # noqa: E402

DT = .01
RED = [1., 0., 0., 1.]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--distance", type=float, default=.90)
    ap.add_argument("--size", type=float, default=.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--innate", action="store_true",
                    help="保留先天那套（默认跑的就是先天那套，这个开关只是写明白）")
    args = ap.parse_args(argv)

    ctx = tb.context({}, args.seed)
    body = tb.clean_body(ctx["model_path"])
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [args.size] * 3
    body.model.geom_rgba[target] = RED
    brain, eyes = tb.brain_for(ctx, body, eyes=True)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    print("眼睛画面 %d x %d，眼肌单元 %d，眼睛行程 %s / %s"
          % (brain.eye_width, brain.eye_height, brain.eye_motor_units,
             np.round(brain.eye_lower, 3), np.round(brain.eye_upper, 3)))
    print("静止时给出的角度（没看任何东西）：%s" % np.round(brain.eye_command(), 4))
    environment = tb.blank_environment()

    def aim(bearing, elevation):
        """把球放到某个方位/高低，返回它在身体坐标系里的真实方位和高低。"""
        x = args.distance * np.cos(elevation) * np.cos(bearing)
        y = args.distance * np.cos(elevation) * np.sin(bearing)
        z = 0.32 + args.distance * np.sin(elevation)
        body.model.geom_pos[target] = [x, y, z]
        mujoco.mj_forward(body.model, body.data)
        offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
        rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
        return (float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0])),
                float(np.arctan2(offset @ rotation[:, 2],
                                 np.linalg.norm(offset @ rotation[:, :2]))))

    places = [(0., 0.), (.30, 0.), (-.30, 0.), (.15, 0.), (-.15, 0.),
              (0., .25), (0., -.25), (.20, .18)]
    print("\n%8s %8s | %s" % ("方位", "高低", " ".join(
        "命令yaw@%d" % s for s in (10, 30, 60, args.steps))))
    print("%8s %8s | %s" % ("", "", " ".join(
        "真实yaw@%d" % s for s in (10, 30, 60, args.steps))))
    for bearing, elevation in places:
        true_bearing, true_elevation = aim(bearing, elevation)
        body.reset(pose="stand", seed=0, joint_noise=0.)
        body.command_eyes(np.zeros(4))
        observation = body.observe()
        for _ in range(20):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
        trail = {}
        for step in range(args.steps):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(brain.eye_command())
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
            want = brain.eye_command()
            real = np.asarray(observation["eye_position"], dtype=float).reshape(4)
            if step + 1 in (10, 30, 60, args.steps):
                trail[step + 1] = (want.copy(), real.copy())
        command_yaw = " ".join("%7.3f" % trail[s][0][0] for s in (10, 30, 60, args.steps))
        real_yaw = " ".join("%7.3f" % trail[s][1][0] for s in (10, 30, 60, args.steps))
        print("%8.3f %8.3f | %s" % (true_bearing, true_elevation, command_yaw))
        print("%8s %8s | %s" % ("", "", real_yaw))
    print("\n参考：命令左右两眼的 yaw 平均 = 靶方位 才算对准；高低那一对看 pitch（第 1、3 个关节）")
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())