# -*- coding: utf-8 -*-
"""把眼球直接转到一个已知角度，看画面里红球的亮块往哪边跑。

大脑不参与：用 body.command_eyes() 直接命令关节角，所以读数只反映"角度正负 = 看哪边"。
红球先摆在正前方，再分别把左眼 yaw / pitch 转 +0.30 弧度，看亮块中心行/列怎么变。

    python tools/probe_eye_axis.py
"""
import sys
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb

DT = .01
EYE_W, EYE_H = 48, 36


def blob(brain):
    rates = np.asarray(brain.network.activity, dtype=float)
    red = rates[np.asarray(brain.groups["retinal_opponent"])[0::3]][:EYE_H * EYE_W]
    patch = red.reshape(EYE_H, EYE_W)
    total = patch.sum()
    if total <= 1e-9:
        return None
    row = float((patch.sum(axis=1) * np.arange(EYE_H)).sum() / total)
    col = float((patch.sum(axis=0) * np.arange(EYE_W)).sum() / total)
    return row, col, float(patch.max()), total


def main():
    ctx = tb.context({}, 0)
    body = tb.clean_body(ctx["model_path"])
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06] * 3
    body.model.geom_rgba[target] = [1., 0., 0., 1.]
    body.model.geom_pos[target] = [.90, 0., .32]
    mujoco.mj_forward(body.model, body.data)
    brain, eyes = tb.brain_for(ctx, body, eyes=True)
    environment = tb.blank_environment()
    print("画面 %d x %d，列中值 %.1f，行中值 %.1f" % (EYE_W, EYE_H, (EYE_W - 1) / 2., (EYE_H - 1) / 2.))

    def settle(angles, steps=40):
        observation = body.observe()
        for _ in range(steps):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(np.asarray(angles, dtype=float))
            observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        return blob(brain)

    base = settle([0., 0., 0., 0.])
    print("眼球归零  亮块中心 (行 %6.2f, 列 %6.2f)  峰值 %.3f" % base[:3])
    for name, angles in (("左眼 yaw = +0.30", [.30, 0., 0., 0.]),
                         ("左眼 yaw = -0.30", [-.30, 0., 0., 0.]),
                         ("左眼 pitch = +0.30", [0., .30, 0., 0.]),
                         ("左眼 pitch = -0.30", [0., -.30, 0., 0.])):
        got = settle(angles)
        if got is None:
            print("%-20s 什么都看不到" % name)
            continue
        print("%-20s 亮块中心 (行 %6.2f, 列 %6.2f)  相对归零 行 %+6.2f 列 %+6.2f"
              % (name, got[0], got[1], got[0] - base[0], got[1] - base[1]))
    print("\n判读：列变大 = 物体在画面里往右跑；列变小 = 往左跑（表中方位正=左侧 → 列小）。")
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())