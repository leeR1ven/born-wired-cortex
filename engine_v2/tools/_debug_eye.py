# -*- coding: utf-8 -*-
"""对照调试：为什么探针里眼睛不动。"""
import sys
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb

DT = .01
ctx = tb.context({}, 0)
print("REFERENCE_SIZE:", tb.REFERENCE_SIZE)
print("gaze_measure(先天) ->", tb.gaze_measure(ctx, sweep=.30, seconds=2.5, distance=.90, scenery=False))

body = tb.clean_body(ctx["model_path"])
target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
body.model.geom_size[target] = [.06] * 3
body.model.geom_rgba[target] = [1., 0., 0., 1.]
brain, eyes = tb.brain_for(ctx, body, eyes=True)
ids = np.asarray(brain.groups["retinal_opponent"])
red = ids[0::3]
print("视网膜细胞 %d，红通道 %d" % (len(ids), len(red)))
base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
environment = tb.blank_environment()
observation = body.observe()
errors = []
for step in range(300):
    lateral = .30 - 2. * .30 * step / 299.
    body.model.geom_pos[target] = [.30 + .90, lateral, .32]
    mujoco.mj_forward(body.model, body.data)
    activation = brain.step(observation, environment=environment,
                            eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
    body.command_eyes(brain.eye_command())
    rates = np.asarray(brain.network.activity)
    offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
    rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
    bearing = float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0]))
    command = brain.eye_command()
    if step % 50 == 0:
        print("步 %3d 靶 %6.3f  命令yaw %6.3f  红细胞最大 %.3f 均值 %.4f  看左 %.4f 看右 %.4f 看下 %.4f 看下_i %.4f"
              % (step, bearing, .5 * (command[0] + command[2]), rates[red].max(), rates[red].mean(),
                 rates[brain.groups["eye_look_left"]].mean(),
                 rates[brain.groups["eye_look_right"]].mean(),
                 rates[brain.groups["eye_look_down"]].mean(),
                 rates[brain.groups["eye_look_up_inhibition"]].mean()))
    observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
    if step > 20:
        errors.append(.5 * (command[0] + command[2]) - bearing)
print("复刻 gaze_measure 的平均误差 %.4f" % float(np.abs(np.array(errors)).mean()))
eyes.close()