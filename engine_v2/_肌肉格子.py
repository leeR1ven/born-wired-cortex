# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, str(Path("tools").resolve()))
import wire_red_gaze as W
from tools import taskbank as tb
DT = .01
table = W.load_table("artifacts/红球位置野_红_大.json")
spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), cell_time=.01, latch=0., cross=0.,
                 motor_time=.20, gain=1.4)
ctx = tb.context({}, 0)
body, brain, eyes, target = W.build(ctx, spec)
W.G.place_ball(body.model, body.data, target, 0., 0., .90)
env = tb.blank_environment()
obs = body.observe()
for _ in range(300):
    act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
    body.command_eyes(brain.eye_command())
    obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
units = brain.network.rates_at(brain.groups["eye_motor"]).reshape(4, brain.eye_motor_units)
cmd = brain.eye_command()
print("眼球 24 个小格子（eye_motor）的放电；第 0 格推到正方向最远、第 23 格推到负方向最远")
print("角度命令 = 24 格的放电平均值 换算来的：左眼 yaw %+.4f（要 %+.4f）、右眼 yaw %+.4f（要 %+.4f）"
      % (cmd[0], W.G.seen_by([0., 0.], "left")[0], cmd[2], W.G.seen_by([0., 0.], "right")[0]))
for joint, name in ((0, "左眼 yaw"), (2, "右眼 yaw")):
    line = units[joint]
    print("\n%s：平均 %.4f  最高 %.4f  最低 %.4f" % (name, line.mean(), line.max(), line.min()))
    print("   " + " ".join("%.2f" % v for v in line))
    print("   第0格 %.3f 第1格 %.3f ... 第22格 %.3f 第23格 %.3f" % (line[0], line[1], line[-2], line[-1]))
eyes.close()