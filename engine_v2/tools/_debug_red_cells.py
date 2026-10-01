# -*- coding: utf-8 -*-
"""把红色指令细胞、眼肌、以及画面里红球的位置一起打出来，看振荡是从哪一环来的。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb
from tools import wire_red_gaze as w

DT = .01
table = w.load_table()
spec = w.spec_of(table, 0.3, None, .10, .60, .20, np.radians(3.))
ctx = tb.context({}, 0)
body, brain, eyes, target = w.build(ctx, spec)
aim = w.aimer(body, target, .90)
true_bearing, true_elevation = aim(.30, 0.)
print("球方位 %+.3f 高低 %+.3f" % (true_bearing, true_elevation))
names = [name for name in brain.groups if name.startswith("eye_red_")]
print("红色指令细胞：", names)
ids = {name: np.asarray(brain.groups[name])[0] for name in names}
muscle = np.asarray(brain.groups["eye_motor"])
environment = tb.blank_environment()
observation = body.observe()
print("%4s | %7s %7s %7s %7s | %7s %7s %7s %7s | %7s | %7s %7s"
      % ("步", "左", "右", "上", "下", "移左", "移右", "移下", "移上", "命令yaw", "肌肉0", "肌肉2"))
for step in range(90):
    activation = brain.step(observation, environment=environment,
                            eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
    body.command_eyes(brain.eye_command())
    observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
    rates = np.asarray(brain.network.activity)
    command = brain.eye_command()
    if step % 5 == 0:
        print("%4d | %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f | %7.3f | %7.3f %7.3f"
              % (step, rates[ids["eye_red_move_left"]], rates[ids["eye_red_move_right_inhibition"]],
                 rates[ids["eye_red_move_up_inhibition"]], rates[ids["eye_red_move_down"]],
                 rates[ids["eye_red_hold_left"]], rates[ids["eye_red_hold_right_inhibition"]],
                 rates[ids["eye_red_hold_down"]], rates[ids["eye_red_hold_up_inhibition"]],
                 .5*(float(command[0]) + float(command[2])),
                 float(rates[muscle[0:24]].mean()), float(rates[muscle[48:72]].mean())))
eyes.close()