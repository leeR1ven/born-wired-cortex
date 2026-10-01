# -*- coding: utf-8 -*-
"""看时间序列：眼睛是稳到一个角度，还是在来回摆。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb
from tools import wire_red_gaze as w

DT = .01


def run(gain, bearing, elevation, steps=120):
    table = w.load_table()
    spec = w.spec_of(table, gain, None, .10, .60, .20, np.radians(3.))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = w.build(ctx, spec)
    aim = w.aimer(body, target, .90)
    true_bearing, true_elevation = aim(bearing, elevation)
    environment = tb.blank_environment()
    observation = body.observe()
    trail = []
    for step in range(steps):
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        command = brain.eye_command()
        trail.append((.5*(float(command[0]) + float(command[2])),
                      .5*(float(command[1]) + float(command[3])),
                      float(np.asarray(observation["eye_position"]).reshape(4)[0])))
    eyes.close()
    return true_bearing, true_elevation, trail


for gain in (0.3, 1.0, 3.0):
    bearing, elevation, trail = run(gain, .30, 0.)
    print("=== 增益 %.1f，球方位 %+.3f 高低 %+.3f ===" % (gain, bearing, elevation))
    print("  步:  " + " ".join("%6d" % s for s in range(0, 120, 10)))
    print("  命令yaw: " + " ".join("%6.3f" % trail[s][0] for s in range(0, 120, 10)))
    print("  真实yaw: " + " ".join("%6.3f" % trail[s][2] for s in range(0, 120, 10)))
    print("  后 40 步命令yaw 的波动（标准差）%.4f" % float(np.std([t[0] for t in trail[-40:]])))