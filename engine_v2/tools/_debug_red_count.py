# -*- coding: utf-8 -*-
"""红色通道到底亮了多少？带红色接线 / 不带，比一比。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb
from tools import wire_red_gaze as w

DT = .01
table = w.load_table()


def look(spec, steps=8):
    ctx = tb.context({}, 0)
    body, brain, eyes, target = w.build(ctx, spec)
    aim = w.aimer(body, target, .90)
    aim(.30, 0.)
    flat = np.asarray(brain.groups["retinal_opponent"])[0::3]
    environment = tb.blank_environment()
    observation = body.observe()
    out = []
    for step in range(steps):
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        rates = np.asarray(brain.network.activity)
        red = rates[flat]
        out.append((float(red.max()), float(red.mean()), int((red > .2).sum()), int((red > .5).sum())))
    eyes.close()
    return out


spec = w.spec_of(table, 0.3, None, .10, .60, .20, np.radians(3.))
for name, chosen in (("不带红色接线（先天那套）", None), ("带红色接线", spec)):
    print("=== %s ===" % name)
    for step, (peak, mean, over2, over5) in enumerate(look(chosen)):
        print("  步 %d：最大 %.3f  均值 %.4f  >0.2 的 %d 个  >0.5 的 %d 个" % (step, peak, mean, over2, over5))