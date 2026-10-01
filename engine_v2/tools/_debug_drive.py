# -*- coding: utf-8 -*-
"""谁在驱动「右」那一档？把贡献最大的红细胞、它们在表里的方位、以及当场的活动列出来。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb
from tools import wire_red_gaze as w

DT = .01
table = w.load_table()
places = np.asarray(table["places"], float)
layer = table["cells"]["retinal_opponent"]
best = np.asarray(layer["best_place"], int)
spec = w.spec_of(table, 0.3, None, .10, .60, .20, np.radians(3.))
ctx = tb.context({}, 0)
body, brain, eyes, target = w.build(ctx, spec)
aim = w.aimer(body, target, .90)
tb_bearing, tb_elev = aim(.30, 0.)
print("球在 %+.3f / %+.3f" % (tb_bearing, tb_elev))
flat = np.asarray(brain.groups["retinal_opponent"])
environment = tb.blank_environment()
observation = body.observe()
for step in range(8):
    activation = brain.step(observation, environment=environment,
                            eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
    body.command_eyes(brain.eye_command())
    observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
rates = np.asarray(brain.network.activity)
lit = flat[np.where(rates[flat] > .2)[0]]
print("亮着的红细胞 %d 个" % len(lit))
angles = places[best[lit]]
print("它们的表方位范围 %+.3f ~ %+.3f，高低范围 %+.3f ~ %+.3f"
      % (angles[:, 0].min(), angles[:, 0].max(), angles[:, 1].min(), angles[:, 1].max()))
print("其中方位为负（会驱动「右」）的 %d 个：%s"
      % (int((angles[:, 0] < 0).sum()), np.round(np.degrees(angles[angles[:, 0] < 0][:, 0]), 1)))
side = {}
for cell, direction, tier, weight in spec["edges"]:
    side.setdefault(direction, {})[cell] = side.setdefault(direction, {}).get(cell, 0.) + weight
for direction in ("left", "right"):
    drive = sum(weight*float(rates[flat[cell]]) for cell, weight in side[direction].items())
    print("「%s」总驱动 = %.3f（它那一档推 %.2f 弧度 -> 角度 %.3f）"
          % (direction, drive, .60, min(1., drive)*.60))
eyes.close()