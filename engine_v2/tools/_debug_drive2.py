# -*- coding: utf-8 -*-
"""接线到底把多少电流送进去了？直接算 Σ 权重x活动，和细胞自己的电压比一比。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import taskbank as tb
from tools import wire_red_gaze as w

DT = .01
table = w.load_table()
spec = w.spec_of(table, .10, .60, .15, np.radians(3.))
ctx = tb.context({}, 0)
body, brain, eyes, target = w.build(ctx, spec)
aim = w.aimer(body, target, .90)
aim(.34, 0.)
flat = np.asarray(brain.groups["retinal_opponent"])
environment = tb.blank_environment()
observation = body.observe()
for step in range(70):
    activation = brain.step(observation, environment=environment,
                            eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
    body.command_eyes(brain.eye_command())
    observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
rates = np.asarray(brain.network.activity)
voltage = np.asarray(brain.network.voltage)
by_side = {}
for cell, direction, tier, weight in spec["edges"]:
    by_side.setdefault((direction, tier), []).append((cell, weight))
print("红色通道亮着的细胞：>0.2 %d 个，>0.5 %d 个，>0.9 %d 个；总活动 %.1f"
      % (int((rates[flat[0::3]] > .2).sum()), int((rates[flat[0::3]] > .5).sum()),
         int((rates[flat[0::3]] > .9).sum()), float(rates[flat[0::3]].sum())))
for (direction, tier), items in sorted(by_side.items()):
    drive = sum(weight*float(rates[flat[cell]]) for cell, weight in items)
    name = "eye_red_%s_%s" % (tier, direction)
    idx = int(np.asarray(brain.groups[name])[0]) if name in brain.groups else None
    print("%-26s 线 %4d 根，总权重 %8.3f，总驱动 %8.4f，细胞的电压 %7.4f、活动 %7.4f"
          % (name, len(items), sum(weight for _, weight in items), drive,
             (voltage[idx] if idx is not None else float("nan")),
             (rates[idx] if idx is not None else float("nan"))))
print("两眼命令角度 %s，静止基线 %s" % (np.round(brain.eye_command(), 4), "0"))
lit = flat[0::3][rates[flat[0::3]] > .5]
angles = np.asarray(table["places"], float)[np.asarray(table["cells"]["retinal_opponent"]["best_place"], int)[
    np.arange(len(table["cells"]["retinal_opponent"]["best_place"])) % 3 == 0][
        np.where(rates[flat[0::3]] > .5)[0]]]
print("亮着的细胞看到的方位范围 %+.3f ~ %+.3f，|方位| 之和 %.2f"
      % (angles[:, 0].min(), angles[:, 0].max(), float(np.abs(angles[:, 0]).sum())))
eyes.close()