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
red = W.red_cells(table)
group = W.eye_of(red, table)
row, column = W.cell_rows_columns(red, table)
sight = W.cell_angles(table, red, group, True)
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
cmd = brain.eye_command()
print("球摆在头坐标系 (0,0)、0.90 米；两眼各自的应转角度：左 %+.4f，右 %+.4f"
      % (W.G.seen_by([0., 0.], "left")[0], W.G.seen_by([0., 0.], "right")[0]))
print("实际眼命令：左 %+.4f，右 %+.4f" % (cmd[0], cmd[2]))
opp = np.asarray(brain.groups["retinal_opponent"])
rates = brain.network.rates_at(opp[np.asarray(red, dtype=int)])
for which, side in ((0, "左"), (1, "右")):
    mine = np.nonzero(group == which)[0]
    order = mine[np.argsort(-rates[mine])]
    total = float((np.abs(sight[mine, 0])*rates[mine]).sum())
    print("\n%s眼：亮着的红细胞 %d 个；|标定方位| x 放电 的总和 = %.3f"
          % (side, int((rates[mine] > .05).sum()), total))
    for k in order[:6]:
        print("   列 %2d 行 %2d  标定方位 %+.3f 放电 %.3f"
              % (column[k], row[k], sight[k, 0], rates[k]))
names = sorted(n for n in brain.groups if n.startswith("eye_red_") and not n.endswith(("relay", "inhibition")))
at = brain.network.rates_at(np.asarray([brain.groups[n][0] for n in names]))
for name, rate in zip(names, at):
    if rate > 1e-3:
        print("指令细胞 %-26s 放电 %.3f" % (name, rate))
eyes.close()