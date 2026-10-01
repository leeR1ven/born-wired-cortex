# -*- coding: utf-8 -*-
"""拆开看：推右的力是谁贡献的。按「细胞在视网膜的哪一列」分组。（索引按细胞群的实际编号）"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402
import eye_geometry as G                                        # noqa: E402
from tools import taskbank as tb                                # noqa: E402

DT = .01


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    yaw, pitch = W.retina_angles(table)
    shape = table["eye_shape"]
    rows, columns = int(shape[1]), int(shape[2])
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    edges = [(int(c), str(d), str(t), float(w)) for c, d, t, w in spec["edges"]]
    motor = np.asarray(brain.groups["eye_motor"], dtype=int)
    opp = np.asarray(brain.groups["retinal_opponent"], dtype=int)
    for bearing in (0.0, 0.30):
        G.place_ball(body.model, body.data, target, bearing, 0., .90)
        obs = body.observe()
        for _ in range(40):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            brain._eye_angle = np.zeros(4)
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        rate = brain.network.activity
        red_rate = rate[opp]
        left = red_rate[:rows*columns*3].reshape(rows, columns, 3)[:, :, 0]
        print("\n=== 球方位 %+0.2f（左眼该转 %+0.3f）===" % (bearing, W.required(body, target)[0][0]))
        print("左眼红细胞：放电>0.05 的 %d 个（共 %d），总和 %.1f，最强 %.2f 在 行%d 列%d"
              % ((left > .05).sum(), left.size, left.sum(), left.max(),
                 *(np.unravel_index(int(np.argmax(left)), left.shape))))
        for line in range(0, rows, 3):
            print("   " + "".join("#" if left[line, c] > .05 else "." for c in range(columns)))
        by = {}
        tops = []
        for cell, direction, tier, weight in edges:
            r = float(red_rate[cell])
            if r <= .01:
                continue
            within = (cell//3) % (rows*columns)
            row, col = divmod(within, columns)
            key = (direction, "col<16" if col < 16 else ("col16-32" if col < 32 else "col>=32"))
            got = by.setdefault(key, [0., 0])
            got[0] += weight*r
            got[1] += 1
            tops.append((weight*r, direction, tier, col, row, r, weight))
        print("  按「推哪个方向 x 细胞在哪一段列」分：")
        for key in sorted(by):
            print("    %-6s %-9s 推力 %+6.2f（%d 个细胞在放电）" % (key[0], key[1], by[key][0], by[key][1]))
        tops.sort(reverse=True)
        print("  最能推的 10 个细胞：")
        for force, direction, tier, col, row, r, weight in tops[:10]:
            print("    %-5s %-4s 列%2d 行%2d 放电%.2f 线粗%.3f -> 力%+.3f  表说这列 yaw %+.3f"
                  % (direction, tier, col, row, r, weight, force, yaw[0][col]))
        m = np.asarray(rate[motor]).reshape(4, 2)
        print("  眼肌放电（左眼yaw 左/右=%.2f/%.2f、左眼pitch 上/下=%.2f/%.2f）"
              % (m[0, 0], m[0, 1], m[1, 0], m[1, 1]))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())