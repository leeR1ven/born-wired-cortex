# -*- coding: utf-8 -*-
"""把「球亮哪些细胞 -> 往哪个方向推 -> 眼肌哪一根亮」这条链逐步量出来。

眼睛硬按在 0 角度不动，球摆在几个方位上，看红色特征细胞各自往哪儿推。
"""
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
    print("接线 %d 根；红细胞 %d 个；视网膜 %d 行 x %d 列" % (len(edges), len(set(c for c, _, _, _ in edges)), rows, columns))
    print("\n%-8s | %-9s | %-22s | %-19s | %s"
          % ("球方位", "左眼该转到", "红色细胞推哪边(权重x放电)", "亮的位置", "眼肌(左j0 右j1, 正=往左)"))
    for bearing in (-0.30, -0.15, 0.0, 0.15, 0.30):
        G.place_ball(body.model, body.data, target, bearing, 0., .90)
        ask = W.required(body, target)[0]
        obs = body.observe()
        for _ in range(40):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            brain._eye_angle = np.zeros(4)
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        rate = brain.network.activity
        push = {}
        for cell, direction, tier, weight in edges:
            push[direction] = push.get(direction, 0.) + weight*float(rate[cell])
        order = ("left", "right", "up", "down")
        shown = " ".join("%s %+.2f" % (k, push.get(k, 0.)) for k in order)
        here = np.asarray(rate[opp[:rows*columns*3]]).reshape(rows, columns, 3)[:, :, 0]
        flat = here.ravel()
        top = int(np.argmax(flat))
        row_top, col_top = divmod(top, columns)
        lit = int((flat > .05).sum())
        m = np.asarray(rate[motor]).reshape(4, 2)
        print("%+8.3f | %+8.3f | %-22s | 行%2d 列%2d (亮%d) | 左眼 yaw %+.2f/%+.2f 右眼 yaw %+.2f/%+.2f"
              % (bearing, ask[0], shown, row_top, col_top, lit,
                 m[0, 0], m[0, 1], m[2, 0], m[2, 1]))
        print("%9s   %s" % ("", "球亮的那列，表里说它是 yaw %s" % np.round(yaw[0][max(0, col_top-6):col_top+7], 3)))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())