# -*- coding: utf-8 -*-
"""只做右眼：球摆在头坐标系 (-0.338, 0.172)，把轨迹和亮点位置打出来。"""
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
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), only=1, rest_speed=.6)
    shape = table["eye_shape"]
    rows, columns = int(shape[1]), int(shape[2])
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    motor = np.asarray(brain.groups["eye_motor"], dtype=int)
    opp = np.asarray(brain.groups["retinal_opponent"], dtype=int)
    G.place_ball(body.model, body.data, target, -0.338, 0.172, .90)
    print("右眼该转到 %+0.3f" % W.required(body, target)[1][0])
    print("%6s %9s %9s %9s | %s" % ("t", "右眼yaw", "球相对右眼", "差", "右眼画面里亮的列（#=亮）"))
    for step in range(300):
        obs = body.observe()
        activation = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                                learn=False)[1]
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        if step % 15:
            continue
        rate = brain.network.activity
        right = rate[opp].reshape(2, rows, columns, 3)[1, :, :, 0]
        lit = right > .05
        cols = np.flatnonzero(lit.any(axis=0))
        m = np.asarray(rate[motor]).reshape(4, 2)
        print("%6.2f %+9.3f %+9.3f %+9.3f | 列 %s %s  眼肌左/右 %5.2f/%5.2f"
              % (step*DT, float(brain.eye_command()[2]), W.required(body, target)[1][0],
                 float(brain.eye_command()[2]) - W.required(body, target)[1][0],
                 (cols.min() if cols.size else "无"), (cols.max() if cols.size else ""),
                 m[2, 0], m[2, 1]))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())