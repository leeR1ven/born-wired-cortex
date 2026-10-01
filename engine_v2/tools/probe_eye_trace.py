# -*- coding: utf-8 -*-
"""球摆在偏左 0.34 弧度，把眼球角度的整条曲线打出来。"""
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
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, only=0)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    motor = np.asarray(brain.groups["eye_motor"], dtype=int)
    edges = [(int(c), str(d), str(t), float(w)) for c, d, t, w in spec["edges"]]
    opp = np.asarray(brain.groups["retinal_opponent"], dtype=int)
    for bearing in (0.34, 0.0):
        G.place_ball(body.model, body.data, target, bearing, 0., .90)
        obs = body.observe()
        print("\n=== 球在头坐标系 %+0.2f 弧度，左眼该转到 %+0.3f ==="
              % (bearing, W.required(body, target)[0][0]))
        print("%6s %8s %8s %8s | %8s %8s %8s" % ("t", "眼球yaw", "球相对眼", "差", "推左", "推右", "眼肌左/右"))
        for step in range(300):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
            if step % 12:
                continue
            rate = brain.network.activity
            red_rate = rate[opp]
            push = {}
            for cell, direction, tier, weight in edges:
                if direction in ("left", "right"):
                    push[direction] = push.get(direction, 0.) + weight*float(red_rate[cell])
            yaw_now = float(brain.eye_command()[0])
            ball_now = W.required(body, target)[0][0]
            m = np.asarray(rate[motor]).reshape(4, 2)
            print("%6.2f %+8.3f %+8.3f %+8.3f | %8.2f %8.2f %5.2f/%5.2f"
                  % (step*DT, yaw_now, ball_now, yaw_now - ball_now,
                     push.get("left", 0.), push.get("right", 0.), m[0, 0], m[0, 1]))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())