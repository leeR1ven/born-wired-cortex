# -*- coding: utf-8 -*-
"""\u91cf\u4e00\u91cf\u65b0\u773c\u808c\uff08\u4e00\u53ea\u773c 8 \u4e2a\u795e\u7ecf\u5143\uff09\u5230\u5e95\u5728\u5e72\u4ec0\u4e48\u3002

\u7b2c\u4e00\u6b65\uff1a\u7403\u6446\u5728\u6b63\u524d\u65b9\uff0c\u8ba9\u773c\u775b\u81ea\u5df1\u8dd1\uff0c\u770b\u5b83\u8dd1\u5230\u54ea\u91cc\u3001\u56db\u4e2a\u795e\u7ecf\u5143\u5404\u653e\u591a\u5c11\u7535\u3002
\u7b2c\u4e8c\u6b65\uff1a\u628a\u773c\u7403\u786c\u6309\u5728\u4e0d\u540c\u89d2\u5ea6\u4e0a\uff0c\u770b\u7ea2\u8272\u90a3\u4e00\u4fa7\u7684\u53cd\u5e94\u6709\u6ca1\u6709\u8ddf\u7740\u53d8 \u2014\u2014
\u5982\u679c\u4e0d\u53d8\uff0c\u8bf4\u660e\u8f6c\u773c\u775b\u6ca1\u5e26\u6765\u65b0\u753b\u9762\uff0c\u95ed\u73af\u662f\u65ad\u7684\u3002
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


def rates(brain):
    return brain.network.rates_at(brain.groups['eye_motor']).reshape(4, 4)


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    G.place_ball(body.model, body.data, target, 0., 0., .90)
    print("\u7403\u6446\u6b63\u524d\u65b9\uff08\u9ad8\u4f4e 0\uff09\uff1a\u5de6\u773c\u8be5\u8f6c\u5230 yaw %+0.3f"
          % W.required(body, target)[0][0])
    obs = body.observe()
    for step in range(150):
        act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                         learn=False)[1]
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        if step % 15 == 0 or step == 149:
            r = rates(brain)
            print("t=%4.2f\u79d2 yaw %+0.3f | \u5927\u529b\u5de6 %.2f \u5c0f\u529b\u5de6 %.2f "
                  "\u5927\u529b\u53f3 %.2f \u5c0f\u529b\u53f3 %.2f"
                  % (step*DT, brain.eye_command()[0], r[0, 0], r[0, 1], r[0, 2], r[0, 3]))
    print("\n\u628a\u5de6\u773c\u786c\u6309\u5728\u56fa\u5b9a\u89d2\u5ea6\u4e0a\uff08\u7403\u4e00\u76f4\u5728\u6b63\u524d\u65b9\uff09\uff1a")
    for want in (.6, .3, .0, -.18, -.3, -.6):
        brain._eye_angle = np.array([want, 0., 0., 0.])
        obs = body.observe()
        for _ in range(40):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            brain._eye_angle = np.array([want, 0., 0., 0.])
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        r = rates(brain)
        print("yaw %+0.2f -> \u5927\u529b\u5de6 %.2f \u5c0f\u529b\u5de6 %.2f \u5927\u529b\u53f3 %.2f \u5c0f\u529b\u53f3 %.2f"
              % (want, r[0, 0], r[0, 1], r[0, 2], r[0, 3]))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())