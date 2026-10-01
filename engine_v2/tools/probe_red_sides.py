# -*- coding: utf-8 -*-
"""\u628a\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u6309\u300c\u5de6\u53f3 \u00d7 \u5927/\u5c0f\u6863\u300d\u5206\u7ec4\uff0c\u770b\u773c\u775b\u5728\u4e0d\u540c\u89d2\u5ea6\u65f6\u54ea\u4e00\u7ec4\u5728\u4eae\u3002"""
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
    groups = {}
    for cell, direction, tier, weight in spec['edges']:
        groups.setdefault((str(direction), str(tier)), []).append(int(cell))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    G.place_ball(body.model, body.data, target, 0., 0., .90)
    print("\u7403\u6446\u6b63\u524d\u65b9\uff1a\u5de6\u773c\u8be5\u8f6c yaw %+0.3f\uff08\u5f80\u53f3\uff09"
          % W.required(body, target)[0][0])
    print("\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u6309\u7ec4\uff1a%s" % {k: len(v) for k, v in sorted(groups.items())})
    print("\n%-8s | %s" % ("\u5de6\u773cyaw", " ".join("%s-%s" % k for k in sorted(groups))))
    for want in (.6, .3, .1, .0, -.1, -.18, -.3, -.6):
        brain._eye_angle = np.array([want, 0., 0., 0.])
        obs = body.observe()
        for _ in range(30):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            brain._eye_angle = np.array([want, 0., 0., 0.])
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        out = []
        for key in sorted(groups):
            r = brain.network.rates_at(np.asarray(groups[key]))
            out.append("%.2f/%d" % (r.max(), int((r > .05).sum())))
        print("%+8.2f | %s" % (want, "      ".join(out)))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())