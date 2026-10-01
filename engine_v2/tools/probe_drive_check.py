# -*- coding: utf-8 -*-
"""\u7b97\u4e00\u7b97\u6bcf\u4e2a\u773c\u808c\u795e\u7ecf\u5143\u5230\u5e95\u6536\u5230\u591a\u5c11\u9a71\u52a8\uff0c\u548c\u5b83\u5b9e\u9645\u653e\u7684\u7535\u5bf9\u4e0d\u5bf9\u3002"""
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
SLOT = ["\u5de6yaw\u5927", "\u5de6yaw\u5c0f", "\u5de6yaw\u53cd\u5927", "\u5de6yaw\u53cd\u5c0f",
        "\u5de6pitch\u5927", "\u5de6pitch\u5c0f", "\u5de6pitch\u53cd\u5927", "\u5de6pitch\u53cd\u5c0f",
        "\u53f3yaw\u5927", "\u53f3yaw\u5c0f", "\u53f3yaw\u53cd\u5927", "\u53f3yaw\u53cd\u5c0f",
        "\u53f3pitch\u5927", "\u53f3pitch\u5c0f", "\u53f3pitch\u53cd\u5927", "\u53f3pitch\u53cd\u5c0f"]


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    total = {}
    for cell, direction, tier, weight in spec['edges']:
        key = (str(direction), str(tier))
        total[key] = total.get(key, 0.) + float(weight)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    ids = np.asarray([int(c) for c, _, _, _ in spec['edges']])
    print("\u63a5\u7ebf\u603b\u91cf\uff1a%s" % {k: round(v, 2) for k, v in sorted(total.items())})
    for place in ((0., 0.), (0., 0.30), (-.30, 0.)):
        G.place_ball(body.model, body.data, target, place[0], place[1], .90)
        brain._eye_angle = np.zeros(4)
        obs = body.observe()
        for _ in range(40):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            brain._eye_angle = np.zeros(4)
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        feature = brain.network.rates_at(ids)
        want = np.zeros(16)
        rank = {'move': 0, 'hold': 1}
        for (cell, direction, tier, weight), rate in zip(spec['edges'], feature):
            side = str(direction)
            axis = 0 if side in ('left', 'right') else 1
            going = 0 if side in ('left', 'down') else 2
            slot = 2*0 + axis
            want[(2*0 + axis)*4 + going + rank[str(tier)]] += float(weight)/total[(side, str(tier))]*rate
        actual = brain.network.rates_at(brain.groups['eye_motor']).reshape(4, 4)
        print("\n\u7403 \u65b9\u4f4d %+.2f \u9ad8\u4f4e %+.2f\uff08\u5de6\u773c\u8be5\u8f6c %+.3f\uff09"
              % (place[0], place[1], W.required(body, target)[0][0]))
        print("  \u7b97\u51fa\u6765\u7684\u9a71\u52a8\uff08\u5de6\u8f74\uff09\uff1a%s"
              % np.round(want[:4], 3))
        print("  \u5b9e\u9645\u653e\u7535\uff08\u5de6\u8f74\uff09\uff1a%s" % np.round(actual.reshape(16)[:4], 3))
        print("  \u5168\u90e8 16 \u4e2a\u5b9e\u9645\uff1a%s" % np.round(actual.reshape(16), 2))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())