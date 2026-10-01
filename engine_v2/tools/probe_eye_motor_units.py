# -*- coding: utf-8 -*-
"""\u91cf\u4e00\u91cf\u773c\u808c\u795e\u7ecf\u5143\u5728\u9759\u6b62\u65f6\u5404\u81ea\u505c\u5728\u54ea\uff1a\u9876\u5728\u4e0a\u9650\u3001\u538b\u5728\u4e0b\u9650\u3001\u8fd8\u662f\u5728\u4e2d\u95f4\u3002

\u7528\u6cd5\uff1apython tools/probe_eye_motor_units.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402
from tools import taskbank as tb                                # noqa: E402

DT = .01
JOINTS = ["\u5de6\u773cyaw", "\u5de6\u773cpitch", "\u53f3\u773cyaw", "\u53f3\u773cpitch"]


def settle(brain, body, eyes, steps=200):
    env = tb.blank_environment()
    obs = body.observe()
    for _ in range(steps):
        act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                         learn=False)[1]
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
    return obs


def report(brain, tag):
    units = brain.network.rates_at(brain.groups['eye_motor']).reshape(4, brain.eye_motor_units)
    print("\n== %s ==" % tag)
    print("\u773c\u808c\u795e\u7ecf\u5143 %d \u4e2a\uff08%d \u6839\u8f74 \u00d7 %d \u4e2a\uff09\uff1b\u9759\u6b62\u65f6\u773c\u7403\u547d\u4ee4\u89d2\u5ea6 %s"
          % (units.size, 4, brain.eye_motor_units, np.round(brain.eye_command(), 4)))
    print("%-10s %6s %6s %6s %9s %12s" % ("\u8f74", "\u9876\u4e0a\u9650", "\u538b\u4e0b\u9650", "\u5728\u4e2d\u95f4",
                                        "\u5747\u503c", "\u53ea\u7b97\u4e2d\u95f4"))
    for joint in range(4):
        row = units[joint]
        high = int((row >= .999).sum())
        low = int((row <= .001).sum())
        kept = row[(row > .001) & (row < .999)]
        after = float(kept.mean()) if kept.size else float('nan')
        print("%-10s %6d %6d %6d %9.4f %12.4f"
              % (JOINTS[joint], high, low, brain.eye_motor_units - high - low, row.mean(), after))
    print("\u6bcf\u6839\u8f74\u4e0a 24 \u4e2a\u795e\u7ecf\u5143\u7684\u653e\u7535\u7387\uff1a")
    for joint in range(4):
        print("  %-10s %s" % (JOINTS[joint],
                              np.array2string(units[joint], precision=2, max_line_width=220)))
    return units


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    settle(brain, body, eyes)
    report(brain, "\u63a5\u4e86\u7ea2\u7403\u7ebf\uff08\u672c\u6b21\u5355\u773c\u8dd1\u7684\u90a3\u5957\uff09")
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())