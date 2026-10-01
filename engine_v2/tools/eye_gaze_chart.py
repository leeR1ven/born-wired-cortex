# -*- coding: utf-8 -*-
"""画一张眼球角度的曲线：球摆在前方时，两只眼各自转到哪儿、怎么抖。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

import wire_red_gaze as W                                        # noqa: E402
import eye_geometry as G                                         # noqa: E402
from tools import taskbank as tb                                 # noqa: E402

DT = .01
PLACES = ((0.00, 0.00, "\u7403\u5728\u6b63\u524d\u65b9"), (0.34, 0.00, "\u7403\u5728\u504f\u5de6 0.34 \u5f27\u5ea6"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def run(body, brain, eyes, target, env, bearing, steps=300):
    G.place_ball(body.model, body.data, target, bearing, 0., .90)
    obs = body.observe()
    left, right, want = [], [], []
    for _ in range(steps):
        activation = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(),
                                dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        command = brain.eye_command()
        ask = W.required(body, target)
        left.append(float(command[0]))
        right.append(float(command[2]))
        want.append((ask[0][0], ask[1][0]))
    return np.asarray(left), np.asarray(right), np.asarray(want)


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), rest_speed=.6, fill=W.load_table(W.WIDE))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    figure, axes = plt.subplots(len(PLACES), 1, figsize=(9, 7))
    for axis, (bearing, elevation, title) in zip(axes, PLACES):
        left, right, want = run(body, brain, eyes, target, env, bearing)
        time = np.arange(len(left))*DT
        axis.plot(time, np.degrees(left), label="\u5de6\u773c")
        axis.plot(time, np.degrees(right), label="\u53f3\u773c")
        axis.axhline(np.degrees(want[:, 0].mean()), color="tab:red", ls="--", lw=1,
                     label="\u5de6\u773c\u8be5\u5230")
        axis.axhline(np.degrees(want[:, 1].mean()), color="tab:green", ls="--", lw=1,
                     label="\u53f3\u773c\u8be5\u5230")
        axis.set_title(title)
        axis.set_ylabel("\u773c\u7403\u5de6\u53f3\u8f6c\uff08\u5ea6\uff09")
        axis.grid(alpha=.3)
        axis.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("\u65f6\u95f4\uff08\u79d2\uff09")
    figure.tight_layout()
    out = ROOT / "artifacts" / "\u773c\u52a8\u66f2\u7ebf.png"
    figure.savefig(out, dpi=110)
    print(out)
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())