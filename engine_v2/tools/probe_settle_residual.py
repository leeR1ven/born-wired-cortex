# -*- coding: utf-8 -*-
"""球停住不动、眼睛稳定之后，两只眼各自还差多少度（直接量画面里红球偏中心多少，
不依赖任何符号约定）。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W
import eye_geometry as G
from tools import taskbank as tb

DT = .01
PLACES = [(0., 0.), (.30, 0.), (-.30, 0.), (.55, 0.), (-.55, 0.), (0., .30), (0., -.30)]
SETTLE = 600


def centroid_offset(image, width, height):
    red = np.asarray(image, dtype=int)
    mask = (red[:, :, 0] > 140) & (red[:, :, 1] < 110) & (red[:, :, 2] < 110)
    if mask.sum() < 5:
        return None
    columns = np.nonzero(mask)[1]
    half = np.radians(60.)/2
    deg_per_pixel = np.degrees(np.arctan(np.tan(half)*width/float(height)))/width
    return float((columns.mean() - width/2.)*deg_per_pixel)


def main():
    spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    width, height = brain.eye_width, brain.eye_height
    print("%8s %10s %10s | %10s %10s | %10s %10s" %
          ("球方位", "左眼要", "右眼要", "左眼实际", "左眼残差", "右眼实际", "右眼残差"))
    worst = []
    for bearing, elevation in PLACES:
        G.place_ball(body.model, body.data, target, bearing, elevation, .90)
        for _ in range(SETTLE):
            command = brain.eye_command()
            body.command_eyes(command)
            obs = body.observe()
            raw = eyes.observe_raw()
            activation = brain.step(obs, environment=env, eye_pixels=raw, dt=DT, learn=False)[1]
            body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        command = brain.eye_command()
        ask = W.required(body, target)
        fresh = eyes.observe_raw()
        residual = [centroid_offset(fresh[i], width, height) for i in (0, 1)]
        worst.append((max(abs(r) for r in residual if r is not None), bearing, elevation))
        print("%8s %10.1f %10.1f | %10.1f %10s | %10.1f %10s" %
              ("%+.2f/%+.2f" % (bearing, elevation), np.degrees(ask[0][0]), np.degrees(ask[1][0]),
               np.degrees(command[0]),
               "%.1f度" % residual[0] if residual[0] is not None else "看不见",
               np.degrees(command[2]),
               "%.1f度" % residual[1] if residual[1] is not None else "看不见"))
    eyes.close()
    print("\n最差的一格：球 %+.2f/%+.2f，红球偏画面中心 %.1f 度" % (worst[0][1], worst[0][2], worst[0][0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())