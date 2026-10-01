# -*- coding: utf-8 -*-
"""\u770b\u770b\u5899\u9762\u7684\u7ea2\u8272\u5ea6(2R-G-B)\u548c\u7403\u5dee\u591a\u5c11\uff0c\u4ee5\u53ca\u73b0\u5728 0.08 \u7684\u9608\u503c\u4f1a\u6f0f\u8fdb\u6765\u591a\u5c11\u5899\u9762\u50cf\u7d20\u3002"""
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


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    G.place_ball(body.model, body.data, target, 0., 0., .90)
    raw = np.asarray(eyes.observe_raw()[0], dtype=float)/255.
    red = 2.*raw[:, :, 0] - raw[:, :, 1] - raw[:, :, 2]
    strong = red > .5          # \u7403\u90a3\u4e00\u5757
    wall = red <= .5
    print("\u5de6\u773c\u753b\u9762\uff1a\u50cf\u7d20 %d\uff0cR-G-B \u5206\u5e03\uff1a\u6700\u5927 %.2f\u3001\u4e2d\u4f4d %.2f\u3001\u6700\u5c0f %.2f"
          % (red.size, red.max(), float(np.median(red)), red.min()))
    for t in (.08, .2, .3, .5, .8):
        print("  \u7ea2\u8272\u5ea6 > %.2f \u7684\u50cf\u7d20\uff1a%3d \u4e2a" % (t, int((red > t).sum())))
    if strong.any():
        rest = red[wall]
        print("\u7403\u90a3\u4e00\u5757\uff08\u7ea2\u8272\u5ea6>0.5\uff09\uff1a%d \u4e2a\u50cf\u7d20\uff0c\u5e73\u5747 %.2f"
              % (int(strong.sum()), float(red[strong].mean())))
        print("\u5176\u4f59\uff08\u5899\u9762\uff09\uff1a%d \u4e2a\u50cf\u7d20\uff0c\u5e73\u5747 %.2f\u3001\u6700\u5927 %.2f\u3001"
              "\u7ea2\u8272\u5ea6 > 0.08 \u7684\u5360 %.1f%%"
              % (rest.size, float(rest.mean()), float(rest.max()),
                 100.*float((rest > .08).mean())))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())