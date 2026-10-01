# -*- coding: utf-8 -*-
"""\u628a\u5de6\u773c\u770b\u5230\u7684\u753b\u9762\u5b58\u6210 PNG\uff0c\u770b\u770b\u5230\u5e95\u54ea\u91cc\u662f\u7ea2\u7684\u3002"""
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
    from PIL import Image
    out = ROOT / "artifacts"
    for tag, place in (("\u6709\u7403", (0., 0.)), ("\u65e0\u7403", None)):
        if place is None:
            body.model.geom_pos[target] = [99., 99., 99.]
            import mujoco
            mujoco.mj_forward(body.model, body.data)
        else:
            G.place_ball(body.model, body.data, target, place[0], place[1], .90)
        raw = eyes.observe_raw()
        for which, side in ((0, "left"), (1, "right")):
            img = Image.fromarray(np.asarray(raw[which]), "RGB").resize((48*6, 36*6), Image.NEAREST)
            path = out / ("%s_%s.png" % (side, tag))
            img.save(path)
            print(path)
        red = np.asarray(raw[0], dtype=float)
        strength = red[:, :, 0] - .5*(red[:, :, 1] + red[:, :, 2])
        print("%s\uff1a\u5de6\u773c\u7ea2\u8272\u5f3a\u5ea6 > 20 \u7684\u50cf\u7d20 %d / %d"
              % (tag, int((strength > 20).sum()), strength.size))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())