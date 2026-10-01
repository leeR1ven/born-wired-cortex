# -*- coding: utf-8 -*-
"""量一下 set_images 那个矩形是「从左上角」还是「从左下角」算的。"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
for extra in (ROOT, ROOT / "tools"):
    sys.path.insert(0, str(extra))
import mujoco
from mujoco import viewer as mjviewer
import wire_red_gaze as W
from tools import taskbank as tb

spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6, fill=W.load_table(W.WIDE))
ctx = tb.context({}, 0)
body, brain, eyes, target = W.build(ctx, spec)
blue = np.zeros((150, 200, 3), dtype=np.uint8); blue[:, :, 2] = 255
green = np.zeros((150, 200, 3), dtype=np.uint8); green[:, :, 1] = 255
viewer = mjviewer.launch_passive(body.model, body.data)
print("viewport =", viewer.viewport, flush=True)
start = time.time()
printed = False
while viewer.is_running() and time.time() - start < 20:
    if not printed and time.time() - start > 1:
        print("1 秒后的 viewport =", viewer.viewport, flush=True)
        printed = True
    viewer.set_images([(mujoco.MjrRect(200, 200, 200, 150), blue),
                       (mujoco.MjrRect(600, 400, 200, 150), green)])
    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "左上角文字", "right")])
    viewer.sync()
    time.sleep(0.02)
print("最后 viewport =", viewer.viewport, flush=True)
viewer.close(); eyes.close()