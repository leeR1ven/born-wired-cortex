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
def block(color):
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    image[:, :] = color
    return image
colors = {"红": (255, 0, 0), "绿": (0, 255, 0), "蓝": (0, 0, 255), "黄": (255, 255, 0)}
viewer = mjviewer.launch_passive(body.model, body.data)
start = time.time()
print_once = True
while viewer.is_running() and time.time() - start < 25:
    view = viewer.viewport
    if print_once and time.time() - start > 0.5:
        print("viewport left=%d bottom=%d width=%d height=%d" % (view.left, view.bottom, view.width, view.height), flush=True)
        for name, rect in (
            ("红 上左", (view.left + 8, view.bottom + view.height - 188, 240, 180)),
            ("绿 上右", (view.left + view.width - 248, view.bottom + view.height - 188, 240, 180)),
            ("蓝 下左", (view.left + 8, view.bottom + 8, 240, 180)),
            ("黄 下右", (view.left + view.width - 248, view.bottom + 8, 240, 180))):
            print("  %s 矩形 x=%d y=%d" % (name, rect[0], rect[1]), flush=True)
        print_once = False
    center = mujoco.MjrRect(400, 300, 240, 180)
    viewer.set_images([
        (mujoco.MjrRect(view.left + 8, view.bottom + view.height - 188, 240, 180), block(colors["红"])),
        (mujoco.MjrRect(view.left + view.width - 248, view.bottom + view.height - 188, 240, 180), block(colors["绿"])),
        (mujoco.MjrRect(view.left + 8, view.bottom + 8, 240, 180), block(colors["蓝"])),
        (mujoco.MjrRect(view.left + view.width - 248, view.bottom + 8, 240, 180), block(colors["黄"])),
        (center, block((255, 0, 255))),
    ])
    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "文字在左上", "")])
    viewer.sync()
    time.sleep(0.03)
viewer.close(); eyes.close()