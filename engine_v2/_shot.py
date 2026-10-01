import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
for extra in (ROOT, ROOT / "tools"):
    sys.path.insert(0, str(extra))
import mujoco
from PIL import Image
import wire_red_gaze as W
from tools import taskbank as tb

spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6, fill=W.load_table(W.WIDE))
ctx = tb.context({}, 0)
body, brain, eyes, target = W.build(ctx, spec)
mujoco.mj_forward(body.model, body.data)
r = mujoco.Renderer(body.model, height=330, width=440)
shots = []
setups = [("A 我设的默认 0.55米 方位28", 0.55, 28.0, -6.0),
          ("B 1.2米 方位120", 1.2, 120.0, -12.0),
          ("C 1.6米 方位90", 1.6, 90.0, -10.0),
          ("D 眼球自己 eye_left", None, None, None)]
for label, distance, azimuth, elevation in setups:
    cam = mujoco.MjvCamera()
    if distance is None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_left")
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.32, 0.0, 0.33]
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
    r.update_scene(body.data, camera=cam)
    image = r.render().copy()
    grey = image.reshape(-1, 3).dot(np.array([1, 1, 1], dtype=float))/3
    print("%-26s 平均亮度 %6.1f   最亮格子占比 %5.1f%%" % (label, grey.mean(), 100*np.mean(grey > 200)))
    shots.append(image)
sheet = Image.new("RGB", (440*2, 330*2), "white")
for k, image in enumerate(shots):
    sheet.paste(Image.fromarray(image), ((k % 2)*440, (k//2)*330))
sheet.save(r"F:\born-wired-cortex\engine_v2\artifacts\_视角对照.png")
print("saved artifacts/_视角对照.png", sheet.size)
r.close(); eyes.close()