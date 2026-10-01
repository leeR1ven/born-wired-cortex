import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("tools").resolve()))
import mujoco
import wire_red_gaze as W
from tools import taskbank as tb
ctx = tb.context({}, 0)
body, brain, eyes, target = W.build(ctx, W.spec_of(W.load_table(), .10, .35, .10, __import__("numpy").radians(1.5), rest_speed=.6, fill=W.load_table(W.WIDE)))
m = body.model
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
print("模型里的相机：", names)
print("眼球行程（弧度）：", body.eye_lower_limits.round(3).tolist(), body.eye_upper_limits.round(3).tolist())
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
print("红球 geom：", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, target))
eyes.close()