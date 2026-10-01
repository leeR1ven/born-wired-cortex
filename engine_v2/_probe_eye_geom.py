import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, ".")
from tools import taskbank as tb
ctx = tb.context({}, 0)
body = tb.clean_body(ctx["model_path"])
m, d = body.model, body.data
base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
print("base xpos", np.round(d.xpos[base], 4))
print("base xmat\n", np.round(d.xmat[base].reshape(3,3), 4))
for side in ("left", "right"):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "eye_"+side)
    print("eye_%s xpos %s" % (side, np.round(d.xpos[i], 4)))
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eye_"+side)
    print("   cam pos %s  fovy %s" % (np.round(m.cam_pos[cam],4), m.cam_fovy[cam]))
    print("   cam quat", np.round(m.cam_quat[cam], 4))
print("nq", m.nq, "nu", m.nu)
print("eye joint names:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)])