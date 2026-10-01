# -*- coding: utf-8 -*-
from pathlib import Path
p = Path("tools/wire_red_gaze.py")
t = p.read_text(encoding="utf-8", newline="")
old = '''    _, rotation = head_frame(body)
    ball = np.asarray(body.data.geom_xpos[target], dtype=float)
    out = []
    for position in eye_world(body):
        v = rotation.T @ (ball - position)
        out.append((float(np.arctan2(v[1], v[0])),
                    -float(np.arctan2(v[2], np.hypot(v[0], v[1])))))
    return out
'''
new = '''    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    rotation = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
    ball = np.asarray(body.data.geom_xpos[target], dtype=float)
    out = []
    for side in ("left", "right"):
        i = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_" + side)
        v = rotation.T @ (ball - np.asarray(body.data.xpos[i], dtype=float))
        out.append((float(np.arctan2(v[1], v[0])),
                    -float(np.arctan2(v[2], np.hypot(v[0], v[1])))))
    return out
'''
assert t.count(old) == 1
p.write_text(t.replace(old, new), encoding="utf-8", newline="")
print("ok")