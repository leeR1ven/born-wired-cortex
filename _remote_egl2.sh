PY=/hy-tmp/bwc/.venv/bin/python
MUJOCO_GL=egl $PY - <<'PY' 2>&1 | head -20
import mujoco, numpy as np, time
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 24, 32)
r.update_scene(d)
img = r.render()
print('先渲染一张 OK', img.shape, int(img.sum()))
t0 = time.perf_counter()
for _ in range(50):
    r.update_scene(d); img = r.render()
print('egl 每张 %.4f 秒' % ((time.perf_counter()-t0)/50))
import os
os._exit(0)
PY
echo "---- 退出码 $? ----"