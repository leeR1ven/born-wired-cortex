which nvidia-smi && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -3
echo "--- 设备节点 ---"
ls /dev/nvidia* 2>/dev/null | head
echo "--- 试 egl 渲染 ---"
PY=/hy-tmp/bwc/.venv/bin/python
MUJOCO_GL=egl $PY - <<'PY' 2>&1 | tail -4
import mujoco, numpy as np, time
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 24, 32)
r.update_scene(d)
img = r.render()
t0 = time.perf_counter()
for _ in range(20):
    r.update_scene(d); img = r.render()
print('egl 每张 %.3f 秒' % ((time.perf_counter()-t0)/20), img.shape)
PY