PY=/hy-tmp/bwc/.venv/bin/python
echo "--- osmesa/egl 库 ---"
ldconfig -p 2>/dev/null | grep -iE "osmesa|libEGL|libGL\.so" | head
echo "--- xvfb ---"
which xvfb-run Xvfb 2>/dev/null
echo "--- 试 osmesa ---"
MUJOCO_GL=osmesa $PY -c "import mujoco,numpy as np; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d=mujoco.MjData(m); r=mujoco.Renderer(m,24,32); mujoco.mj_forward(m,d); r.update_scene(d); print('osmesa 渲染 OK', r.render().shape)" 2>&1 | tail -3
echo "--- 试 egl ---"
MUJOCO_GL=egl $PY -c "import mujoco,numpy as np; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d=mujoco.MjData(m); r=mujoco.Renderer(m,24,32); mujoco.mj_forward(m,d); r.update_scene(d); print('egl 渲染 OK', r.render().shape)" 2>&1 | tail -3