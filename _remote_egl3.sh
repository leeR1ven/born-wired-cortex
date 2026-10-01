cd /hy-tmp/bwc
PY=/hy-tmp/bwc/.venv/bin/python
MUJOCO_GL=egl $PY egl_test.py > /hy-tmp/bwc/egl_out.log 2>&1
echo "退出码 $?"
cat /hy-tmp/bwc/egl_out.log | head -8