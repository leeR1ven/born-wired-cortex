cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
PY=/hy-tmp/bwc/.venv/bin/python
nproc
timeout 1800 $PY tools/evolve_chase.py --candidates 24 --generations 1 --workers 24 --seconds 6 --seed 5 --quiet --out-prefix artifacts/chase_rate > /hy-tmp/bwc/logs/chase_rate.log 2>&1
echo "exit=$?"
grep -v GLFW /hy-tmp/bwc/logs/chase_rate.log | tail -5