cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=osmesa
PY=/hy-tmp/bwc/.venv/bin/python
timeout 1800 $PY tools/evolve_chase.py --candidates 6 --generations 1 --workers 6 --seconds 6 --seed 9 --out-prefix artifacts/chase_smoke2 > /hy-tmp/bwc/logs/chase_smoke2.log 2>&1
echo "exit=$?"
grep -v GLFW /hy-tmp/bwc/logs/chase_smoke2.log | tail -14