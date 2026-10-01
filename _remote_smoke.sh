PY=/hy-tmp/bwc/.venv/bin/python
ls -la $PY 2>/dev/null || echo "没有 .venv"
$PY -c "import numpy, mujoco; print('numpy', numpy.__version__, 'mujoco', mujoco.__version__)" 2>&1 | tail -2
cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
timeout 1800 $PY tools/evolve_chase.py --candidates 4 --generations 1 --workers 4 --seconds 3 --seed 9 --out-prefix artifacts/chase_smoke > /hy-tmp/bwc/logs/chase_smoke.log 2>&1
echo "exit=$?"
tail -12 /hy-tmp/bwc/logs/chase_smoke.log