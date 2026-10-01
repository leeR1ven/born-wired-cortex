#!/bin/bash
cd /hy-tmp/bwc/engine_v2
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/hy-tmp/bwc/.venv/bin/python
echo "########## D 关眼睛 + 挂 nvfix ##########"
for i in 1 2; do
  env LD_LIBRARY_PATH=/tmp/nvfix $PY tools/chase_red_ball.py --seconds 6 --no-eye 2>&1 | grep -E "走了"
done
echo "########## E 挂 nvfix，跑的时候盯显卡 ##########"
( for k in $(seq 1 12); do LD_LIBRARY_PATH=/tmp/nvfix nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; sleep 1; done ) > /tmp/_gpuwatch.txt &
env LD_LIBRARY_PATH=/tmp/nvfix $PY tools/chase_red_ball.py --seconds 6 2>&1 | grep -E "走了"
wait
echo "--- 跑的时候显卡读数 ---"
cat /tmp/_gpuwatch.txt