#!/bin/bash
cd /hy-tmp/bwc/engine_v2
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/tmp/nvfix
PY=/hy-tmp/bwc/.venv/bin/python
seq 8 | xargs -P8 -I{} $PY tools/chase_red_ball.py --seconds 6 > /tmp/_p8c_out.txt 2>&1
echo "挂显卡 8 路并行：$(grep -c '走了' /tmp/_p8c_out.txt) 只跑完"