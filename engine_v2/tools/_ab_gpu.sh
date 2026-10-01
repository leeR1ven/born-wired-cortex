#!/bin/bash
cd /hy-tmp/bwc/engine_v2
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/hy-tmp/bwc/.venv/bin/python
run() {
  local tag="$1"; shift
  local t0=$(date +%s.%N)
  "$@" > /tmp/_ab_$tag.txt 2>&1
  local t1=$(date +%s.%N)
  echo "$tag 墙钟 $(echo "$t1 - $t0" | bc)"
  grep -E "走了|用时" /tmp/_ab_$tag.txt | head -2
}
echo "########## A 正常（不认识 CUDA） ##########"
run A1 env LD_LIBRARY_PATH= $PY tools/chase_red_ball.py --seconds 6
run A2 env LD_LIBRARY_PATH= $PY tools/chase_red_ball.py --seconds 6
echo "########## B 挂 nvfix（认识 CUDA） ##########"
run B1 env LD_LIBRARY_PATH=/tmp/nvfix $PY tools/chase_red_ball.py --seconds 6
run B2 env LD_LIBRARY_PATH=/tmp/nvfix $PY tools/chase_red_ball.py --seconds 6
echo "########## C 关了眼睛（看眼睛占多少） ##########"
run C1 env LD_LIBRARY_PATH= $PY tools/chase_red_ball.py --seconds 6 --no-eye