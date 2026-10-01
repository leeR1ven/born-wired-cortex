#!/bin/bash
cd /hy-tmp/bwc/engine_v2
export MUJOCO_GL=egl PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/hy-tmp/bwc/.venv/bin/python

echo "===== 旧考试 (CPU) ====="
$PY /tmp/_val_exam.py old > /tmp/_v_old.json 2>/tmp/_v_old.err || tail -5 /tmp/_v_old.err
cat /tmp/_v_old.json

echo "===== 新考试 (CPU) ====="
$PY /tmp/_val_exam.py new > /tmp/_v_new.json 2>/tmp/_v_new.err || tail -5 /tmp/_v_new.err
cat /tmp/_v_new.json

echo "===== 新考试 (GPU) ====="
source /hy-tmp/bwc/_gpu_env.sh
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
$PY /tmp/_val_exam.py newgpu > /tmp/_v_gpu.json 2>/tmp/_v_gpu.err || tail -5 /tmp/_v_gpu.err
cat /tmp/_v_gpu.json