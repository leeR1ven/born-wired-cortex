#!/bin/bash
# 一份题、一只狗、跑一场；同时开 N 个进程 —— 跟真正跑训练一模一样。比墙钟。
cd /hy-tmp/bwc/engine_v2
export MUJOCO_GL=egl PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/hy-tmp/bwc/.venv/bin/python

batch() {
  local tag=$1 n=$2 envs=$3
  local s=$(date +%s)
  seq $n | xargs -P$n -I{} env $envs $PY /tmp/_val_exam.py $tag > /tmp/_ab_$tag.out 2>&1
  local e=$(date +%s)
  local ok=$(grep -c '"mode"' /tmp/_ab_$tag.out)
  local bad=$(grep -c 'refused\|Error\|Traceback' /tmp/_ab_$tag.out)
  local worst=$(grep -o '"wall": [0-9.]*' /tmp/_ab_$tag.out | grep -o '[0-9.]*' | sort -n | tail -1)
  echo "$tag：跑完 $ok/$n 只，报错 $bad 处，墙钟 $((e-s)) 秒，单只最慢 ${worst}s"
}

echo "########## 1. 现在线上这套：CPU / 24 进程 ##########"
batch new 24 ""

echo "########## 2. 只挂显卡：GPU / 24 进程 ##########"
batch newgpu 24 "LD_LIBRARY_PATH=/hy-tmp/bwc/nvfix"

echo "########## 3. 挂显卡 + 多开进程：GPU / 32 进程 ##########"
batch newgpu 32 "LD_LIBRARY_PATH=/hy-tmp/bwc/nvfix"

echo "########## 4. 挂显卡 + 再多开：GPU / 40 进程 ##########"
batch newgpu 40 "LD_LIBRARY_PATH=/hy-tmp/bwc/nvfix"

echo "########## \u7ed3\u679c\u662f\u4e0d\u662f\u4e00\u6837 ##########"
for f in /tmp/_ab_new.out /tmp/_ab_newgpu.out; do
  echo "--- $f ---"
  grep -o '"chase_settled": [0-9.]*' $f | sort | uniq -c | head -5
done