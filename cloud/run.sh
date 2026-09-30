#!/usr/bin/env bash
# 挂长跑：接着现在最好的那批爹，一代 300 只、生完立刻挑爹，一直迭代下去。
#
#   WORKERS=20 GENERATIONS=20 bash cloud/run.sh
#
# 后台跑，日志写进 logs/，关掉终端也不停。断了怎么接：看日志最后一行说的那个
# keep 文件、和那一代的编号，这样接：
#   PARENTS=artifacts/题1f_演化_云_g07_keep.jsonl START_GEN=8 bash cloud/run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/engine_v2"
PY="${PY:-python3}"
export PYTHONIOENCODING=utf-8
export MUJOCO_GL="${MUJOCO_GL:-egl}"

KIDS="${CANDIDATES:-300}"
GEN="${GENERATIONS:-20}"
WORKERS="${WORKERS:-$(nproc)}"
PARENTS="${PARENTS:-artifacts/题1f_演化_g01_keep.jsonl}"
START_GEN="${START_GEN:-2}"
PREFIX="${PREFIX:-artifacts/题1f_演化_云}"

if [ ! -f "$PARENTS" ]; then
    echo "找不到爹的台账 $PARENTS —— 先确认仓库是新的（git pull），或者用 PARENTS=... 指一个"
    exit 1
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/上云演化_$(date +%Y%m%d_%H%M).log"

echo "日志     $LOG"
echo "爹       $PARENTS"
echo "规模     一代 $KIDS 只 × $GEN 代，$WORKERS 个进程"
nohup "$PY" -X utf8 -u tools/evolve.py --parents "$PARENTS" --candidates "$KIDS" \
    --generations "$GEN" --workers "$WORKERS" --genes 6 --sigma .55 --cross-rate .5 \
    --start-gen "$START_GEN" --seed 101 --quiet --out-prefix "$PREFIX" >> "$LOG" 2>&1 &
echo "开跑了，pid $!"
echo "看进度：tail -f $LOG"
echo "看成绩单：grep 代收工 $LOG"