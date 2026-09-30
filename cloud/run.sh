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
# 这台机器真正能用的核数/内存（不是宿主机的 nproc / free）
source "$ROOT/cloud/machine.sh"
# 解释器：优先用 cloud/new_python.sh 造在仓库里的 .venv（镜像只有 python 3.8 时用它），
# 其次用 PY=... 指定的，最后才用系统的 python3。
if [ -n "${PY:-}" ]; then :
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY=python3
fi
export PYTHONIOENCODING=utf-8
export MUJOCO_GL="${MUJOCO_GL:-egl}"

KIDS="${CANDIDATES:-300}"
GEN="${GENERATIONS:-20}"
# 进程数默认按 cgroup 配额来，别按 nproc：容器里 nproc 报的是宿主机核数（这台报 96），
# 照它开就是 96 个进程抢 24 核的配额，内存也会超（一只约 1.9 GB）。
WORKERS="${WORKERS:-$(cpu_cores)}"
BY_RAM=$(( $(mem_gb) / 2 ))
if [ "$WORKERS" -gt "$BY_RAM" ]; then
    echo "内存只给 $(mem_gb) GB，进程数从 $WORKERS 降到 $BY_RAM（一只约 1.9 GB）"
    WORKERS="$BY_RAM"
fi
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