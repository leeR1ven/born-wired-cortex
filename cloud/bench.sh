#!/usr/bin/env bash
# 量这台机器开几个进程最划算：同一批孩子，换进程数各跑一遍，报每只多少秒、每小时多少只。
#
#   bash cloud/bench.sh                 # 默认试 8 / 16 / 24 个进程
#   WORKERS="8 12" KIDS=48 bash cloud/bench.sh
#
# 注意：每一档的头十几秒是 python/torch 起进程的开销，孩子数给少了会把这一档显得偏慢。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/engine_v2"
# 解释器：优先用 cloud/new_python.sh 造在仓库里的 .venv（镜像只有 python 3.8 时用它），
# 其次用 PY=... 指定的，最后才用系统的 python3。
if [ -n "${PY:-}" ]; then :
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY=python3
fi
export PYTHONIOENCODING=utf-8
export MUJOCO_GL="${MUJOCO_GL:-egl}"

CORES=$(nproc)
RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 16)
# 日志目录自己建一下：精简的上云包里 logs/ 可能根本没跟着来
mkdir -p "$ROOT/logs"
KIDS="${KIDS:-64}"
# 爹用随包带来的那份 keep 台账。不传的话 evolve.py 会退回它自己的默认值
# （artifacts/题1f_十秒赛5000.jsonl，6 MB 的比赛台账），精简的上云包里没有，一跑就退。
PARENTS="${PARENTS:-artifacts/题1f_演化_g01_keep.jsonl}"
if [ ! -f "$PARENTS" ]; then echo "找不到爹的台账 $PARENTS"; exit 1; fi
WORKER_LIST="${WORKERS:-8 16 24}"

echo "这台机器：$CORES 核 / ${RAM_GB} GB 内存（cgroup 限 24 核）；每一档跑 $KIDS 只孩子，爹用 $PARENTS"
echo
printf '%8s %10s %12s %14s\n' "进程数" "花多久" "每只多少秒" "每小时多少只"
for W in $WORKER_LIST; do
    if [ "$W" -gt "$CORES" ]; then echo "跳过 $W：这台只有 $CORES 核"; continue; fi
    if [ "$W" -gt $((RAM_GB / 2)) ]; then echo "跳过 $W：内存不够（一个进程要 1.1 GB 左右）"; continue; fi
    LOG="$ROOT/logs/上云测速_${W}进程.log"
    START=$(date +%s)
    "$PY" -X utf8 -u tools/evolve.py --parents "$PARENTS" --candidates "$KIDS" --generations 1 \
        --workers "$W" --quiet --out-prefix "artifacts/_上云测速_$W" --seed 5 > "$LOG" 2>&1
    TAKEN=$(( $(date +%s) - START ))
    awk -v w="$W" -v s="$TAKEN" -v k="$KIDS" \
        'BEGIN{printf "%8d %9ds %13.1f %14.0f\n", w, s, s*w/k, k*3600.0/s}'
    echo "         （明细 $LOG）"
done
echo
echo "挑每小时最多的那一档当进程数，然后：WORKERS=<那一档> bash cloud/run.sh"