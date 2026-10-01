#!/bin/bash
# 七批跑完自动打包 + 自动关机
#
# 注意（本轮踩的坑）：看门狗如果先于训练启动，第一句 pgrep 就找不到 evolve_chase，
# 会直接当成「已经跑完」冲过去 —— 上一版就是这么掉进 _watchdog4.sh 的，
# 那样会打包错批次、还会照样关机。所以先「等训练起来」，起来了再「等它跑完」。
set -u
ENGINE=/hy-tmp/bwc/engine_v2
LOG=/hy-tmp/bwc/logs/watchdog.log
TAG=七批
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "$TAG 看门狗开始：先等训练起来（最多 10 分钟）"
UP=0
for _ in $(seq 1 60); do
  if pgrep -f 'evolv[e]_chase' > /dev/null; then UP=1; break; fi
  sleep 10
done
if [ "$UP" -ne 1 ]; then
  say "等了 10 分钟训练也没起来，看门狗退出（不打包、不关机）"
  exit 0
fi

say "训练起来了，等它跑完"
while pgrep -f 'evolv[e]_chase' > /dev/null; do sleep 60; done
say "训练进程没了，等 2 分钟再确认一次"
sleep 120
while pgrep -f 'evolv[e]_chase' > /dev/null; do
  say "又出现了训练进程，接着等"
  sleep 300
done

LAST=$(ls -1 "$ENGINE"/artifacts/追球_云_${TAG}_g*_keep.jsonl 2>/dev/null | tail -1)
say "最后一份 keep：${LAST:-（没有）}"
tail -4 /hy-tmp/bwc/logs/追球_云_${TAG}_*.log >> "$LOG" 2>/dev/null
mkdir -p /hy-tmp/bwc/备份
TAR=/hy-tmp/bwc/备份/追球_云_${TAG}_$(date +%Y%m%d_%H%M).tar.gz
cd /hy-tmp/bwc || exit 1
tar czf "$TAR" \
  engine_v2/artifacts/追球_云_${TAG}_g*.jsonl \
  engine_v2/artifacts/追球_云_${TAG}_起点.jsonl \
  engine_v2/tools/chase_task.py engine_v2/tools/evolve_chase.py \
  engine_v2/tools/chase_red_ball.py _gpu_env.sh \
  logs/追球_云_${TAG}_*.log 2>/dev/null
say "打包好了：$TAR（$(du -h "$TAR" 2>/dev/null | cut -f1)）"
say "等 20 分钟，然后关机"
sleep 1200
say "现在执行 /usr/local/bin/shutdown"
/usr/local/bin/shutdown >> "$LOG" 2>&1
say "关机命令返回码 $?"