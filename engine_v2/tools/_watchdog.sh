#!/bin/bash
# 训练跑完自动关机（用户 2026-09-30 要求）
set -u
ENGINE=/hy-tmp/bwc/engine_v2
LOG=/hy-tmp/bwc/logs/watchdog.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "看门狗开始：先等 evolve_chase 跑完"
while pgrep -f 'evolv[e]_chase' > /dev/null; do
  sleep 60
done
say "训练进程没了，等 2 分钟再确认一次"
sleep 120
if pgrep -f 'evolv[e]_chase' > /dev/null; then
  say "又出现了训练进程，重新等"
  exec "$0"
fi

LAST=$(ls -1 "$ENGINE"/artifacts/追球_云_二批_g0*_keep.jsonl 2>/dev/null | tail -1)
say "最后一份 keep：${LAST:-（没有）}"
tail -4 /hy-tmp/bwc/logs/追球_云_二批_*.log >> "$LOG" 2>/dev/null

mkdir -p /hy-tmp/bwc/备份
TAR=/hy-tmp/bwc/备份/追球_云_二批_$(date +%Y%m%d_%H%M).tar.gz
cd /hy-tmp/bwc || exit 1
tar czf "$TAR" \
  engine_v2/artifacts/追球_云_二批_g0*.jsonl \
  engine_v2/artifacts/追球_云_起点2.jsonl \
  engine_v2/tools/chase_task.py engine_v2/tools/evolve_chase.py \
  engine_v2/tools/verify_chase.py engine_v2/tools/chase_red_ball.py \
  engine_v2/born_wired/embodied.py \
  logs/追球_云_二批_*.log 2>/dev/null
say "打包好了：$TAR（$(du -h "$TAR" 2>/dev/null | cut -f1)）"

say "等 10 分钟（万一你想先看一眼），然后关机"
sleep 600
say "现在执行 /usr/local/bin/shutdown"
/usr/local/bin/shutdown >> "$LOG" 2>&1
say "关机命令返回码 $?（如果没关掉，明早在控制台点关机就行）"