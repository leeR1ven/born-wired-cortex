#!/bin/bash
# 三批：每代 300 只、迭代 10 代；追球每趟 20 秒；
# 球匀速、速度非常慢（0.15 米/秒），随机曲线背离狗 —— 先让狗学会「朝着球走」
# 用户 2026-10-01：一代只生 24 只可能性太小，要几百只
cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_三批_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --parents artifacts/追球_云_三批_起点.jsonl \
  --candidates 300 --generations 10 --workers 24 \
  --seconds 6 --chase-seconds 20 --curve 1.0 --flee 0.15 \
  --genes 8 --sigma 0.7 --keep-stable 10 --keep-fast 6 \
  --seed 20261001 \
  --verify-top 0 --out-prefix artifacts/追球_云_三批 \
  > $LOG 2>&1 < /dev/null &
sleep 10
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"
grep -v GLFW $LOG | head -4
echo $LOG > /hy-tmp/bwc/logs/当前跑.txt
