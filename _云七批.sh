#!/bin/bash
# 七批：追到球、跑得快、不摔——这三样吃重（用户 2026-10-01）。
#
# 权重（合计 1.00，在 tools/evolve_chase.py 的 SCORE_WEIGHTS 里）：
#   真追住球 .27 | 跑得快 .22 | 站得稳 .15 | 球在正前方 .10 | 跑完落后 .08
#   静止球 .05 | 追近了 .04 | 最近贴到 .03 | 两边都会 .03 | 走得直 .03
# 「站得稳」原来是「没摔」的是一/否二值格，抬权重也不改名次（摔了本来就直接沉底），
# 所以换成量「离摔倒还有多远」：全程最低直立分量 lowest_up_z，0.3 算摔、0.9 给满分。
# 「跑得快」那格从「除以 5 米」改成「除以 6 米」—— 量到最快的狗 20 秒跑 6.15 米，
# 原来除以 5 的话好狗全顶到 1.0，那一格根本分不出快慢。
#
# 球速：0.17 起步、每代 +0.003、封顶 0.20（狗自己 0.20~0.23 米/秒）。
# 融合：综合分最高前 8 只平均成「融合爹」，一半孩子从它变异来（整数基因取整）。
cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_七批_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --parents artifacts/追球_云_七批_起点.jsonl \
  --candidates 300 --generations 10 --workers 24 \
  --seconds 6 --chase-seconds 20 --curve 1.0 \
  --flee 0.17 --flee-ramp 0.003 --flee-max 0.20 \
  --fuse-rate 0.5 --fuse-top 8 \
  --genes 8 --sigma 0.7 --keep-stable 10 --keep-fast 6 \
  --seed 20261005 \
  --verify-top 0 --out-prefix artifacts/追球_云_七批 \
  > $LOG 2>&1 < /dev/null &
sleep 15
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"
echo $LOG > /hy-tmp/bwc/logs/当前跑.txt
grep -v GLFW $LOG | grep -v CUDA | head -8