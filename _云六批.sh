#!/bin/bash
# 六批：五批的融合有个 bug（整数基因被平均成小数，300 只里 107 只建不出来），修完重跑。
#
# 选爹：综合评分（见 evolve_chase.py 的 SCORE_WEIGHTS）—— 用户 2026-10-01：
#   真追住球 .30 / 球在正前方 .15 / 跑完落后 .13 / 追近了 .06 / 最近贴到 .05 /
#   两边都会 .05 / 没摔 .07 / 跑得快 .09 / 走得直 .04 / 静止球 .06
# 生孩子：综合分最高的前 8 只平均成一个「融合爹」，一半孩子从它变异来。
# 球速：0.17 起步，每代 +0.003，封顶 0.20（狗自己只有 0.20~0.23 米/秒，量过）。
cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_六批_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --parents artifacts/追球_云_六批_起点.jsonl \
  --candidates 300 --generations 10 --workers 24 \
  --seconds 6 --chase-seconds 20 --curve 1.0 \
  --flee 0.17 --flee-ramp 0.003 --flee-max 0.20 \
  --fuse-rate 0.5 --fuse-top 8 \
  --genes 8 --sigma 0.7 --keep-stable 10 --keep-fast 6 \
  --seed 20261004 \
  --verify-top 0 --out-prefix artifacts/追球_云_六批 \
  > $LOG 2>&1 < /dev/null &
sleep 15
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"
echo $LOG > /hy-tmp/bwc/logs/当前跑.txt
grep -v GLFW $LOG | grep -v CUDA | head -8