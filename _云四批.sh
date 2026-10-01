#!/bin/bash
# 四批：接着三批挑出来的爹往下迭代。
#
# 这一批只开了一处加速：考试瘦身 —— 静止球从 4 趟（左/右 x 3米/4米）减成随机 1 趟
# （用户 2026-10-01：模型已经真的会追红球了，红球不用换那么多种摆法各测一遍）。
# 测下来单只快 1.30 倍，而且追球那几个读数跟老考试**逐位相同**。
#
# 显卡那条路 2026-10-01 量过了，这一批**不开**：
#   单进程快 3.37 倍（40.72 -> 12.09 ms/步，两边各自重跑逐位相同、400 步后身体位置一样），
#   但 24 个进程一起抢一张 4090 时，墙钟跟 24 个 CPU 进程一模一样（都是 112 秒）——
#   小内核太多，显卡当瓶颈，一张卡顶 24 个核封顶。再往上开到 32/40 个进程直接显存爆掉
#   （每个进程约 600 MB，40 个 > 24 GB）。所以这一批走 CPU，显卡留给单进程的活儿用。
#   要真吃显卡得把「一进程算多只狗」批量化，那是大工程，另说。
cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_四批_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --parents artifacts/追球_云_四批_起点.jsonl \
  --candidates 300 --generations 10 --workers 24 \
  --seconds 6 --chase-seconds 20 --curve 1.0 --flee 0.15 \
  --genes 8 --sigma 0.7 --keep-stable 10 --keep-fast 6 \
  --seed 20261002 \
  --verify-top 0 --out-prefix artifacts/追球_云_四批 \
  > $LOG 2>&1 < /dev/null &
sleep 15
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"
echo $LOG > /hy-tmp/bwc/logs/当前跑.txt
grep -v GLFW $LOG | head -5