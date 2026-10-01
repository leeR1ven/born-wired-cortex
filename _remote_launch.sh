cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_融合_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --candidates 5000 --generations 3 --workers 24 --seconds 6 --seed 20260930 \
  --quiet --out-prefix artifacts/追球_云_融合 \
  > $LOG 2>&1 < /dev/null &
sleep 5
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"