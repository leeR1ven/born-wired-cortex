cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_二批_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --parents artifacts/追球_云_起点2.jsonl \
  --candidates 1000 --generations 8 --workers 24 --seconds 6 --seed 20260930 \
  --genes 8 --sigma 0.7 \
  --verify-top 8 --quiet --out-prefix artifacts/追球_云_二批 \
  > $LOG 2>&1 < /dev/null &
sleep 10
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"
grep -v GLFW $LOG | head -4
echo $LOG > /hy-tmp/bwc/logs/当前跑.txt