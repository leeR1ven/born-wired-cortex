cd /hy-tmp/bwc/engine_v2
cp -f artifacts/追球_云_融合_g00.jsonl artifacts/追球_云_融合_g00_第一批.jsonl
pkill -f 'evolv[e]_chase'; sleep 4
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
STAMP=$(date +%Y%m%d_%H%M)
LOG=/hy-tmp/bwc/logs/追球_云_复试_${STAMP}.log
setsid nohup /hy-tmp/bwc/.venv/bin/python tools/evolve_chase.py \
  --parents artifacts/追球_云_融合_起点.jsonl \
  --candidates 5000 --generations 3 --workers 24 --seconds 6 --seed 20261001 \
  --verify-top 8 --quiet --out-prefix artifacts/追球_云_复试 \
  > $LOG 2>&1 < /dev/null &
sleep 8
echo "日志：$LOG"
echo "进程数：$(pgrep -fc 'evolve_chase')"
head -3 $LOG | grep -v GLFW