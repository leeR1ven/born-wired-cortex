cd /hy-tmp/bwc/engine_v2
echo "进程数：$(pgrep -fc 'evolve_chase')"
echo "已测：$(wc -l < artifacts/追球_云_复试_g00.jsonl 2>/dev/null || echo 0) / 5000"
grep -v GLFW /hy-tmp/bwc/logs/追球_云_复试_20260930_1117.log | tail -3