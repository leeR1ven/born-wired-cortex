#!/bin/bash
# 修好看门狗脚本里的 exec 写法，然后重启它（临时脚本）
sed -i 's|exec "$0"|exec /bin/bash /hy-tmp/bwc/_watchdog.sh|' /hy-tmp/bwc/_watchdog.sh
bash -n /hy-tmp/bwc/_watchdog.sh && echo "语法OK"
grep -n 'exec' /hy-tmp/bwc/_watchdog.sh
pkill -f '^bash /hy-tmp/bwc/_watchdog\.sh$'
sleep 2
setsid nohup bash /hy-tmp/bwc/_watchdog.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo "看门狗进程数：$(pgrep -fc '^bash /hy-tmp/bwc/_watchdog\.sh$')"
echo "训练进程数：$(pgrep -fc 'evolv[e]_chase')"
tail -2 /hy-tmp/bwc/logs/watchdog.log