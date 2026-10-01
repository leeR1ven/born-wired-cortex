cd /hy-tmp/bwc/engine_v2
ls artifacts/云端/
python3 - <<'PY'
import io, json
r = json.loads(io.open('artifacts/云端/题1f_演化_云_冠军.jsonl', encoding='utf-8').readline())
print('冠军 gen', r['gen'], 'index', r['index'], 'seed', r['seed'], '走了', round(r['travelled_m'], 3), '米 没摔', r['upright'])
PY