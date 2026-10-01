cd /hy-tmp/bwc/engine_v2
python3 - <<'PY'
import io, json
for line in io.open('artifacts/chase_smoke_g00.jsonl', encoding='utf-8'):
    r = json.loads(line)
    print(r['index'], r['status'], r.get('error'))
PY
PY2=/hy-tmp/bwc/.venv/bin/python
$PY2 - <<'PY'
import io, json
for line in io.open('/hy-tmp/bwc/engine_v2/artifacts/chase_smoke_g00.jsonl', encoding='utf-8'):
    r = json.loads(line)
    print(r['index'], r['status'], (r.get('error') or '')[:300])
PY