cd /hy-tmp/bwc/engine_v2
echo "进程数：$(pgrep -fc 'evolve_chase')"
echo "已测：$(wc -l < artifacts/追球_云_融合_g00.jsonl 2>/dev/null || echo 0) / 5000"
grep -v GLFW /hy-tmp/bwc/logs/追球_云_融合_20260930_1102.log | tail -4
PY=/hy-tmp/bwc/.venv/bin/python
$PY - <<'PY'
import io, json
rows = []
try:
    for line in io.open('/hy-tmp/bwc/engine_v2/artifacts/追球_云_融合_g00.jsonl', encoding='utf-8'):
        rows.append(json.loads(line))
except FileNotFoundError:
    pass
ok = [r for r in rows if r.get('status') == 'ok']
print('已建出来', len(ok), '只')
if ok:
    reach = sorted(ok, key=lambda r: (-r['covered'], -r['moving'], r['score']))
    print('朝球走最多的：')
    for r in reach[:5]:
        print('   第%d只 到了%d/4趟 最近%.2f米 走了%.2f米 没摔%s 路%s%s'
              % (r['index'], r['covered'], r['best'], r['travelled_m'], r['upright'],
                 r['chase']['route'], ' 对调' if r['chase']['flip'] else ''))
    print('两边都会的：%d 只' % sum(1 for r in ok if r.get('both')))
PY