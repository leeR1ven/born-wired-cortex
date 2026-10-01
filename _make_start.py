import io, json, sys
src = '/hy-tmp/bwc/engine_v2/artifacts/追球_云_融合_g00.jsonl'
rows = [json.loads(l) for l in io.open(src, encoding='utf-8') if l.strip()]
ok = [r for r in rows if r.get('status') == 'ok']
ok.sort(key=lambda r: (-int(r['moving']), -r['covered'], r['score']))
keep = ok[:8]
out = '/hy-tmp/bwc/engine_v2/artifacts/追球_云_融合_起点.jsonl'
with io.open(out, 'w', encoding='utf-8') as h:
    for r in keep:
        h.write(json.dumps(r, ensure_ascii=False) + '\n')
print('已测 %d 只，起点取最好的 %d 只' % (len(rows), len(keep)))
for r in keep:
    print('   第%d只 到了%d/4 最近%.2f米 走了%.2f米 两边%s 路%s' % (r['index'], r['covered'], r['best'], r['travelled_m'], r['both'], r['chase']['route']))
PY