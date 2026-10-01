import io, json
p = r"F:\born-wired-cortex\engine_v2\artifacts\追球_云_三批_起点.jsonl"
rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
print("起点行数：", len(rows))
keys = {}
for r in rows:
    c = r.get("chase") or {}
    keys[tuple(sorted(c))] = keys.get(tuple(sorted(c)), 0) + 1
for k, v in keys.items():
    print(v, "行 → chase 字段：", k)
print("样例：", rows[0].get("chase"))
print("有 avoidance_gain 的行数：", sum(1 for r in rows if "avoidance_gain" in r["genome"]))