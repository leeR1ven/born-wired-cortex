# -*- coding: utf-8 -*-
"""从二批台账里挑出「会追球」的当三批起点（用户 2026-10-01：接着追球往下练）。"""
import io, json, glob, os

d = r"F:\born-wired-cortex\engine_v2\artifacts\云端\二批"
out = r"F:\born-wired-cortex\engine_v2\artifacts\追球_云_三批_起点.jsonl"
rows = []
for path in sorted(glob.glob(os.path.join(d, "追球_云_二批_g0[0-7].jsonl"))):
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)

print("建出来的总数：", len(rows))
for bar in (1, 2):
    print("  球跑着还能追住 %d 趟的：%d 只" % (bar, sum(1 for r in rows if r.get("flee_covered", 0) >= bar)))
print("  6 秒里最后落后 < 1 米的：", sum(1 for r in rows if r.get("flee_settled", 9) <= 1.0))
print("  6 秒里最后落后 < 2 米的：", sum(1 for r in rows if r.get("flee_settled", 9) <= 2.0))
print("  追上过球的（caught>0）：", sum(1 for r in rows if r.get("caught", 0) > 0))

# 起点：追球追得最好的那批 + 静止球也走得到的
picked, seen = [], set()
def take(cands, n):
    got = 0
    for row in cands:
        mark = tuple(sorted((k, round(float(v), 6)) for k, v in row["genome"].items()))
        if mark in seen:
            continue
        seen.add(mark)
        picked.append(row)
        got += 1
        if got >= n:
            break

take(sorted(rows, key=lambda r: (-r.get("flee_covered", 0), r.get("flee_settled", 9),
                                 -r.get("flee_kept", 0), -r.get("caught", 0))), 30)
take(sorted(rows, key=lambda r: (r.get("flee_settled", 9), -r.get("caught", 0))), 15)
take(sorted(rows, key=lambda r: (-r.get("caught", 0), r.get("score", 9))), 15)

io.open(out, "w", encoding="utf-8", newline="\n").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked))
print("起点写好：%s（%d 只）" % (out, len(picked)))
print("起点里最好几只：")
for r in picked[:8]:
    print("  第%d代第%d只 追住%d/%d趟 落后%.2f米 正前%3.0f%% 追上过%d次 没摔=%s"
          % (r.get("gen"), r.get("index"), r.get("flee_covered", 0), r.get("flee_trials", 0),
             r.get("flee_settled", float("nan")), r.get("flee_kept", 0) * 100,
             r.get("caught", 0), r.get("upright")))