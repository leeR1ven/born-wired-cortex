from pathlib import Path
p = Path("book/_抽取论文逐句.py")
s = p.read_text(encoding="utf-8")
old = """    per_sec = {}
    for r in rows:
        if r[5] != "其它":
            per_sec.setdefault(r[2], Counter())[r[5]] += 1
    for r in rows:
        if r[5] == "其它":
            c = per_sec.get(r[2])
            r[5] = c.most_common(1)[0][0] if c else "其它"
"""
new = """    # 兜底按**小节**（R1..R21 / 5.1..5.17）算，比按大章节准得多；连小节都没有就留"未分类"
    per_sub, per_sec = {}, {}
    for r in rows:
        if r[5] != "其它":
            key = r[3] or r[2]
            per_sub.setdefault(key, Counter())[r[5]] += 1
            per_sec.setdefault(r[2], Counter())[r[5]] += 1
    for r in rows:
        if r[5] == "其它":
            c = per_sub.get(r[3] or r[2]) or per_sec.get(r[2])
            top = c.most_common(1) if c else []
            r[5] = (top[0][0] + "?") if top else "未分类"
"""
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding="utf-8", newline="\n")
print("改成按小节兜底（带 ? 表示是猜的）")
