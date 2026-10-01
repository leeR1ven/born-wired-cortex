# -*- coding: utf-8 -*-
"""小补丁：分号也断句 + 没命中关键词的句子按本章节的主部件兜底。"""
from pathlib import Path
p = Path("book/_抽取论文逐句.py")
s = p.read_text(encoding="utf-8")

old_split = """        if ch in "。！？":"""
new_split = """        if ch in "。！？；;":"""
assert s.count(old_split) == 1
s = s.replace(old_split, new_split)

# 兜底：两遍分类。第一遍按关键词，第二遍把"其它"归到本章节的主部件
old_main = """            comp, kind = classify(sent, section)
            rows.append((n, lineno, section, sub, kind, comp, sent))
"""
new_main = """            comp, kind = classify(sent, section)
            rows.append([n, lineno, section, sub, kind, comp, sent])

    # 第二遍：没命中关键词的，归到本章节里出现最多的那个部件（章节本身就是语境）
    per_sec = {}
    for r in rows:
        if r[5] != "其它":
            per_sec.setdefault(r[2], Counter())[r[5]] += 1
    for r in rows:
        if r[5] == "其它":
            c = per_sec.get(r[2])
            r[5] = c.most_common(1)[0][0] if c else "其它"
    rows = [tuple(r) for r in rows]
"""
assert s.count(old_main) == 1
s = s.replace(old_main, new_main)

# 分号拆出来的短从句（<12 字）原来会被丢掉，放低到 8
s = s.replace("if len(sent) < 12:", "if len(sent) < 8:")
p.write_text(s, encoding="utf-8", newline="\n")
print("补丁写好")
