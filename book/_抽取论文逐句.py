# -*- coding: utf-8 -*-
"""把论文一句一句拆出来，编号、归类，写成一张"一句不漏"的账。

用户 2026-10-01：论文要一句一句看，任何一句都可能是重要信息；还要做好分类。

用法：
    python book/_抽取论文逐句.py

产物：
    book/03_论文逐句清单.md     每句一行：编号 | 行号 | 章节 | 类型 | 部件 | 原句
    book/03_论文逐句统计.md     分类统计（按 章节 x 类型、按 部件）
"""
import io
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "Born_wired_manuscript.md"
OUT = ROOT / "book"

# 参考文献不是知识点，跳过；图注要留（图注里全是结论）
STOP_AT = re.compile(r"^##\s+References")
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")

# 部件关键词（先匹配到的赢，所以顺序 = 优先级）
COMPONENT = [
    ("前额叶", r"prefrontal|return line|互惠"),
    ("记忆与时间", r"hippocamp|memory region|motor memory|replay|time ring|tick 2\.2|episod"),
    ("眼睛与视觉", r"\bgaze|retina|retinal|eye|eyes|ocular|vergence|pupil|stereo"),
    ("视觉", r"\bvisual|image|picture|pixel|colou?r|red ball|texture|macro-pixel"),
    ("听觉", r"auditory|\bsound|cochlea|pinna|tone|binaural|hearing"),
    ("身体与运动", r"\bmotor|leg|joint|MuJoCo|Go2|torque|walk|stand|posture|vestibular|propriocept|body state|locomot"),
    ("特征与表示", r"feature|thermometer|representation|encoder|encoding|sparse code|receptive field"),
    ("学习与可塑性", r"plastic|Hebbian|co-activat|learning rule|synapse|weight|trace|dopamine|reward|nursery|lesson"),
    ("本能与标定", r"instinct|calibrat|repertoire|innate wiring|\bname|\bnames\b|blur"),
    ("演化与题库", r"evolution|mutant|candidate|task bank|bank of task|selection|generation|score|fitness"),
    ("融合", r"\bmerge|fusion|blend|added into one"),
    ("规模与性能", r"\bcost|scale|larger|wider|\bms\b|Hz|latency|memory footprint|how large"),
    ("身体之外的类比", r"like a|as if|metaphor|analogy"),
]

KIND = [
    ("边界与不声明", r"\bdoes not\b|\bcannot\b|\bnot claim|no claim|do not\b|limitat|remains? (an )?open|future work|would be needed|falsif"),
    ("规则", r"\brule\b|\bif\b|\beach\b|\bper\b|every tick|is written|receives|projects to|must be|threshold"),
    ("参数", r"\d+(\.\d+)?\s*(%|m\b|ms\b|s\b|rad\b|degrees?\b|cells?\b|Hz\b|Nm\b|MB\b|words?\b)"),
    ("证据", r"\bR\d+\b|Fig(ure)?\.?\s*\d|we measured|was measured|table \d|supplement"),
    ("结论", r"\bwe show|demonstrat|indicat|therefore|\bthus\b|in short|the result is"),
    ("定义", r"we (define|call|use the term)|is called|refers to|means that|that is,\s"),
]

ABBR = {"e.g", "i.e", "cf", "vs", "etc", "Fig", "No", "Dr", "al", "approx", "ca", "resp", "St"}


def sentences(text):
    """把一段话拆成句子。数字小数点、常见缩写不拆。"""
    out, buf = [], ""
    i = 0
    while i < len(text):
        ch = text[i]
        buf += ch
        if ch in "。！？；;":
            out.append(buf.strip())
            buf = ""
        elif ch == ".":
            prev = text[max(0, i - 30):i + 1]
            m = re.search(r"([A-Za-z]+)\.$", prev)
            nxt = text[i + 1:i + 3]
            ok_next = (i + 1 >= len(text)) or nxt.startswith(" ") or nxt.startswith("\n")
            if m and m.group(1) in ABBR:
                pass
            elif re.match(r"\d\.$", prev) and nxt[:1].isdigit():
                pass
            elif ok_next:
                out.append(buf.strip())
                buf = ""
        i += 1
    if buf.strip():
        out.append(buf.strip())
    return [s for s in out if len(s) > 1]


def paragraphs(lines):
    """把被硬折行的段落接回去，同时记住它从第几行来。"""
    block, start = [], None
    for n, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if line.strip() == "":
            if block:
                yield start, " ".join(block)
                block, start = [], None
            continue
        if HEADING.match(line) or line.lstrip().startswith(("*", ">", "|", "-", "```", "#")):
            if block:
                yield start, " ".join(block)
                block, start = [], None
            yield n, line.strip()
            continue
        if not block:
            start = n
        block.append(line.strip())
        if re.search(r"[.!?。！？]\s*$", line.strip()):
            yield start, " ".join(block)
            block, start = [], None
    if block:
        yield start, " ".join(block)


def classify(text, section):
    low = text.lower()
    comp = "其它"
    for name, pat in COMPONENT:
        if re.search(pat, low):
            comp = name
            break
    kind = "陈述"
    for name, pat in KIND:
        if re.search(pat, text if name == "证据" else low):
            kind = name
            break
    return comp, kind


def main():
    lines = io.open(PAPER, encoding="utf-8").read().splitlines()
    rows, section, sub = [], "(前言)", ""
    n = 0
    for lineno, para in paragraphs(lines):
        if STOP_AT.match(para):
            break
        m = HEADING.match(para)
        if m:
            if len(m.group(1)) == 2:
                section, sub = m.group(2).strip(), ""
            else:
                sub = m.group(2).strip()
            continue
        if para.startswith("|") or para.startswith("*") and len(para) < 4:
            continue
        for sent in sentences(para):
            if len(sent) < 8:
                continue
            n += 1
            comp, kind = classify(sent, section)
            rows.append([n, lineno, section, sub, kind, comp, sent])

    # 第二遍：没命中关键词的，归到本章节里出现最多的那个部件（章节本身就是语境）
    # 兜底按**小节**（R1..R21 / 5.1..5.17）算，比按大章节准得多；连小节都没有就留"未分类"
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
    rows = [tuple(r) for r in rows]

    with io.open(OUT / "03_论文逐句清单.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 论文逐句清单（一句不漏的账）\n\n")
        fh.write("> 由 `book/_抽取论文逐句.py` 自动生成。"
                 "编号连续即无遗漏；分类是机器粗分，写书时逐条人工过。\n\n")
        fh.write("| # | 行 | 章节 | 小节 | 类型 | 部件 | 原句 |\n|---|---|---|---|---|---|---|\n")
        for num, lineno, sec, s, kind, comp, sent in rows:
            fh.write("| %d | %d | %s | %s | %s | %s | %s |\n"
                     % (num, lineno, sec.replace("|", "/"), s.replace("|", "/"),
                        kind, comp, sent.replace("|", "/")))

    per_section = Counter((r[2], r[4]) for r in rows)
    per_comp = Counter(r[5] for r in rows)
    per_kind = Counter(r[4] for r in rows)
    with io.open(OUT / "03_论文逐句统计.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 论文逐句统计\n\n")
        fh.write("总句数：**%d**\n\n## 按类型\n\n| 类型 | 句数 |\n|---|---|\n" % len(rows))
        for k, v in per_kind.most_common():
            fh.write("| %s | %d |\n" % (k, v))
        fh.write("\n## 按部件\n\n| 部件 | 句数 |\n|---|---|\n")
        for k, v in per_comp.most_common():
            fh.write("| %s | %d |\n" % (k, v))
        fh.write("\n## 按章节\n\n| 章节 | 句数 |\n|---|---|\n")
        sec_count = Counter(r[2] for r in rows)
        for k, v in sec_count.most_common():
            fh.write("| %s | %d |\n" % (k, v))

    print("总句数：%d" % len(rows))
    print("按类型：" + "、".join("%s %d" % (k, v) for k, v in per_kind.most_common()))
    print("按部件：" + "、".join("%s %d" % (k, v) for k, v in per_comp.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
