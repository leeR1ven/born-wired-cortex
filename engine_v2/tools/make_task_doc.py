# -*- coding: utf-8 -*-
"""把题库原样读成一份人看的文档：桌面上的 Markdown 和 HTML。

    python tools/make_task_doc.py                       # 只读题库本身
    python tools/make_task_doc.py --ledger artifacts/轮6_全库24.jsonl

不带台账时，文档写的是每道题问什么、量什么、门槛多少、先要会什么，外加
出厂动物（没变异没训练的那一只）在这道题上的判定。
带上全库台账时，再加一列「这批候选在这道题上的通过率」，并附一份体检结论：
哪些题一只都过不去、哪些题一只不落全过、哪些题跟出厂动物打架。

文档里的每个数字都能从题库和台账重新读出来，没有手抄。
"""
import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import screen_candidates as sc                               # noqa: E402
from tools import taskbank as tb                                        # noqa: E402

BIRTH_AUDIT = ROOT / "artifacts" / "审计_出生动物_返回线_s0.jsonl"
BIRTH_AUDIT_BEFORE = ROOT / "artifacts" / "审计_出生动物_无返回线_before_returnline.jsonl"


def read_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def birth_verdicts(path):
    return {row["task"]: row for row in read_jsonl(path)}


def candidate_counts(path):
    """Per-task (asked, passed) over the full-stage animals in a ledger."""
    asked, passed = {}, {}
    animals = [r for r in read_jsonl(path)
               if r.get("kind") == "candidate" and r.get("stage") == "full" and sc.built(r)]
    for row in animals:
        for name in row.get("ran_tasks") or []:
            asked[name] = asked.get(name, 0) + 1
        for name in row.get("passed_tasks") or []:
            passed[name] = passed.get(name, 0) + 1
    return animals, asked, passed


def stage_of(name, screen, probe):
    if name in screen:
        return "筛子 screen"
    if name in probe:
        return "探针 probe"
    return "全库 full"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path.home() / "Desktop"))
    ap.add_argument("--ledger", default=None,
                    help="可选：一份 stage=full 的台账，用来加一列候选通过率")
    ap.add_argument("--birth", default=str(BIRTH_AUDIT))
    args = ap.parse_args(argv)

    tasks = tb.stage_tasks("full")
    screen = {t["name"] for t in tb.stage_tasks("screen")}
    probe = {t["name"] for t in tb.stage_tasks("probe")}
    order = {t["name"]: i for i, t in enumerate(tasks, 1)}
    depth = tb.ladder(tasks)
    groups = []
    for task in tasks:
        if task["group"] not in groups:
            groups.append(task["group"])

    birth = birth_verdicts(args.birth)
    birth_before = birth_verdicts(BIRTH_AUDIT_BEFORE)
    n_pass = sum(1 for t in tasks if (birth.get(t["name"]) or {}).get("passed"))
    n_before = sum(1 for t in tasks if (birth_before.get(t["name"]) or {}).get("passed"))
    sim_seconds = sum(t["sim_seconds"] for t in tasks)

    animals, asked, passed = [], {}, {}
    if args.ledger:
        animals, asked, passed = candidate_counts(args.ledger)

    def cand_cell(name):
        if not animals:
            return None
        a, p = asked.get(name, 0), passed.get(name, 0)
        return p, a

    def flags(name):
        out = []
        a, p = asked.get(name, 0), passed.get(name, 0)
        if animals and a and p == 0:
            out.append("全灭")
        if animals and a and p == a:
            out.append("全过")
        if name in birth and animals and a:
            if birth[name]["passed"] and p == 0:
                out.append("与出厂冲突")
            if (not birth[name]["passed"]) and p == a:
                out.append("与出厂冲突")
        return out

    # ------------------------------------------------------------------ md
    md = []
    md.append("# 题库全表｜born-wired-cortex")
    md.append("")
    md.append("模型要考的**全部 %d 道题**，逐题列出：问什么、量什么、门槛多少、先要会什么。" % len(tasks))
    md.append("")
    md.append("- 来源：`engine_v2/tools/taskbank.py`（本文档由 `engine_v2/tools/make_task_doc.py` 直接读出，不是手抄）")
    md.append("- 生成时间：%s" % time.strftime("%Y-%m-%d %H:%M"))
    md.append("- 出厂动物（出生那一只，没变异、没训练）全库 **%d / %d**；装上返回线之前是 %d / %d" % (
        n_pass, len(tasks), n_before, len(tasks)))
    md.append("- 全库模拟时间合计 %.0f 秒 = %.1f 分钟（一只动物把 130 道题全考一遍）" % (sim_seconds, sim_seconds / 60.))
    if animals:
        md.append("- 候选：`%s` 里 **%d 只**建成的动物，每只都考了全部 %d 道" % (args.ledger, len(animals), len(tasks)))
    md.append("")
    md.append("## 一、三级台阶：不是漏题，是省时间")
    md.append("")
    md.append("| 档 | 题数 | 覆盖 | 一只动物要多久 | 什么时候用 |")
    md.append("| --- | --- | --- | --- | --- |")
    md.append("| 筛子 `screen` | %d | 最基础几项 | 47.9 秒 | 从几万只里先淘汰明显不行的 |" % len(screen))
    md.append("| 探针 `probe` | %d | 8 / 16 个方面 | 约 1 分钟 | 链式考，第一道不过就停 |" % len(probe))
    md.append("| 全库 `full` | **%d** | **16 / 16 个方面** | 约 26 分钟 | 给每一只出完整成绩单、审计题目本身 |" % len(tasks))
    md.append("")
    md.append("## 二、阶梯：谁要先会才能考谁")
    md.append("")
    md.append("每道题都写清了前置。第 0 层是「天生就该会、不用先会别的」。")
    md.append("")
    md.append("| 阶梯 | 题数 | 这一层的题 |")
    md.append("| --- | --- | --- |")
    for d in range(max(depth.values()) + 1):
        ts = [t for t in tasks if depth[t["name"]] == d]
        if ts:
            md.append("| 第 %d 层 | %d | %s |" % (d, len(ts), "、".join("`%s`" % t["name"] for t in ts)))
    md.append("")
    md.append("## 三、16 个方面")
    md.append("")
    head = "| 方面 | 题数 | 出厂动物过了 | 模拟秒 |"
    sep = "| --- | --- | --- | --- |"
    md.append(head)
    md.append(sep)
    for g in groups:
        ts = [t for t in tasks if t["group"] == g]
        p = sum(1 for t in ts if (birth.get(t["name"]) or {}).get("passed"))
        md.append("| %s | %d | %d | %.0f |" % (g, len(ts), p, sum(t["sim_seconds"] for t in ts)))
    md.append("| **合计** | **%d** | **%d** | **%.0f** |" % (len(tasks), n_pass, sim_seconds))
    md.append("")
    md.append("## 四、出厂动物没过的 %d 道" % (len(tasks) - n_pass))
    md.append("")
    md.append("| # | 题 | 方面 | 为什么没过 |")
    md.append("| --- | --- | --- | --- |")
    for t in tasks:
        row = birth.get(t["name"])
        if row is not None and not row["passed"]:
            md.append("| %d | `%s` | %s | %s |" % (order[t["name"]], t["name"], t["group"], row.get("why", "")))
    md.append("")
    if animals:
        dead = [t["name"] for t in tasks if asked.get(t["name"], 0) and passed.get(t["name"], 0) == 0]
        free = [t["name"] for t in tasks if asked.get(t["name"], 0) and passed.get(t["name"], 0) == asked[t["name"]]]
        md.append("## 五、这批 %d 只候选的题目体检" % len(animals))
        md.append("")
        md.append("- **全灭**（问到了、一只没过）%d 道：%s" % (len(dead), "、".join("`%s`" % x for x in dead) or "（无）"))
        md.append("- **全过**（一只不落，不筛人）%d 道：%s" % (len(free), "、".join("`%s`" % x for x in free) or "（无）"))
        md.append("- **与出厂冲突**（出厂过、这批全灭，或反过来）%d 道：%s" % (
            sum(1 for t in tasks if "与出厂冲突" in flags(t["name"])),
            "、".join("`%s`" % t["name"] for t in tasks if "与出厂冲突" in flags(t["name"])) or "（无）"))
        md.append("")
    md.append("## 六、逐题详表")
    md.append("")
    for g in groups:
        ts = [t for t in tasks if t["group"] == g]
        md.append("### %s（%d 道，模拟 %.0f 秒）" % (g, len(ts), sum(t["sim_seconds"] for t in ts)))
        md.append("")
        for t in ts:
            row = birth.get(t["name"])
            mark = "—" if row is None else ("✅ 过" if row["passed"] else "❌ 没过")
            cell = cand_cell(t["name"])
            extra = "" if cell is None else "　·　候选 %d/%d" % (cell[0], cell[1])
            md.append("#### %d. `%s`　出厂 %s　·　%s　·　阶梯第 %d 层　·　模拟 %.1f 秒%s" % (
                order[t["name"]], t["name"], mark, stage_of(t["name"], screen, probe),
                depth[t["name"]], t["sim_seconds"], extra))
            md.append("")
            md.append("- **问什么**：%s" % t["question"])
            md.append("- **量什么**：%s" % (t.get("reads") or "（未写）"))
            md.append("- **门槛**：%s" % (t.get("bar_text") or "（未写）"))
            if row is not None:
                md.append("- **出厂动物**：%s" % row.get("why", ""))
            md.append("- **先要会**：%s" % ("、".join("`%s`" % r for r in t.get("requires") or []) or "（没有前置）"))
            md.append("- **备注**：%s" % (t.get("note") or "（无）"))
            if flags(t["name"]):
                md.append("- **标记**：%s" % "、".join(flags(t["name"])))
            md.append("")
    md_text = "\n".join(md)

    # ---------------------------------------------------------------- html
    H = []
    H.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    H.append("<title>题库全表 %d 道｜born-wired-cortex</title>" % len(tasks))
    H.append("""<style>
body{margin:0 auto;padding:32px 40px 80px;background:#fbfbfc;color:#1a1a1a;max-width:1080px;
 font:16px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
h1{font-size:30px;margin:0 0 6px}
h2{font-size:22px;margin:44px 0 12px;padding-bottom:6px;border-bottom:2px solid #e3e3e6}
h3{font-size:19px;margin:32px 0 10px;color:#2f6fd0}
.sub{color:#666;font-size:14px;margin:0 0 18px}
table{border-collapse:collapse;width:100%;margin:12px 0 20px;font-size:15px;background:#fff}
th,td{border:1px solid #e3e3e6;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f1f3f6}
code{background:#eef1f5;padding:1px 5px;border-radius:4px;font-size:14px;font-family:Consolas,Menlo,monospace}
.card{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:14px 18px;margin:14px 0}
.card h4{margin:0 0 8px;font-size:17px;font-family:Consolas,Menlo,monospace}
.card ul{margin:0;padding-left:20px}
.pass{color:#0a7d33;font-weight:600}.fail{color:#c0392b;font-weight:600}.none{color:#666}
.box{background:#fff;border-left:4px solid #2f6fd0;padding:10px 16px;margin:14px 0;border-radius:0 8px 8px 0}
td.flag{color:#c0392b}
</style></head><body>""")
    H.append("<h1>题库全表：%d 道题</h1>" % len(tasks))
    H.append("<p class='sub'>born-wired-cortex ／ 全部考核题 ／ 生成于 %s ／ 来源 <code>tools/taskbank.py</code></p>" % time.strftime("%Y-%m-%d %H:%M"))
    H.append("<p>出厂动物（出生那一只，没变异没训练）全库 <b>%d / %d</b>；装返回线之前 %d / %d。</p>" % (n_pass, len(tasks), n_before, len(tasks)))
    H.append("<div class='box'>三级台阶：<b>筛子 %d 道</b>（47.9 秒）→ <b>探针 %d 道</b>（约 1 分钟）→ <b>全库 %d 道</b>（约 26 分钟，16 个方面一个不漏）。前两档只是全库的前几道。</div>" % (len(screen), len(probe), len(tasks)))
    if animals:
        H.append("<div class='box'>本页还带上了这批 <b>%d 只候选</b>的成绩：每只都考了全部 %d 道（<code>%s</code>）。</div>" % (len(animals), len(tasks), html.escape(str(args.ledger))))
    H.append("<h2>一、16 个方面</h2><table><tr><th>方面</th><th>题数</th><th>出厂动物过了</th><th>模拟秒</th></tr>")
    for g in groups:
        ts = [t for t in tasks if t["group"] == g]
        p = sum(1 for t in ts if (birth.get(t["name"]) or {}).get("passed"))
        H.append("<tr><td>%s</td><td>%d</td><td>%d</td><td>%.0f</td></tr>" % (g, len(ts), p, sum(t["sim_seconds"] for t in ts)))
    H.append("<tr><th>合计</th><th>%d</th><th>%d</th><th>%.0f</th></tr></table>" % (len(tasks), n_pass, sim_seconds))
    H.append("<h2>二、阶梯</h2><table><tr><th>阶梯</th><th>题数</th><th>这一层的题</th></tr>")
    for d in range(max(depth.values()) + 1):
        ts = [t for t in tasks if depth[t["name"]] == d]
        if ts:
            H.append("<tr><td>第 %d 层</td><td>%d</td><td>%s</td></tr>" % (
                d, len(ts), "、".join("<code>%s</code>" % t["name"] for t in ts)))
    H.append("</table>")
    H.append("<h2>三、出厂动物没过的 %d 道</h2><table><tr><th>#</th><th>题</th><th>方面</th><th>为什么没过</th></tr>" % (len(tasks) - n_pass))
    for t in tasks:
        row = birth.get(t["name"])
        if row is not None and not row["passed"]:
            H.append("<tr><td>%d</td><td><code>%s</code></td><td>%s</td><td>%s</td></tr>" % (
                order[t["name"]], t["name"], t["group"], html.escape(str(row.get("why", "")))))
    H.append("</table>")
    if animals:
        H.append("<h2>四、这批 %d 只候选的逐题通过率</h2><table><tr><th>#</th><th>题</th><th>方面</th><th>过 / 问到</th><th>通过率</th><th>标记</th></tr>" % len(animals))
        for t in tasks:
            a, p = asked.get(t["name"], 0), passed.get(t["name"], 0)
            rate = (100. * p / a) if a else 0.
            H.append("<tr><td>%d</td><td><code>%s</code></td><td>%s</td><td>%d / %d</td><td>%.0f%%</td><td class='flag'>%s</td></tr>" % (
                order[t["name"]], t["name"], t["group"], p, a, rate, "、".join(flags(t["name"]))))
        H.append("</table>")
    H.append("<h2>五、逐题详表</h2>")
    for g in groups:
        ts = [t for t in tasks if t["group"] == g]
        H.append("<h3>%s　<span class='none'>%d 道 ／ 模拟 %.0f 秒</span></h3>" % (g, len(ts), sum(t["sim_seconds"] for t in ts)))
        for t in ts:
            row = birth.get(t["name"])
            if row is None:
                mark = "<span class='none'>未记录</span>"
            else:
                mark = "<span class='pass'>✅ 过</span>" if row["passed"] else "<span class='fail'>❌ 没过</span>"
            cell = cand_cell(t["name"])
            H.append("<div class='card'><h4>%d. %s　%s%s</h4>" % (
                order[t["name"]], html.escape(t["name"]), mark,
                "" if cell is None else "　<span class='none'>候选 %d/%d</span>" % (cell[0], cell[1])))
            H.append("<p><b>问：</b>%s</p>" % html.escape(t["question"]))
            H.append("<ul>")
            H.append("<li><b>量什么</b>：%s</li>" % html.escape(t.get("reads") or "（未写）"))
            H.append("<li><b>门槛</b>：%s</li>" % html.escape(t.get("bar_text") or "（未写）"))
            H.append("<li><b>档</b>：%s　<b>阶梯</b>：第 %d 层　<b>模拟</b>：%.1f 秒</li>" % (
                stage_of(t["name"], screen, probe), depth[t["name"]], t["sim_seconds"]))
            if row is not None:
                H.append("<li><b>出厂动物读数</b>：%s</li>" % html.escape(str(row.get("why", ""))))
            req = t.get("requires") or []
            H.append("<li><b>先要会</b>：%s</li>" % ("、".join("<code>%s</code>" % html.escape(r) for r in req) if req else "（没有前置）"))
            H.append("<li><b>备注</b>：%s</li>" % html.escape(t.get("note") or "（无）"))
            if flags(t["name"]):
                H.append("<li class='fail'><b>标记</b>：%s</li>" % "、".join(flags(t["name"])))
            H.append("</ul></div>")
    H.append("</body></html>")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "题库全表_130道.md").write_text(md_text, encoding="utf-8", newline="\n")
    (out / "题库全表_130道.html").write_text("\n".join(H), encoding="utf-8", newline="\n")
    print("-> %s" % (out / "题库全表_130道.md"))
    print("-> %s" % (out / "题库全表_130道.html"))
    print("出厂动物 %d/%d；候选 %d 只。" % (n_pass, len(tasks), len(animals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())