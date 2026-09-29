# -*- coding: utf-8 -*-
"""题目体检：拿一份全库台账，看每道题有没有在筛人、有没有人过得去。

    python tools/task_health.py --ledger artifacts/轮6_全库24.jsonl [--out logs/题目体检.md]

读的是一份**全库**（stage=full）台账：每只动物都被问了每一道题，所以
「问到的只数」就是「建出来的只数」，每道题的分母一样，通过率可以直接横向比。

会标出四类可疑题：
  全灭     —— 问到了若干只、一只都没过（题目本身有问题，或者这道本能谁都没有）
  全过     —— 一只不落全过（不筛人，等于不占筛选预算）
  与出厂冲突 —— 出厂动物过了、这批一只没过（或反过来）
  太便宜   —— 模拟秒数很小，几乎不花时间，也不筛人
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import screen_candidates as sc                               # noqa: E402
from tools import taskbank as tb                                        # noqa: E402


def read(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


built = sc.built        # the same rule the round itself uses: an error means no animal


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--birth", default=str(ROOT / "artifacts" / "审计_出生动物_返回线_s0.jsonl"),
                    help="the birth animal's full-bank audit, one line per task")
    ap.add_argument("--out", default=None)
    ap.add_argument("--among", type=int, default=1,
                    help="how many task-runs count as 'was asked' (guards thin ledgers)")
    args = ap.parse_args(argv)

    tasks = tb.stage_tasks("full")
    depth = tb.ladder(tasks)
    meta = {t["name"]: t for t in tasks}

    rows = [r for r in read(args.ledger)
            if r.get("kind") == "candidate" and r.get("scaffold") == 0]
    animals = [r for r in rows if built(r) and r.get("stage") == "full"]
    if not animals:
        print("台账里没有 stage=full 的建成的动物，没得体检。")
        return 1

    birth = {}
    if Path(args.birth).exists():
        for row in read(args.birth):
            birth[row["task"]] = bool(row["passed"])

    asked, passed = {}, {}
    for row in animals:
        ran = row.get("ran_tasks") or list(tb.stage_tasks(row.get("stage") or "full"))
        for name in ran:
            asked[name] = asked.get(name, 0) + 1
        for name in row.get("passed_tasks") or []:
            passed[name] = passed.get(name, 0) + 1

    n = len(animals)
    lines = []
    lines.append("# 题目体检｜%d 只动物、%d 道题" % (n, len(tasks)))
    lines.append("")
    lines.append("- 台账：`%s`" % args.ledger)
    lines.append("- 每道题的通过率 = 过了的只数 ÷ 被问到的只数（全库档下分母都等于 %d）" % n)
    lines.append("")

    dead, free, flipped, cheap = [], [], [], []
    for task in tasks:
        name = task["name"]
        a, p = asked.get(name, 0), passed.get(name, 0)
        rate = (p / a) if a else 0.
        tag = []
        if a and p == 0:
            tag.append("全灭")
            dead.append(name)
        if a and p == a:
            tag.append("全过")
            free.append(name)
        if name in birth:
            if birth[name] and a and p == 0:
                tag.append("与出厂冲突")
                flipped.append(name)
            if (not birth[name]) and a and p == a:
                tag.append("与出厂冲突")
                flipped.append(name)
        if p == 0 and task["sim_seconds"] <= 2.0:
            tag.append("太便宜")
            cheap.append(name)
        lines.append("| %s | %d | %s | %s | %d | %s | %.1f | %s |" % (
            name, depth[name], task["group"], task.get("bar_text", "")[:0] or "-",
            p, "%.0f%%" % (100. * rate), task["sim_seconds"], "、".join(tag) or ""))

    head = "| 题 | 阶梯 | 方面 | 过了/问到 | 通过率 | 出厂 | 模拟秒 | 标记 |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    table = [head, sep]
    for task in tasks:
        name = task["name"]
        a, p = asked.get(name, 0), passed.get(name, 0)
        b = "-" if name not in birth else ("✅" if birth[name] else "❌")
        lines_ = []
        if a and p == 0:
            lines_.append("全灭")
        if a and p == a:
            lines_.append("全过")
        if name in birth and ((birth[name] and a and p == 0) or ((not birth[name]) and a and p == a)):
            lines_.append("与出厂冲突")
        table.append("| `%s` | %d | %s | %d / %d | %.0f%% | %s | %.1f | %s |" % (
            name, depth[name], task["group"], p, a, 100. * (p / a if a else 0.), b,
            task["sim_seconds"], "、".join(lines_) or ""))

    text = "\n".join(lines[:4]) + "\n" + "\n".join(table) + "\n"
    text += "\n## 汇总\n\n"
    text += "- 全灭（问到 %d 只、一只没过）：%d 道 %s\n" % (n, len(dead), "、".join("`%s`" % x for x in dead) or "（无）")
    text += "- 全过（一只不落）：%d 道 %s\n" % (len(free), "、".join("`%s`" % x for x in free) or "（无）")
    text += "- 与出厂动物冲突：%d 道 %s\n" % (len(flipped), "、".join("`%s`" % x for x in flipped) or "（无）")
    text += "- 又没过又几乎不花时间（设计上白花钱）：%d 道 %s\n" % (len(cheap), "、".join("`%s`" % x for x in cheap) or "（无）")
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())