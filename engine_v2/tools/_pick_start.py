# -*- coding: utf-8 -*-
"""从上一批台账里挑出值得当爹的模型，拼成新起点（临时脚本）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evolve_chase import rank                                    # noqa: E402


def read(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def by_index(path, wanted):
    got = {}
    for row in read(path):
        if row.get("index") in wanted and row.get("genome") and row.get("chase"):
            got[row["index"]] = row
    return got


def main():
    art = ROOT / "artifacts"
    fresh = [r for r in read(art / "追球_云_复试_g00_旧版.jsonl")
             if r.get("status") == "ok" and r.get("genome") and r.get("chase")]
    fresh.sort(key=rank)
    best = fresh[:40]
    walkers = [r for r in fresh if r["upright"] and r["moving"] >= 3][:20]
    old = by_index(art / "追球_云_融合_g00_第一批.jsonl", {118, 172, 13, 70, 267, 213, 149, 88})
    picked, seen = [], set()
    for row in best + walkers + list(old.values()):
        key = (int(row["seed"]), int(row["index"]))
        if key in seen or not row.get("upright"):
            continue
        seen.add(key)
        picked.append(row)
    out = art / "追球_云_起点2.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked),
                   encoding="utf-8", newline="\n")
    print("新起点 %d 只 -> %s" % (len(picked), out))
    for row in picked[:20]:
        print("  第%d只 转%d/%d 两边%s 最近%.2f米 走%.2f米 路%s%s"
              % (row["index"], row["covered"], row["trials"],
                 "会" if row["both"] else "不会", row["score"], row["travelled_m"],
                 (row.get("chase") or {}).get("route", "?"),
                 " 对调" if (row.get("chase") or {}).get("flip") else ""))


if __name__ == "__main__":
    main()