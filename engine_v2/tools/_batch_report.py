# -*- coding: utf-8 -*-
"""把云端这批训练的成绩单打出来（临时脚本）。

    python tools/_batch_report.py                      # 看本地拉回来的那一份
    python tools/_batch_report.py --dir <目录>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts" / "云端" / "二批"


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(DEFAULT))
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)
    folder = Path(args.dir)
    if not folder.exists():
        print("这个目录还没有：%s" % folder)
        return 1

    logs = sorted(folder.glob("*二批_*.log")) + sorted(folder.glob("watchdog.log"))
    if logs:
        print("==== 云端日志（收工 / 复试） ====")
        for log in logs:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if ("收工" in line or "复试" in line or "看门狗" in line
                        or "关机" in line or "跑完" in line):
                    print("  " + line.strip())
        print()

    keeps = sorted(folder.glob("*_keep.jsonl"))
    families = []
    for path in keeps:
        for row in read_jsonl(path):
            verify = row.get("verify") or {}
            row["_file"] = path.name
            row["_hits"] = int(verify.get("hits", -1))
            row["_looks"] = int(verify.get("looks", -1))
            row["_shift"] = float(verify.get("shifted_m", float("nan")) or float("nan"))
            families.append(row)
    if not families:
        print("还没有 keep 文件（可能训练还在跑）")
        return 0
    families.sort(key=lambda r: (-r["_hits"], -r["covered"], r["score"]))
    print("==== 谁最会追球（按复试「18 个没见过的新位置里贴到球几个」排） ====")
    print("%-9s %-6s %-8s %-8s %-8s %-7s %-9s %s"
          % ("第几只", "批次", "复试贴到", "看球位置", "考试趟数", "最近", "走了", "追球接法"))
    for row in families[:args.top]:
        chase = row.get("chase") or {}
        print("%-9d %-6s %-8s %-8s %-8d %-7.2f %-9.2f %s%s"
              % (row["index"], row["_file"].split("_g")[1][:2] if "_g" in row["_file"] else "?",
                 "%d/18" % row["_hits"] if row["_hits"] >= 0 else "没复试",
                 "%d/18" % row["_looks"] if row["_looks"] >= 0 else "-",
                 row["covered"], row["score"], row["travelled_m"],
                 chase.get("route", "?"), " 对调" if chase.get("flip") else ""))
    print()
    print("==== 每一批的爹（挑出来继续繁殖的） ====")
    for path in keeps:
        rows = read_jsonl(path)
        print("  %s：%s" % (path.name, "、".join(
            "第%d只(贴到%s/18、考试%d/4趟)" % (r["index"],
                                        (r.get("verify") or {}).get("hits", "?"),
                                        r["covered"]) for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())