# -*- coding: utf-8 -*-
"""复核「会朝球走」的候选：换没用过的位置，而且**每个位置**都配一趟「球搬走」的对照。

考试里见过的是 0.5 弧度 x 3.0/4.0 米。这里换成 -0.7~0.7 弧度 x 2.5/3.5/4.5 米，
全是它没见过的位置。

用户 2026-09-30：只换位置不够 —— 万一某只只对球摆在某个地方有反应呢？所以每个位置
都跑两趟：一趟球摆在那儿、一趟球搬走（同一个种子、别的什么都不动）。两趟落点差不到
0.05 米，说明那个位置上它根本没在看球；那一趟的「贴到球」不算数，只能算碰巧路过。

    python tools/verify_chase.py --ledger artifacts/追球_云_复试_g00.jsonl --top 6
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import chase_red_ball as C                                          # noqa: E402

REACH_BAR = 1.0
LOOK_BAR = .05


def numbers(text):
    return [float(part) for part in str(text).replace(" ", "").split(",") if part]


def pick(rows, top):
    ok = [row for row in rows if row.get("status") == "ok" and row.get("both")]
    ok.sort(key=lambda row: (-row["covered"], -row["moving"], row["score"]))
    rest = [row for row in rows if row.get("status") == "ok" and not row.get("both")]
    rest.sort(key=lambda row: (-row["covered"], -row["moving"], row["score"]))
    seen, out = set(), []
    for row in ok + rest:
        mark = json.dumps(row["chase"], sort_keys=True) + str(row["seed"])
        if mark in seen:
            continue
        seen.add(mark)
        out.append(row)
        if len(out) >= top:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=6.)
    ap.add_argument("--bearings", default="-0.7,-0.5,-0.3,0.3,0.5,0.7")
    ap.add_argument("--distances", default="2.5,3.5,4.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in
            Path(args.ledger).read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = pick(rows, args.top)
    if not chosen:
        print("这份台账里没有建出来的候选：%s" % args.ledger)
        return 1
    spec = C.gaze_spec()
    bearings = numbers(args.bearings)
    distances = numbers(args.distances)
    report = []
    for row in chosen:
        genome = dict(row["genome"])
        chase = dict(row["chase"])
        seed = int(row["seed"])
        print("\n==== 第 %d 只（考试：到了 %d/4 趟、最近 %.2f 米、两边都会 %s）"
              "  路 %s%s pivot=%s gain=%.2f turn_gain=%.3f 拐弯力 %.4f ===="
              % (row["index"], row["covered"], row["best"], row["both"],
                 chase["route"], " 对调" if chase["flip"] else "", chase.get("pivot", 8),
                 chase.get("gain", 1.), chase.get("turn_gain", 0.),
                 genome.get("avoidance_gain", float("nan"))))
        print("%8s %9s | %9s %9s %9s | %s"
              % ("球偏(弧度)", "距离(米)", "最近(米)", "球搬走差(米)", "走了(米)", "判定"))
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        # 球搬走之后每个位置的世界都一样，所以「没球」那一趟只跑一次（用户 2026-09-30）。
        C.restore(brain, saved)
        off = C.measure(body, brain, eyes, geom, args.seconds, bearings[0], distances[0],
                        seed=seed, ball=False)
        no_ball = np.asarray(off["end_xy"])
        results = []
        for distance in distances:
            for bearing in bearings:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, args.seconds, bearing, distance,
                               seed=seed)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"]) - no_ball))
                moving = bool(on["travelled_m"] >= 2.0 and on["lowest_up_z"] >= .3)
                close = float(on["range_min"]) <= REACH_BAR
                looks = bool(shifted >= LOOK_BAR)
                verdict = ("贴到球了（真在看）" if moving and close and looks else
                           "贴到球了（搬走球也一样=碰巧）" if close and moving else
                           "走起来了" if moving else "没怎么动")
                results.append(dict(bearing=bearing, distance=distance,
                                    closest=float(on["range_min"]), shifted_m=shifted,
                                    travelled_m=on["travelled_m"], moving=moving,
                                    close=close, looks_at_ball=looks,
                                    lowest_up_z=on["lowest_up_z"]))
                print("%8.2f %9.1f | %9.2f %9.2f %9.2f | %s"
                      % (bearing, distance, on["range_min"], shifted, on["travelled_m"],
                         verdict))
        eyes.close()
        real = sum(1 for r in results if r["close"] and r["moving"] and r["looks_at_ball"])
        looks = sum(1 for r in results if r["looks_at_ball"])
        moved = sum(1 for r in results if r["moving"])
        print("  -> %d 个新位置里：真的贴到球 %d 个；有反应的位置 %d 个；走起来 %d 个"
              % (len(results), real, looks, moved))
        report.append(dict(index=row["index"], seed=seed, chase=chase,
                           covered=row["covered"], best=row["best"], both=row["both"],
                           real=real, looks=looks, moved=moved, total=len(results),
                           rows=results))
    print("\n---- 汇总 ----")
    print("%6s %8s %12s %12s %s" % ("第几只", "考试趟数", "真贴到球", "有反应位置", "结论"))
    for item in report:
        share = item["real"]/item["total"]
        print("%6d %8d %9d/%d %10d/%-3d %s"
              % (item["index"], item["covered"], item["real"], item["total"],
                 item["looks"], item["total"],
                 "真会追球" if share >= .5 else
                 ("一半会追" if share >= .25 else
                  ("偶尔会追" if item["real"] else "不追"))))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())