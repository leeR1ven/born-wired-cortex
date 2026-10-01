# -*- coding: utf-8 -*-
"""一批候选一起考：谁真的会追红球。

考题把球摆三个地方各跑一遍 —— 狗的左前方、正前方、右前方。会追的狗在左边就往左拐、
在右边就往右拐，正前方就直接走过去。**只会原地转圈**的狗只会在某一侧碰巧撞上，另一侧
过不了，所以「左右两边都得过」这条卡死了那种假货。

每个候选项 = 一套基因（冠军那套，只改转向相关的几个数）+ 一种追球接法（往左拐接哪几个
转向细胞、往右拐接哪几个、增益多大）。另外带上「同一套基因但不接追球」的对照，用来确认
分数不是基因自己带来的。

    python tools/hunt_chase.py --out artifacts/追球_候选.json
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import chase_red_ball as C                                           # noqa: E402

PLACES = (("左前", .60), ("正前", .00), ("右前", -.60))
# 就一个来源：眼睛位置排。转向的自己一对新细胞（chase_turn_left / chase_turn_right），
# 平时全灭，不借 wall_turn 那条已经被别的反射占着的路。
KNOBS = ("turn_gain", "gain")


def one(job):
    """一个候选项：建一次狗，三个球位各跑一遍。"""
    genome = dict(C.load_champion()["genome"])
    for name, value in job["genome"].items():
        genome[name] = float(value)
    chase = None
    if job.get("chase"):
        chase = dict(job["chase"])
    started = time.perf_counter()
    rows = []
    for label, bearing in PLACES:
        # 每一趟都新建一只狗。建一次连跑三趟的话，上一趟的电压、适应、眼睛角度会带到
        # 下一趟去（实测：左前那趟走完 8.8 米之后，正前那趟一步都不走，只有 0.05 米）。
        body, brain, eyes, geom = C.build(genome, chase, seed=job.get("seed", 0))
        got = C.measure(body, brain, eyes, geom, job["seconds"], bearing, 1.5, seed=job.get("seed", 0))
        eyes.close()
        tail = got["headings"][-int(1./C.DT):]
        rows.append(dict(place=label, bearing=bearing,
                         heading_end=float(np.mean(tail)), heading_last=got["heading_end"],
                         range0=got["range0"], range_end=got["range_end"], range_min=got["range_min"],
                         settled=float(np.mean(got["ranges"][-int(1./C.DT):])),
                         travelled_m=got["travelled_m"], straightness=got["straightness"],
                         lowest_up_z=got["lowest_up_z"]))
    fell = any(row["lowest_up_z"] < .3 for row in rows)
    reached = [row["range_min"] < .9 for row in rows]
    faces = [abs(row["heading_end"]) < .45 for row in rows]
    passed = bool(all(reached) and all(faces) and not fell)
    return dict(name=job["name"], genome=job["genome"], chase=job.get("chase"), rows=rows,
                fell=fell, passed=passed,
                score=float(np.mean([row["settled"] for row in rows])),
                seconds=time.perf_counter() - started)


def jobs_for(turns, gains, seconds, seed, with_control=True, flips=(False, True)):
    out = []
    for turn in turns:
        for gain in gains:
            for flip in flips:
                name = ("拧髋 %.2f 弧度 / 位置排增益 %.1f%s"
                        % (turn, gain, " / 左右对调" if flip else ""))
                out.append(dict(name=name, seconds=seconds, seed=seed, genome={},
                                chase={"turn_gain": turn, "gain": gain, "flip": flip,
                                       "source": "position"}))
        if with_control:
            out.append(dict(name="对照：拧髋 %.2f 但不接追球" % turn, seconds=seconds, seed=seed,
                            genome={}, chase=None))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=6.)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--turns", default="0.05,0.1,0.2,0.4", help="拧髋的力，弧度，逗号分隔")
    ap.add_argument("--gains", default="1", help="位置排到转向细胞的总增益，逗号分隔")
    ap.add_argument("--no-flip", action="store_true", help="只试一个方向，不对调")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "追球_候选.json"))
    args = ap.parse_args(argv)

    turns = [float(x) for x in args.turns.split(",") if x]
    gains = [float(x) for x in args.gains.split(",") if x]
    flips = (False,) if args.no_flip else (False, True)
    jobs = jobs_for(turns, gains, args.seconds, args.seed, not args.no_control, flips)
    print("候选 %d 个 x 3 个球位，%d 个进程" % (len(jobs), args.workers), flush=True)
    started = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = []
        for done, got in enumerate(pool.imap_unordered(one, jobs), 1):
            results.append(got)
            print("[%d/%d] %-46s 追球分 %6.2f  过=%s %s"
                  % (done, len(jobs), got["name"], got["score"], got["passed"],
                     "摔了" if got["fell"] else ""), flush=True)
    results.sort(key=lambda row: (not row["passed"], row["score"]))
    print("\n=== 排名（先看「过」的）===", flush=True)
    print("%-46s %7s %5s %6s | %s" % ("候选", "追球分", "过", "最低直立", "三个球位最后离球多远(米)"))
    for row in results:
        print("%-46s %7.2f %5s %6.3f | %s"
              % (row["name"], row["score"], row["passed"],
                 min(r["lowest_up_z"] for r in row["rows"]),
                 "  ".join("%s %.2f" % (r["place"], r["range_end"]) for r in row["rows"])))
    print("\n一共花了 %.1f 秒" % (time.perf_counter() - started), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(seconds=args.seconds, seed=args.seed, results=results),
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())