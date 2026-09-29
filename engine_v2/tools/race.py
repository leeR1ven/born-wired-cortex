# -*- coding: utf-8 -*-
"""第一题：在会走的模型上随机生成更多，挑出「一直走、不摔、也不停」的。

    python tools/race.py draw --parents artifacts/轮6_全库24.jsonl \
        --candidates 40 --workers 8 --seconds 30 --out artifacts/题1_一直走
    python tools/race.py report --ledger artifacts/题1_一直走.jsonl --keep 8 \
        --write-keep artifacts/题1_一直走_优秀.jsonl

这不是「过 / 不过」，是**打分排序**：同一段路、同样秒数，读它自己走出来的数。
重点是它有没有**停下来**，所以每一段十秒各记一次路程：起点那段和末段一比，
末段几乎不走了就是「停了」。判定分三组：

  一直走   全程直立，并且末段还在走（这是要留的那一组，按平均速度排）
  停过     全程直立，但末段基本不动了（现在的出厂动物就在这一组）
  倒过     中途直立程度掉下去过

父代要求已经通过哪几道题、动几个基因、动多大，都由命令行说清；
变异用的还是筛选器那一套 ``mutate``，所以这里的候选和筛选轮里的候选是同一种
动物。速度、路程、分段路程都是它自己走出来的读数，没有哪一项是我替它定的。
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402

from tools import screen_candidates as sc                               # noqa: E402
from tools import taskbank as tb                                        # noqa: E402

WALK_NEED = "walk_flat,keeps_walking_without_being_told"
WINDOW = 10.        # 每段十秒
STOP_M = 0.10       # 末段十秒走不到这个数，就算「停了」


def read_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append(row, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parents_of(path, need, scaffold=0, limit=None):
    """Genomes of the animals in ``path`` that already pass every task in ``need``."""
    need = {name for name in (need or []) if name}
    pool = []
    for row in read_jsonl(path):
        if row.get("kind") != "candidate" or row.get("scaffold") != scaffold:
            continue
        if not sc.built(row):
            continue
        if need - set(row.get("passed_tasks") or []):
            continue
        pool.append(row)
    pool.sort(key=lambda row: -row.get("n_passed", 0))
    return pool[:limit] if limit else pool


def per_window(timeline, seconds):
    """How far it got in each ten-second stretch, off its own position samples."""
    windows = [0.] * max(1, int(math.ceil(seconds / WINDOW)))
    for before, after in zip(timeline, timeline[1:]):
        step = math.hypot(after["position"][0] - before["position"][0],
                          after["position"][1] - before["position"][1])
        index = min(len(windows) - 1, int(after["time"] // WINDOW))
        windows[index] += float(step)
    return windows


def run_one(job):
    """One candidate: build it from its genome and see how long it keeps walking."""
    index, seed, genome, seconds, scenario, terrain = job
    started = time.perf_counter()
    row = dict(kind="race", index=index, seed=int(seed), genome=genome,
               seconds=float(seconds), scenario=scenario, terrain=terrain,
               revision=tb.revision())
    try:
        ctx = tb.context(genome, seed)
        result = tb.run_seed(seed=ctx["seed"], duration=float(seconds),
                             model_path=ctx["model_path"], scenario=scenario,
                             terrain_start=terrain,
                             controller_parameters_for_seed=ctx["parameters"])
    except Exception as exc:                        # a refused genome is not an animal
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc))
        row["wall_seconds"] = time.perf_counter() - started
        return row
    metrics = result["metrics"]
    timeline = result["timeline"]
    windows = per_window(timeline, seconds)
    last = windows[-1]
    row.update(
        status=result["status"], error=result["error"],
        upright=bool(result["behavior_pass"]),
        displacement_m=metrics["total_displacement_m"],
        path_m=metrics["horizontal_path_m"],
        mean_speed_mps=metrics["mean_speed_mps"],
        max_speed_mps=metrics["max_speed_mps"],
        min_up_z=metrics["minimum_up_z"],
        straightness=metrics["straightness"],
        lateral_m=metrics["lateral_m"],
        yaw_path_rad=metrics["yaw_path_rad"],
        eye_travel_rad=metrics.get("eye_travel_rad"),
        windows_m=[round(float(value), 4) for value in windows],
        last_window_m=float(last),
        kept_going=bool(result["behavior_pass"] and last >= STOP_M),
        stalled=bool(last < STOP_M),
        wall_seconds=time.perf_counter() - started,
    )
    return row


def draw(args):
    pool = parents_of(args.parents, args.need.split(",") if args.need else [],
                      scaffold=args.scaffold, limit=args.parents_limit)
    if not pool:
        print("台账 %s 里没有通过 %s 的父代，看看要不要放宽 --need" % (args.parents, args.need))
        return 1
    print("父代 %d 只，最会的那只过了 %d/%d 道；要求父代已通过 %s"
          % (len(pool), pool[0]["n_passed"], pool[0]["n_tasks"], args.need))
    ledger = Path("%s.jsonl" % args.out)
    done = sum(1 for row in read_jsonl(ledger) if row.get("kind") == "race")
    if done and args.fresh:
        Path(ledger).unlink()
        done = 0
    elif done:
        print("这个台账里已经有 %d 只跑过了，接着往下排（--fresh 可以重开）" % done)
    rng = np.random.default_rng(args.seed)
    jobs = []
    for step in range(done + args.candidates):
        parent = pool[int(rng.integers(len(pool)))]
        genome, _touched = sc.mutate(parent["genome"], args.scaffold, rng,
                                     genes=args.genes, sigma=args.sigma)
        if step < done:
            continue
        jobs.append((step, int(rng.integers(1000000)), genome, args.seconds,
                     args.scenario, args.terrain))
    print("要跑 %d 只，每只 %g 秒、%d 个进程；结果写进 %s"
          % (len(jobs), args.seconds, args.workers, ledger))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        for row in pool_exec.map(run_one, jobs):
            append(row, ledger)
            if row["status"] == "ok":
                print("%4d  %7.3f 米  %6.3f 米/秒  直立最低 %.3f  %s  分段 %s  %.0f 秒"
                      % (row["index"], row["displacement_m"], row["mean_speed_mps"],
                         row["min_up_z"],
                         "一直走" if row["kept_going"] else ("停过" if row["upright"] else "倒过"),
                         row["windows_m"], row["wall_seconds"]), flush=True)
            else:
                print("%4d  没建出来：%s" % (row["index"], row.get("error")), flush=True)
    print("这批 %d 只跑了 %.1f 分钟" % (len(jobs), (time.perf_counter() - started) / 60.))
    return 0


def report(args):
    rows = [row for row in read_jsonl(args.ledger) if row.get("kind") == "race"]
    ok = [row for row in rows if row.get("status") == "ok"]
    refused = [row for row in rows if row.get("status") != "ok"]
    if not ok:
        print("台账里没有跑成的动物（%d 条，%d 条没建出来）" % (len(rows), len(refused)))
        return 1
    going = [row for row in ok if row["kept_going"]]
    stalled = [row for row in ok if row["upright"] and row["stalled"]]
    fell = [row for row in ok if not row["upright"]]
    for group in (going, stalled, fell):
        group.sort(key=lambda row: -float(row["mean_speed_mps"] or 0.))
    print("跑成的 %d 只：一直走 %d 只，走一会儿就停 %d 只，倒过 %d 只（另 %d 条基因被引擎拒绝）"
          % (len(ok), len(going), len(stalled), len(fell), len(refused)))
    header = "%4s %9s %10s %8s %8s %8s  %s" % ("#", "位移(米)", "平均(米/秒)", "最快", "直度", "眼球转(弧度)", "每十秒走了多少")
    for title, group in (("一直走（要留的这一组，按平均速度排）", going),
                         ("走一会儿就停", stalled),
                         ("倒过", fell)):
        print("\n%s：%d 只" % (title, len(group)))
        if not group:
            continue
        print(header)
        for row in group[:args.top]:
            print("%4d %9.3f %10.3f %8.3f %8.3f %8.1f  %s" % (
                row["index"], row["displacement_m"], row["mean_speed_mps"],
                row["max_speed_mps"], row["straightness"],
                row["eye_travel_rad"] or 0., row["windows_m"]))
    # Hard bars can leave a group empty, and an empty group says nothing about
    # which animal stalls last.  Rank the upright ones by how far they still
    # got in the final stretch, so the answer to "does any of them keep going"
    # is a number even when the bar catches none of them.
    upright = [row for row in ok if row["upright"]]
    upright.sort(key=lambda row: (-float(row["last_window_m"]), -float(row["mean_speed_mps"] or 0.)))
    print("\n全程直立、按末段（最后十秒）还在走多远排：%d 只" % len(upright))
    print("%4s %12s %10s %10s  %s" % ("#", "末段(米)", "全程(米)", "平均(米/秒)", "每十秒走了多少"))
    for row in upright[:args.top]:
        print("%4d %12.3f %10.3f %10.3f  %s" % (
            row["index"], row["last_window_m"], row["displacement_m"],
            row["mean_speed_mps"], row["windows_m"]))

    if args.write_keep:
        # "Keeps going" is not the same as "fast": the one that still has
        # something left in the final ten seconds is the one that stalls last.
        # Both orders are worth having, so the caller says which one to write.
        if args.keep_by == "last":
            keep = sorted(going, key=lambda row: (-float(row["last_window_m"]),
                                                  -float(row["mean_speed_mps"] or 0.)))[:args.keep]
        else:
            keep = going[:args.keep]
        Path(args.write_keep).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keep),
            encoding="utf-8", newline="\n")
        print("\n一直走的那 %d 只（按「%s」挑）写进了 %s"
              % (len(keep), "末段还走得最远" if args.keep_by == "last" else "平均速度最快",
                 args.write_keep))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("draw", help="draw candidates and see how long they keep walking")
    d.add_argument("--parents", required=True, help="台账：父代从这里挑")
    d.add_argument("--need", default=WALK_NEED, help="父代必须已经通过这几道题（逗号分隔）")
    d.add_argument("--parents-limit", type=int, default=None)
    d.add_argument("--candidates", type=int, default=40)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--genes", type=int, default=sc.MUTATION_GENES)
    d.add_argument("--sigma", type=float, default=sc.MUTATION_SIGMA)
    d.add_argument("--scaffold", type=int, default=0)
    d.add_argument("--seconds", type=float, default=30.)
    d.add_argument("--scenario", default="autonomous")
    d.add_argument("--terrain", default="origin")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", required=True)
    d.add_argument("--fresh", action="store_true", help="不续跑，重开一个台账")

    r = sub.add_parser("report", help="分组排序")
    r.add_argument("--ledger", required=True)
    r.add_argument("--top", type=int, default=15)
    r.add_argument("--keep", type=int, default=8)
    r.add_argument("--write-keep", default=None)
    r.add_argument("--keep-by", choices=("last", "speed"), default="last",
                   help="留哪一批：last = 末段还在走得最远的；speed = 平均速度最快的")
    args = ap.parse_args(argv)
    return draw(args) if args.command == "draw" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())