# -*- coding: utf-8 -*-
"""第一题：给每只模型固定十秒，看它走多远、倒没倒，然后按速度挑前几批。

    python tools/race.py draw --parents artifacts/题1_一直走.jsonl \
        --candidates 256 --window 10 --workers 8 --field open --out artifacts/题1d_十秒赛
    python tools/race.py report --ledger artifacts/题1d_十秒赛.jsonl --keep 32 \
        --write-keep artifacts/题1d_最快32只.jsonl

默认是**固定窗口档**：每只只跑 `--window` 秒（默认十秒），时间到就结束，读它这
十秒里走了多远、有没有倒。不倒的按走得远排，走两步就倒的单独列出来。这样每只的
代价是固定的、可以算得出来，几万只也排得起队。

要看距离目标（走够多少米才结束）就用 `--goal`，那时 `--cap/--gate/--stall` 才
生效。场地 `--field open` 把摆设和四堵墙都挪到六十米外，只剩无限地板，所以读数
只反映走得好不好，不掺撞东西。
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
FAR = (60., 60., -8.)
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")


def field_props(name):
    """What the arena holds for this run.

    ``furnished`` is the room as it ships: props and walls 3.5 m away, so
    nothing can walk more than a few metres without meeting something.
    ``clean`` moves the props away and leaves the walls.  ``open`` moves the
    walls away too: the floor is an infinite plane, so the reading is about the
    walk rather than about the furniture.
    """
    if name == "furnished":
        return None
    props = {geom: FAR for geom in tb.SCENERY}
    if name == "open":
        props.update({geom: FAR for geom in WALLS})
    return props


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
    """The animals in ``path`` that qualify as parents.

    A full-bank ledger is read as "it passed every task in ``need``".  A race
    ledger has no task list on its rows - it has the walk it just did - so a row
    there qualifies by having stayed upright.
    """
    need = {name for name in (need or []) if name}
    pool = []
    for row in read_jsonl(path):
        if row.get("scaffold", scaffold) != scaffold:
            continue
        if row.get("kind") == "race":
            if row.get("status") == "ok" and row.get("upright"):
                pool.append(row)
            continue
        if row.get("kind") != "candidate" or not sc.built(row):
            continue
        if need - set(row.get("passed_tasks") or []):
            continue
        pool.append(row)
    # Older race ledgers call the same reading ``displacement_m``: a parent pool
    # is worth reading either way rather than sorting every one of them at zero.
    pool.sort(key=lambda row: -float(row.get("travelled_m")
                                     or row.get("displacement_m")
                                     or row.get("n_passed") or 0.))
    return pool[:limit] if limit else pool


def run_one(job):
    """One candidate: give it its window and read how far it got."""
    started = time.perf_counter()
    row = dict(kind="race", index=job["index"], seed=int(job["seed"]), genome=job["genome"],
               mode=job["mode"], window_seconds=float(job["window"]),
               goal_m=None if job["goal"] is None else float(job["goal"]),
               cap_seconds=float(job["cap"]), gate=job["gate"],
               stall_seconds=job["stall"], field=job["field"], revision=tb.revision())
    try:
        ctx = tb.context(job["genome"], job["seed"])
        result = tb.run_seed(seed=ctx["seed"], duration=float(job["cap"]),
                             model_path=ctx["model_path"], scenario=job["scenario"],
                             terrain_start=job["terrain"], props=field_props(job["field"]),
                             distance_goal_m=(None if job["goal"] is None else float(job["goal"])),
                             pace_gate=job["gate"], stall_seconds=job["stall"],
                             controller_parameters_for_seed=ctx["parameters"])
    except Exception as exc:                        # a refused genome is not an animal
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc))
        row["wall_seconds"] = time.perf_counter() - started
        return row
    metrics = result["metrics"]
    ended = result.get("ended") or {}
    upright = bool(result["behavior_pass"])
    reached = bool(upright and ended.get("reached_goal"))
    seconds = float(ended.get("elapsed_s") or job["cap"])
    travelled = float(ended.get("travelled_m") or metrics["total_displacement_m"])
    if not upright:
        verdict = "倒过"
    elif reached:
        verdict = "到目标"
    elif job["mode"] == "window":
        verdict = "走满窗口"
    elif ended.get("too_slow"):
        verdict = "太慢"
    elif ended.get("stalled"):
        verdict = "停了"
    else:
        verdict = "时间用尽"
    row.update(
        status=result["status"], error=result["error"], upright=upright, reached=reached,
        verdict=verdict, sim_seconds=seconds, travelled_m=travelled,
        speed_mps=(travelled / seconds if seconds else None),
        displacement_m=metrics["total_displacement_m"],
        path_m=metrics["horizontal_path_m"],
        mean_speed_mps=metrics["mean_speed_mps"],
        max_speed_mps=metrics["max_speed_mps"],
        min_up_z=metrics["minimum_up_z"],
        straightness=metrics["straightness"],
        lateral_m=metrics["lateral_m"],
        yaw_path_rad=metrics["yaw_path_rad"],
        eye_travel_rad=metrics.get("eye_travel_rad"),
        wall_seconds=time.perf_counter() - started,
    )
    return row


def draw(args):
    pool = parents_of(args.parents, args.need.split(",") if args.need else [],
                      scaffold=args.scaffold, limit=args.parents_limit)
    if not pool:
        print("台账 %s 里没有合格的父代" % args.parents)
        return 1
    if pool[0].get("kind") == "race":
        print("父代 %d 只，来自筛选台账（走得最远那只 %.3f 米）"
              % (len(pool), float(pool[0].get("travelled_m")
                                  or pool[0].get("displacement_m") or 0.)))
    else:
        print("父代 %d 只，最会的那只过了 %d/%d 道；要求已通过 %s"
              % (len(pool), pool[0]["n_passed"], pool[0]["n_tasks"], args.need))
    if args.goal is None:
        mode, cap = "window", float(args.window)
        gate, stall = None, None
    else:
        mode = "distance"
        cap = float(args.cap or 420.)
        gate = tuple(float(part) for part in args.gate.split(",")) if args.gate else None
        stall = args.stall
    ledger = Path("%s.jsonl" % args.out)
    done = sum(1 for row in read_jsonl(ledger) if row.get("kind") == "race")
    if done and args.fresh:
        Path(ledger).unlink()
        done = 0
    elif done:
        print("这个台账里已经有 %d 只跑过了，接着往下排（--fresh 重开）" % done)
    rng = np.random.default_rng(args.seed)
    jobs = []
    for step in range(done + args.candidates):
        parent = pool[int(rng.integers(len(pool)))]
        genome, _touched = sc.mutate(parent["genome"], args.scaffold, rng,
                                     genes=args.genes, sigma=args.sigma)
        seed = int(rng.integers(1000000))
        if step < done:
            continue
        jobs.append(dict(index=step, seed=seed, genome=genome, mode=mode,
                         window=args.window, goal=args.goal, cap=cap, gate=gate,
                         stall=stall, scenario=args.scenario, terrain=args.terrain,
                         field=args.field))
    if mode == "window":
        print("要跑 %d 只：每只固定 %g 秒、%s 场地、%d 个进程；写进 %s"
              % (len(jobs), cap, args.field, args.workers, ledger))
    else:
        print("要跑 %d 只：目标 %g 米、最多 %g 秒、%s 场地、%d 个进程；写进 %s"
              % (len(jobs), args.goal, cap, args.field, args.workers, ledger))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        for row in pool_exec.map(run_one, jobs):
            append(row, ledger)
            if row["status"] == "ok":
                print("%5d  %-8s %6.3f 米 / %5.1f 秒  %5.3f 米/秒  直立最低 %.3f  本机 %.0f 秒"
                      % (row["index"], row["verdict"], row["travelled_m"], row["sim_seconds"],
                         row["speed_mps"] or 0., row["min_up_z"], row["wall_seconds"]), flush=True)
            else:
                print("%5d  没建出来：%s" % (row["index"], row.get("error")), flush=True)
    print("这批 %d 只共 %.1f 分钟" % (len(jobs), (time.perf_counter() - started) / 60.))
    return 0


def report(args):
    rows = [row for row in read_jsonl(args.ledger) if row.get("kind") == "race"]
    ok = [row for row in rows if row.get("status") == "ok"]
    refused = [row for row in rows if row.get("status") != "ok"]
    if not ok:
        print("台账里没有跑成的动物（%d 条，%d 条没建出来）" % (len(rows), len(refused)))
        return 1
    upright = sorted([row for row in ok if row["upright"]],
                     key=lambda row: -float(row["travelled_m"] or 0.))
    down = sorted([row for row in ok if not row["upright"]],
                  key=lambda row: -float(row["travelled_m"] or 0.))
    window = ok[0].get("window_seconds")
    print("跑成的 %d 只：全程没倒 %d 只，倒过 %d 只（另 %d 条基因被引擎拒绝）"
          % (len(ok), len(upright), len(down), len(refused)))
    header = "%5s %9s %9s %8s %8s %9s %9s" % (
        "#", "走了(米)", "速度(米/秒)", "全程路子", "直度", "眼球转", "本机(秒)")
    print("\n全程没倒，按 %g 秒走了多远排（前 %d 只）：" % (window or 0., args.top))
    print(header)
    for row in upright[:args.top]:
        print("%5d %9.3f %9.3f %8.3f %8.3f %9.1f %9.0f" % (
            row["index"], row["travelled_m"], row["speed_mps"] or 0., row["path_m"],
            row["straightness"], row["eye_travel_rad"] or 0., row["wall_seconds"]))
    if down:
        print("\n倒过的（%d 只，倒之前走了这么多）：" % len(down))
        print(header)
        for row in down[:min(args.top, 10)]:
            print("%5d %9.3f %9.3f %8.3f %8.3f %9.1f %9.0f" % (
                row["index"], row["travelled_m"], row["speed_mps"] or 0., row["path_m"],
                row["straightness"], row["eye_travel_rad"] or 0., row["wall_seconds"]))
    if args.write_keep:
        keep = upright[:args.keep]
        Path(args.write_keep).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keep),
            encoding="utf-8", newline="\n")
        print("\n最快的 %d 只（全程没倒的里面挑）写进了 %s" % (len(keep), args.write_keep))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("draw", help="draw candidates and race them a fixed window")
    d.add_argument("--parents", required=True, help="台账：父代从这里挑")
    d.add_argument("--need", default=WALK_NEED, help="用全库台账当父代时，要求它已通过这几道题")
    d.add_argument("--parents-limit", type=int, default=None)
    d.add_argument("--candidates", type=int, default=256)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--genes", type=int, default=sc.MUTATION_GENES)
    d.add_argument("--sigma", type=float, default=sc.MUTATION_SIGMA)
    d.add_argument("--scaffold", type=int, default=0)
    d.add_argument("--window", type=float, default=10., help="固定窗口档：每只只跑这么多秒")
    d.add_argument("--goal", type=float, default=None, help="距离档：走够这么多米就结束（给了它才用距离档）")
    d.add_argument("--cap", type=float, default=None, help="距离档最多跑多少秒")
    d.add_argument("--gate", default=None, help="距离档太慢门槛：「秒,米」")
    d.add_argument("--stall", type=float, default=None, help="距离档：多少秒没挪 5 厘米算停了")
    d.add_argument("--scenario", default="autonomous")
    d.add_argument("--terrain", default="origin")
    d.add_argument("--field", choices=("furnished", "clean", "open"), default="open",
                   help="场地：furnished 原样；clean 挪走摆设；open 连墙都挪走，只剩无限地板")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", required=True)
    d.add_argument("--fresh", action="store_true", help="不续跑，重开一个台账")

    r = sub.add_parser("report", help="按速度分组排序")
    r.add_argument("--ledger", required=True)
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--keep", type=int, default=32)
    r.add_argument("--write-keep", default=None)
    args = ap.parse_args(argv)
    return draw(args) if args.command == "draw" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())