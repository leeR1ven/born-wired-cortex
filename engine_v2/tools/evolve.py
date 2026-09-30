# -*- coding: utf-8 -*-
"""一代只生几百只、每只随机性大，测完马上挑爹生下一代：迭代快。

    python tools/evolve.py --generations 2 --candidates 300 --workers 8 \
        --out-prefix artifacts/题1f_演化 --seed 11

跟 race.py 的区别：race.py 是"一次生几千只、只生一代"；这里是"一代只生几百只、
每只随机性大（默认动六根线、每根动半格），生完立刻测试，按成绩挑出爹，马上生下一
代"。每一代的账单独存一份，另外存一份跨代冠军，日志里每一代一行成绩单。

挑爹的规矩：从这一代里挑「没摔的里面最快的」几只 + 「全场最快的」几只。两边都留，
是为了不让"会摔但快"的基因断种——只留没摔的，速度就永远长不出来。

断了下一次拿上一代的 keep 名单接着跑：

    python tools/evolve.py --parents artifacts/题1f_演化_g03_keep.jsonl --generations 5 \
        --out-prefix artifacts/题1f_演化 --start-gen 4
"""
import argparse
import json
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

FAR = (60., 60., -8.)
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")


def open_props():
    import tools.taskbank as tb
    props = {geom: FAR for geom in tb.SCENERY}
    props.update({name: FAR for name in WALLS})
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


def fingerprint(genome):
    return tuple(sorted((key, round(float(value), 6)) for key, value in genome.items()))


def rank(row):
    """先看有没有摔，再看了多远。"""
    return (0 if row["upright"] else 1, -float(row["travelled_m"] or 0.))


def run_one(job):
    import tools.taskbank as tb
    started = time.perf_counter()
    row = dict(kind="evolve", gen=job["gen"], index=job["index"], seed=job["seed"],
               genome=job["genome"], touched=job["touched"], window_seconds=job["window"],
               field=job["field"], revision=tb.revision())
    try:
        ctx = tb.context(job["genome"], job["seed"])
        result = tb.run_seed(seed=ctx["seed"], duration=float(job["window"]),
                             model_path=ctx["model_path"], scenario="autonomous",
                             terrain_start="origin", props=open_props(),
                             controller_parameters_for_seed=ctx["parameters"])
    except Exception as exc:
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc))
        row["wall_seconds"] = time.perf_counter() - started
        return row
    metrics = result["metrics"]
    row.update(status=result["status"], error=result["error"],
               upright=bool(result["behavior_pass"]),
               travelled_m=metrics["total_displacement_m"],
               path_m=metrics["horizontal_path_m"],
               min_up_z=metrics["minimum_up_z"],
               straightness=metrics["straightness"],
               wall_seconds=time.perf_counter() - started)
    return row


def pick_parents(rows, keep_stable, keep_fast):
    ok = [row for row in rows if row.get("status") == "ok" and row.get("genome")]
    stable = sorted([row for row in ok if row["upright"]], key=lambda row: -row["travelled_m"])[:keep_stable]
    fast = sorted(ok, key=lambda row: -row["travelled_m"])[:keep_fast]
    seen, out = set(), []
    for row in stable + fast:
        mark = fingerprint(row["genome"])
        if mark in seen:
            continue
        seen.add(mark)
        out.append(row)
    return out


def make_child(parents, rng, args):
    """单亲大变异，或者两个爹各出一半再大变异。"""
    if len(parents) >= 2 and rng.random() < args.cross_rate:
        first, second = (parents[int(v)] for v in rng.choice(len(parents), size=2, replace=False))
        genome = {}
        for key in set(first["genome"]) | set(second["genome"]):
            mine, theirs = first["genome"].get(key), second["genome"].get(key)
            if mine is None:
                genome[key] = theirs
                continue
            if theirs is None:
                genome[key] = mine
                continue
            roll = rng.random()
            genome[key] = mine if roll < 0.4 else (theirs if roll < 0.8 else 0.5 * (mine + theirs))
    else:
        genome = dict(parents[int(rng.integers(len(parents)))]["genome"])
    return sc.mutate(genome, 0, rng, genes=args.genes, sigma=args.sigma)


def generation_line(gen, rows, seconds):
    ok = [row for row in rows if row.get("status") == "ok"]
    upright = sorted([row for row in ok if row["upright"]], key=lambda row: -row["travelled_m"])
    if not ok:
        return "第 %d 代：%d 只一只都没建出来" % (gen, len(rows))
    pick = max(ok, key=lambda row: row["travelled_m"])
    mid = upright[len(upright) // 2]["travelled_m"] if upright else 0.
    return ("第 %d 代收工：%d 只里建出来 %d 只（没摔 %d、摔 %d）；没摔的中位 %.3f 米、"
            "最好 %.3f 米；全场最快 %.3f 米 %s；本代花了 %.1f 分钟"
            % (gen, len(rows), len(ok), len(upright), len(ok) - len(upright), mid,
               upright[0]["travelled_m"] if upright else 0., pick["travelled_m"],
               "没摔" if pick["upright"] else "摔了", seconds / 60.))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parents", default=str(ROOT / "artifacts" / "题1f_十秒赛5000.jsonl"),
                    help="头一代的爹从这份台账里挑")
    ap.add_argument("--candidates", type=int, default=300, help="一代生几只")
    ap.add_argument("--generations", type=int, default=2, help="一共生几代")
    ap.add_argument("--keep-stable", type=int, default=6, help="留几只没摔的当爹")
    ap.add_argument("--keep-fast", type=int, default=4, help="再留几只全场最快的当爹（哪怕会摔）")
    ap.add_argument("--genes", type=int, default=6, help="每只随机动几个数字")
    ap.add_argument("--sigma", type=float, default=.55, help="每个数字动多大（一格=它自己的量级）")
    ap.add_argument("--cross-rate", type=float, default=.5, help="多少比例的孩子是两个爹凑的")
    ap.add_argument("--window", type=float, default=10., help="每只跑多少秒")
    ap.add_argument("--field", choices=("furnished", "clean", "open"), default="open")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-gen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-prefix", default=str(ROOT / "artifacts" / "题1f_演化"))
    ap.add_argument("--quiet", action="store_true", help="不逐只打印，只在每代收工时打印一行")
    args = ap.parse_args(argv)

    parents = pick_parents(read_jsonl(args.parents), args.keep_stable, args.keep_fast)
    if not parents:
        print("这份台账里挑不出爹：%s" % args.parents)
        return 1
    print("头一代的爹 %d 只：%s" % (len(parents), "、".join(
        "#%s %.2f米%s" % (row.get("index"), row["travelled_m"], "" if row["upright"] else "(摔)")
        for row in parents)), flush=True)

    rng = np.random.default_rng(args.seed)
    champion = None
    with ProcessPoolExecutor(args.workers) as pool:
        for gen in range(args.start_gen, args.start_gen + args.generations):
            jobs = []
            for index in range(args.candidates):
                genome, touched = make_child(parents, rng, args)
                jobs.append(dict(gen=gen, index=index, seed=int(rng.integers(1000000)),
                                 genome=genome, touched=sorted(touched), window=args.window,
                                 field=args.field))
            ledger = "%s_g%02d.jsonl" % (args.out_prefix, gen)
            print("第 %d 代：生 %d 只、每只 %.0f 秒、%d 个进程" % (gen, len(jobs), args.window, args.workers),
                  flush=True)
            started = time.perf_counter()
            rows = []
            for row in pool.map(run_one, jobs):
                rows.append(row)
                append(row, ledger)
                if row["status"] == "ok" and (champion is None or rank(row) < rank(champion)):
                    champion = row
                    print("  跨代冠军易主：%d 代第 %d 只  %.3f 米  %s" % (
                        row["gen"], row["index"], row["travelled_m"],
                        "没摔" if row["upright"] else "摔了"), flush=True)
                if not args.quiet and row["status"] == "ok":
                    print("%5d  %-6s %6.3f 米  直立最低 %6.3f  直度 %.3f  本机 %.0f 秒" % (
                        row["index"], "没摔" if row["upright"] else "倒了", row["travelled_m"],
                        row["min_up_z"], row["straightness"], row["wall_seconds"]), flush=True)
            print(generation_line(gen, rows, time.perf_counter() - started), flush=True)
            parents = pick_parents(rows, args.keep_stable, args.keep_fast)
            keep = "%s_g%02d_keep.jsonl" % (args.out_prefix, gen)
            Path(keep).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in parents),
                                  encoding="utf-8", newline="\n")
            print("  下一代的爹 %d 只（%s）：%s" % (len(parents), keep, "、".join(
                "%.2f米%s" % (row["travelled_m"], "" if row["upright"] else "(摔)") for row in parents)),
                  flush=True)
            if champion is not None:
                Path("%s_冠军.jsonl" % args.out_prefix).write_text(
                    json.dumps(champion, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if champion is not None:
        print("跑完。跨代冠军：%d 代第 %d 只，%.3f 米，%s" % (
            champion["gen"], champion["index"], champion["travelled_m"],
            "没摔" if champion["upright"] else "摔了"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())