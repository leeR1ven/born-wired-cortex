# -*- coding: utf-8 -*-
"""把两只模型融合成一只，看能不能既要速度又不要摔。

两种合的法子：

    # 一、按比例混合（两只数字各取一部分）
    python tools/merge_models.py --rule blend --ledger artifacts/题1f_十秒赛5000.jsonl \
        --fast-top 5 --stable-top 4 --alphas 0.25,0.5,0.75 --seeds 2 --out artifacts/题1f_融合_混合.jsonl

    # 二、挑连接：以一只为底，一次只换另一只的一根线（用户说的"挑重要的连接"）
    python tools/merge_models.py --rule swap --ledger artifacts/题1f_十秒赛5000.jsonl \
        --fast-index 2646,523 --stable-index 623 --seeds 1 --out artifacts/题1f_融合_换线.jsonl

    # 三、先按平均值合一只，再从它身上随机生出好几只（用户说的"平均完再随机生成几只"）
    python tools/merge_models.py --rule seed --ledger artifacts/题1f_十秒赛5000.jsonl \
        --fast-index 2646,523 --stable-index 623 --children 40 --out artifacts/题1f_融合_平均再随机.jsonl

    # 五、快的那群、稳的那群各平均成一份，再按比例兑，扫"快基因占多少"
    python tools/merge_models.py --rule ratio --ledger artifacts/题1f_十秒赛5000.jsonl \
        --fast-top 10 --stable-top 6 --alphas 0,0.05,0.1,0.15,0.2,0.25,0.3,0.4 --seeds 3

    # 四、一大群一起合（用户说的"一大堆好的模型一起融合"），不只两两合
    python tools/merge_models.py --rule pool --ledger artifacts/题1f_十秒赛5000.jsonl \
        --pool-index 623,2010,1802,890,447,1999,2646,523,2382 --anchor-index 623 \
        --subsets 24 --children 12 --out artifacts/题1f_融合_一群.jsonl

背景：这批模型**骨架完全相同**（同一套神经元、同一套连线），不同的只是那几十个数字；
哪只跑得快、哪只会摔，差的就是这些数字。所以"融合"在这里就是把两边的数字重新组合。

- `--fast-top` 从台账里"倒过"的、走得最远的取前 N 只当"快"的那一边。
- `--stable-top` 从"全程没倒"的里面取前 N 只当"稳"的那一边。
- blend：每一对 x 每一颗种子跑一遍，alpha 是"快那只占多少"。
- swap：以一只为底、一次只把一根线换成另一只的值（两个方向都试：把快的线换进稳的、
  把稳的线换进快的），这样能看出**是哪一根线让人摔**、哪一根线带来速度。
  "另一个模型里没有这根线" 这种情况不生成候选（没得换）。

跑完按「先看有没有摔、再看走了多远」排序打印，台账写进 `--out`。
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

from tools import screen_candidates as sc                            # noqa: E402

FAR = (60., 60., -8.)
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")


def open_props():
    import tools.taskbank as tb
    props = {geom: FAR for geom in tb.SCENERY}
    props.update({name: FAR for name in WALLS})
    return props


def blend(first, second, alpha):
    """alpha 全是第一只（快那只），1-alpha 是第二只（稳那只）。"""
    out = {}
    for key in set(first) | set(second):
        mine = first.get(key, second.get(key))
        theirs = second.get(key, mine)
        out[key] = alpha * mine + (1. - alpha) * theirs
    return out


def run_one(job):
    from tools import taskbank as tb
    started = time.perf_counter()
    row = dict(kind="merge", index=job["index"], seed=job["seed"], rule=job["rule"],
               fast=job["fast"], stable=job["stable"], alpha=job["alpha"],
               genome=job["genome"], revision=tb.revision())
    try:
        ctx = tb.context(job["genome"], job["seed"])
        result = tb.run_seed(seed=ctx["seed"], duration=job["seconds"],
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
               speed_mps=metrics["total_displacement_m"] / job["seconds"],
               path_m=metrics["horizontal_path_m"],
               min_up_z=metrics["minimum_up_z"],
               straightness=metrics["straightness"],
               wall_seconds=time.perf_counter() - started)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--rule", choices=("blend", "swap", "seed", "pool", "ratio", "gene"), default="blend")
    ap.add_argument("--fast-top", type=int, default=10, help="倒过的里面走得最远的前 N 只")
    ap.add_argument("--stable-top", type=int, default=6, help="全程没倒的里面前 N 只")
    ap.add_argument("--fast-index", default=None, help="指定台账编号，逗号分隔")
    ap.add_argument("--stable-index", default=None)
    ap.add_argument("--alphas", default="0.5", help="快那只占多少，逗号分隔（只对 blend）")
    ap.add_argument("--children", type=int, default=40, help="平均完再随机生几只（只对 seed）")
    ap.add_argument("--genes", type=int, default=sc.MUTATION_GENES, help="每只随机动几个数值（只对 seed）")
    ap.add_argument("--sigma", type=float, default=sc.MUTATION_SIGMA, help="每个数值动多大（只对 seed）")
    ap.add_argument("--draw-seed", type=int, default=7, help="随机生成的骰子（只对 seed）")
    ap.add_argument("--pool-index", default=None, help="一起合的那群编号，逗号分隔（只对 pool）")
    ap.add_argument("--anchor-index", type=int, default=None, help="拉伸时朝哪只拉，不给就朝最快的（只对 pool）")
    ap.add_argument("--pool-stat", choices=("mean", "median"), default="mean")
    ap.add_argument("--stretches", default="0,0.25,0.5,0.75", help="全体平均之后，再朝锚那只拉多少")
    ap.add_argument("--subsets", type=int, default=24, help="随机抽几只一组的次数（只对 pool）")
    ap.add_argument("--base-index", type=int, default=None, help="一只底子，只动它身上一个数字（只对 gene）")
    ap.add_argument("--gene", default=None, help="动哪个数字，逗号分隔可给多个（只对 gene）")
    ap.add_argument("--values", default=None, help="这个数字取哪些值，逗号分隔（只对 gene）")
    ap.add_argument("--subset-size", type=int, default=4, help="每组抽几只（只对 pool）")
    ap.add_argument("--subset-sizes", default=None, help="一次比好几个抽取只数，逗号分隔（只对 pool）")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=10.)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.ledger).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ok = [row for row in rows if row.get("status") == "ok" and row.get("genome")]

    def pick(condition, text, top):
        if text:
            wanted = {int(v) for v in text.split(",")}
            return [row for row in ok if row["index"] in wanted]
        pool = [row for row in ok if condition(row)]
        pool.sort(key=lambda row: -row["travelled_m"])
        return pool[:top]

    fast = pick(lambda row: not row["upright"], args.fast_index, args.fast_top)
    stable = pick(lambda row: row["upright"], args.stable_index, args.stable_top)
    if not fast or not stable:
        print("台账里挑不出两边（要倒过的一边和没倒的一边各至少一只）")
        return 1
    print("快的那边 %d 只：%s" % (len(fast), "、".join(
        "#%d %.2f米" % (row["index"], row["travelled_m"]) for row in fast)))
    print("稳的那边 %d 只：%s" % (len(stable), "、".join(
        "#%d %.2f米" % (row["index"], row["travelled_m"]) for row in stable)))

    jobs = []
    if args.rule == "blend":
        for alpha in (float(v) for v in args.alphas.split(",")):
            for first in fast:
                for second in stable:
                    genome = blend(first["genome"], second["genome"], alpha)
                    for seed in range(1, args.seeds + 1):
                        jobs.append(dict(seed=seed, fast=first["index"], stable=second["index"],
                                         alpha=alpha, rule="混合%.2f" % alpha, genome=genome,
                                         seconds=args.seconds))
    elif args.rule == "swap":
        for first in fast:
            for second in stable:
                for base, other, label in ((second, first, "稳底+换快的"),
                                           (first, second, "快底+换稳的")):
                    for key in sorted(set(base["genome"]) | set(other["genome"])):
                        if base["genome"].get(key) == other["genome"].get(key):
                            continue
                        if key not in other["genome"]:
                            continue
                        genome = dict(base["genome"])
                        genome[key] = other["genome"][key]
                        for seed in range(1, args.seeds + 1):
                            jobs.append(dict(seed=seed, fast=first["index"],
                                             stable=second["index"], alpha=None,
                                             rule="%s:%s" % (label, key), genome=genome,
                                             seconds=args.seconds))
    elif args.rule == "seed":
        import numpy as np
        rng = np.random.default_rng(args.draw_seed)
        alpha = float(args.alphas.split(",")[0])
        for first in fast:
            for second in stable:
                genome = blend(first["genome"], second["genome"], alpha)
                jobs.append(dict(seed=1, fast=first["index"], stable=second["index"], alpha=alpha,
                                 rule="平均", genome=genome, seconds=args.seconds))
                for _ in range(args.children):
                    child, _touched = sc.mutate(genome, 0, rng, genes=args.genes, sigma=args.sigma)
                    jobs.append(dict(seed=1, fast=first["index"], stable=second["index"], alpha=None,
                                     rule="平均+随机", genome=child, seconds=args.seconds))

    elif args.rule == "pool":
        import numpy as np
        rng = np.random.default_rng(args.draw_seed)
        if args.pool_index:
            wanted = {int(v) for v in args.pool_index.split(",")}
            chosen = [row for row in ok if row["index"] in wanted]
        else:
            chosen = fast + stable
        if len(chosen) < 2:
            print("一起合的那群不足两只")
            return 1
        label = "(%d只)" % len(chosen)
        anchor = None
        if args.anchor_index is not None:
            anchor = next((row for row in chosen if row["index"] == args.anchor_index), None)
        if anchor is None:
            anchor = max(chosen, key=lambda row: row["travelled_m"])

        def pool_stat(rows, how):
            keys = set()
            for row in rows:
                keys |= set(row["genome"])
            out = {}
            for key in keys:
                values = [float(row["genome"][key]) for row in rows if key in row["genome"]]
                out[key] = float(np.median(values)) if how == "median" else sum(values) / len(values)
            return out

        mean_genome = pool_stat(chosen, "mean")
        for stretch in ([float(v) for v in args.stretches.split(",") if v.strip()]
                        if args.stretches.strip() else []):
            genome = {key: value + stretch * (float(anchor["genome"].get(key, value)) - value)
                      for key, value in mean_genome.items()}
            jobs.append(dict(seed=1, fast=0, stable=0, alpha=stretch,
                             rule="全体平均朝#%d拉%.2f%s" % (anchor["index"], stretch, label),
                             genome=genome, seconds=args.seconds))
        jobs.append(dict(seed=1, fast=0, stable=0, alpha=None,
                         rule="全体中位%s" % label, genome=pool_stat(chosen, "median"),
                         seconds=args.seconds))
        sizes = ([int(v) for v in args.subset_sizes.split(",")] if args.subset_sizes
                 else [args.subset_size])
        for size in sizes:
            size = min(size, len(chosen))
            for _ in range(args.subsets):
                picked = [chosen[int(j)] for j in rng.choice(len(chosen), size=size, replace=False)]
                genome = pool_stat(picked, "mean")
                for seed in range(1, args.seeds + 1):
                    jobs.append(dict(seed=seed, fast=0, stable=0, alpha=None,
                                     rule="随机抽%d只平均" % size, genome=genome,
                                     seconds=args.seconds))
        for _ in range(args.children):
            child, _touched = sc.mutate(mean_genome, 0, rng, genes=args.genes, sigma=args.sigma)
            jobs.append(dict(seed=1, fast=0, stable=0, alpha=None,
                             rule="全体平均+随机%s" % label, genome=child, seconds=args.seconds))

    elif args.rule == "ratio":
        # 快的那群先平均成一份"快基因"，稳的那群平均成一份"稳基因"，
        # 再按 alpha 把两份兑起来，扫出"快基因占多少最合适"。
        def group_mean(rows):
            keys = set()
            for row in rows:
                keys |= set(row["genome"])
            out = {}
            for key in keys:
                values = [float(row["genome"][key]) for row in rows if key in row["genome"]]
                out[key] = sum(values) / len(values)
            return out

        fast_mean = group_mean(fast)
        stable_mean = group_mean(stable)
        for alpha in (float(v) for v in args.alphas.split(",")):
            genome = blend(fast_mean, stable_mean, alpha)
            for seed in range(1, args.seeds + 1):
                jobs.append(dict(seed=seed, fast=fast[0]["index"], stable=stable[0]["index"],
                                 alpha=alpha, rule="快基因占%.2f" % alpha, genome=genome,
                                 seconds=args.seconds))

    elif args.rule == "gene":
        # 拿一只当底子，一次只改它身上一个数字，看这个数字单独说了多少话。
        if not args.gene or not args.values:
            print("要 --gene 和 --values")
            return 1
        base = None
        if args.base_index is not None:
            base = next((row for row in ok if row["index"] == args.base_index), None)
        if base is None:
            print("台账里找不到底子 #%s" % args.base_index)
            return 1
        for name in args.gene.split(","):
            was = base["genome"].get(name, "没这根")
            for value in (float(v) for v in args.values.split(",")):
                genome = dict(base["genome"])
                genome[name] = value
                for seed in range(1, args.seeds + 1):
                    jobs.append(dict(seed=seed, fast=0, stable=0, alpha=None,
                                     rule="%s=%g(原%s)" % (name, value, was), genome=genome,
                                     seconds=args.seconds))

    for index, job in enumerate(jobs):
        job["index"] = index
    print("要跑 %d 只，每只 %.0f 秒、开阔场地、%d 个进程" % (len(jobs), args.seconds, args.workers))
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(run_one, jobs):
            results.append(row)
            pair = ("#%d+#%d" % (row["fast"], row["stable"])) if row["fast"] else "一起合"
            print("%5d  %-11s %-34s 种子%d  %-6s %6.3f 米  直立最低 %6.3f  直度 %.3f"
                  % (row["index"], pair, row["rule"], row["seed"],
                     row.get("status") if row["status"] != "ok" else ("没倒" if row["upright"] else "倒了"),
                     0. if row.get("travelled_m") is None else row["travelled_m"],
                     0. if row.get("min_up_z") is None else row["min_up_z"],
                     0. if row.get("straightness") is None else row["straightness"]),
                  flush=True)
    print("这批 %d 只共 %.1f 分钟" % (len(jobs), (time.perf_counter() - started) / 60.))

    good = [row for row in results if row.get("status") == "ok"]
    groups = {}
    for row in good:
        groups.setdefault((row["fast"], row["stable"], row["rule"]), []).append(row)
    ranking = []
    for key, group in groups.items():
        speeds = sorted(item["travelled_m"] for item in group)
        ranking.append((sum(1 for item in group if not item["upright"]),
                        speeds[len(speeds) // 2], key, group))
    ranking.sort(key=lambda item: (item[0], -item[1]))
    print("\n按「先看摔不摔、再看中位速度」排（前 20）：")
    print("%9s %9s %-34s %6s %10s %9s" % ("快的", "稳的", "怎么合的", "摔倒", "中位(米)", "最好(米)"))
    for falls, mid, key, group in ranking[:20]:
        print("%9d %9d %-34s %6d %10.3f %9.3f"
              % (key[0], key[1], key[2], falls, mid, max(item["travelled_m"] for item in group)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
                                  encoding="utf-8", newline="\n")
        print("\n台账 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())