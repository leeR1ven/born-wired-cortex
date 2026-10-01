# -*- coding: utf-8 -*-
"""追球演化：一代只生几十只、每只换一套随机连接，考完立刻挑爹、马上生下一代。

跟 tools/evolve.py 的区别只有一个 —— 考的不只是「直着走多远」，还有「球跑它追不追」。
题在 tools/chase_task.py，判分全都留着（没摔 / 朝球转了几趟 / 两边会不会 / 球躲着跑时
跟住没有 / 最后离球多远 / 正对球多少 / 顶在正前方几成时间 / 走了多远 / 直度）。

每一只候选身上有两份随机：
  1. 基因：冠军那份 50 个数字上随机改几笔（tools/evolve.py 的 make_child），
     再加上「拐弯的力」avoidance_gain 单独随机（用户 2026-09-30：直接随机生成一堆
     模型然后测试；冠军那份的 avoidance_gain 被演化压到了 0.0003，不单独摇就永远
     摇不到能拐弯的量级）。
  2. 追球接法：眼睛偏到哪一边 → 往哪边拐，这套接法是每只自己随机长出来的
     （走哪条路、往哪边拐、从眼位那一排的哪些细胞接出来、接多粗）。

    python tools/evolve_chase.py --candidates 60 --generations 40 --workers 8 ^
        --parents artifacts/追球_云_二批_g07_keep.jsonl --out-prefix artifacts/追球_三批

繁衍方式（用户 2026-10-01）：**每代少生、多迭代** —— 一代只生几十只，生完立刻考、
考完立刻挑爹、马上生下一代。代与代之间转得快，比「一代生上万只、却要等半天」管用。

挑爹的规矩（用户 2026-10-01）：第一眼看「真追住了几趟」，再看不摔、两边都会、
跑完落后几米、球在正前方几成时间。摔了但会追的也留几只当爹 —— 只留没摔的，
追球这套本事可能永远长不出来。
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

from tools import evolve as ev                                          # noqa: E402

CHAMPION = ROOT / "artifacts" / "云端" / "题1f_演化_云_冠军.jsonl"


def random_chase(rng):
    """这一只身上那套追球接法，全部随机长出来。

    走哪条路：'motor' = 新加的转向细胞直接拧髋；'orient' = 报给模型本来那个
    「凑过去看」的细胞（拐弯的力是基因 avoidance_gain）。两条路哪个能用，让挑爹去说。
    """
    route = "orient" if rng.random() < .5 else "motor"
    return dict(route=route,
                source="position" if rng.random() < .75 else "muscle",
                flip=bool(rng.integers(2)),
                pivot=int(rng.integers(5, 12)),
                gain=float(np.round(rng.uniform(.25, 2.), 3)),
                turn_gain=float(np.round(rng.uniform(.02, .3), 4)),
                turn_time=float(np.round(rng.uniform(.1, .5), 3)))


def random_avoidance(rng):
    """拐弯的力：avoidance_gain（steering→髋 那根线）。0.001 到 0.5，取对数均匀。"""
    return float(10 ** rng.uniform(-3., -.52))


def mutate_chase(base, rng):
    """追球接法：从爹那套抄一份，每一格有概率重新摇；没有爹就整套重新摇。

    这样「会追球的接法」才传得下去、也才有机会越攒越准（用户 2026-10-01：每代少生、
    多迭代）。数值格子按倍数抖（gain / turn_gain / turn_time），不是加减，
    这样 0.02 和 0.3 这种量级不会被一下子抹平。
    """
    fresh = random_chase(rng)
    if not isinstance(base, dict) or not base:
        return fresh
    out = dict(fresh)
    for key in ("route", "source", "flip", "pivot"):
        if key in base and rng.random() < .75:
            out[key] = base[key]
    for key, span in (("gain", .5), ("turn_gain", 1.2), ("turn_time", 1.2)):
        if key in base and rng.random() < .7:
            out[key] = float(np.round(max(.001, float(base[key])
                                           * float(np.exp(rng.normal(0., span)))), 5))
    return out


def cross_chase(first, second, rng):
    """两个爹的追球接法各出一半 —— 跟基因那边的两爹融合是一个道理。

    用户 2026-10-01：孩子应该是「之前最好的模型融合后随机生成」。基因那边一直是
    两爹融合（tools/evolve.py 的 make_child），但追球接法原来只从**一个**爹那儿抄，
    少了一半的融合。这里补上：数值格子取 40%/40%/20%（谁/A+B 平均），
    离散格子（走哪条路、往哪边拐）各 50%。
    """
    out = {}
    for key in set(first) | set(second):
        mine, theirs = first.get(key), second.get(key)
        if mine is None or theirs is None:
            out[key] = theirs if mine is None else mine
            continue
        roll = rng.random()
        if isinstance(mine, float) and isinstance(theirs, float):
            out[key] = mine if roll < .4 else (theirs if roll < .8 else .5 * (mine + theirs))
        else:
            out[key] = mine if roll < .5 else theirs
    return out


def pick_chase(parents, rng):
    """给孩子一套追球接法：一半的孩子拿两个爹的接法融合，一半从一个爹那儿抄；
    之后一样要过 mutate_chase 的「按概率重摇」。

    没有爹带着接法，就整套重新摇。
    """
    donors = [row for row in parents if isinstance(row.get("chase"), dict)]
    if not donors:
        return mutate_chase(None, rng)
    if len(donors) >= 2 and rng.random() < .5:
        i, j = (int(v) for v in rng.choice(len(donors), size=2, replace=False))
        base = cross_chase(donors[i]["chase"], donors[j]["chase"], rng)
    else:
        base = donors[int(rng.integers(len(donors)))]["chase"]
    return mutate_chase(base, rng)


def rank(row):
    # 用户 2026-10-01：这一轮要的就是「一直追着红球」。所以排序最前面是「追住几趟」，
    # 再是两边会不会、跑完落后几米、球在正前方几成时间、追近了没、最近贴到几米；
    # 静止球那套老判分（走到球跟前几趟、平均贴多近）退到后面，只当参考。
    # —— 上一批丢追球本事，就是因为老判分排在最前面。
    return (0 if row.get("upright") else 1,
            -int(row.get("chase_covered", 0)),
            -int(bool(row.get("both"))),
            float(row.get("chase_mean", 99.)),
            -float(row.get("chase_in_view", 0.)),
            float(row.get("chase_settled", 99.)),
            -float(row.get("chase_approach", -99.)),
            float(row.get("chase_min", 99.)),
            -int(row.get("covered", 0)),
            float(row.get("score", 99.)),
            -int(row.get("moving", 0)),
            -float(row.get("chase_travelled", 0.)),
            -float(row.get("travelled_m", 0.)),
            -float(row.get("straightness", 0.)))

def run_one(job):
    import tools.chase_task as ct
    started = time.perf_counter()
    row = dict(kind="chase", gen=job["gen"], index=job["index"], seed=job["seed"],
               genome=job["genome"], chase=job["chase"], touched=job["touched"],
               window_seconds=job["seconds"], label=job.get("label"))
    try:
        got = ct.exam(job["genome"], job["seed"], seconds=job["seconds"],
                      chase=job["chase"], flee=job["flee"],
                      chase_seconds=job.get("chase_seconds", 20.),
                      curve=job.get("curve", 1.))

    except Exception as exc:
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc),
                   wall_seconds=time.perf_counter() - started)
        return row
    slim = [{key: value for key, value in item.items() if key != "ball_track"}
            for item in got["rows"]]
    row.update(status="ok", upright=got["upright"], moving=got["moving"],
               score=got["score"], kept=got["kept"],
               travelled_m=got["travelled_m"], straightness=got["straightness"],
               covered=got["covered"], trials=got["trials"], both=got["both"],
               flee_covered=got["flee_covered"], flee_trials=got["flee_trials"],
               flee_settled=got["flee_settled"], flee_kept=got["flee_kept"],
               caught=got["caught"], blind=got["blind"], best=got["best"],
               toward_deg=got["toward_deg"],
               chase_covered=got["chase_covered"], chase_trials=got["chase_trials"],
               chase_settled=got["chase_settled"], chase_mean=got["chase_mean"],
               chase_best=got["chase_best"],
               chase_min=got["chase_min"], chase_in_view=got["chase_in_view"],
               chase_approach=got["chase_approach"], chase_travelled=got["chase_travelled"],
               shifted_m=max([item["shifted_m"] for item in got["rows"]
                              if item["shifted_m"] == item["shifted_m"]] or [float("nan")]),
               rows=slim, wall_seconds=time.perf_counter() - started)

    return row


def verify_one(job):
    """复试：换它没见过的位置再考一遍，而且**每一个位置**都配一趟「把球搬走」的对照。

    用户 2026-09-30：只换球的位置还不够 —— 万一某只只对球摆在某个地方有反应呢？
    所以每个位置都跑两趟：一趟球摆在那儿、一趟球搬走（同一个种子、别的什么都不动）。
    两趟落点差不到 0.05 米，说明那个位置上它根本没在看球，那一趟的「贴到球」不算数。

    考过的位置是 0.5 弧度 x 3.0/4.0 米；复试换成 -0.7~0.7 弧度 x 2.5/3.5/4.5 米。
    """
    import numpy as np
    import chase_red_ball as C
    import tools.chase_task as ct
    started = time.perf_counter()
    genome, chase, seed = job["genome"], job["chase"], job["seed"]
    bar = float(job.get("bar", .05))
    row = dict(index=job["index"], seed=seed)
    try:
        spec = ct.gaze()
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        # 用户 2026-09-30：球搬走之后，狗面对的世界每一个位置都是一样的（球不在了），
        # 所以「没球」那一趟只跑一次就够 —— 拿这一趟的落点跟 18 个有球的位置逐个比。
        C.restore(brain, saved)
        off = C.measure(body, brain, eyes, geom, job["seconds"], job["bearings"][0],
                        job["distances"][0], seed=seed, ball=False)
        no_ball = np.asarray(off["end_xy"])
        hits, moved, looks, lowest, shifts, detail = 0, 0, 0, 1., [], []
        for distance in job["distances"]:
            for bearing in job["bearings"]:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                               seed=seed)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"]) - no_ball))
                moving = bool(on["travelled_m"] >= 2.0 and on["lowest_up_z"] >= .3)
                close = float(on["range_min"]) <= 1.0
                looks_here = bool(shifted >= bar)
                hits += int(moving and close and looks_here)
                moved += int(moving)
                looks += int(looks_here)
                shifts.append(shifted)
                lowest = min(lowest, float(on["lowest_up_z"]))
                detail.append(dict(bearing=bearing, distance=distance,
                                   closest=float(on["range_min"]), shifted_m=shifted,
                                   moving=moving, looks_at_ball=looks_here))
        eyes.close()
        row.update(status="ok", hits=hits, trials=len(detail), moved=moved, looks=looks,
                   lowest_up_z=lowest,
                   shifted_m=float(np.mean(shifts)) if shifts else float("nan"),
                   shifted_max=float(np.max(shifts)) if shifts else float("nan"), at=detail)
    except Exception as exc:
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc))
    row["wall_seconds"] = time.perf_counter() - started
    return row


def looks_at_ball(row, least=2):
    """复试：它有没有几个位置是真的在看球（球搬走行为就变）。一个都没有就是碰巧。"""
    return bool(row.get("status") == "ok" and row.get("looks", 0) >= least)


FOLLOW_SETTLED = 1.5     # 跑完落后不到这么近 = 这一趟「追住了」（跟 chase_task 一个口径）
FOLLOW_SHIFT = .05       # 有球那趟的落点跟「球搬走」那趟差这么多米，才算它在看球
FOLLOW_MOVE = 2.0        # 这一趟至少得真的走了这么多米，才算它在追


def pick_follows(rows):
    """从已经考完的账里数「它是不是真在追球」，一个脑子都不重建。

    一趟「追住了」要同时满足：球在跑（flee > 0）、没趴下（lowest_up_z >= .3）、
    真的动了（>= 2 米）、跑完落后不到 1.2 米。

    再把「球搬走」对照拿出来：有球那趟的落点跟没球那趟差 >= 0.05 米，才算它
    看着球在动。两趟落点几乎没变 = 球在不在对它毫无影响，那它「追住了球」是
    碰巧，follows 记 0（用户 2026-09-30：确认不是碰巧）。
    """
    out = []
    for row in rows:
        runs = [item for item in (row.get("rows") or []) if item.get("flee", 0.) > 0.]
        if not runs:
            continue
        follows = [item for item in runs
                   if float(item.get("lowest_up_z", 0.)) >= .3
                   and float(item.get("travelled_m", 0.)) >= FOLLOW_MOVE
                   and float(item.get("settled", 99.)) <= FOLLOW_SETTLED]
        looks = [item for item in runs
                 if float(item.get("shifted_m", 0.)) >= FOLLOW_SHIFT]
        picked = dict(row)
        picked["follow_runs"] = len(follows)
        picked["follow_trials"] = len(runs)
        picked["follow_looks"] = len(looks)
        picked["follow_blind"] = bool(not looks)
        picked["follows"] = 0 if not looks else len(follows)
        out.append(picked)
    return out


def pick_parents(rows, keep_stable, keep_fast, follow_pool=0):
    """挑爹：第一眼看「真追住了几趟」（follows），再按 rank 的其余次序排。

    follow_pool > 0 时只在 rank 前 N 名里数追球 —— 一只摔得满地打滚的模型，就算
    碰巧「追住了」也没意义，不必为它费事。
    """
    ok = [row for row in rows if row.get("status") == "ok" and row.get("genome")]
    if follow_pool > 0:
        ok = sorted(ok, key=rank)[:follow_pool]
    scored = pick_follows(ok)

    def key(row):
        return (0 if row.get("upright") else 1, -row["follows"]) + rank(row)

    stable = sorted([row for row in scored if row.get("upright")], key=key)[:keep_stable]
    fast = sorted(scored, key=key)[:keep_fast]
    seen, out = set(), []
    for row in stable + fast:
        mark = ev.fingerprint(row["genome"])
        if mark in seen:
            continue
        seen.add(mark)
        out.append(row)
    return out

def generation_line(gen, rows, seconds):
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return "第 %d 代：%d 只一只都没建出来" % (gen, len(rows))
    scored = pick_follows(ok)
    best = min(scored, key=rank)
    upright = [row for row in ok if row.get("upright")]
    followers = [row for row in scored if row["follows"] > 0]
    both = [row for row in scored if row["follows"] >= 2]
    aware = [row for row in scored if not row["follow_blind"]]
    return ("第 %d 代收工：%d 只里建出来 %d 只（没摔 %d）；真追住球的 %d 只，两边都追住的 %d 只"
            "（球挪走行为会变的 %d 只）；追得最好那只落后 %.2f 米、球在正前方 %3.0f%% 的时间、"
            "最近贴到 %.2f 米%s；本代花了 %.1f 分钟"
            % (gen, len(rows), len(ok), len(upright), len(followers), len(both), len(aware),
               best.get("chase_settled", float("nan")),
               best.get("chase_in_view", 0.) * 100.,
               best.get("chase_min", float("nan")),
               "" if best.get("upright") else "(摔了)",
               seconds / 60.))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parents", default=str(CHAMPION), help="头一代的爹；给冠军那一行就行")
    ap.add_argument("--candidates", type=int, default=60, help="一代生几只（少生、多迭代）")
    ap.add_argument("--generations", type=int, default=24, help="一共生几代")

    ap.add_argument("--keep-stable", type=int, default=6)
    ap.add_argument("--keep-fast", type=int, default=4)
    ap.add_argument("--genes", type=int, default=6, help="每只随机动几个数字")
    ap.add_argument("--sigma", type=float, default=.55, help="每个数字动多大（一格=它自己的量级）")
    ap.add_argument("--cross-rate", type=float, default=.5)
    ap.add_argument("--seconds", type=float, default=6., help="静止球那几趟每趟考多少秒")
    ap.add_argument("--chase-seconds", type=float, default=20., help="追着跑的球那几趟每趟考多少秒")
    ap.add_argument("--curve", type=float, default=1., help="球逃跑路线的弯度；0 = 直线逃")
    ap.add_argument("--flee", type=float, default=.6, help="球躲着狗跑的速度；0 = 不考这一项")

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-gen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out-prefix", default=str(ROOT / "artifacts" / "追球_演化"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verify-top", type=int, default=0,
                    help="每一代挑完爹之后，拿前几名做复试（没见过的位置 + 球挪走对照）；0 = 不复试")
    ap.add_argument("--follow-top", type=int, default=0,
                    help="只在成绩前 N 名里数「真追住了几趟」；0 = 全都数")

    args = ap.parse_args(argv)

    parents = ev.read_jsonl(args.parents)
    if not parents:
        print("这份台账里挑不出爹：%s" % args.parents)
        return 1
    print("头一代的爹 %d 只：%s" % (len(parents), "、".join(
        "第%s代第%s只" % (row.get("gen"), row.get("index")) for row in parents[:4])), flush=True)

    rng = np.random.default_rng(args.seed)
    champion = None
    with ProcessPoolExecutor(args.workers) as pool:
        for gen in range(args.start_gen, args.start_gen + args.generations):
            jobs = []
            for index in range(args.candidates):
                genome, touched = ev.make_child(parents, rng, args)
                genome = dict(genome)
                if rng.random() < .5:
                    # 一半的孩子把拐弯的力重新摇一个量级（原来那支被压到 0.0003，
                    # 不重摇就跳不出来）；另一半留着爹的值，让好的量级攒得住。
                    genome["avoidance_gain"] = random_avoidance(rng)
                touched = sorted(set(list(touched) + ["avoidance_gain"]))
                jobs.append(dict(gen=gen, index=index, seed=int(rng.integers(1000000)),
                                 genome=genome, touched=touched, seconds=args.seconds,
                                 chase_seconds=args.chase_seconds, curve=args.curve,
                                 flee=args.flee, chase=pick_chase(parents, rng)))

            ledger = "%s_g%02d.jsonl" % (args.out_prefix, gen)
            print("第 %d 代：生 %d 只、每只考 4 趟不动球（每趟 %.0f 秒）+ 2 趟追着跑的球"
                  "（每趟 %.0f 秒，球沿随机曲线逃）+ 1 趟没球对照、%d 个进程"
                  % (gen, len(jobs), args.seconds, args.chase_seconds, args.workers), flush=True)

            started = time.perf_counter()
            rows = []
            for row in pool.map(run_one, jobs):
                rows.append(row)
                ev.append(row, ledger)
                if row["status"] == "ok" and (champion is None or rank(row) < rank(champion)):
                    champion = row
                    print("  跨代冠军易主：%d 代第 %d 只  追住 %d/%d 趟、跑完落后 %.2f 米、"
                          "球在正前方 %3.0f%% 的时间  %s"
                          % (row["gen"], row["index"], row["chase_covered"], row["chase_trials"],
                             row["chase_settled"], row["chase_in_view"] * 100.,
                             "没摔" if row["upright"] else "摔了"), flush=True)

                if not args.quiet:
                    print("%5d  %-14s 追住 %d/%d 趟  平均 %6.2f 米  落后 %6.2f 米  最近 %5.2f 米  "
                          "正前方 %3.0f%%  静球到了 %d/%d 趟  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("chase_covered", 0), row.get("chase_trials", 0),
                             row.get("chase_mean", float("nan")),
                             row.get("chase_settled", float("nan")),
                             row.get("chase_min", float("nan")),
                             row.get("chase_in_view", 0.) * 100.,
                             row.get("covered", 0), row.get("trials", 0),
                             row.get("travelled_m", float("nan")),
                             "  (两边都会)" if row.get("both") else "",
                             (row.get("chase") or {}).get("route", "?"),
                             " 对调" if (row.get("chase") or {}).get("flip") else "",
                             row.get("wall_seconds", 0.)), flush=True)

            print(generation_line(gen, rows, time.perf_counter() - started), flush=True)
            parents = pick_parents(rows, args.keep_stable, args.keep_fast, args.follow_top)
            counted = pick_follows([row for row in rows if row.get("status") == "ok"])
            print("  这一代自己走出来的、真追住球的 %d 只，其中两边都追住的 %d 只"
                  % (sum(1 for row in counted if row["follows"] > 0),
                     sum(1 for row in counted if row["follows"] >= 2)), flush=True)

            if args.verify_top > 0 and parents:
                # 用户 2026-09-30：确认不是碰巧 —— 挑完爹之后，前几名再做一次复试，
                # 球挪走行为不变的（习惯性往一边拐）从爹名单里剔掉。
                ranked = sorted([row for row in rows if row.get("status") == "ok"], key=rank)[:args.verify_top]
                jobs = [dict(index=row["index"], seed=row["seed"], genome=row["genome"],
                             chase=row["chase"], seconds=args.seconds,
                             bearings=[float(v) for v in args.verify_bearings.split(",")],
                             distances=[float(v) for v in args.verify_distances.split(",")])
                        for row in ranked]
                print("  复试 %d 只：18 个没见过的位置，每个位置各配一趟「球搬走」对照"
                      % len(jobs), flush=True)
                checked = list(pool.map(verify_one, jobs))
                ev.append(dict(kind="verify", gen=gen, rows=checked),
                          "%s_g%02d_verify.jsonl" % (args.out_prefix, gen))
                by_index = {row["index"]: row for row in checked}
                for row in parents:
                    got = by_index.get(row["index"])
                    if got is not None:
                        row["verify"] = got
                kept, dropped = [], []
                for row in parents:
                    got = row.get("verify")
                    (kept if got is None or looks_at_ball(got) else dropped).append(row)
                if dropped:
                    print("  复试剔掉 %d 只（球搬走行为不变 = 碰巧）：%s"
                          % (len(dropped), "、".join(
                              "第%d只(看球位置%d/18)" % (row["index"], row["verify"]["looks"])
                              for row in dropped)), flush=True)
                print("  复试留下 %d 只：%s" % (len(kept), "、".join(
                    "第%d只(贴到%d/%d、看球位置%d/18)"
                    % (row["index"], row["verify"]["hits"], row["verify"]["trials"],
                       row["verify"]["looks"]) for row in kept if row.get("verify"))
                    or "（没有一只进了复试）"), flush=True)
                parents = kept or parents
            keep = "%s_g%02d_keep.jsonl" % (args.out_prefix, gen)
            Path(keep).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                          for row in parents), encoding="utf-8", newline="\n")
            print("  下一代的爹 %d 只（%s）：%s" % (len(parents), keep, "、".join(
                "第%d只 追住%d/%d趟 落后%.2f米%s" % (row["index"], row.get("follow_runs", 0),
                                                  row.get("follow_trials", 0),
                                                  row.get("chase_settled", float("nan")),
                                                  "" if row["upright"] else "(摔)")
                for row in parents)), flush=True)

            if champion is not None:
                Path("%s_冠军.jsonl" % args.out_prefix).write_text(
                    json.dumps(champion, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if champion is not None:
        print("跑完。跨代冠军：%d 代第 %d 只，追住 %d/%d 趟、跑完落后 %.2f 米、"
              "球在正前方 %3.0f%% 的时间、最近贴到 %.2f 米，%s"
              % (champion["gen"], champion["index"], champion["chase_covered"],
                 champion["chase_trials"], champion["chase_settled"],
                 champion["chase_in_view"] * 100., champion["chase_min"],
                 "没摔" if champion["upright"] else "摔了"), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())