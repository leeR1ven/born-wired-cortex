# -*- coding: utf-8 -*-
"""把 evolve_chase.py 改成「追着跑的球」第一优先 + 每代少生多迭代的繁衍方式。

用户 2026-10-01：
  1) 这一轮要的是「一直追着红球」，所以挑爹第一眼看的就是「真追住了几趟」。
     老版把「贴到静止球」排在前面 —— 上一批追球的本事就是那样一代代丢光的。
  2) 繁衍方式换成「每代少生、多迭代」：一代只生几十只，生完立刻考、考完立刻挑爹、
     马上生下一代。代与代之间转得快，比一代生几千只要等半天更有用。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
lines = s.split("\n")


def find(marker, start=0):
    hits = [i for i, line in enumerate(lines) if line == marker]
    if not hits:
        hits = [i for i, line in enumerate(lines) if line.startswith(marker)]
    hits = [i for i in hits if i >= start]
    assert hits, "没找到这一行：" + marker
    return hits[0]


def block(start, end, new, tag):
    global lines
    i = find(start)
    j = find(end, i + 1)
    if new == "":
        lines[i:j] = []
    else:
        lines[i:j] = new.split("\n")
    print("改好", tag)


NEW_RANK = r'''def rank(row):
    # 用户 2026-10-01：这一轮要的就是「一直追着红球」。所以排序最前面是「追住几趟」，
    # 再是两边会不会、跑完落后几米、球在正前方几成时间、追近了没、最近贴到几米；
    # 静止球那套老判分（走到球跟前几趟、平均贴多近）退到后面，只当参考。
    # —— 上一批丢追球本事，就是因为老判分排在最前面。
    return (0 if row.get("upright") else 1,
            -int(row.get("chase_covered", 0)),
            -int(bool(row.get("both"))),
            float(row.get("chase_settled", 99.)),
            -float(row.get("chase_in_view", 0.)),
            -float(row.get("chase_approach", -99.)),
            float(row.get("chase_min", 99.)),
            -int(row.get("covered", 0)),
            float(row.get("score", 99.)),
            -int(row.get("moving", 0)),
            -float(row.get("chase_travelled", 0.)),
            -float(row.get("travelled_m", 0.)),
            -float(row.get("straightness", 0.)))
'''

NEW_EXAM = r'''        got = ct.exam(job["genome"], job["seed"], seconds=job["seconds"],
                      chase=job["chase"], flee=job["flee"],
                      chase_seconds=job.get("chase_seconds", 20.),
                      curve=job.get("curve", 1.))
'''

NEW_ROW = r'''    slim = [{key: value for key, value in item.items() if key != "ball_track"}
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
               chase_settled=got["chase_settled"], chase_best=got["chase_best"],
               chase_min=got["chase_min"], chase_in_view=got["chase_in_view"],
               chase_approach=got["chase_approach"], chase_travelled=got["chase_travelled"],
               shifted_m=max([item["shifted_m"] for item in got["rows"]
                              if item["shifted_m"] == item["shifted_m"]] or [float("nan")]),
               rows=slim, wall_seconds=time.perf_counter() - started)
'''

NEW_PICK = r'''FOLLOW_SETTLED = 1.2     # 跑完落后不到这么近 = 这一趟「追住了」
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
'''

NEW_GENLINE = r'''def generation_line(gen, rows, seconds):
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
'''

NEW_ARGS1 = r'''    ap.add_argument("--candidates", type=int, default=60, help="一代生几只（少生、多迭代）")
    ap.add_argument("--generations", type=int, default=40, help="一共生几代")
'''

NEW_ARGS2 = r'''    ap.add_argument("--seconds", type=float, default=6., help="静止球那几趟每趟考多少秒")
    ap.add_argument("--chase-seconds", type=float, default=20., help="追着跑的球那几趟每趟考多少秒")
    ap.add_argument("--curve", type=float, default=1., help="球逃跑路线的弯度；0 = 直线逃")
    ap.add_argument("--flee", type=float, default=.6, help="球躲着狗跑的速度；0 = 不考这一项")
'''

NEW_ARGS3 = r'''    ap.add_argument("--verify-top", type=int, default=0,
                    help="每一代挑完爹之后，拿前几名做复试（没见过的位置 + 球挪走对照）；0 = 不复试")
    ap.add_argument("--follow-top", type=int, default=0,
                    help="只在成绩前 N 名里数「真追住了几趟」；0 = 全都数")
'''

NEW_JOBS = r'''                jobs.append(dict(gen=gen, index=index, seed=int(rng.integers(1000000)),
                                 genome=genome, touched=touched, seconds=args.seconds,
                                 chase_seconds=args.chase_seconds, curve=args.curve,
                                 flee=args.flee, chase=random_chase(rng)))
'''

NEW_GENPRINT = r'''            print("第 %d 代：生 %d 只、每只考 4 趟不动球（每趟 %.0f 秒）+ 2 趟追着跑的球"
                  "（每趟 %.0f 秒，球沿随机曲线逃）+ 1 趟没球对照、%d 个进程"
                  % (gen, len(jobs), args.seconds, args.chase_seconds, args.workers), flush=True)
'''

NEW_CHAMPPRINT = r'''                    print("  跨代冠军易主：%d 代第 %d 只  追住 %d/%d 趟、跑完落后 %.2f 米、"
                          "球在正前方 %3.0f%% 的时间  %s"
                          % (row["gen"], row["index"], row["chase_covered"], row["chase_trials"],
                             row["chase_settled"], row["chase_in_view"] * 100.,
                             "没摔" if row["upright"] else "摔了"), flush=True)
'''

NEW_ROWPRINT = r'''                if not args.quiet:
                    print("%5d  %-14s 追住 %d/%d 趟  落后 %6.2f 米  最近 %5.2f 米  正前方 %3.0f%%  "
                          "静球到了 %d/%d 趟  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("chase_covered", 0), row.get("chase_trials", 0),
                             row.get("chase_settled", float("nan")),
                             row.get("chase_min", float("nan")),
                             row.get("chase_in_view", 0.) * 100.,
                             row.get("covered", 0), row.get("trials", 0),
                             row.get("travelled_m", float("nan")),
                             "  (两边都会)" if row.get("both") else "",
                             (row.get("chase") or {}).get("route", "?"),
                             " 对调" if (row.get("chase") or {}).get("flip") else "",
                             row.get("wall_seconds", 0.)), flush=True)
'''

NEW_PARENTS = r'''            parents = pick_parents(rows, args.keep_stable, args.keep_fast, args.follow_top)
            counted = pick_follows([row for row in rows if row.get("status") == "ok"])
            print("  这一代自己走出来的、真追住球的 %d 只，其中两边都追住的 %d 只"
                  % (sum(1 for row in counted if row["follows"] > 0),
                     sum(1 for row in counted if row["follows"] >= 2)), flush=True)
'''

NEW_KEEPPRINT = r'''            print("  下一代的爹 %d 只（%s）：%s" % (len(parents), keep, "、".join(
                "第%d只 追住%d/%d趟 落后%.2f米%s" % (row["index"], row.get("follow_runs", 0),
                                                  row.get("follow_trials", 0),
                                                  row.get("chase_settled", float("nan")),
                                                  "" if row["upright"] else "(摔)")
                for row in parents)), flush=True)
'''

NEW_FINALPRINT = r'''        print("跑完。跨代冠军：%d 代第 %d 只，追住 %d/%d 趟、跑完落后 %.2f 米、"
              "球在正前方 %3.0f%% 的时间、最近贴到 %.2f 米，%s"
              % (champion["gen"], champion["index"], champion["chase_covered"],
                 champion["chase_trials"], champion["chase_settled"],
                 champion["chase_in_view"] * 100., champion["chase_min"],
                 "没摔" if champion["upright"] else "摔了"), flush=True)
'''

block("def rank(row):", "def run_one(job):", NEW_RANK, "rank（追球第一优先）")
block('        got = ct.exam(job["genome"], job["seed"], seconds=job["seconds"],',
      "    except Exception as exc:", NEW_EXAM, "exam 调用（加追球秒数/弯度）")
block('    row.update(status="ok", upright=got["upright"], moving=got["moving"],',
      "    return row", NEW_ROW, "run_one 写进新判分")
block("def pick_parents(rows, keep_stable, keep_fast):",
      "def generation_line(gen, rows, seconds):", NEW_PICK, "pick_follows + 新 pick_parents")
block("def generation_line(gen, rows, seconds):", "def main(argv=None):",
      NEW_GENLINE, "generation_line")
block('    ap.add_argument("--candidates", type=int, default=200, help="一代生几只")',
      '    ap.add_argument("--keep-stable", type=int, default=6)', NEW_ARGS1, "默认少生多代")
block('    ap.add_argument("--seconds", type=float, default=6., help="每趟考多少秒")',
      '    ap.add_argument("--workers", type=int, default=8)', NEW_ARGS2, "考多少秒")
block('    ap.add_argument("--verify-top", type=int, default=8,',
      "    args = ap.parse_args(argv)", NEW_ARGS3, "复试默认关")
block('                jobs.append(dict(gen=gen, index=index, seed=int(rng.integers(1000000)),',
      '            ledger = "%s_g%02d.jsonl" % (args.out_prefix, gen)', NEW_JOBS, "job 里带追球秒数")
block('            print("第 %d 代：生 %d 只、每只考 4 趟不动球 + 2 趟躲着跑的球、%d 个进程"',
      "            started = time.perf_counter()", NEW_GENPRINT, "开跑那行提示")
block('                    print("  跨代冠军易主：', "                if not args.quiet:",
      NEW_CHAMPPRINT, "冠军易主那行")
block("                if not args.quiet:",
      "            print(generation_line(gen, rows, time.perf_counter() - started), flush=True)",
      NEW_ROWPRINT, "每只一行")
block("            parents = pick_parents(rows, args.keep_stable, args.keep_fast)",
      "            if args.verify_top > 0 and parents:", NEW_PARENTS, "挑爹那一行")
block('            print("  下一代的爹 %d 只（%s）：%s" % (len(parents), keep, "、".join(',
      "            if champion is not None:", NEW_KEEPPRINT, "爹名单那行")
block('        print("跑完。跨代冠军：', "    return 0", NEW_FINALPRINT, "收工那行")

old_doc = ["    python tools/evolve_chase.py --candidates 300 --generations 3 --workers 8 ^",
           "        --parents artifacts/云端/题1f_演化_云_冠军.jsonl --out-prefix artifacts/追球_演化",
           "",
           "挑爹的规矩：先把摔了的排后面，再看「朝球转了几趟」，再看「球躲着跑时跟住几趟」，",
           "再看「最后离球多远」，最后看「走了多远」。摔了但会追的也留几只当爹 —— 只留没摔的，",
           "追球这套本事可能永远长不出来。"]
i = find(old_doc[0])
assert lines[i:i + len(old_doc)] == old_doc, "文档那段对不上"
lines[i:i + len(old_doc)] = r'''    python tools/evolve_chase.py --candidates 60 --generations 40 --workers 8 ^
        --parents artifacts/追球_云_二批_g07_keep.jsonl --out-prefix artifacts/追球_三批

繁衍方式（用户 2026-10-01）：**每代少生、多迭代** —— 一代只生几十只，生完立刻考、
考完立刻挑爹、马上生下一代。代与代之间转得快，比「一代生上万只、却要等半天」管用。

挑爹的规矩（用户 2026-10-01）：第一眼看「真追住了几趟」，再看不摔、两边都会、
跑完落后几米、球在正前方几成时间。摔了但会追的也留几只当爹 —— 只留没摔的，
追球这套本事可能永远长不出来。'''.split("\n")
print("改好 文档那一段")

io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
print("写回", p)