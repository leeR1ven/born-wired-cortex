# -*- coding: utf-8 -*-
"""evolve_chase.py：改成「按综合评分留模型 + 好模型融合生后代 + 球慢慢提速」。

三处：
  1. rank() —— 从「追住几趟一票定生死」换成一张摊开的综合评分表
  2. fuse() —— 综合分最高的前几只平均成一个「融合爹」，一半孩子从它变异来
  3. --flee-ramp —— 每过一代球速加一点点（狗才 0.20~0.23 米/秒，不能一次加太狠）
"""
import io, sys
P = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
t = io.open(P, encoding="utf-8").read()

def rep(old, new, why):
    global t
    n = t.count(old)
    if n != 1:
        print("!! %s : 找到 %d 处" % (why, n)); sys.exit(1)
    t = t.replace(old, new); print("ok  %s" % why)

# ---------- 1. rank -> 综合评分 ----------
rep('''def rank(row):
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
            -float(row.get("straightness", 0.)))''',
'''# =============================== 综合评分 ===============================
# 用户 2026-10-01：要「根据整体评分来保留模型」，整体都很优秀的才拿去融合生后代。
# 所以不再是「追住几趟」一票定生死，而是把每一格摊开、各按一个权重加起来。
# 每一格先换算到 0~1（1 = 最好），十个权重加起来正好 1.00。摔了的直接沉底。
#
#   这一格          权重   0 分长什么样        1 分长什么样       怎么换算
#   -------------  -----  ------------------  -----------------  ----------------------
#   真追住球        .30   一趟都没追住        两趟都追住         追住趟数 / 总趟数
#   球在正前方      .15   球老跑到侧面去      球一直待在正前方   已经是 0~1
#   跑完落后        .13   落后 3 米以上       贴在球屁股上       (3 - 落后) / 3
#   追近了          .06   20 秒一点没追近     追近 1.5 米        追近的米数 / 1.5
#   最近贴到        .05   最近都没进 1.5 米   贴到 0 米          (1.5 - 最近) / 1.5
#   两边都会        .05   只认一边            左、右都追住       是/否
#   没摔            .07   摔了                没摔               是/否
#   跑得快          .09   一步没挪            20 秒跑满 5 米     跑了多少米 / 5
#   走得直          .04   原地打转            直着走             直度（已经是 0~1）
#   静止球          .06   没走到球跟前        走到了             静止球那趟到了没有
SCORE_WEIGHTS = (("follows", .30), ("in_view", .15), ("settled", .13), ("approach", .06),
                 ("closest", .05), ("both", .05), ("upright", .07), ("fast", .09),
                 ("straight", .04), ("static", .06))


def clamp01(value):
    """夹到 0~1。"""
    return 0. if value < 0. else (1. if value > 1. else float(value))


def overall(row, detail=False):
    """这一只的整体分，0~1，越大越好；摔了的给 -1，直接沉底。

    权重表在 SCORE_WEIGHTS，每一项怎么换算在那一坨注释里写着，没有藏起来的东西。
    """
    if not row.get("upright"):
        return (-1., {}) if detail else -1.
    trials = max(int(row.get("follow_trials", row.get("chase_trials", 0)) or 0), 1)
    follows = int(row.get("follows", row.get("chase_covered", 0)) or 0)
    settled = float(row.get("chase_settled", 99.))
    parts = {
        "follows": clamp01(follows / float(trials)),
        "in_view": clamp01(float(row.get("chase_in_view", 0.))),
        "settled": clamp01((3. - settled) / 3.),
        "approach": clamp01(float(row.get("chase_approach", -99.)) / 1.5),
        "closest": clamp01((1.5 - float(row.get("chase_min", 99.))) / 1.5),
        "both": 1. if row.get("both") else 0.,
        "upright": 1.,
        "fast": clamp01(float(row.get("chase_travelled", 0.)) / 5.),
        "straight": clamp01(float(row.get("straightness", 0.))),
        "static": 1. if int(row.get("covered", 0) or 0) > 0 else 0.,
    }
    score = sum(weight * parts[name] for name, weight in SCORE_WEIGHTS)
    return (score, parts) if detail else score


def rank(row):
    # 用户 2026-10-01：按整体评分留模型。所以第一眼看摔没摔，第二眼就是综合分。
    # 分数一样时（很少见）才轮到后面这些细项分先后。
    return (0 if row.get("upright") else 1,
            -overall(row),
            -int(row.get("chase_covered", 0)),
            -int(bool(row.get("both"))),
            float(row.get("chase_settled", 99.)),
            -float(row.get("chase_in_view", 0.)),
            -float(row.get("chase_travelled", 0.)),
            float(row.get("score", 99.)))


def top_overall(rows, count):
    """综合分最高的前 count 只（摔了的不要，没建出来的不要）。"""
    okay = [row for row in (rows or [])
            if row.get("status") == "ok" and row.get("genome") and row.get("upright")]
    return sorted(okay, key=rank)[:max(int(count), 1)]


def fuse(rows):
    """把一堆好模型平均成一个「融合爹」。

    用户 2026-10-01：整体都很优秀的模型融合后产生后代。
    数值取平均（用户说的「按平均数值就行」）；接法里那几个离散的 —— 走哪条路、
    从哪儿接出来、左右要不要对调 —— 按多数票，因为「motor」和「orient」平均一下
    是没有意义的。不搞加权、不搞花活。
    """
    if not rows:
        return None, None
    columns = {}
    for row in rows:
        for name, value in (row.get("genome") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                columns.setdefault(name, []).append(float(value))
    genome = {name: float(np.mean(values)) for name, values in columns.items() if values}

    chases = [row["chase"] for row in rows if isinstance(row.get("chase"), dict)]
    chase = None
    if chases:
        chase = {}
        for name in sorted(set().union(*[set(item) for item in chases])):
            values = [item[name] for item in chases if name in item]
            if not values:
                continue
            if all(isinstance(v, bool) for v in values):
                chase[name] = bool(sum(values) * 2 >= len(values))
            elif all(isinstance(v, str) for v in values):
                chase[name] = max(sorted(set(values)), key=values.count)
            elif all(isinstance(v, int) for v in values):
                chase[name] = int(round(float(np.mean(values))))
            else:
                chase[name] = float(np.mean([float(v) for v in values]))
    return (genome or None), chase''',
"rank -> \u7efc\u5408\u8bc4\u5206 + fuse")

# ---------- 2. 选爹不再让 follows 打头 ----------
rep('''    def key(row):
        return (0 if row.get("upright") else 1, -row["follows"]) + rank(row)''',
'''    def key(row):
        # 用户 2026-10-01：整体评分说了算。rank 第一眼是摔没摔、第二眼就是综合分，
        # 所以这里不再拿「追住几趟」压在最前面。
        return rank(row)''',
"\u9009\u7238\u6539\u770b\u7efc\u5408\u5206")

# ---------- 3. 繁衍循环：融合爹 + 球速递进 ----------
rep('''        for gen in range(args.start_gen, args.start_gen + args.generations):
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
                                 flee=args.flee, chase=pick_chase(parents, rng)))''',
'''        for gen in range(args.start_gen, args.start_gen + args.generations):
            # 球速递进（用户 2026-10-01：稍微提速 / 慢慢提速）。狗自己才 0.20~0.23
            # 米/秒，球一超过这个数谁都追不上，所以默认一步只加一丁点、还封了顶。
            flee = min(args.flee_max, args.flee + args.flee_ramp * (gen - args.start_gen))
            if gen == args.start_gen or flee != last_flee:
                print("  这一代球速 %.3f 米/秒（每代 +%.3f，上限 %.3f）"
                      % (flee, args.flee_ramp, args.flee_max), flush=True)
            last_flee = flee

            # 融合爹（用户 2026-10-01：整体都很优秀的模型融合后产生后代）
            fuse_ids = "、".join("第%s只" % row.get("index") for row in fused_from)
            if fused_genome is not None:
                print("  融合爹：综合分最高的 %d 只平均成一只（%s）" % (len(fused_from), fuse_ids),
                      flush=True)

            jobs = []
            for index in range(args.candidates):
                if fused_genome is not None and rng.random() < args.fuse_rate:
                    # 从融合爹身上变异来
                    genome, touched = sc.mutate(dict(fused_genome), 0, rng,
                                                genes=args.genes, sigma=args.sigma)
                    chase = (mutate_chase(dict(fused_chase), rng) if fused_chase
                             else pick_chase(parents, rng))
                else:
                    # 剩下的一部分还是从单个爹那儿抄，保住多样性，别让种群太快长成一个样
                    genome, touched = ev.make_child(parents, rng, args)
                    chase = pick_chase(parents, rng)
                genome = dict(genome)
                if rng.random() < .5:
                    # 一半的孩子把拐弯的力重新摇一个量级（原来那支被压到 0.0003，
                    # 不重摇就跳不出来）；另一半留着爹的值，让好的量级攒得住。
                    genome["avoidance_gain"] = random_avoidance(rng)
                touched = sorted(set(list(touched) + ["avoidance_gain"]))
                jobs.append(dict(gen=gen, index=index, seed=int(rng.integers(1000000)),
                                 genome=genome, touched=touched, seconds=args.seconds,
                                 chase_seconds=args.chase_seconds, curve=args.curve,
                                 flee=flee, chase=chase))''',
"\u7e41\u884d\u5faa\u73af")

rep('''            print("第 %d 代：生 %d 只、每只考 1 趟不动球（位置随机、每趟 %.0f 秒）+ 2 趟追着跑的球"
                  "（每趟 %.0f 秒，球沿随机曲线逃）+ 1 趟没球对照、%d 个进程"
                  % (gen, len(jobs), args.seconds, args.chase_seconds, args.workers), flush=True)''',
'''            print("第 %d 代：生 %d 只（其中 %.0f%% 从融合爹变异来）、每只考 1 趟不动球（位置随机、"
                  "每趟 %.0f 秒）+ 2 趟追着跑的球（每趟 %.0f 秒、球 %.3f 米/秒沿随机曲线逃）"
                  "+ 1 趟没球对照、%d 个进程"
                  % (gen, len(jobs), args.fuse_rate * 100., args.seconds, args.chase_seconds,
                     flee, args.workers), flush=True)''',
"\u6a2a\u5e45")

rep('''            parents = pick_parents(rows, args.keep_stable, args.keep_fast, args.follow_top)''',
'''            parents = pick_parents(rows, args.keep_stable, args.keep_fast, args.follow_top)
            fused_from = top_overall(parents, args.fuse_top)
            fused_genome, fused_chase = fuse(fused_from)''',
"\u6bcf\u4ee3\u6536\u5de5\u540e\u91cd\u7b97\u878d\u5408\u7238")

rep('''    parents = ev.read_jsonl(args.parents)
    if not parents:
        print("这份台账里挑不出爹：%s" % args.parents)
        return 1''',
'''    parents = ev.read_jsonl(args.parents)
    if not parents:
        print("这份台账里挑不出爹：%s" % args.parents)
        return 1
    last_flee = None
    fused_from = top_overall(parents, args.fuse_top)
    fused_genome, fused_chase = fuse(fused_from)''',
"\u5934\u4e00\u4ee3\u7684\u878d\u5408\u7238")

# ---------- 4. 新增旋钮 ----------
rep('''    ap.add_argument("--cross-rate", type=float, default=.5)''',
'''    ap.add_argument("--cross-rate", type=float, default=.5)
    ap.add_argument("--fuse-rate", type=float, default=.5,
                    help="多少比例的孩子从「融合爹」变异来（用户 2026-10-01：整体优秀的融合生后代）")
    ap.add_argument("--fuse-top", type=int, default=8, help="拿综合分最高的前几只来融合")
    ap.add_argument("--flee-ramp", type=float, default=.0,
                    help="每过一代球速再加这么多米/秒（用户 2026-10-01：慢慢提速）")
    ap.add_argument("--flee-max", type=float, default=.25, help="球速上限，别超过狗自己跑得动的速度")''',
"\u65b0\u65cb\u94ae")

# ---------- 5. import ----------
rep('''from tools import evolve as ev                                          # noqa: E402''',
'''from tools import evolve as ev                                          # noqa: E402
from tools import screen_candidates as sc                               # noqa: E402''',
"\u8865 import")

# ---------- 6. 每只那一行加上综合分 ----------
rep('''                    print("%5d  %-14s 追住 %d/%d 趟''',
'''                    print("%5d  %-14s 综合 %5.3f  追住 %d/%d 趟''',
"\u6bcf\u53ea\u90a3\u4e00\u884c\u52a0\u7efc\u5408\u5206")

rep('''                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",''',
'''                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             overall(row),''',
"\u6bcf\u53ea\u90a3\u4e00\u884c\u7684\u53c2\u6570")

io.open(P, "w", encoding="utf-8", newline="\n").write(t)
print("\n\u5168\u90e8\u6539\u5b8c")