# -*- coding: utf-8 -*-
"""「朝球走 + 追着跑的球」这道题 —— 用户 2026-10-01。

两场考试一起考，判分全都留着：

一、静止球（老判分）：
    球不动，**随机**摆在狗看得见的地方（左右 ±0.55 弧度、3.0~4.0 米，位置从种子里摇），
    只跑 1 趟、6 秒。主判分是「这一趟离球最近多少米」。
    用户 2026-10-01：模型已经真的会追红球了，红球就不用换那么多种摆法各测一遍，随机摆
    一次就够 —— 省下来的时间拿去多迭代几代。

二、追着跑的球（这一轮的重点）：
    球一开始在两眼正前方 2.5 米，然后一直逃 —— 每一拍都往背对狗的方向挪，
    只是叠了一个缓慢摆动的小偏角，所以跑的是一条弯线；地图无限大，没有场地边可绕
    （用户 2026-10-01：始终远离狗、跑的是弯线）。左、右各一趟，每趟 20 秒
    （用户：每个模型要考更久，才能确定是真的能追着球）。球比狗慢一点，会追的狗越追越近。
    主判分：「20 秒结束时落后几米」「球有几成时间还在正前方」。

三、对照：把球搬走，同样跑一趟 20 秒。球搬走之后哪个位置的世界都一样，所以只跑一趟就够，
        不用每个位置各测一遍（用户 2026-09-30）。
    球在不在、行为一样 = 它根本没在看球。

判分（全都留着，一个都不丢）：
    upright / moving / travelled / straightness    走路本身
    closest / reached / covered                    静止球那趟（只 1 趟了）
    chase_settled    追着跑的球，跑完平均落后多少米（越小越好，主判分）
    chase_covered    2 趟里追住了几趟（跑完落后 < 1 米）
    both             左、右两趟都追住了（用户 2026-09-30：同一只狗两边都要试，
                     不然会漏掉「只认某一边」的模型。改成由追球那两趟来判，
                     因为静止球那趟只剩随机一次了）
    chase_in_view    球有几成时间还在正前方 ±0.6 弧度内
    chase_approach   追了 20 秒，差距缩小了多少米（正数 = 越追越近）
    blind            球搬走之后落点几乎没变（< 0.05 米）= 根本没在看球
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import chase_red_ball as C                                          # noqa: E402

BEARINGS = (-.55, .55)        # 静止球随机摆在这个范围里（正 = 狗的左前方；负 = 右前方）
DISTANCES = (3.0, 4.0)        # 距离也在这个区间里随机取
CENTER_GAP = .15              # 离正前方至少这么远：摆正中间考不出会不会拐弯
SECONDS = 6.
FACE_BAR = .5
KEEP_BAR = .35
NEAR_BAR = 1.5
CATCH_BAR = 1.2
FALLEN = .3
BLIND_BAR = .05
REACH_BAR = 1.0               # 静止球：贴到这么近 = 这一趟「走到了球跟前」
MOVE_BAR = 2.0

# 追着跑的球（用户 2026-10-01）
CHASE_SECONDS = 20.           # 每趟考多少秒（球匀速逃，20 秒够看出会不会追；
                              # 再久球就跑出视野了，后面的时间白跑）
CHASE_BEARINGS = (.5, -.5)    # 球起初在左前方 / 右前方
CHASE_DISTANCE = 2.5          # 球起初离眼睛多远（米）
CHASE_FLEE = .6               # 球逃跑的速度默认值（米/秒）；真正跑多少由 --flee 说了算
CHASE_ESCAPE = 14.           # 球离狗超过这么远就不再挪（给读数一个上界）
CHASE_CURVE = 1.0             # 随机曲线的弯度（0 = 老版直线逃）
CHASE_NEAR = 1.5              # 跑完落后不到这么近 = 这一趟「追住了」（球匀速逃，追住了就贴在 1 米多）
VIEW_BAR = .6                 # 球偏在正前方这么多弧度以内算「还在视野里」

_SPEC = None


def gaze():
    global _SPEC
    if _SPEC is None:
        _SPEC = C.gaze_spec()
    return _SPEC


def _trip(genome, seed, spec, bearing, distance, seconds, chase, flee=0., curve=0.):
    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
    got = C.measure(body, brain, eyes, geom, seconds, bearing, distance,
                    seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)
    eyes.close()
    return got


def _row(got, bearing, distance, flee):
    fell = got["lowest_up_z"] < FALLEN
    headings = np.abs(np.asarray(got["headings"], dtype=float))
    kept = float(np.mean(headings <= KEEP_BAR))
    moving = bool(got["travelled_m"] >= MOVE_BAR)
    start = np.asarray(got["start_xy"], dtype=float)
    ball0 = np.asarray(got["ball0_xy"], dtype=float)
    walk = np.asarray(got["end_xy"], dtype=float) - start
    to_ball = ball0 - start
    scale = float(np.linalg.norm(walk))*float(np.linalg.norm(to_ball))
    toward_deg = (float("nan") if scale < 1e-9
                  else float(np.degrees(np.arccos(np.clip(float(walk @ to_ball)/scale, -1., 1.)))))
    closest = float(got["range_min"])
    reached = bool(closest <= REACH_BAR and moving and not fell)
    tail = int(round(1./C.DT))
    settled = float(np.mean(np.asarray(got["ranges"], dtype=float)[-tail:]))
    return dict(bearing=bearing, distance=distance, flee=flee, facing=got["facing"],
                kept=kept, closest=closest, reached=reached, toward_deg=toward_deg,
                range_min=closest, range0=float(got["range0"]),
                settled=settled, approach=float(got["range0"]) - settled,
                in_view=float(np.mean(headings <= VIEW_BAR)),
                travelled_m=got["travelled_m"], straightness=got["straightness"],
                heading_end=got["heading_end"], lowest_up_z=got["lowest_up_z"],
                upright=not fell, moving=moving,
                turned=bool(reached and toward_deg <= 25.), caught=reached,
                kept_in_front=bool(not fell and moving and settled <= 2.0),
                end_xy=got["end_xy"], shifted_m=float("nan"), blind=False)


def _chase_row(got, bearing, flee=CHASE_FLEE, fell=None):
    headings = np.abs(np.asarray(got["headings"], dtype=float))
    tail = int(round(1./C.DT))
    settled = float(np.mean(np.asarray(got["ranges"], dtype=float)[-tail:]))
    low = float(got["lowest_up_z"])
    moved = bool(got["travelled_m"] >= MOVE_BAR)
    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=flee,
                settled=settled, mean=float(np.mean(np.asarray(got["ranges"], dtype=float))),
                range_min=float(got["range_min"]),
                range0=float(got["range0"]), approach=float(got["range0"]) - settled,
                in_view=float(np.mean(headings <= VIEW_BAR)),
                travelled_m=got["travelled_m"], straightness=got["straightness"],
                lowest_up_z=low, upright=bool(low >= FALLEN), moving=moved,
                kept_in_front=bool(moved and settled <= CHASE_NEAR),
                end_xy=got["end_xy"], shifted_m=float("nan"), blind=False,
                ball_track=got["ball_track"],
                track=[[float(p[0]), float(p[1])] for p in
                       np.asarray(got["start_xy"], dtype=float).reshape(1, 2)])


def exam(genome, seed, seconds=SECONDS, spec=None, loud=False, aware=True, chase=None,
         flee=CHASE_FLEE, chase_seconds=CHASE_SECONDS, curve=CHASE_CURVE):
    """一个模型考一场：静止球 1 趟（随机摆）+ 追着跑的球 2 趟 + 没球对照 1 趟。"""
    spec = gaze() if spec is None else spec
    rows = []
    # 静止球只跑 1 趟，位置从种子里摇 —— 同一只狗每次考都摆同一个位置，不会这次考左、
    # 下次考右。用户 2026-10-01：模型已经真的会追红球了，红球不用换那么多种摆法各测
    # 一遍，随机摆一次就够。
    rng = np.random.default_rng(int(seed)*7919 + 13)
    bearing = float(np.round(rng.uniform(*BEARINGS), 3))
    if abs(bearing) < CENTER_GAP:          # 摆正中间考不出会不会拐，往外挪一点
        bearing = float(np.copysign(CENTER_GAP, bearing if bearing else 1.))
    distance = float(np.round(rng.uniform(*DISTANCES), 3))
    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
    got = C.measure(body, brain, eyes, geom, seconds, bearing, distance,
                    seed=seed, ball=True)
    row = _row(got, bearing, distance, 0.)
    rows.append(row)
    if loud:
        print("   静止球在 %+.2f 弧度、%.1f 米：最近贴到 %.2f 米%s、位移离球 %4.1f 度、"
              "走了 %.2f 米"
              % (bearing, distance, row["closest"],
                 "  到了" if row["reached"] else "", row["toward_deg"],
                 row["travelled_m"]), flush=True)
    eyes.close()

    # 2026-10-01 试过让这三趟共用一只狗（省两次建狗），结果追球读数全变了 —— 建狗这点
    # 开销本来就可以忽略，不值得为它改行为。所以还是每趟各建一只。
    # 球搬走之后，狗面对的世界每个位置都一样，所以对照只跑一次就够（用户 2026-09-30）。
    chase_rows, no_ball = [], None
    if aware:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],
                         CHASE_DISTANCE, seed=seed, ball=False)
        no_ball = np.asarray(free["end_xy"], dtype=float)
        eyes.close()
    for bearing in CHASE_BEARINGS:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,
                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)
        row = _chase_row(got, bearing, flee)
        row["ball_track"] = got["ball_track"]
        row["track"] = [[float(p[0]), float(p[1])]
                        for p in np.asarray(got["start_xy"], dtype=float).reshape(1, 2)]
        if aware and no_ball is not None:
            row["shifted_m"] = float(np.linalg.norm(np.asarray(got["end_xy"]) - no_ball))
            row["blind"] = bool(row["shifted_m"] < BLIND_BAR)
        chase_rows.append(row)
        if loud:
            print("   追着跑的球、从 %+.2f 弧度起步：%.0f 秒后落后 %.2f 米（最近 %.2f）、"
                  "球在正前方 %3.0f%% 的时间、缩小了 %.2f 米%s"
                  % (bearing, chase_seconds, row["settled"], row["range_min"],
                     row["in_view"]*100., row["approach"],
                     "  追住了" if row["kept_in_front"] else ""), flush=True)
        eyes.close()
    rows += chase_rows

    still = [row for row in rows if row["flee"] == 0.]
    reached = [row for row in still if row["reached"]]
    # 「两边都会」改由追球那两趟来判（静止球只剩随机一次，判不了左右）。
    caught = [row for row in chase_rows if row["kept_in_front"] and row["upright"]]
    left_done = any(row["bearing"] > 0 for row in caught)
    right_done = any(row["bearing"] < 0 for row in caught)
    near = [row for row in chase_rows if row["kept_in_front"] and row["upright"]]
    return dict(
        rows=rows, chase_rows=chase_rows,
        score=float(np.mean([row["closest"] for row in still])),
        best=float(np.min([row["closest"] for row in still])),
        kept=float(np.mean([row["kept"] for row in still])),
        travelled_m=float(np.mean([row["travelled_m"] for row in still])),
        straightness=float(np.mean([row["straightness"] for row in still])),
        toward_deg=float(np.nanmean([row["toward_deg"] for row in still])),
        covered=len(reached), trials=len(still),
        moving=sum(1 for row in still if row["moving"]),
        both=bool(left_done and right_done),
        chase_trials=len(chase_rows), chase_covered=len(near),
        chase_settled=float(np.mean([row["settled"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_mean=float(np.mean([row["mean"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_best=float(np.min([row["settled"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_min=float(np.min([row["range_min"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_in_view=float(np.mean([row["in_view"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_approach=float(np.mean([row["approach"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_travelled=float(np.mean([row["travelled_m"] for row in chase_rows])) if chase_rows else float("nan"),
        flee_trials=len(chase_rows), flee_covered=len(near),
        flee_settled=float(np.mean([row["settled"] for row in chase_rows])) if chase_rows else float("nan"),
        flee_kept=float(np.mean([row["in_view"] for row in chase_rows])) if chase_rows else float("nan"),
        caught=len(near),
        blind=bool(aware and chase_rows and all(row["blind"] for row in chase_rows)),
        upright=bool(all(row["upright"] for row in rows)))


def closest_of(row):
    return row["closest"]