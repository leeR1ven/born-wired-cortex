# -*- coding: utf-8 -*-
"""让一只动物一直走，看它到底为什么慢下来。

    python tools/probe_long_walk.py --seconds 60 --clean --label birth
    python tools/probe_long_walk.py --seconds 60 --genome-file artifacts/题1_一直走_末段最远.jsonl --index 0

每五秒打一行：它在哪、这一段走了多远（所以看得出来是匀速还是衰减），以及脑
子里那几个「想动 / 累了 / 想歇」的细胞各是多少 —— 一个动物停下来，要么是路
上有东西，要么是这几个细胞把它按住了，这一行数字就是用来分开这两件事的。

``--clean`` 把场地里那些摆设（坎、方块、柱子……）全部挪到六十米外，只剩地板
和墙：这样「走不远」就不可能是撞上东西。地板是无限平面，四面墙在 ±3.5 米，
所以不撞墙、不撞东西时它爱走多远走多远。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank as tb                                        # noqa: E402

FAR = (60., 60., -8.)


WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")


def clean_props():
    return {name: FAR for name in tb.SCENERY}


def open_props():
    """Nothing but floor: the props *and* the room's four walls, out of the way.

    The arena is a 7 by 7 metre room, so "walk twenty metres" is not something
    an animal can do inside it no matter how well it walks - it hits a wall at
    3.5 m.  The floor is an infinite plane, so moving the walls away turns the
    same arena into an open field where distance is limited by the walker.
    """
    props = dict(clean_props())
    props.update({name: FAR for name in WALLS})
    return props


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="autonomous")
    ap.add_argument("--terrain", default="origin")
    ap.add_argument("--clean", action="store_true", help="把摆设挪走，只剩地板和墙")
    ap.add_argument("--open", action="store_true", help="摆设和四面墙全挪走，只剩无限地板")
    ap.add_argument("--genome-file", default=None, help="从一份台账里取第 --index 只的基因")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--label", default="birth")
    ap.add_argument("--out", default=None, help="把逐五秒的读数写成一份 jsonl")
    args = ap.parse_args(argv)

    genome = {}
    if args.genome_file:
        rows = [json.loads(line) for line in Path(args.genome_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if row.get("genome")]
        genome = rows[args.index]["genome"]
        args.label = rows[args.index].get("label") or args.label
    ctx = tb.context(genome, args.seed)
    props = open_props() if args.open else (clean_props() if args.clean else None)
    print("%s：%g 秒，%s场地，版本 %s" % (args.label, args.seconds,
                                        "开阔的（连墙都挪走了）" if args.open
                                        else ("清空的" if args.clean else "有摆设的"),
                                        ctx["revision"]))
    started = time.perf_counter()
    result = tb.run_seed(seed=ctx["seed"], duration=args.seconds, model_path=ctx["model_path"],
                         scenario=args.scenario, terrain_start=args.terrain,
                         controller_parameters_for_seed=ctx["parameters"], props=props)
    wall = time.perf_counter() - started
    timeline = result["timeline"]
    metrics = result["metrics"]
    print("状态 %s  全程直立 %s  用时 %.0f 秒" % (result["status"], result["behavior_pass"], wall))
    print("%6s %8s %8s %9s %8s %8s %7s %7s %7s %7s" % (
        "秒", "x", "y", "本段(米)", "本段速度", "累计(米)", "想动", "累了", "想歇", "起步"))
    rows = []
    previous = None
    step = max(1, int(round(5. / .5)))
    for sample in timeline[::step]:
        position = sample["position"]
        moved = None
        if previous is not None:
            moved = ((position[0] - previous["position"][0]) ** 2
                     + (position[1] - previous["position"][1]) ** 2) ** .5
        travelled = (position[0] ** 2 + position[1] ** 2) ** .5
        row = dict(t=sample["time"], x=position[0], y=position[1], moved_m=moved,
                   travelled_m=travelled, drive=sample["drive"], fatigue=sample["fatigue"],
                   rest=sample["rest"], initiation=sample["initiation"],
                   eye_angle_rad=sample.get("eye_angle_rad"))
        rows.append(row)
        print("%6.1f %8.3f %8.3f %9s %8s %8.3f %7.3f %7.3f %7.3f %7.3f" % (
            row["t"], row["x"], row["y"],
            "-" if moved is None else "%.3f" % moved,
            "-" if moved is None else "%.3f" % (moved / 5.),
            travelled, row["drive"], row["fatigue"], row["rest"], row["initiation"]))
        previous = sample
    print("位移 %.3f 米，路程 %.3f 米，平均 %.3f 米/秒，最快 %.3f，直立最低 %.3f，直度 %.3f"
          % (metrics["total_displacement_m"], metrics["horizontal_path_m"],
             metrics["mean_speed_mps"], metrics["max_speed_mps"],
             metrics["minimum_up_z"], metrics["straightness"]))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                  encoding="utf-8", newline="\n")
        print("逐五秒读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())