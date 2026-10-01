# -*- coding: utf-8 -*-
"""扫一遍「眼睛偏到哪边、身子就往哪边拐」这套接法的参数，看这套接法到底能不能朝球走。

一次只跑一趟球（默认球在左前方 0.6 弧度、1.5 米），把 turn_gain / gain / flip /
source / pivot 排成网格，一个个读回来：最后离球多远、正对着球多少、摔没摔、走了多远、
收尾时两个转向细胞还亮着多少。

    python tools/sweep_chase_wiring.py --seconds 6 --turn-gains 0.01,0.02,0.04,0.08
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import chase_red_ball as C                                          # noqa: E402


def one(genome, chase, seconds, bearing, distance, seed, spec):
    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
    got = C.measure(body, brain, eyes, geom, seconds, bearing, distance, seed=seed)
    rates = [float(np.mean(brain.network.rates_at(brain.groups[name])))
             for name in ("chase_turn_left", "chase_turn_right")]
    eyes.close()
    return got, rates


def numbers(text):
    return [float(part) for part in str(text).replace(" ", "").split(",") if part]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--champion", default=str(C.CHAMPION))
    ap.add_argument("--seconds", type=float, default=6.)
    ap.add_argument("--bearing", type=float, default=.6)
    ap.add_argument("--distance", type=float, default=1.5)
    ap.add_argument("--turn-gains", default="0.01,0.02,0.04,0.08,0.16")
    ap.add_argument("--gains", default="1.0")
    ap.add_argument("--sources", default="position")
    ap.add_argument("--pivots", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    genome = dict(C.load_champion(Path(args.champion))["genome"])
    spec = C.gaze_spec()
    per = 16
    pivots = [int(v) for v in numbers(args.pivots)] or [per//2]

    rows = []
    for turn_gain, gain, source, pivot, flip in itertools.product(
            numbers(args.turn_gains), numbers(args.gains),
            str(args.sources).split(","), pivots, (False, True)):
        chase = dict(turn_gain=turn_gain, gain=gain, source=source, pivot=pivot,
                     flip=flip)
        started = time.perf_counter()
        try:
            got, rates = one(genome, chase, args.seconds, args.bearing, args.distance,
                             args.seed, spec)
            row = dict(chase=chase, settled=got["settled"], facing=got["facing"],
                       range_end=got["range_end"], travelled_m=got["travelled_m"],
                       straightness=got["straightness"], lowest_up_z=got["lowest_up_z"],
                       turn_cells=rates, heading0=got["heading0"],
                       heading_end=got["heading_end"], heading_last=got["heading_last"])
        except Exception as exc:
            row = dict(chase=chase, error="%s: %s" % (type(exc).__name__, exc))
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        print("拧髋 %5.3f 增益 %.1f %-8s 正前方偏在 %2d 号%s | 最后离球 %5.2f 米  正对球 %+5.2f  "
              "走了 %5.2f 米 直度 %.2f 最低 %s  转向细胞 %s"
              % (turn_gain, gain, source, pivot, " 对调" if flip else "    ",
                 row.get("settled", float("nan")), row.get("facing", float("nan")),
                 row.get("travelled_m", float("nan")), row.get("straightness", float("nan")),
                 "%.2f" % row["lowest_up_z"] if "lowest_up_z" in row else "--",
                 np.round(row["turn_cells"], 2) if "turn_cells" in row else row.get("error", "")),
              flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())