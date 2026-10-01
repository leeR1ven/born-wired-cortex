# -*- coding: utf-8 -*-
"""扫「眼位那一排 → orienting（凑过去看）→ steering → 髋」这条路。

拐弯的力是基因 avoidance_gain，往哪边拐由 flip 定。球摆在左前方 0.6 弧度、1.2 米，
跑几秒看：狗是不是把球转到了正前方。

    python tools/sweep_orient_route.py --seconds 3
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


def numbers(text):
    return [float(part) for part in str(text).replace(" ", "").split(",") if part]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--champion", default=str(C.CHAMPION))
    ap.add_argument("--seconds", type=float, default=3.)
    ap.add_argument("--bearing", type=float, default=.6)
    ap.add_argument("--distance", type=float, default=1.2)
    ap.add_argument("--avoidance", default="0.05,0.15,0.3")
    ap.add_argument("--weights", default="0.5,1.0,2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    genome = dict(C.load_champion(Path(args.champion))["genome"])
    spec = C.gaze_spec()
    rows = []
    for avoidance, weight, flip in itertools.product(
            numbers(args.avoidance), numbers(args.weights), (False, True)):
        bench = dict(genome, avoidance_gain=avoidance)
        chase = dict(route="orient", gain=weight, source="position", flip=flip)
        started = time.perf_counter()
        try:
            body, brain, eyes, geom = C.build(bench, chase, seed=args.seed, spec=spec)
            got = C.measure(body, brain, eyes, geom, args.seconds, args.bearing, args.distance,
                            seed=args.seed)
            orient = [float(np.mean(brain.network.rates_at(brain.groups["orienting"][k])))
                      for k in (0, 1)]
            steer = [float(np.mean(brain.network.rates_at(brain.groups["steering"][k])))
                     for k in (0, 1)]
            eyes.close()
            row = dict(avoidance=avoidance, weight=weight, flip=flip, settled=got["settled"],
                       facing=got["facing"], travelled_m=got["travelled_m"],
                       lowest_up_z=got["lowest_up_z"], orient=orient, steering=steer,
                       heading0=got["heading0"], heading_end=got["heading_end"])
        except Exception as exc:
            row = dict(avoidance=avoidance, weight=weight, flip=flip,
                       error="%s: %s" % (type(exc).__name__, exc))
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        print("拐弯力 %4.2f 眼位总增益 %4.1f%s | 最后离球 %5.2f 米  正对球 %+5.2f  走了 %5.2f 米  "
              "最低直立 %s  凑过去细胞 %s  转向细胞 %s"
              % (avoidance, weight, " 对调" if flip else "    ",
                 row.get("settled", float("nan")), row.get("facing", float("nan")),
                 row.get("travelled_m", float("nan")),
                 "%.2f" % row["lowest_up_z"] if "lowest_up_z" in row else "--",
                 np.round(row["orient"], 2) if "orient" in row else row.get("error", ""),
                 np.round(row["steering"], 2) if "steering" in row else ""), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())