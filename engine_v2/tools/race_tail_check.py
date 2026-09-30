# -*- coding: utf-8 -*-
"""第一题的复验：十秒窗口里走最远的那几只，真的是「一直在走」吗？

    python tools/race_tail_check.py --ledger artifacts/题1d_十秒赛.jsonl --top 12 --seed 1

固定十秒窗口只有一个数：十秒走了多远。可一个动物完全可以是「冲四秒、站六秒」，
十秒窗口照样好看，但它并不是走得好的那只。这个工具把前几名重跑一遍，另外记两
个数出来：

* ``末段`` —— 最后两秒它还在不在走（米/秒）。站着不动的那几只，这里接近 0。
* ``僵住`` —— 窗口里「想动」细胞低于 0.2 的步数占比。这是它走走停停的直接证据。
* ``直立最低`` —— 这一趟最低的直立程度，用来看有没有快倒没倒。

报告里的每只都用同一颗种子重跑，所以同名次之间的数字可以直接比。
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAR = (60., 60., -8.)
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")


def open_props():
    """只剩无限地板：摆设和四面墙全挪走，读数只反映走得怎么样。"""
    import tools.taskbank as tb
    props = {geom: FAR for geom in tb.SCENERY}
    props.update({name: FAR for name in WALLS})
    return props


def one(payload):
    index, genome, seed, seconds = payload
    from tools import taskbank as tb
    ctx = tb.context(genome, seed)
    result = tb.run_seed(seed=ctx["seed"], duration=seconds, model_path=ctx["model_path"],
                         scenario="autonomous", terrain_start="origin",
                         controller_parameters_for_seed=ctx["parameters"], props=open_props())
    timeline = result["timeline"]
    metrics = result["metrics"]
    tail = float("nan")
    recent = [sample for sample in timeline if sample["time"] >= seconds - 2.]
    if len(recent) >= 2:
        first, last = recent[0]["position"], recent[-1]["position"]
        span = max(1e-6, recent[-1]["time"] - recent[0]["time"])
        tail = ((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2) ** .5 / span
    frozen = sum(1 for sample in timeline if sample["drive"] < .2)
    return dict(index=index, travelled_m=metrics["total_displacement_m"],
                tail_mps=tail, frozen=frozen, steps=len(timeline),
                min_up_z=metrics["minimum_up_z"], straightness=metrics["straightness"],
                upright=result["behavior_pass"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10.)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.ledger).read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [row for row in rows if row.get("status") == "ok" and row.get("genome")]
    top = sorted(ok, key=lambda row: -row["travelled_m"])[:args.top]
    print("复验 %s 的前 %d 名：开阔场地、固定 %.0f 秒、种子 %d"
          % (args.ledger, len(top), args.seconds, args.seed))
    print("%6s %10s %10s %10s %8s %9s %8s"
          % ("编号", "十秒走(米)", "末两秒(米/秒)", "僵住步", "直度", "直立最低", "全程直立"))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, [(row["index"], row["genome"], args.seed, args.seconds)
                                      for row in top]))
    for row in results:
        print("#%-5d %10.4f %10.4f %5d/%-4d %8.3f %9.3f %8s"
              % (row["index"], row["travelled_m"], row["tail_mps"], row["frozen"], row["steps"],
                 row["straightness"], row["min_up_z"], row["upright"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())