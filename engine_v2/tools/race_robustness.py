# -*- coding: utf-8 -*-
"""同一只动物换几颗随机种子重跑，看它的成绩是不是运气。

    python tools/race_robustness.py --ledger artifacts/题1f_最快64只.jsonl --top 6 --seeds 5

十秒窗口只跑一次，容易把"这只正好起脚顺"当成"这只走得快"。这里把前几名拿出来，
每只换 `--seeds` 颗种子重跑同一件事，然后报中位数、最差、最好、以及有没有摔倒。
挑父代的时候用中位数，不要用单次最快。
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
    return dict(index=index, seed=seed, travelled_m=result["metrics"]["total_displacement_m"],
                path_m=result["metrics"]["horizontal_path_m"],
                min_up_z=result["metrics"]["minimum_up_z"],
                straightness=result["metrics"]["straightness"],
                upright=bool(result["behavior_pass"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=10.)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.ledger).read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [row for row in rows if row.get("status") == "ok" and row.get("genome")]
    top = sorted(ok, key=lambda row: -row["travelled_m"])[:args.top]
    jobs = [(row["index"], row["genome"], seed, args.seconds)
            for row in top for seed in range(1, args.seeds + 1)]
    print("%s 的前 %d 名，每只换 %d 颗种子、每颗 %.0f 秒、开阔场地"
          % (args.ledger, len(top), args.seeds, args.seconds))
    print("%6s %9s %9s %9s %9s %8s %9s" % ("编号", "中位(米)", "最差(米)", "最好(米)", "台账那次", "摔倒", "中位直度"))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, jobs))
    by_index = {}
    for row in results:
        by_index.setdefault(row["index"], []).append(row)
    for row in top:
        group = sorted(by_index[row["index"]], key=lambda item: item["travelled_m"])
        values = [item["travelled_m"] for item in group]
        mid = values[len(values)//2]
        print("%6d %9.3f %9.3f %9.3f %9.3f %8d %9.3f"
              % (row["index"], mid, values[0], values[-1], row["travelled_m"],
                 sum(1 for item in group if not item["upright"]),
                 sorted(item["straightness"] for item in group)[len(group)//2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())