# -*- coding: utf-8 -*-
"""视网膜行列 -> 每只眼自己的角度：把接线用的那张标定表打印出来、留下来。

位置野表（artifacts/红球位置野_红_*.json）记的是「球摆在头坐标系的哪个方位时，哪个细胞最亮」。
接线要的是「这个细胞答话时，球偏在这只眼自己的哪儿」，中间靠 cell_rows_columns + 每列每行取中位
换来（tools/wire_red_gaze.py 的 retina_angles）。这个工具把那一步的成色量出来：同一列里的细胞
给出的角度是否一致（不一致这张表就不能用）、以及行列到角度的曲线是不是中间密、边缘疏。

    python tools/measure_retina_angles.py --table artifacts\红球位置野_红_大.json \
        --out artifacts\视网膜角度表.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wire_red_gaze as W                                                   # noqa: E402
from tools import eye_geometry as G                                         # noqa: E402

BIG = ROOT / "artifacts" / "红球位置野_红_大.json"


def spread_per_number(seen, number, count):
    """每一个列号（行号）里的角度有多散：个数、中位、标准差。"""
    rows = []
    for value in range(count):
        taken = seen[number == value]
        if len(taken):
            rows.append((value, int(len(taken)), float(np.median(taken)), float(np.std(taken))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=str(BIG))
    ap.add_argument("--floor", type=float, default=.15, help="位置野峰值低于这个的细胞不接（同接线）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    table = W.load_table(args.table)
    shape = table["eye_shape"]
    eyes, rows, columns = int(shape[0]), int(shape[1]), int(shape[2])
    places = np.asarray(table["places"], dtype=float)
    best = np.asarray(table["cells"]["retinal_opponent"]["best_place"], dtype=int)
    red = W.red_cells(table, args.floor)
    group = W.eye_of(red, table)
    row, column = W.cell_rows_columns(red, table)
    yaw, pitch = W.retina_angles(table, args.floor)

    print("表 %s：%d 个位置、%d 只眼、每只眼 %d 行 x %d 列"
          % (Path(args.table).name, len(places), eyes, rows, columns))
    print("球离 %.2f 米、球高 %.2f 米（tools/eye_geometry.py）；红色通道值得接线的细胞 %d 个"
          % (float(table["options"]["distance"]), G.BALL_Z, len(red)))
    worst_column = 0.
    worst_row = 0.
    for which, side in ((0, "左"), (1, "右")):
        mine = np.nonzero(group == which)[0]
        seen = G.seen_by_many(places[best[red[mine]]], "left" if which == 0 else "right")
        by_column = spread_per_number(seen[:, 0], column[mine], columns)
        by_row = spread_per_number(seen[:, 1], row[mine], rows)
        worst_column = max(worst_column, max(entry[3] for entry in by_column))
        worst_row = max(worst_row, max(entry[3] for entry in by_row))
        print("\n%s眼：方位 %+.3f ~ %+.3f 弧度，高低 %+.3f ~ %+.3f 弧度"
              % (side, yaw[which].min(), yaw[which].max(), pitch[which].min(), pitch[which].max()))
        print("  同一列里的方位标准差最大 %.4f 弧度（第 %d 列，%d 个细胞），%d 列都有细胞"
              % (max(entry[3] for entry in by_column),
                 max(by_column, key=lambda entry: entry[3])[0],
                 max(by_column, key=lambda entry: entry[3])[1], len(by_column)))
        print("  同一行里的高低标准差最大 %.4f 弧度（第 %d 行），%d 行都有细胞"
              % (max(entry[3] for entry in by_row),
                 max(by_row, key=lambda entry: entry[3])[0], len(by_row)))
        print("  列 -> 方位（中位）：%s" % " ".join("%.2f" % entry[2] for entry in by_column))
        print("  行 -> 高低（中位）：%s" % " ".join("%.2f" % entry[2] for entry in by_row))
    print("\n结论：同一列（行）内角度标准差最大 %.4f 弧度 = %.2f 度；"
          "一个细胞 %.2f 度（相机上下 60 度 / %d 行）"
          % (max(worst_column, worst_row), np.degrees(max(worst_column, worst_row)),
             np.degrees(np.radians(60.)/rows), rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(table=Path(args.table).name, floor=args.floor,
                 yaw=[[None if np.isnan(v) else round(float(v), 4) for v in line] for line in yaw],
                 pitch=[[None if np.isnan(v) else round(float(v), 4) for v in line] for line in pitch]),
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("标定表 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())