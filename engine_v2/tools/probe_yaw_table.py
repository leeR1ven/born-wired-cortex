# -*- coding: utf-8 -*-
"""把接线用的列->角度表（两只眼）整行打出来，找右眼反方向的原因。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402


def main():
    table = W.load_table()
    yaw, pitch = W.retina_angles(table)
    for which, side in ((0, "左"), (1, "右")):
        line = yaw[which]
        print("\n%s眼 列->方位（共 %d 列）:" % (side, len(line)))
        for start in range(0, len(line), 12):
            chunk = line[start:start+12]
            print("  列%2d-%2d: %s" % (start, start+len(chunk)-1,
                  " ".join("  nan " if not np.isfinite(v) else "%+6.3f" % v for v in chunk)))
    print("\n每只眼的细胞数（列上）：")
    red = W.red_cells(table)
    group = W.eye_of(red, table)
    row, col = W.cell_rows_columns(red, table)
    for which, side in ((0, "左"), (1, "右")):
        mine = group == which
        have = np.zeros(48, bool)
        have[np.unique(col[mine])] = True
        print("  %s眼：列 %s" % (side, "".join("#" if f else "." for f in have)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())