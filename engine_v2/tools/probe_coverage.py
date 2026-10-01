# -*- coding: utf-8 -*-
"""看位置野表覆盖了每只眼的哪些列、哪些行（用 wide 表）。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402


def report(path):
    table = W.load_table(path)
    shape = table["eye_shape"]
    rows, columns = int(shape[1]), int(shape[2])
    red = W.red_cells(table)
    group = W.eye_of(red, table)
    row, col = W.cell_rows_columns(red, table)
    yaw, pitch = W.retina_angles(table)
    print("\n%s：红色通道值得接线的细胞 %d 个" % (Path(path).name, len(red)))
    for which, side in ((0, "左"), (1, "右")):
        mine = group == which
        have_col = np.zeros(columns, bool)
        have_row = np.zeros(rows, bool)
        have_col[np.unique(col[mine])] = True
        have_row[np.unique(row[mine])] = True
        print("  %s眼 列：%s" % (side, "".join("#" if f else "." for f in have_col)))
        print("  %s眼 行：%s" % (side, "".join("#" if f else "." for f in have_row)))
        good = np.isfinite(yaw[which])
        print("  %s眼 列->方位：%s" % (side, " ".join("nan" if not good[c] else "%+.2f" % yaw[which][c]
                                              for c in range(0, columns, 3))))


if __name__ == '__main__':
    for path in (sys.argv[1:] or [str(W.TABLE), str(ROOT / "artifacts" / "红球位置野_红_宽.json")]):
        report(path)