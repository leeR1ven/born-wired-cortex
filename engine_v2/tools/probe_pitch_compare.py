# -*- coding: utf-8 -*-
"""对比两种扫场范围算出来的「行->高低」标定。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402


def main():
    for name in ("红球位置野_红.json", "红球位置野_红_大.json", "红球位置野_红_宽.json"):
        path = ROOT / "artifacts" / name
        table = W.load_table(path)
        yaw, pitch = W.retina_angles(table)
        places = np.asarray(table["places"], dtype=float)
        best = np.asarray(table["cells"]["retinal_opponent"]["best_place"], dtype=int)
        red = W.red_cells(table)
        row, col = W.cell_rows_columns(red, table)
        b = places[best[red]][:, 0]
        print("\n%s（方位 %+.2f~%+.2f、高低 %+.2f~%+.2f）"
              % (name, places[:, 0].min(), places[:, 0].max(),
                 places[:, 1].min(), places[:, 1].max()))
        for which, side in ((0, "左"), (1, "右")):
            mine = (W.eye_of(red, table) == which)
            line = pitch[which]
            print("  %s眼 行->高低：%s" % (side, " ".join(
                "%+.2f" % v if np.isfinite(v) else " nan " for v in line[::3])))
            # 同一行里，样本的方位离正中越远，算出的高低差多少
            mid, far = [], []
            for r in range(int(table["eye_shape"][1])):
                taken = np.isfinite(line[r])
                here = b[mine & (row == r)]
                if not taken or not len(here):
                    continue
                if np.abs(here).max() > .45:
                    far.append(r)
            print("  %s眼 有细胞落在 |方位|>0.45 的行数：%d" % (side, len(far)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())