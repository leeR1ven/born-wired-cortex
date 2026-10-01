# -*- coding: utf-8 -*-
"""把视网膜那张表『照原样画』和『按真实位置还原画』并排存出来。

表本身是：画面中段格子密、边缘格子疏（每个格子负责画面上一小块）。直接当均匀网格画，
中段就被放大、边缘被压扭；按格子真实的边界还原，才是这只眼实际看到的画面（中段清楚、边缘糊）。
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"F:\born-wired-cortex\engine_v2")
for extra in (ROOT, ROOT / "tools"):
    sys.path.insert(0, str(extra))

import wire_red_gaze as W
import eye_geometry as G
from tools import taskbank as tb

SCALE = 6


def as_seen(eyes, sheet):
    """把非均匀采样的表还原成均匀画面：每个输出像素取它所在格子里的那个数。"""
    low_rows, high_rows = eyes._rows
    low_columns, high_columns = eyes._columns
    drawn_h, drawn_w = int(high_rows[-1]), int(high_columns[-1])
    which_rows = np.clip(np.searchsorted(low_rows, np.arange(drawn_h), side="right") - 1,
                         0, eyes.height - 1)
    which_columns = np.clip(np.searchsorted(low_columns, np.arange(drawn_w), side="right") - 1,
                            0, eyes.width - 1)
    return np.asarray(sheet)[np.ix_(which_rows, which_columns)]


def main():
    spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    G.place_ball(body.model, body.data, target, .25, .10, .90)
    sheet = eyes.observe_raw()
    print("表 %s   oversample=%d  centre_gain=%.1f" %
          (np.asarray(sheet[0]).shape, eyes.oversample, eyes.centre_gain))
    print("格子左边界（列，前 12 个）：", np.asarray(eyes._columns[0])[:12].tolist())
    print("格子宽度（列，两端和中间）：", np.asarray(eyes._columns[1])[:4].tolist(), np.asarray(eyes._columns[1])[22:26].tolist(), np.asarray(eyes._columns[1])[-4:].tolist())
    tiles = []
    for i in (0, 1):
        raw = np.repeat(np.repeat(np.asarray(sheet[i], dtype=np.uint8), SCALE, 0), SCALE, 1)
        back = as_seen(eyes, sheet[i])
        up = Image.fromarray(back).resize((raw.shape[1], raw.shape[0]), Image.NEAREST)
        tiles.append((raw.copy(), np.asarray(up).copy()))
    high = tiles[0][0].shape[0]
    canvas = Image.new("RGB", (tiles[0][0].shape[1]*2 + 24, high*2 + 36), "white")
    for k in range(2):
        canvas.paste(Image.fromarray(tiles[k][0]), (0, k*(high+18)))
        canvas.paste(Image.fromarray(tiles[k][1]), (tiles[k][0].shape[1] + 24, k*(high+18)))
    canvas.save(str(ROOT / "artifacts" / "_视网膜对照.png"))
    print("左列=照原样画（看着扭曲）   右列=按真实位置还原（实际看到的）")
    print("saved artifacts/_视网膜对照.png", canvas.size)
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())