# -*- coding: utf-8 -*-
import json, io, sys
from pathlib import Path
import numpy as np
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
import wire_red_gaze as W
t = W.load_table()
print("eye_shape =", t["eye_shape"], " peak size =", len(t["cells"]["retinal_opponent"]["peak"]))
red = W.red_cells(t)
g = W.eye_of(red, t)
print("\u63a5\u7ebf\u7684\u7ea2\u8272\u7ec6\u80de %d \u4e2a\uff1b\u5f52\u5230\u7b2c\u51e0\u53ea\u773c\uff1a%s" % (len(red), np.bincount(g)))
r, c = W.cell_rows_columns(red, t)
print("\u884c\u8303\u56f4 %d~%d\uff0c\u5217\u8303\u56f4 %d~%d" % (r.min(), r.max(), c.min(), c.max()))
yaw, pitch = W.retina_angles(t)
print("yaw \u8868\u5f62\u72b6 %s\uff1b\u7b2c0\u884c\u524d\u4e94\u4e2a %s\uff1b\u7b2c1\u884c\u524d\u4e94\u4e2a %s"
      % (yaw.shape, np.round(yaw[0][:5], 3), np.round(yaw[1][:5], 3)))
print("yaw \u8868\u6709\u6548\u5217\u6570\uff1a\u5de6\u773c %d\u3001\u53f3\u773c %d" % (np.isfinite(yaw[0]).sum(), np.isfinite(yaw[1]).sum()))