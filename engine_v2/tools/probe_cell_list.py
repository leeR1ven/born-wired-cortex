# -*- coding: utf-8 -*-
"""直接接线模式的细胞清单：旧的 24 级阶梯、红色指令细胞还在不在。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402
from tools import taskbank as tb                                # noqa: E402


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), only=0)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    groups = brain.groups
    print("细胞群共 %d 个，细胞总数 %d" % (len(groups), brain.network.n_neurons))
    for name in sorted(groups):
        print("  %-34s %6d" % (name, len(np.atleast_1d(groups[name]))))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())