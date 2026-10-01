# -*- coding: utf-8 -*-
"""\u628a\u773c\u808c\u795e\u7ecf\u5143\u8eab\u4e0a\u771f\u5b9e\u7684\u8fde\u63a5\u6743\u91cd\u8bfb\u51fa\u6765\u3002"""
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
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    syn = brain.network.synapses
    dst = np.asarray(syn.dst)
    weights = np.asarray(syn.weights)
    signs = np.asarray(syn.signs)
    budgets = np.asarray(syn.budgets)
    motor = np.asarray(brain.groups['eye_motor'])
    print("\u773c\u808c\u795e\u7ecf\u5143 %d \u4e2a\uff0c\u9884\u7b97 %s" % (len(motor), budgets[motor]))
    for slot in range(4):
        cell = int(motor[slot])
        pick = dst == cell
        print("slot %d (\u5de6yaw \u524d%d)\uff1a\u6536\u5230 %3d \u6839\u7ebf\uff0c\u6743\u91cd\u548c %.3f\uff0c"
              " \u524d\u4e94\u6839 %s"
              % (slot, slot, int(pick.sum()), float(weights[pick].sum()),
                 np.round(weights[pick][:5], 4)))
    # \u770b\u770b\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u81ea\u5df1\u7684\u9884\u7b97\u4e0e\u51fa\u8fb9
    red_cell = int(spec['edges'][0][0])
    out = np.asarray(syn.src) == red_cell
    print("\u7b2c\u4e00\u4e2a\u7ea2\u8272\u7279\u5f81\u7ec6\u80de %d\uff1a\u51fa\u8fb9 %s" % (red_cell, np.round(weights[out], 4)))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())