# -*- coding: utf-8 -*-
"""谁在推「左眼 yaw 往右」那根眼肌：把来源细胞按列排出来。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wire_red_gaze as W                                       # noqa: E402
import eye_geometry as G                                        # noqa: E402
from tools import taskbank as tb                                # noqa: E402

DT = .01


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), gain=1.4, motor_time=.2, only=0)
    yaw, pitch = W.retina_angles(table)
    shape = table["eye_shape"]
    rows, columns = int(shape[1]), int(shape[2])
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    motor = np.asarray(brain.groups["eye_motor"], dtype=int)
    opp = np.asarray(brain.groups["retinal_opponent"], dtype=int)
    start = int(opp[0])
    src, dst = brain.synapses.src, brain.synapses.dst
    weight = brain.synapses._weights
    band = {}
    for cell, direction, tier, w in spec["edges"]:
        key = (str(direction), str(tier))
        got = band.setdefault(key, [0., 0, 0.])
        got[0] += float(w)          # 表里的粗（未归一）
        got[1] += 1
    print("每档的线数与表里的粗（未归一）：")
    for key in sorted(band):
        print("   %-6s %-4s %4d 根，粗合计 %.3f" % (key[0], key[1], band[key][1], band[key][0]))
    print("\n眼肌收到的线（按来源的列分组）与「表里的粗 vs 实际落在突触上的粗」：")
    for k in (0, 1, 2, 3):
        cell = int(motor[k])
        mine = np.nonzero(dst == cell)[0]
        print("\n眼肌细胞号 %d（%s）：%d 根线" % (cell, ("左眼yaw正","左眼yaw负","左眼pitch正","左眼pitch负")[k], len(mine)))
        if not len(mine):
            continue
        spans = {}
        for position in mine:
            col = (((int(src[position]) - start)//3) % (rows*columns)) % columns
            got = spans.setdefault(col, [0., 0.])
            got[0] += 1
            got[1] += float(weight[position])
        print("   线的粗细合计 %.3f；落在 %d 个列上；列 %s"
              % (sum(v[1] for v in spans.values()), len(spans),
                 " ".join("%d:%.3f" % (c, spans[c][1]) for c in sorted(spans)))  )
    # 现在量球在正前方时，谁真的在放电并且推这根肌肉
    G.place_ball(body.model, body.data, target, 0.0, 0., .90)
    obs = body.observe()
    for _ in range(40):
        act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                         learn=False)[1]
        brain._eye_angle = np.zeros(4)
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
    rate = brain.network.activity
    cell = int(motor[1])
    mine = np.nonzero(dst == cell)[0]
    flow = weight[mine]*rate[src[mine]]
    order = np.argsort(-flow)[:12]
    print("\n球在正前方时，推「左眼yaw负」的来源里最强的 12 根：")
    for position in mine[order]:
        source = int(src[position])
        col = (((source - start)//3) % (rows*columns)) % columns
        row = (((source - start)//3) % (rows*columns)) // columns
        print("   列%2d 行%2d 放电%.2f 线粗%.4f -> 力%.4f  表说这列 yaw %+.3f"
              % (col, row, rate[source], weight[position], weight[position]*rate[source], yaw[0][col]))
    print("   合计激励 %.4f（接到这根细胞上的线共 %d 根，其中在放电 %d 根）"
          % (flow.sum(), len(mine), int((rate[src[mine]] > .05).sum())))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())