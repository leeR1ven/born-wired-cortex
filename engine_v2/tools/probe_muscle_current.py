# -*- coding: utf-8 -*-
"""眼肌细胞到底收到多少电流：把 synapse 的 exc/inh、电压、适应都打出来。"""
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
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    motor = np.asarray(brain.groups["eye_motor"], dtype=int)
    net = brain.network
    print("眼肌细胞的编号 %s" % motor.tolist())
    print("它们的时间常数 %s；适应增益 %s；预算 %s"
          % (np.round(net._tau[motor], 3), np.round(net._gain[motor], 3),
             np.round(brain.synapses.budgets[motor], 3)))
    for bearing in (0.0, 0.30):
        G.place_ball(body.model, body.data, target, bearing, 0., .90)
        obs = body.observe()
        for step in range(40):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            brain._eye_angle = np.zeros(4)
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        rate = net.activity
        exc, inh = brain.synapses.currents(rate)
        print("\n=== 球方位 %+0.2f（该转 %+0.3f）===" % (bearing, W.required(body, target)[0][0]))
        for k, cell in enumerate(motor):
            joint, going = divmod(k, 2)
            print("  %-8s 激励%6.3f 抑制%6.3f 电压%6.3f 适应%5.3f 放电%5.3f"
                  % (("左眼yaw","左眼pitch","右眼yaw","右眼pitch")[joint]
                     + ("正" if going == 0 else "负"),
                     exc[cell], inh[cell], net.voltage[cell], net.adaptation[cell],
                     net.rate[cell] if hasattr(net, "rate") else net.activity[cell]))
        src, dst = brain.synapses.src, brain.synapses.dst
        for k, cell in enumerate(motor[:2]):
            mine = dst == cell
            total = brain.synapses._weights[mine].sum()
            print("  [核对] 编号%d 收到 %d 根线，总粗 %.3f，w_max 之和 %.3f"
                  % (cell, mine.sum(), total, brain.synapses.w_max[mine].sum()))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())