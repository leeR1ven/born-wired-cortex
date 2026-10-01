# -*- coding: utf-8 -*-
"""看看红球摆在几个位置上时，红色指令细胞各放多少电、两只眼各自被指到哪。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wire_red_gaze as W                                       # noqa: E402
from tools import taskbank as tb                                # noqa: E402

DT = .01
PLACES = [(-.35, .32), (-.35, -.22), (-.17, .32), (.34, .32), (.34, -.22), (0., 0.)]


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(3.), cell_time=.01, latch=0., cross=0.,
                     motor_time=.35)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    aim = W.aimer(body, target, .90)
    env = tb.blank_environment()
    names = sorted(n for n in brain.groups if n.startswith('eye_red_') and 'relay' not in n)
    print('指令细胞：', names)
    for bearing, elevation in PLACES:
        true_b, true_e = aim(bearing, elevation)
        obs = body.observe()
        for _ in range(120):
            act = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(), dt=DT,
                             learn=False)[1]
            body.command_eyes(brain.eye_command())
            obs = body.step(np.asarray(body.home_angles), duration=DT, activation=act)
        rates = brain.network.rates_at(np.asarray([brain.groups[n][0] for n in names]))
        cmd = brain.eye_command()
        print('\n球 方位 %+.3f 高低 %+.3f  ->  真实方位 %+.3f 高低 %+.3f' % (bearing, elevation,
                                                                   true_b, true_e))
        print('  眼命令 左眼(yaw,pitch) %+.3f %+.3f | 右眼 %+.3f %+.3f' % (cmd[0], cmd[1], cmd[2], cmd[3]))
        print('  误差 yaw %+.3f pitch %+.3f' % (.5*(cmd[0]+cmd[2]) - true_b,
                                                .5*(cmd[1]+cmd[3]) + true_e))
        for name, rate in zip(names, rates):
            if rate > 1e-4:
                print('   %-34s %.3f' % (name, rate))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())