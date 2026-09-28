"""把机器狗的模型做大：细胞数和分辨率能推到多大，代价是多少？

视网膜是唯一的分辨率旋钮：每只眼 EYE_HEIGHT x EYE_WIDTH 个像素、每个像素 3 个
颜色通道各占一个神经元，再加两层同样大小的视网膜细胞，所以像素这一摊的细胞数
= 2*H*W*3*3。运动/本体/联想单元只是小数点后面的零头。
这里量：细胞数、连接数、建模时间、大脑每步耗时（开/关学习）、眼睛渲染耗时、物理耗时。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
import born_wired.embodied as embodied
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes


def measure(body, observation, environment, height, width, motor, proprio, association):
    started = time.perf_counter()
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=motor, proprio_units=proprio,
                               association_units=association,
                               eye_width=width, eye_height=height)
    build = time.perf_counter() - started
    eyes = RawEyes(body, width=width, height=height)
    pixels = np.zeros(brain.eye_shape, dtype=np.uint8)
    ear = np.zeros((2, 160))
    calls = dict(eye_pixels=pixels, ear_waveform=ear, environment=environment, dt=.01)
    timings = {}
    for learn in (True, False):
        for _ in range(3):
            target, activation = brain.step(observation, learn=learn, **calls)
            body.step(target, duration=.01, activation=activation)
        started = time.perf_counter()
        for _ in range(10):
            target, activation = brain.step(observation, learn=learn, **calls)
        timings['brain_ms_learn' if learn else 'brain_ms_frozen'] = (time.perf_counter()-started)/10*1000
    started = time.perf_counter()
    for _ in range(10):
        observation = body.observe()
    timings['physics_ms'] = (time.perf_counter()-started)/10*1000
    started = time.perf_counter()
    for _ in range(5):
        eyes.observe_raw()
    timings['render_ms'] = (time.perf_counter()-started)/5*1000
    eyes.close()
    cells = int(brain.network.n_neurons)
    retina = int(brain.groups['photoreceptors'].size
                 + brain.groups['retinal_interneurons'].size
                 + brain.groups['retinal_opponent'].size)
    return dict(height=height, width=width, motor=motor, proprio=proprio,
                association=association, cells=cells, retina_cells=retina,
                edges=int(len(brain.synapses.src)), build_s=build, **timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/probe_model_scale.json')
    args = parser.parse_args()
    body = Go2Body(str(ROOT / 'models' / 'reflex_arena.xml'))
    observation = body.observe()
    environment = {name: np.zeros(4) for name in
                   ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
    plans = [(36, 48, 200, 64, 256),
             (90, 120, 200, 64, 256),
             (90, 120, 800, 256, 1024),
             (120, 160, 800, 256, 1024),
             (120, 160, 1600, 512, 2048),
             (144, 192, 1600, 512, 2048),
             (180, 240, 1600, 512, 2048)]
    print('%-16s %10s %10s %10s %14s %14s %10s %10s %8s'
          % ('分辨率', '运动/本体/联想', '细胞', '视网膜占', '连接', '建模s',
             '大脑ms', '冻结ms', '物理+渲染'))
    rows = []
    for height, width, motor, proprio, association in plans:
        row = measure(body, observation, environment, height, width, motor, proprio, association)
        rows.append(row)
        print('%-16s %10s %10d %10d %14d %14.1f %10.2f %10.2f %8.1f'
              % ('%dx%d' % (height, width), '%d/%d/%d' % (motor, proprio, association),
                 row['cells'], row['retina_cells'], row['edges'], row['build_s'],
                 row['brain_ms_learn'], row['brain_ms_frozen'],
                 row['physics_ms'] + row['render_ms']))
        total = row['brain_ms_learn'] + row['physics_ms'] + row['render_ms']
        print('%-16s 一步合计 %.2f ms -> 墙钟最多 %.0f Hz（仿真步长固定 10 ms）'
              % ('', total, 1000./total))
    (ROOT/args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print()
    print('写到 %s' % (ROOT/args.output))


if __name__ == '__main__':
    main()
