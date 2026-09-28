"""大脑一步要多久：host 引擎和 device 引擎，同一张图同一批输入。

只计 brain.step，不含 MuJoCo 渲染，所以看到的是引擎本身的开销。先空跑几步
把 CUDA 和缓存热起来，再计时。

用法：python tools/probe_engine_speed.py [步数] [--eye-width N --eye-height N
        --motor N --proprio N --association N]
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired import torch_execution
from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes

MODEL = str(ROOT / 'models' / 'reflex_arena.xml')


def measure(device, steps, *, motor, proprio, association, eye_width, eye_height,
            seed=7, dt=.01, warmup=3):
    chosen = torch_execution.set_default(device)
    body = Go2Body(MODEL)
    observation = body.reset('stand')
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=motor, proprio_units=proprio,
                               association_units=association, learning_interval=.05,
                               eye_width=eye_width, eye_height=eye_height, seed=seed)
    environment = {name: np.zeros(4) for name in
                   ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
    environment['foot_support'] = np.zeros(4, dtype=bool)
    eyes = RawEyes(body, width=eye_width, height=eye_height)
    pixels = eyes.observe_raw()
    ear = np.zeros((2, 160))
    for _ in range(warmup):
        brain.step(observation, environment=environment, eye_pixels=pixels,
                   ear_waveform=ear, locomotion=.3, dt=dt, learn=True)
    started = time.perf_counter()
    for _ in range(steps):
        brain.step(observation, environment=environment, eye_pixels=pixels,
                   ear_waveform=ear, locomotion=.3, dt=dt, learn=True)
    elapsed = time.perf_counter() - started
    peak = 0.
    if chosen is not None:
        peak = torch_execution.torch_module().cuda.max_memory_allocated()/2**20
    return dict(engine=torch_execution.name_of(chosen), cells=int(brain.network.n_neurons),
                edges=int(len(brain.synapses.src)), per_step=1000*elapsed/steps, peak=peak)


def main():
    arguments = list(sys.argv[1:])
    steps, settings = 60, dict(motor=800, proprio=256, association=1024,
                               eye_width=160, eye_height=120)
    names = {'--motor': 'motor', '--proprio': 'proprio', '--association': 'association',
             '--eye-width': 'eye_width', '--eye-height': 'eye_height'}
    while arguments:
        argument = arguments.pop(0)
        if argument in names:
            settings[names[argument]] = int(arguments.pop(0))
        else:
            steps = int(argument)
    print(f"配置：眼 {settings['eye_width']}x{settings['eye_height']}  "
          f"运动/本体/联想 {settings['motor']}/{settings['proprio']}/{settings['association']}")
    for device in ('cpu', None):
        result = measure(device, steps, **settings)
        memory = '' if not result['peak'] else f"   显存峰值 {result['peak']:.0f} MiB"
        print(f"  {result['engine']:10s} {result['cells']:>8d} 细胞 {result['edges']:>9d} 连接 "
              f"大脑一步 {result['per_step']:7.2f} ms{memory}")


if __name__ == '__main__':
    main()
