"""两个引擎跑同一张图：差多少，各自重跑是不是一模一样。

host 引擎（NumPy）和 device 引擎（torch）从同一份代码、同一个种子建出控制器，
用同一批输入跑同样多步。报告三件事：
  1. 每个引擎自己的重跑一致性 —— 同一个模型、同一个输入，跑两遍结果是否逐位相同；
  2. 两个引擎之间的最大差 —— 归约次序不同带来的浮点尾差有多大；
  3. 身体位置 —— 这些尾差有没有改变行为。

用法：python tools/compare_engines.py [步数] [--eye-width N --eye-height N
        --motor N --proprio N --association N]
"""
import hashlib, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired import torch_execution
from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes

MODEL = str(ROOT / 'models' / 'reflex_arena.xml')
FIELDS = ('activity', 'voltage', 'weights')


def digest(array):
    return hashlib.md5(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def run(device, steps, *, motor, proprio, association, eye_width, eye_height,
        seed=7, dt=.01):
    chosen = torch_execution.set_default(device)
    body = Go2Body(MODEL)
    observation = body.reset('stand')
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=motor, proprio_units=proprio,
                               association_units=association, learning_interval=0.,
                               eye_width=eye_width, eye_height=eye_height, seed=seed)
    environment = {name: np.zeros(4) for name in
                   ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
    environment['foot_support'] = np.zeros(4, dtype=bool)
    eyes = RawEyes(body, width=eye_width, height=eye_height)
    pixels = eyes.observe_raw()
    ear = np.zeros((2, 160))
    started = time.perf_counter()
    for _ in range(steps):
        target, activation = brain.step(observation, environment=environment,
                                        eye_pixels=pixels, ear_waveform=ear,
                                        locomotion=.3, dt=dt, learn=True)
        observation = body.step(target, duration=dt, activation=activation)
    elapsed = time.perf_counter() - started
    return dict(engine=torch_execution.name_of(chosen), seconds=elapsed,
                cells=int(brain.network.n_neurons), edges=int(len(brain.synapses.src)),
                activity=np.asarray(brain.network.activity, dtype=float),
                voltage=np.asarray(brain.network.voltage, dtype=float),
                weights=brain.synapses.weights.copy(),
                position=np.asarray(body.data.qpos[:3], dtype=float))


def gap(left, right):
    if not left.size:
        return 0., 0.
    difference = float(np.max(np.abs(left - right)))
    scale = float(np.max(np.abs(left)))
    return difference, difference/max(scale, 1e-30)


def main():
    arguments = list(sys.argv[1:])
    steps, settings = 200, dict(motor=200, proprio=64, association=256,
                                eye_width=48, eye_height=36)
    names = {'--motor': 'motor', '--proprio': 'proprio', '--association': 'association',
             '--eye-width': 'eye_width', '--eye-height': 'eye_height'}
    while arguments:
        argument = arguments.pop(0)
        if argument in names:
            settings[names[argument]] = int(arguments.pop(0))
        else:
            steps = int(argument)
    print(f"步数 {steps}   眼 {settings['eye_width']}x{settings['eye_height']}   "
          f"运动/本体/联想 {settings['motor']}/{settings['proprio']}/{settings['association']}")
    host = run('cpu', steps, **settings)
    host_again = run('cpu', steps, **settings)
    device = run(None, steps, **settings)
    if device['engine'] == 'host':
        print('  没有可用设备，两次都跑在 host 上')
        return
    device_again = run(None, steps, **settings)
    for label, first, second in (('host  ', host, host_again), ('device', device, device_again)):
        same = all(np.array_equal(first[field], second[field]) for field in FIELDS)
        print(f"  {label} {first['cells']:>7d} 细胞 {first['edges']:>8d} 连接 "
              f"{1000*first['seconds']/steps:6.2f} ms/步   重跑逐位相同 {same}")
    print(f"  host {1000*host['seconds']/steps:.2f} ms/步  ->  device "
          f"{1000*device['seconds']/steps:.2f} ms/步，快 {host['seconds']/device['seconds']:.2f} 倍")
    for field in FIELDS:
        absolute, relative = gap(host[field], device[field])
        print(f'  {field:9s} 最大差 {absolute:.3e}   相对 {relative:.3e}')
    print(f"  身体位置 host {np.round(host['position'], 6)}   device "
          f"{np.round(device['position'], 6)}")


if __name__ == '__main__':
    main()
