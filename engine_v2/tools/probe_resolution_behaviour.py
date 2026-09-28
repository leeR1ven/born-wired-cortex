"""同一套本能连接，换不同的眼睛分辨率，狗还站得住、走得动吗。

用法：python tools/probe_resolution_behaviour.py [步数] [分辨率:宽x高 ...]
协议与 tools/measure_sound_walking.py 完全一致（同参数、同站姿种子、眼部肌肉
一起动、眼睛每步都用真实画面渲染），只把眼睛的分辨率换掉。量身体自己的数据。
"""
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes

ARENA = ROOT / 'models' / 'reflex_arena.xml'
CONFIG = ROOT / 'live_config.json'
ENVIRONMENT = ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')
PLANS = [(48, 36), (72, 54), (96, 72), (120, 90), (144, 108)]
SEED, LOCOMOTION = 0, .6


def run(width, height, ticks):
    body = Go2Body(model_path=ARENA)
    parameters = dict(json.loads(CONFIG.read_text(encoding='utf-8-sig'))['parameters'])
    # Size under test is this probe's own argument. The live window's copies of
    # the same knobs in the config would otherwise be passed twice, which is an
    # error rather than an override.
    for name in ('eye_width', 'eye_height', 'motor_units', 'proprio_units',
                 'association_units'):
        parameters.pop(name, None)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               eye_width=width, eye_height=height, **parameters)
    eyes = RawEyes(body, width=width, height=height)
    observation = body.reset(seed=SEED)
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment['foot_support'] = np.zeros(4, dtype=bool)
    ear = np.zeros((2, 160))
    start = body.data.qpos[:2].copy()
    low, upright, lowest_step = 1., 1., None
    started = time.perf_counter()
    for step in range(ticks):
        target, activation = brain.step(observation, environment=environment,
                                        eye_pixels=eyes.observe_raw(), ear_waveform=ear,
                                        dt=.01, learn=False, locomotion=LOCOMOTION)
        body.command_eyes(brain.eye_command())
        observation = body.step(target, duration=.01, activation=activation)
        height_now = float(body.data.qpos[2])
        if step > 100:
            low = min(low, height_now)
            upright = min(upright, float(observation['body_up'][2]))
            if lowest_step is None or height_now < lowest_step:
                lowest_step = height_now
    eyes.close()
    return dict(width=width, height=height, cells=int(brain.network.n_neurons),
                edges=int(len(brain.synapses.src)),
                forward=float(body.data.qpos[0]-start[0]),
                sideways=float(np.linalg.norm(body.data.qpos[:2]-start)),
                end_height=float(body.data.qpos[2]), low_height=float(low),
                min_up_z=float(upright), seconds=time.perf_counter()-started)


def main():
    arguments = sys.argv[1:]
    ticks = int(arguments.pop(0)) if arguments else 1000
    plans = PLANS
    if arguments:
        plans = [tuple(reversed([int(v) for v in a.split('x')])) for a in arguments]
    rows = []
    print('%-10s %9s %9s %9s %9s %9s %9s %9s %8s' %
          ('分辨率', '细胞', '连接', '前进m', '位移m', '末高度', '最低高度', '最小up_z', '用时s'))
    for width, height in plans:
        row = run(width, height, ticks)
        rows.append(row)
        print('%-10s %9d %9d %9.3f %9.3f %9.3f %9.3f %9.5f %8.1f' %
              ('%dx%d' % (width, height), row['cells'], row['edges'], row['forward'],
               row['sideways'], row['end_height'], row['low_height'], row['min_up_z'],
               row['seconds']))
    (ROOT/'artifacts/probe_resolution_behaviour.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
