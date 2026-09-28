"""模型的指纹：细胞、连接、权重、20 步活动、200 步行走。

改引擎或者改参数以后，指纹必须一模一样，否则说明动的不是"零开销"的东西。
用法：python tools/check_model_fingerprint.py [输出文件]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired import torch_execution
from born_wired.stereo_senses import RawEyes

MODEL = str(ROOT / 'models' / 'reflex_arena.xml')


def digest(array):
    return hashlib.md5(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def measure(motor=200, proprio=64, association=256, learning_interval=0.,
            eye_width=48, eye_height=36):
    body = Go2Body(MODEL)
    observation = body.reset('stand')
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=motor, proprio_units=proprio,
                               association_units=association, learning_interval=learning_interval,
                               eye_width=eye_width, eye_height=eye_height)
    environment = {name: np.zeros(4) for name in
                   ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
    environment['foot_support'] = np.zeros(4, dtype=bool)
    eyes = RawEyes(body, width=eye_width, height=eye_height)
    pixels = eyes.observe_raw()
    ear = np.zeros((2, 160))
    result = dict(cells=int(brain.network.n_neurons), edges=int(len(brain.synapses.src)),
                  engine=torch_execution.name_of(torch_execution.resolve()),
                  weights=digest(brain.synapses.weights),
                  groups={name: int(brain.groups[name].size) for name in
                          ('photoreceptors', 'retinal_opponent', 'retina', 'motor',
                           'proprio', 'association', 'binocular', 'eye_gaze')
                          if name in brain.groups})
    for tick in range(20):
        target, activation = brain.step(observation, environment=environment, eye_pixels=pixels,
                                        ear_waveform=ear, locomotion=.3, dt=.01, learn=True)
    result['weights_after_20'] = digest(brain.synapses.weights)
    result['activity_after_20'] = digest(np.asarray(brain.network.activity))
    for tick in range(200):
        target, activation = brain.step(observation, environment=environment, eye_pixels=pixels,
                                        ear_waveform=ear, locomotion=.5, dt=.01, learn=True)
        observation = body.step(target, duration=.01, activation=activation)
    result['walk'] = dict(x=float(body.data.qpos[0]), y=float(body.data.qpos[1]),
                          height=float(body.data.qpos[2]),
                          up_z=float(observation['body_up'][2]))
    result['weights_after_walk'] = digest(brain.synapses.weights)
    result['activity_after_walk'] = digest(np.asarray(brain.network.activity))
    return result


if __name__ == '__main__':
    cadence = 0.
    eye_width, eye_height = 48, 36
    target = None
    engine = None
    arguments = list(sys.argv[1:])
    while arguments:
        argument = arguments.pop(0)
        if argument == '--cadence':
            cadence = float(arguments.pop(0))
        elif argument == '--eye-width':
            eye_width = int(arguments.pop(0))
        elif argument == '--eye-height':
            eye_height = int(arguments.pop(0))
        elif argument == '--engine':
            engine = arguments.pop(0)
        else:
            target = Path(argument)
    if engine is not None:
        torch_execution.set_default(engine)
    engine = torch_execution.name_of(torch_execution.resolve())
    if target is None:
        # One file per engine: the two engines agree to the fifteenth digit, not
        # to the last one, so each is compared against its own record.
        target = ROOT/'artifacts'/f'model_fingerprint_{engine.replace(":", "_")}.json'
    found = measure(learning_interval=cadence, eye_width=eye_width, eye_height=eye_height)
    text = json.dumps(found, indent=2, sort_keys=True)
    if target.exists():
        old = target.read_text(encoding='utf-8')
        print('一致' if old.strip() == text.strip() else '不一致')
        if old.strip() != text.strip():
            print(json.dumps(json.loads(old), indent=2, sort_keys=True))
    else:
        print('没有旧指纹，写入新的')
    target.write_text(text, encoding='utf-8')
    print(text)
