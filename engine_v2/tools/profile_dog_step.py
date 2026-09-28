"""某个分辨率下一步的时间花在哪儿。用法：python tools/_profile_scale.py 120 90"""
import cProfile, pstats, io, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes

width, height = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (48, 36)
body = Go2Body(str(ROOT / 'models' / 'reflex_arena.xml'))
observation = body.observe()
environment = {name: np.zeros(4) for name in ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                           motor_units=200, proprio_units=64, association_units=256,
                           eye_width=width, eye_height=height)
pixels = np.zeros(brain.eye_shape, dtype=np.uint8)
ear = np.zeros((2, 160))
print('细胞 %d  连接 %d' % (brain.network.n_neurons, len(brain.synapses.src)))
calls = dict(eye_pixels=pixels, ear_waveform=ear, environment=environment, dt=.01)
for learn in (True, False):
    for _ in range(4):
        brain.step(observation, learn=learn, **calls)
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(15):
        brain.step(observation, learn=learn, **calls)
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('tottime').print_stats(10)
    print('===== learn=%s =====' % learn)
    print('\n'.join(stream.getvalue().splitlines()[4:20]))
