"""实时窗口真正会跑的那套配置，一步要多久、狗走不走得动。

用法：python tools/probe_live_config.py [步数]
和 tools/live_dog.py 的 build_brain 完全同一条路径：读同一个 live_config.json，
带眼肌肉，眼睛按大脑自己的分辨率渲染，耳朵给静音波形。
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body, DEFAULT_MODEL
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes

CONFIG = ROOT / 'live_config.json'
ENVIRONMENT = ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')


def main():
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    body = Go2Body(config.get('model', DEFAULT_MODEL))
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **config.get('parameters', {}))
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    observation = body.reset(seed=0)
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment['foot_support'] = np.zeros(4, dtype=bool)
    ear = np.zeros((2, 160))
    start = body.data.qpos[:2].copy()
    brain_ms, physics_ms, render_ms = 0., 0., 0.
    low, upright = 1., 1.
    for step in range(ticks):
        mark = time.perf_counter()
        pixels = eyes.observe_raw()
        middle = time.perf_counter()
        target, activation = brain.step(observation, environment=environment, eye_pixels=pixels,
                                        ear_waveform=ear, dt=.01, locomotion=.6)
        after = time.perf_counter()
        if brain.eye_encoder is not None:
            body.command_eyes(brain.eye_command())
        observation = body.step(target, duration=.01, activation=activation)
        done = time.perf_counter()
        render_ms += middle-mark
        brain_ms += after-middle
        physics_ms += done-after
        if step > 100:
            low = min(low, float(body.data.qpos[2]))
            upright = min(upright, float(observation['body_up'][2]))
    eyes.close()
    scale = 1000./ticks
    step_total = brain_ms*scale + physics_ms*scale + render_ms*scale
    print('分辨率 %dx%d  细胞 %d  连接 %d  眼肌肉 %s' %
          (brain.eye_width, brain.eye_height, brain.network.n_neurons,
           len(brain.synapses.src), brain.eye_encoder is not None))
    print('视网膜神经元 %d  学习节拍 %.3f s  仿真 %.1f s' %
          (brain.groups['photoreceptors'].size, brain.network.learning_interval, ticks*.01))
    print('一步：大脑 %.2f ms + 物理 %.2f ms + 眼睛渲染 %.2f ms = %.2f ms -> 墙钟 %.0f Hz'
          % (brain_ms*scale, physics_ms*scale, render_ms*scale, step_total, 1000./step_total))
    print('前进 %.3f m  最低高度 %.3f m  最小 up_z %.5f' %
          (float(body.data.qpos[0]-start[0]), low, upright))


if __name__ == '__main__':
    main()
