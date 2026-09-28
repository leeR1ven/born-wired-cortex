"""Which construction makes the eye render slow?

Builds the same renderer step by step, timing two camera renders after each
step, so the object that costs the milliseconds can be named.

    python tools/probe_render_context.py
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body


def measure(body, renderer, repeat=30):
    cameras = tuple(mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                    for name in ('eye_left', 'eye_right'))
    for _ in range(4):
        for camera in cameras:
            renderer.update_scene(body.data, camera=int(camera))
            renderer.render()
    stamp = time.perf_counter()
    for _ in range(repeat):
        for camera in cameras:
            renderer.update_scene(body.data, camera=int(camera))
            renderer.render()
    return (time.perf_counter() - stamp)/repeat*1000


def step(name, milliseconds):
    print('%-40s %8.3f' % (name, milliseconds))


def main():
    body = Go2Body(ROOT / 'models/reflex_arena.xml')
    renderer = mujoco.Renderer(body.model, height=360, width=480)
    step('mujoco.Renderer right after Go2Body', measure(body, renderer))

    from born_wired.reflex_senses import ReflexSenses
    senses = ReflexSenses(body)
    step('after ReflexSenses built', measure(body, renderer))

    from born_wired.binaural_senses import BinauralSenses
    ears = BinauralSenses(body, window_samples=160)
    step('after BinauralSenses built', measure(body, renderer))

    from born_wired.stereo_senses import RawEyes
    eyes = RawEyes(body, width=160, height=120)
    step('after RawEyes built (own renderer)', measure(body, renderer))

    stamp = time.perf_counter()
    for _ in range(30):
        senses.observe()
        eyes.observe_raw()
    step('senses + RawEyes per step', (time.perf_counter()-stamp)/30*1000)

    stamp = time.perf_counter()
    for _ in range(30):
        body.step(np.clip(np.asarray(body.data.qpos[body._qpos]).copy(), body.lower_limits, body.upper_limits), duration=.01)
        eyes.observe_raw()
    step('body.step + RawEyes per step', (time.perf_counter()-stamp)/30*1000)

    from born_wired.embodied import EmbodiedController
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               motor_units=200, proprio_units=64, association_units=256)
    step('after EmbodiedController built', measure(body, renderer))
    stamp = time.perf_counter()
    for _ in range(30):
        eyes.observe_raw()
    step('RawEyes per step with brain built', (time.perf_counter()-stamp)/30*1000)
    renderer.close()
    eyes.close()


if __name__ == '__main__':
    main()
