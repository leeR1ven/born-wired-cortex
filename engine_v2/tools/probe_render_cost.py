"""How the two eye renders spend their time, and at what size.

The live loop draws two camera pictures every step.  This probe separates the
scene update from the picture readback and repeats the pair at several sizes,
so the cost can be attributed to pixel count or to fixed per-render overhead.

    python tools/probe_render_cost.py
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body


def timed(function, repeat):
    for _ in range(3):
        function()
    stamp = time.perf_counter()
    for _ in range(repeat):
        function()
    return (time.perf_counter() - stamp)/repeat*1000


def main():
    body = Go2Body(ROOT / 'models/reflex_arena.xml')
    cameras = tuple(mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                    for name in ('eye_left', 'eye_right'))
    print('geoms %d  cameras %d  nq %d  nv %d' % (body.model.ngeom, body.model.ncam,
                                                  body.model.nq, body.model.nv))
    print('%-12s %10s %10s %10s %10s' % ('drawn size', 'update ms', 'render ms', 'pair ms', 'readback MB'))
    for width, height in ((120, 90), (240, 180), (480, 360), (640, 480), (960, 720)):
        renderer = mujoco.Renderer(body.model, height=height, width=width)
        picture = renderer.render()
        assert picture.shape == (height, width, 3)
        update = timed(lambda: [renderer.update_scene(body.data, camera=int(camera)) for camera in cameras], 20)
        render = timed(lambda: renderer.render(), 20)
        pair = timed(lambda: [renderer.render() for _ in cameras], 20)
        print('%-12s %10.3f %10.3f %10.3f %10.3f' % ('%dx%d' % (width, height), update, render, pair,
                                                      height*width*3/1e6))
        renderer.close()


if __name__ == '__main__':
    main()
