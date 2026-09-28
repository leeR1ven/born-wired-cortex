"""Diagnostic counterfactuals, not an autonomous behavior script."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.reflex_controller import ReflexController
from born_wired.reflex_senses import ReflexSenses

p = argparse.ArgumentParser()
p.add_argument('--seconds', type=float, default=60)
p.add_argument('--case', choices=['autonomous', 'driven', 'no_reflex', 'rest'], default='autonomous')
p.add_argument('--output', default='artifacts/environment_probe.json')
a = p.parse_args()
config = json.loads((ROOT/'live_config.json').read_text(encoding='utf-8-sig'))
b = Go2Body(ROOT/'models/reflex_arena.xml')
c = ReflexController(b.home_angles, b.lower_limits, b.upper_limits, **config['parameters'])
s = ReflexSenses(b)
obs = b.observe()
trace = []
begin = time.perf_counter()
for i in range(round(a.seconds/.01)):
    env = s.observe()
    q, act = c.step(obs, environment=env, autonomy=a.case=='autonomous',
                    locomotion=.65 if a.case in ('driven', 'no_reflex') else 0.,
                    reflexes=a.case!='no_reflex', dt=.01)
    obs = b.step(q, duration=.01, activation=act)
    if i % 50 == 0:
        rate = c.network.activity
        rotation = b.data.xmat[b._base].reshape(3, 3)
        trace.append(dict(t=float(b.data.time), position=b.data.qpos[:3].tolist(),
                          yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                          drive=float(rate[c.groups['locomotion']][0]),
                          phase=rate[c.groups['phase']].tolist(), rays=env['ray_distance'].tolist(),
                          touch=env['body_touch_force_n'].tolist(),
                          state=c.diagnostics()['reflex_activity'], up=float(obs['body_up'][2])))
result = dict(case=a.case, parameters=config['parameters'], seconds=a.seconds, wall_seconds=time.perf_counter()-begin,
              diagnostics=c.diagnostics(), trace=trace)
out = ROOT/a.output
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
with mujoco.Renderer(b.model, height=480, width=640) as renderer:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = 125, -22, 2.8
    camera.lookat[:] = b.data.qpos[:3]
    renderer.update_scene(b.data, camera=camera)
    Image.fromarray(renderer.render()).save(out.with_suffix('.png'))
print(json.dumps(dict(file=str(out), final=trace[-1]), ensure_ascii=False))
