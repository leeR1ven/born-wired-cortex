"""The same frames through the honest retina and through the old one.

The retina used to warp positions by |t|**fovea before reading a single drawn
pixel, which magnifies the middle of the view without bound. This runs the same
scene and the same eye loop twice - once through RawEyes as it is now (the
cells stand closer together near the middle of the picture) and once through
LegacyEyes, which reproduces the warped sampling - and prints the angle the
eyes settle at, the drive the distance bank reads and how many of its cells are
lit. It is the measurement behind the numbers in tests/test_eye_muscles.py and
docs/视网膜采样与全区放大_20260927.md; its output is
artifacts/bank_honest_vs_warped.log.

  python tools/compare_honest_and_warped_retina.py
"""

import sys, json
from pathlib import Path
import mujoco, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.embodied import EmbodiedController, EYE_DISTANCE_GAIN
from born_wired.go2_body import Go2Body
from born_wired.stereo_senses import RawEyes

ARENA = ROOT / "models" / "reflex_arena.xml"
SCENERY = ("east_wall","west_wall","north_wall","south_wall","red_pillar","blue_box",
           "front_block","curb","low_step","platform","ramp","passage_a","passage_b",
           "sound_low","sound_high")
ENVIRONMENT = ("body_touch","foot_obstacle","foot_load","foot_slip")
BALL=.06; DT=.01

class LegacyEyes(RawEyes):
    """The sampling the working copy used before today: positions warped by
    |t|**fovea and each cell reading a single drawn pixel."""
    def __init__(self, body, width=48, height=36, oversample=4, fovea=2.):
        self._fovea = fovea
        super().__init__(body, width=width, height=height, oversample=oversample)
    def _patches(self, count, drawn):
        even = np.linspace(-1., 1., count)
        warped = np.sign(even)*np.abs(even)**self._fovea
        index = np.rint((warped+1.)*.5*(drawn-1)).astype(int)
        return index, index+1
    def _report(self, picture):
        return picture[np.ix_(self._rows[0], self._columns[0])]

def arena(distance):
    body = Go2Body(model_path=ARENA)
    for name in SCENERY:
        g = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g >= 0: body.model.geom_pos[g] = [60.,60.,-8.]
    t = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[t] = [BALL]*3
    body.model.geom_pos[t] = [.30+distance, 0., .32]
    mujoco.mj_forward(body.model, body.data)
    return body

def run(distance, cls, w, h, steps=400):
    body = arena(distance)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                              motor_units=20, proprio_units=8, association_units=8,
                              eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                              eye_width=w, eye_height=h)
    eyes = cls(body, width=w, height=h)
    obs = body.observe()
    env = {n: np.zeros(4) for n in ENVIRONMENT}
    env['foot_support'] = np.zeros(4, dtype=bool)
    for _ in range(steps):
        target, activation = brain.step(obs, environment=env, eye_pixels=eyes.observe_raw(),
                                        dt=DT, learn=False)
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
    px = np.stack([np.asarray(eyes.observe_raw()[i], dtype=float) for i in (0,)])
    eyes.close()
    rates = brain.network.activity
    g = brain.groups
    cmd = brain.eye_command()
    inn = float(rates[g['eye_convergence_in']][0]); out = float(rates[g['eye_convergence_out']][0])
    bank = rates[g['eye_distance']].copy()
    return dict(cmd_asym=round(float(cmd[2]-cmd[0]),4),
                drive=round(EYE_DISTANCE_GAIN*(inn-out),4),
                bank=[round(float(v),3) for v in bank], lit=int((bank>.02).sum()))

for distance in (2.5,1.5,1.0,.6,.4,.25):
    a = run(distance, RawEyes, 48, 36)
    b = run(distance, LegacyEyes, 48, 36)
    print(json.dumps(dict(distance=distance, honest=a, warped=b), ensure_ascii=False), flush=True)
