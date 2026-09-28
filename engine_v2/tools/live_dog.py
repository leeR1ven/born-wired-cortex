"""Persistent MuJoCo window, with explicit configuration reload and live status."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import argparse
import numpy as np
import mujoco
from mujoco import viewer as viewer_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.runtime import reload_runtime_modules
from born_wired import torch_execution

CONFIG = ROOT / 'live_config.json'
STATUS = ROOT / 'artifacts/live_status.json'


def read_config():
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    for name in ('locomotion', 'flexion'):
        value = config.get(name, 0.)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
            raise ValueError(f'invalid {name}')
    return config


def build_brain(body, config):
    # Reload only on an explicit configuration revision. Physics and viewer
    # remain the same objects, so the window never closes during model edits.
    # The whole runtime module set is reloaded in one pass; reloading a hand
    # picked subset once paired a fresh controller with a stale module and
    # stopped the window mid-run.
    reload_runtime_modules()
    import born_wired.feature_routed as feature
    extra = {}
    if config.get('controller') in ('reflex', 'embodied'):
        import born_wired.reflex_controller as reflex
        cls = reflex.ReflexController
        if config.get('controller') == 'embodied':
            import born_wired.embodied as embodied
            cls = embodied.EmbodiedController
            # The arena carries four eye muscles. Handing their limits to the
            # controller is what makes them part of the graph; without this the
            # window would run an animal that cannot look, and every eye_* knob
            # in the configuration would be inert.
            if body.has_eyes:
                extra['eye_limits'] = (body.eye_lower_limits, body.eye_upper_limits)
    else:
        cls = feature.FeatureRoutedController
    return cls(body.home_angles, body.lower_limits, body.upper_limits,
               **extra, **config.get('parameters', {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', type=Path, default=STATUS)
    parser.add_argument('--allow-multiple', action='store_true', help='Development-only separate viewer')
    args = parser.parse_args()
    status_path = args.status
    if not args.allow_multiple and status_path.exists():
        try:
            prior = json.loads(status_path.read_text(encoding='utf-8'))
            if prior.get('viewer_running') and time.time()-prior.get('wall_time',0) < 4:
                return
        except (OSError, ValueError):
            pass
    config = read_config()
    body = Go2Body(config.get('model', r'C:\mujoco_models\unitree_go2\scene.xml'))
    brain = build_brain(body, config)
    observation = body.observe()
    controls = dict(locomotion=config.get('locomotion', 0.), flexion=config.get('flexion', 0.), cues=np.zeros(4), features=np.zeros(4), autonomy=True, startle=0.)
    events = dict(reload=False, reset=False, push=False, paused=False)
    revision = config.get('revision', 1)
    updates = 0
    last_error = None
    stamp = CONFIG.stat().st_mtime_ns
    next_status = 0.
    # Which engine is running, and how fast, belongs in the status file: it is
    # the difference between a number a reader can check and one they must take
    # on trust.
    paced = dict(stamp=time.time(), sim=body.data.time, brain_ms=0., speed=0.)
    senses = None
    eyes = ears = None
    eye_pixels = None
    next_eye_time = 0.
    reset_token = config.get('body_reset_token', 0)
    body_resets = 0

    def key(code):
        if code in (87, 83):
            controls['locomotion'] = float(np.clip(controls['locomotion'] + (.1 if code == 87 else -.1), 0, 1))
        elif code in (70, 86):
            controls['flexion'] = float(np.clip(controls['flexion'] + (.1 if code == 70 else -.1), 0, 1))
        elif code == 32:
            controls['locomotion'] = controls['flexion'] = 0.
            controls['features'][:] = controls['cues'][:] = 0
        elif code == 80: events['paused'] = not events['paused']
        elif code == 72: events['reload'] = True
        elif code == 82: events['reset'] = True
        elif code == 84: events['push'] = True
        elif code == 69: controls['autonomy'] = not controls['autonomy']
        elif code == 74: controls['startle'] = 1.
        elif code in (65, 68, 88):
            controls['features'][:] = 0
            if code != 88: controls['features'][0 if code == 65 else 3] = 1
        elif code == 67: controls['cues'][0] = 1 - controls['cues'][0]

    with viewer_module.launch_passive(body.model, body.data, key_callback=key) as view:
        view.cam.azimuth, view.cam.elevation, view.cam.distance = 135, -20, 2.2
        while view.is_running():
            start = time.perf_counter()
            if start >= next_status:
                new_stamp = CONFIG.stat().st_mtime_ns
                if new_stamp != stamp:
                    events['reload'] = True
                    stamp = new_stamp
            if events['reload']:
                events['reload'] = False
                try:
                    candidate_config = read_config()
                    candidate = build_brain(body, candidate_config)
                    # This explicit development reload initializes a new brain;
                    # it is counted and never reported as uninterrupted learning.
                    resume_error = last_error is not None
                    brain, config = candidate, candidate_config
                    controls['locomotion'] = config.get('locomotion', 0.)
                    controls['flexion'] = config.get('flexion', 0.)
                    revision = config.get('revision', revision)
                    if config.get('body_reset_token', 0) != reset_token:
                        reset_token = config.get('body_reset_token', 0)
                        events['reset'] = True
                    updates += 1
                    last_error = None
                    if resume_error: events['paused'] = False
                    # Sensor adapters hold renderers and audio windows built
                    # from the modules loaded when they were created, so they
                    # are dropped here and rebuilt from the reloaded modules on
                    # the next control step.
                    if eyes is not None:
                        eyes.close()
                    senses = eyes = ears = None
                except Exception:
                    last_error = traceback.format_exc()
            if events['reset']:
                observation = body.reset()
                body_resets += 1
                next_eye_time = 0.
                events['reset'] = False
                last_error = None
                events['paused'] = False
            if events['push']:
                body.apply_force([0, 30, 0], .2)
                events['push'] = False
            if not events['paused']:
                try:
                    inputs = dict(flexion=controls['flexion'], locomotion=controls['locomotion'],
                                  cues=controls['cues'], features=controls['features'], dt=.01)
                    if config.get('controller') in ('reflex', 'embodied'):
                        if senses is None:
                            from born_wired.reflex_senses import ReflexSenses
                            senses = ReflexSenses(body)
                        inputs['environment'] = senses.observe()
                        inputs['autonomy'] = controls['autonomy']
                        inputs['startle'] = controls['startle']
                        controls['startle'] = 0.
                    if config.get('controller') == 'embodied':
                        if eyes is None:
                            from born_wired.stereo_senses import RawEyes
                            from born_wired.binaural_senses import BinauralSenses
                            # The eyes are built to the picture size the brain
                            # was built at, so the raw frames the retina sees
                            # are always the frames its wiring expects.
                            eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
                            ears = BinauralSenses(body, window_samples=160)
                        eye_pixels = eyes.observe_raw()
                        inputs['eye_pixels'] = eye_pixels
                        inputs['ear_waveform'] = ears.observe()
                    brain_started = time.perf_counter()
                    target, activation = brain.step(observation, **inputs)
                    paced['brain_ms'] = 1000*(time.perf_counter() - brain_started)
                    if brain.eye_encoder is not None:
                        body.command_eyes(brain.eye_command())
                    observation = body.step(target, duration=.01, activation=activation)
                except Exception:
                    last_error = traceback.format_exc()
                    events['paused'] = True
            with view.lock():
                view.cam.lookat[:] = body.data.qpos[:3]
            if start >= next_status:
                now = time.time()
                if now > paced['stamp']:
                    paced['speed'] = (body.data.time - paced['sim'])/(now - paced['stamp'])
                paced['sim'], paced['stamp'] = body.data.time, now
                activity = brain.network.activity
                state = {name: float(activity[brain.groups[name]].mean()) for name in
                         ('locomotion', 'curiosity', 'fatigue', 'rest', 'brake', 'initiation',
                          'appetitive', 'aversive', 'orienting', 'binocular', 'rhythm_recruitment',
                          'wall_contact') if name in brain.groups}
                labels = 'BORN WIRED CORTEX\nLive physics / local learning ON\nRevision / neurons\nSimulation time\nExternal drive / flexion\nRhythm / forward drive\nCuriosity / fatigue / rest\nAppetitive / aversive\nBinocular / wall contact\nAutonomy / obstacle brake\nBody height / brain reloads\nE autonomy, J startle\nW/S drive, F/V flex\nA/D feature, X clear\nT push, P pause\nH reload, R body reset\nSpace clear external inputs\nBottom: LEFT EYE / RIGHT EYE / BOTH FUSED'
                values = f'\n\n{revision} / {brain.network.n_neurons}\n{body.data.time:.1f}s\n{controls["locomotion"]:.2f} / {controls["flexion"]:.2f}\n{state.get("rhythm_recruitment", 0):.2f} / {state.get("locomotion", 0):.2f}\n{state.get("curiosity", 0):.2f} / {state.get("fatigue", 0):.2f} / {state.get("rest", 0):.2f}\n{state.get("appetitive", 0):.2f} / {state.get("aversive", 0):.2f}\n{state.get("binocular", 0):.3f} / {state.get("wall_contact", 0):.2f}\n{controls["autonomy"]} / {state.get("brake", 0):.2f}\n{observation["body_height"]:.3f}m / {updates}'
                if last_error: values += '\nERROR: see live_error.txt'
                view.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values)])
                if eye_pixels is not None:
                    # Display magnification only; the brain receives raw pixels.
                    # The third panel is the two pictures averaged, so a target the
                    # eyes have not converged on shows up as two ghosts, not one.
                    panels = (eye_pixels[0], eye_pixels[1],
                              ((eye_pixels[0].astype(np.uint16)
                                + eye_pixels[1].astype(np.uint16))//2).astype(np.uint8))
                    # Zoom is display magnification only; it is sized so three
                    # panels still fit the window at the largest eye.
                    zoom = max(1, min(3, 900 // (3*panels[0].shape[1])))
                    height, width = panels[0].shape[:2]
                    view.set_images([(mujoco.MjrRect(12+index*(width*zoom+6), 12,
                                                     width*zoom, height*zoom),
                                      np.repeat(np.repeat(panel, zoom, 0), zoom, 1))
                                     for index, panel in enumerate(panels)])
                status = dict(pid=os.getpid(), viewer_running=view.is_running(), simulation_time=body.data.time,
                              revision=revision, neurons=brain.network.n_neurons, edges=len(brain.synapses.src),
                              auditory_neurons=getattr(getattr(brain,'auditory',None),'n_neurons',0),
                              body_height=observation['body_height'], body_up_z=float(observation['body_up'][2]),
                              position=body.data.qpos[:3].tolist(), paused=events['paused'],
                              brain_reloads=updates, error=last_error,
                              body_resets=body_resets, learning_interval=brain.network.learning_interval,
                              eye_width=brain.eye_width, eye_height=brain.eye_height,
                              eye_muscles=int(brain.eye_encoder is not None),
                              retina_neurons=int(brain.groups['photoreceptors'].size),
                              engine=torch_execution.name_of(torch_execution.resolve()),
                              brain_ms=paced['brain_ms'], sim_speed=paced['speed'],
                              neural_state=state, external_drive=controls['locomotion'], autonomy=controls['autonomy'],
                              wall_time=time.time())
                temp = status_path.with_suffix('.tmp')
                temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
                temp.replace(status_path)
                if last_error: (ROOT/'artifacts/live_error.txt').write_text(last_error, encoding='utf-8')
                next_status = start + .5
            view.sync()
    if eyes is not None:
        eyes.close()


if __name__ == '__main__':
    main()
