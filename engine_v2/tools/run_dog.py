"""Run the neural controller in MuJoCo, optionally recording a real simulation GIF.

Keyboard: W/S increase/decrease rhythmic drive; F/R flexion drive; C toggle
cue 0; Space clear drives. These are sensory inputs, never joint trajectories.
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body, DEFAULT_MODEL
from born_wired.innate import InnateController
from born_wired.feature_routed import FeatureRoutedController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--demo', action='store_true', help='scheduled sensory-input demonstration')
    parser.add_argument('--feature-routes', action='store_true', help='enable hidden-layer feature routes; A/D select feature, X clears it')
    parser.add_argument('--record', type=Path, help='save a GIF (implies headless)')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    body = Go2Body(args.model)
    observation = body.reset('crouch' if args.demo else 'stand')
    controller_type = FeatureRoutedController if args.feature_routes else InnateController
    brain = controller_type(body.home_angles, body.lower_limits, body.upper_limits)
    drives = dict(flexion=0., locomotion=0., cue=0.)
    features = np.zeros(4)
    viewer = renderer = None
    frames = []
    if args.record:
        from PIL import Image, ImageDraw, ImageFont
        renderer = mujoco.Renderer(body.model, height=360, width=640)
        font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 16)
    elif not args.headless:
        from mujoco import viewer as viewer_module
        def key(code):
            if code == 32:
                for name in drives:
                    drives[name] = 0.
                features[:] = 0
            elif code in (87, 83):
                drives['locomotion'] = float(np.clip(drives['locomotion'] + (.2 if code == 87 else -.2), 0, 1))
            elif code in (70, 82):
                drives['flexion'] = float(np.clip(drives['flexion'] + (.2 if code == 70 else -.2), 0, 1))
            elif code == 67:
                drives['cue'] = 1 - drives['cue']
            elif code in (65, 68, 88):
                features[:] = 0
                if code != 88:
                    features[0 if code == 65 else 3] = 1
        viewer = viewer_module.launch_passive(body.model, body.data, key_callback=key)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -18, 1.35
    try:
        for tick in range(round(args.seconds / .01)):
            started = time.perf_counter()
            t = tick * .01
            if args.demo:
                # Experiment stimuli only. All joint commands are computed by
                # the neural graph, including when a learned cue is tested.
                drives['flexion'] = .8 if 3 <= t < 9 else 0.
                drives['cue'] = float(3 <= t < 9 or 12 <= t < 16)
                drives['locomotion'] = .8 if 19 <= t < 29 else 0.
            feature_input = {'features': features} if args.feature_routes else {}
            target, activation = brain.step(observation, flexion=drives['flexion'],
                                            locomotion=drives['locomotion'],
                                            cues=[drives['cue'], 0, 0, 0], dt=.01, **feature_input)
            observation = body.step(target, duration=.01, activation=activation)
            if renderer and tick % 10 == 0:
                camera.lookat[:] = body.data.qpos[:3]
                renderer.update_scene(body.data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 51), fill='#10202d')
                draw.text((10, 3), 'MuJoCo 真实仿真 · 神经连接产生运动 · 局部学习开启', font=font, fill='white')
                draw.text((10, 27), f"{t:4.1f}s   屈伸输入 {drives['flexion']:.1f}   节律输入 {drives['locomotion']:.1f}   提示 {drives['cue']:.0f}", font=font, fill='#acd8ee')
                frames.append(frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
            if viewer:
                if not viewer.is_running():
                    break
                viewer.cam.lookat[:] = body.data.qpos[:3]
                viewer.sync()
                time.sleep(max(0, .01 - (time.perf_counter() - started)))
        if frames:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(brain.diagnostics())
    finally:
        if viewer:
            viewer.close()
        if renderer:
            renderer.close()


if __name__ == '__main__':
    main()
