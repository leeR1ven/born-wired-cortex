# -*- coding: utf-8 -*-
"""看一眼冠军跑得有多快（临时脚本）。

跟 tools/evolve.py 里那趟走路考得一模一样：同一份基因、同一个种子、同样的
「空旷场地（四面墙和所有摆设都挪到天边）」、同样 10 秒、同样 learn=True、耳朵和眼球
都照常喂进去。唯一的区别是这里把每一步的位置记下来，还顺手录了动画。

    python tools/_walk_movie.py --champion artifacts/云端/题1f_演化_云_冠军.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco                                                    # noqa: E402
from PIL import Image, ImageDraw, ImageFont                      # noqa: E402

from born_wired.binaural_senses import BinauralSenses            # noqa: E402
from born_wired.go2_body import Go2Body                          # noqa: E402
from born_wired.reflex_senses import ReflexSenses                # noqa: E402
from born_wired.stereo_senses import RawEyes                     # noqa: E402
from tools import taskbank as tb                                 # noqa: E402
from tools import evolve as ev                                   # noqa: E402
from tools.validate_reflex_v3 import (NEURAL_DT, PHYSICS_DT,     # noqa: E402
                                      base_position, make_controller)

PANEL = (380, 285)
FOOT = 118
STRIDE = 10                       # 每 0.1 秒记一帧 = 10 帧/秒
CHAMPION = ROOT / "artifacts" / "云端" / "题1f_演化_云_冠军.jsonl"


def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--champion", default=str(CHAMPION))
    ap.add_argument("--seconds", type=float, default=10.)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "冠军跑得快不快.gif"))
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in
            Path(args.champion).read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[0]
    genome = dict(row["genome"])
    seed = int(row["seed"])
    print("这只：第%s代第%s只、种子 %s；当时考出来是 %.2f 米、最低直立分量 %.2f"
          % (row.get("gen"), row.get("index"), seed, row.get("travelled_m", float("nan")),
             row.get("min_up_z", float("nan"))), flush=True)

    ctx = tb.context(genome, seed)
    steps = int(round(args.seconds/NEURAL_DT))
    body = Go2Body(model_path=ctx["model_path"], timestep=PHYSICS_DT)
    initial_yaw = float(np.random.default_rng(seed).uniform(-.12, .12))
    body.reset(seed=seed, joint_noise=.01, tilt=(0., 0., initial_yaw))
    body.data.qpos[:2] = (0., 0.)
    for prop_name, place in ev.open_props().items():
        prop = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, prop_name)
        if prop >= 0:
            body.model.geom_pos[prop] = list(place)
    mujoco.mj_forward(body.model, body.data)
    controller = make_controller(body, seed=seed,
                                 controller_parameters=ctx["parameters"])
    senses = ReflexSenses(body)
    observation = body.observe()
    environment = senses.observe()
    eyes = (RawEyes(body, width=controller.eye_width, height=controller.eye_height)
            if body.model.ncam >= 2 else None)
    ears = BinauralSenses(body, window_samples=160)
    base = body._base
    pixels = None

    renderer = mujoco.Renderer(body.model, height=PANEL[1], width=PANEL[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 62.
    cam.elevation = -13.
    cam.distance = 2.5

    track, shots, stamps = [], [], []
    for step_index in range(steps):
        if eyes is not None and step_index % 10 == 0:
            pixels = eyes.observe_raw()
        target, activation = controller.step(observation, environment=environment,
                                            autonomy=True, locomotion=0.,
                                            dt=NEURAL_DT, learn=True, startle=0.,
                                            eye_pixels=pixels,
                                            ear_waveform=ears.observe())
        if getattr(controller, "eye_encoder", None) is not None:
            body.command_eyes(controller.eye_command())
        observation = body.step(target, duration=NEURAL_DT, activation=activation)
        environment = senses.observe()
        position = base_position(body)
        track.append(np.asarray(position, dtype=float).copy())
        if step_index % STRIDE == 0:
            cam.lookat[:] = [position[0], position[1], position[2] + .04]
            renderer.update_scene(body.data, camera=cam)
            shots.append(Image.fromarray(renderer.render()).copy())
            stamps.append(step_index*NEURAL_DT)

    track = np.asarray(track)
    walked = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(track[:, :2], axis=0),
                                                            axis=1))])
    hop = np.linalg.norm(track[:, :2] - track[0, :2], axis=1)
    window = int(round(.5/NEURAL_DT))
    speed = np.zeros(len(track))
    for i in range(1, len(track)):
        j = max(0, i - window)
        speed[i] = (walked[i] - walked[j])/((i - j)*NEURAL_DT)
    print("\n跑了 %.1f 秒（模拟时间）：" % args.seconds)
    print("  从起点直线拉出去 %.2f 米，脚下的路走了 %.2f 米" % (hop[-1], walked[-1]))
    print("  全程平均 %.2f 米/秒 = %.2f 公里/小时"
          % (hop[-1]/args.seconds, hop[-1]/args.seconds*3.6))
    print("  最稳的时候（3 秒之后那段的中位数）%.2f 米/秒 = %.2f 公里/小时"
          % (np.median(speed[300:]), np.median(speed[300:])*3.6))
    print("  最快的一小段 %.2f 米/秒 = %.2f 公里/小时" % (speed.max(), speed.max()*3.6))
    print("  最低的直立分量 %.3f（小于 0.3 就是摔了）" % track[:, 2].min())

    big, mid, small = font(15), font(13), font(12)
    span = track[:, :2]
    lo = span.min(axis=0) - .6
    hi = span.max(axis=0) + .6
    side = max(hi - lo)
    lo = lo - (side - (hi - lo))/2.
    hi = lo + side

    def to_px(xy):
        u = (xy[0] - lo[0])/(hi[0] - lo[0])
        v = (xy[1] - lo[1])/(hi[1] - lo[1])
        return int(u*(PANEL[0] - 24) + 12), int(PANEL[1] - 12 - v*(PANEL[1] - 24))

    frames = []
    for k, shot in enumerate(shots):
        i = min(int(round(stamps[k]/NEURAL_DT)), len(track) - 1)
        canvas = Image.new("RGB", (PANEL[0]*2, PANEL[1] + FOOT), (22, 22, 26))
        canvas.paste(shot, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), "跟在它后面拍的", font=mid, fill=(255, 220, 60))
        left = PANEL[0]
        draw.rectangle([left, 0, left + PANEL[0], PANEL[1]], fill=(18, 26, 18))
        gx = np.ceil(lo[0])
        while gx <= hi[0]:
            x, _ = to_px((gx, lo[1]))
            draw.line([x, 12, x, PANEL[1] - 12], fill=(38, 52, 38))
            draw.text((x + 2, 2), "%.0f米" % gx, font=small, fill=(90, 120, 90))
            gx += 1.
        gy = np.ceil(lo[1])
        while gy <= hi[1]:
            _, y = to_px((lo[0], gy))
            draw.line([left + 12, y, left + PANEL[0] - 12, y], fill=(38, 52, 38))
            draw.text((left + 14, y - 14), "%.0f米" % gy, font=small, fill=(90, 120, 90))
            gy += 1.
        path_px = [to_px(point) for point in track[:i + 1:5]]
        if len(path_px) > 1:
            draw.line(path_px, fill=(255, 90, 90), width=3)
        sx, sy = to_px(track[0, :2])
        draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(90, 160, 255))
        draw.text((sx + 7, sy - 16), "起点", font=small, fill=(120, 190, 255))
        dx, dy = to_px(track[i, :2])
        draw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=(255, 240, 90))
        draw.text((left + 6, PANEL[1] - 20), "俯视：它走过的路线（每格 1 米）",
                  font=mid, fill=(255, 220, 60))
        base_y = PANEL[1]
        draw.rectangle([0, base_y, canvas.width, canvas.height], fill=(14, 14, 18))
        draw.text((10, base_y + 8), "秒表 %4.1f 秒" % (i*NEURAL_DT), font=big,
                  fill=(230, 230, 230))
        draw.text((10, base_y + 34), "已经走了 %.2f 米" % float(walked[i]), font=big,
                  fill=(120, 230, 140))
        draw.text((10, base_y + 60), "此刻 %.2f 米/秒（%.1f 公里/小时）"
                  % (speed[i], speed[i]*3.6), font=big, fill=(255, 200, 90))
        draw.text((10, base_y + 86),
                  "全程平均 %.2f 米/秒 = %.2f 公里/小时   人走路约 1.4，狗小跑约 2~3"
                  % (hop[-1]/args.seconds, hop[-1]/args.seconds*3.6),
                  font=small, fill=(190, 190, 200))
        bx0, bx1 = PANEL[0] + 16, canvas.width - 16
        by = base_y + 20
        draw.rectangle([bx0, by, bx1, by + 16], fill=(40, 40, 46))
        draw.rectangle([bx0, by, bx0 + int((bx1 - bx0)*min(1., speed[i]/3.)), by + 16],
                       fill=(255, 120, 80))
        draw.text((bx0, by + 22), "速度条（满格 = 3 米/秒）", font=small, fill=(190, 190, 200))
        mark = bx0 + int((bx1 - bx0)*min(1., hop[-1]/args.seconds/3.))
        draw.line([mark, by + 2, mark, by + 14], fill=(120, 230, 140), width=3)
        frames.append(canvas)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000*STRIDE*NEURAL_DT), loop=0, optimize=True)
    print("\n动画 -> %s（%d 帧、%.1f 秒、%.1f MB）"
          % (out, len(frames), len(frames)*STRIDE*NEURAL_DT, out.stat().st_size/1e6))

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    times = np.arange(len(track))*NEURAL_DT
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=130)
    axes[0].plot(times, walked, color="#3aa6ff", lw=2)
    axes[0].set_xlabel("秒（模拟时间）")
    axes[0].set_ylabel("累计走了（米）")
    axes[0].set_title("%.0f 秒走了 %.1f 米" % (args.seconds, walked[-1]))
    axes[0].grid(alpha=.3)
    axes[1].plot(times, speed, color="#ff8a3a", lw=2)
    axes[1].axhline(hop[-1]/args.seconds, color="#3ad07a", ls="--",
                    label="全程平均 %.2f 米/秒" % (hop[-1]/args.seconds))
    axes[1].axhline(1.4, color="#888", ls=":", label="人走路 1.4 米/秒")
    axes[1].set_xlabel("秒（模拟时间）")
    axes[1].set_ylabel("米/秒")
    axes[1].set_title("速度（每半秒量一段）")
    axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    chart = out.with_suffix(".png")
    fig.savefig(chart)
    print("曲线 -> %s" % chart)
    xy = out.with_name(out.stem + "_路线.png")
    fig2, ax2 = plt.subplots(figsize=(4.6, 4.6), dpi=130)
    ax2.plot(track[:, 0], track[:, 1], color="#e05555", lw=2)
    ax2.scatter([track[0, 0]], [track[0, 1]], color="#3aa6ff", zorder=3, label="起点")
    ax2.scatter([track[-1, 0]], [track[-1, 1]], color="#ffd23a", zorder=3,
                label="%.0f 秒后" % args.seconds)
    ax2.set_aspect("equal")
    ax2.grid(alpha=.3)
    ax2.legend(fontsize=8)
    ax2.set_title("走过的路线（俯视，每格 1 米）")
    fig2.tight_layout()
    fig2.savefig(xy)
    print("路线 -> %s" % xy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())