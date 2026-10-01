# -*- coding: utf-8 -*-
"""把一只会追球的候选录成动画（临时脚本）。

跟 tools/chase_task.py 考的一模一样：同一只狗（基因 + 它自己那套追球接法）、
同一个种子、同一套摆球法（球摆在两眼正中间的正前方），每段 6 秒。
镜头跟着狗和球的中间，所以能看清它是绕过去、还是真的朝球拐。

    python tools/_chase_movie.py --ledger <keep.jsonl> --index 335
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco                                                    # noqa: E402
from PIL import Image, ImageDraw, ImageFont                      # noqa: E402

import chase_red_ball as C                                       # noqa: E402

DT = .01
PANEL = (420, 300)
FOOT = 96
STRIDE = 10
SEGMENTS = ((.5, 3.0), (-.5, 3.0), (.3, 4.5))


def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def run_segment(genome, chase, seed, spec, bearing, distance, seconds, cam):
    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
    renderer = mujoco.Renderer(body.model, height=PANEL[1], width=PANEL[0])
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    body.reset(seed=seed, joint_noise=.01)
    C.G.place_in_front(body.model, body.data, geom, bearing, 0., distance)
    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()
    senses = C.ReflexSenses(body)
    observation = body.observe()
    environment = senses.observe()
    pixels = eyes.observe_raw()
    track, ranges, shots, stamps = [], [], [], []
    for step in range(int(round(seconds/DT))):
        if getattr(brain, "eye_encoder", None) is not None:
            body.command_eyes(brain.eye_command())
        observation = body.observe()
        if step % 10 == 0:
            pixels = eyes.observe_raw()
        target, activation = brain.step(observation, environment=environment, autonomy=True,
                                        locomotion=0., dt=DT, learn=False, eye_pixels=pixels,
                                        startle=0.)
        observation = body.step(target, duration=DT, activation=activation)
        environment = senses.observe()
        here = np.asarray(body.data.xpos[base], dtype=float).copy()
        track.append(here)
        ranges.append(float(np.linalg.norm(ball - here)))
        if step % STRIDE == 0:
            middle = (here + ball)/2.
            apart = float(np.linalg.norm(ball[:2] - here[:2]))
            cam.lookat[:] = [middle[0], middle[1], middle[2] + .05]
            cam.distance = float(np.clip(apart*1.6, 3.0, 8.0))
            cam.azimuth = 95.
            cam.elevation = -30.
            renderer.update_scene(body.data, camera=cam)
            shots.append(Image.fromarray(renderer.render()).copy())
            stamps.append(step*DT)
    eyes.close()
    renderer.close()
    return dict(track=np.asarray(track), ranges=np.asarray(ranges), ball=ball,
                shots=shots, stamps=stamps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=6.)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in
            Path(args.ledger).read_text(encoding="utf-8").splitlines() if line.strip()]
    row = [r for r in rows if int(r["index"]) == args.index][0]
    genome, chase, seed = dict(row["genome"]), dict(row["chase"]), int(row["seed"])
    print("第%d只：路 %s%s pivot=%s gain=%.2f 拐弯力 %.4f  复试贴到 %s/18"
          % (row["index"], chase["route"], " 对调" if chase["flip"] else "",
             chase.get("pivot"), chase.get("gain", 1.), genome.get("avoidance_gain", 0.),
             (row.get("verify") or {}).get("hits", "?")), flush=True)

    spec = C.gaze_spec()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    big, mid, small = font(15), font(13), font(12)
    frames, notes = [], []
    for bearing, distance in SEGMENTS:
        got = run_segment(genome, chase, seed, spec, bearing, distance, args.seconds, cam)
        track, ball, ranges = got["track"], got["ball"], got["ranges"]
        print("  球在 %+.1f 弧度、%.1f 米：最近贴到 %.2f 米，走了 %.2f 米"
              % (bearing, distance, ranges.min(),
                 float(np.linalg.norm(track[-1, :2] - track[0, :2]))), flush=True)
        notes.append((bearing, distance, float(ranges.min())))
        lo = np.minimum(track[:, :2].min(axis=0), ball[:2]) - .5
        hi = np.maximum(track[:, :2].max(axis=0), ball[:2]) + .5
        side = max(hi - lo)
        lo = lo - (side - (hi - lo))/2.
        hi = lo + side
        for k, shot in enumerate(got["shots"]):
            i = min(int(round(got["stamps"][k]/DT)), len(track) - 1)
            canvas = Image.new("RGB", (PANEL[0]*2, PANEL[1] + FOOT), (22, 22, 26))
            canvas.paste(shot, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 4), "球摆在 %s%.1f 弧度、%.1f 米处" % ("左" if bearing > 0 else "右",
                      abs(bearing), distance), font=mid, fill=(255, 220, 60))
            left = PANEL[0]
            draw.rectangle([left, 0, left + PANEL[0], PANEL[1]], fill=(18, 26, 18))

            def to_px(xy, lo=lo, hi=hi):
                u = (xy[0] - lo[0])/(hi[0] - lo[0])
                v = (xy[1] - lo[1])/(hi[1] - lo[1])
                return int(u*(PANEL[0] - 24) + 12 + left), int(PANEL[1] - 12 - v*(PANEL[1] - 24))

            bx, by = to_px(ball[:2])
            draw.ellipse([bx - 7, by - 7, bx + 7, by + 7], fill=(255, 40, 40))
            draw.text((bx + 9, by - 16), "红球", font=small, fill=(255, 120, 120))
            path_px = [to_px(point[:2]) for point in track[:i + 1:5]]
            if len(path_px) > 1:
                draw.line(path_px, fill=(255, 200, 90), width=3)
            sx, sy = to_px(track[0, :2])
            draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(90, 160, 255))
            dx, dy = to_px(track[i, :2])
            draw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=(255, 255, 120))
            draw.text((left + 6, PANEL[1] - 20), "俯视：黄线是它走过的路", font=mid,
                      fill=(255, 220, 60))
            base_y = PANEL[1]
            draw.rectangle([0, base_y, canvas.width, canvas.height], fill=(14, 14, 18))
            draw.text((10, base_y + 8), "秒表 %4.1f 秒" % (i*DT), font=big, fill=(230, 230, 230))
            draw.text((10, base_y + 34), "离红球 %.2f 米（最近 %.2f）"
                      % (ranges[i], ranges.min()), font=big, fill=(120, 230, 140))
            draw.text((10, base_y + 60), "身子朝球那一边偏：%s"
                      % ("在拐了" if abs(ranges[i] - ranges[max(0, i - 50)]) > .02 else "直走"),
                      font=big, fill=(255, 200, 90))
            draw.text((10, base_y + 82), "全程离球最近 %.2f 米 —— 小于 1 米就算「走到球跟前」"
                      % ranges.min(), font=small, fill=(190, 190, 200))
            frames.append(canvas)

    out = Path(args.out) if args.out else (ROOT / "artifacts" /
                                           ("第%d只追球.gif" % args.index))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000*STRIDE*DT), loop=0, optimize=True)
    print("\n动画 -> %s（%d 帧、%.1f 秒、%.1f MB）"
          % (out, len(frames), len(frames)*STRIDE*DT, out.stat().st_size/1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())