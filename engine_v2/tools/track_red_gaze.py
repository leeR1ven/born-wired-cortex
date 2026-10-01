# -*- coding: utf-8 -*-
"""两只眼一起盯着红球：静态斜视检查 + 移动小球跟踪 + 视频。

-eye 每只眼只被自己那只眼的红细胞驱动，所以两只眼是各转各的：各自把球对到自己画面
正中，球越近两只眼越往里转。这个工具就是量这件事。

    python tools/track_red_gaze.py --mode strabismus          # 球摆在正前方不同距离
    python tools/track_red_gaze.py --mode path --gif artifacts\眼睛追球.gif
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
# 图上写的都是中文，matplotlib 自带那套字体没有汉字，会画成一排方框。
for _name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    if any(_name == _font.name for _font in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [_name]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
from PIL import Image                                              # noqa: E402

import wire_red_gaze as W                                          # noqa: E402
from tools import taskbank as tb                                   # noqa: E402
from tools import eye_geometry as G                                # noqa: E402

DT = .01
BIG = ROOT / "artifacts" / "红球位置野_红_大.json"


def ids(body, kind, name):
    return mujoco.mj_name2id(body.model, kind, name)


def head_frame(body):
    base = ids(body, mujoco.mjtObj.mjOBJ_BODY, "base")
    return (np.asarray(body.data.xpos[base], dtype=float),
            np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3))


def eye_world(body):
    out = []
    for side in ("left", "right"):
        i = ids(body, mujoco.mjtObj.mjOBJ_BODY, "eye_" + side)
        out.append(np.asarray(body.data.xpos[i], dtype=float))
    return out


def place(body, target, bearing, elevation, distance):
    G.place_ball(body.model, body.data, target, bearing, elevation, distance)


def wanted(body, target):
    """每只眼要转到多少弧度才算正对球 —— 这只眼自己的坐标系，不是头坐标系。"""
    origin, rotation = head_frame(body)
    ball = np.asarray(body.data.geom_xpos[target], dtype=float)
    out = []
    for side, position in zip(("left", "right"), eye_world(body)):
        v = rotation.T @ (ball - position)
        yaw = float(np.arctan2(v[1], v[0]))
        pitch = -float(np.arctan2(v[2], np.hypot(v[0], v[1])))
        out.append(dict(side=side, yaw=yaw, pitch=pitch,
                        distance=float(np.linalg.norm(ball - position))))
    return out


def redness(image):
    """画面里红得比别的颜色多的那些像素，重心就是球的像。"""
    r = image[:, :, 0].astype(float)
    rest = np.maximum(image[:, :, 1], image[:, :, 2]).astype(float)
    weight = np.clip(r - rest, 0., None)
    if weight.sum() <= 0:
        return None
    rows, columns = np.indices(weight.shape)
    return float((columns*weight).sum()/weight.sum()), float((rows*weight).sum()/weight.sum())


def run_path(args):
    table = W.load_table(args.table)
    spec = W.spec_of(table, args.split, args.move_push, args.hold_push, np.radians(args.dead),
                     cell_time=args.cell_time, latch=args.latch, cross=args.cross,
                     motor_time=args.motor_time, gain=args.gain)
    ctx = tb.context({}, args.seed)
    body, brain, eyes, target = W.build(ctx, spec)
    environment = tb.blank_environment()
    period = float(args.seconds)
    frames, trace = [], []
    steps = int(round(period/DT)) + args.settle
    for step in range(steps):
        t = max(0., (step - args.settle)*DT)
        phase = 2.*np.pi*t/period
        if args.path == "circle":
            bearing, elevation = args.span*np.cos(phase), args.lift*np.sin(phase)
        else:
            bearing, elevation = args.span*np.sin(phase), args.lift*.0
        place(body, target, bearing, elevation, args.distance)
        observation = body.observe()
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
        command = brain.eye_command()
        body.command_eyes(command)
        body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        physical = np.asarray(body.eye_angles, dtype=float)
        ask = wanted(body, target)
        image = eyes.observe_raw()
        spots = [redness(image[0]), redness(image[1])]
        origin, rotation = head_frame(body)
        offset = rotation.T @ (np.asarray(body.data.geom_xpos[target], dtype=float) - origin)
        row = dict(step=step, t=round(t, 3),
                   bearing=round(float(np.arctan2(offset[1], offset[0])), 4),
                   elevation=round(float(np.arctan2(offset[2], np.hypot(offset[0], offset[1]))), 4),
                   distance=args.distance)
        for k, side in enumerate(("left", "right")):
            row[side+"_yaw"] = round(float(physical[2*k]), 4)
            row[side+"_pitch"] = round(float(physical[2*k+1]), 4)
            row[side+"_want_yaw"] = round(ask[k]["yaw"], 4)
            row[side+"_want_pitch"] = round(ask[k]["pitch"], 4)
            row[side+"_error"] = round(float(physical[2*k]) - ask[k]["yaw"], 4)
            row[side+"_pitch_error"] = round(float(physical[2*k+1]) - ask[k]["pitch"], 4)
            row[side+"_spot"] = None if spots[k] is None else [round(spots[k][0], 1),
                                                               round(spots[k][1], 1)]
        trace.append(row)
        if step >= args.settle and (step - args.settle) % args.every == 0 and len(frames) < args.frames:
            frames.append(draw(body, target, image, ask, physical, row, args))
    eyes.close()
    moving = [row for row in trace if row["t"] > 0]
    tail = [row for row in moving if row["t"] > period*.5]
    report = dict(options=vars(args), rows=len(trace),
                  left_error_median=float(np.median([r["left_error"] for r in tail])),
                  right_error_median=float(np.median([r["right_error"] for r in tail])),
                  left_worst=float(np.max(np.abs([r["left_error"] for r in tail]))),
                  right_worst=float(np.max(np.abs([r["right_error"] for r in tail]))),
                  convergence_actual=float(np.mean([r["left_yaw"] - r["right_yaw"] for r in tail])),
                  convergence_wanted=float(np.mean([r["left_want_yaw"] - r["right_want_yaw"]
                                                    for r in tail])),
                  trace=trace)
    print("跟球 %d 步（每 %.2f 秒一个来回）；后半程：左眼偏离中位 %.4f 最差 %.4f，"
          "右眼中位 %.4f 最差 %.4f"
          % (len(moving), period, report["left_error_median"], report["left_worst"],
             report["right_error_median"], report["right_worst"]))
    print("两眼的夹角（会聚）：实际 %.4f 弧度、该有 %.4f 弧度"
          % (report["convergence_actual"], report["convergence_wanted"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print("读数 -> %s" % args.out)
    if args.gif and frames:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=args.frame_ms,
                       loop=0, optimize=True)
        print("动画 -> %s（%d 帧）" % (args.gif, len(frames)))
        still = Path(args.gif).with_suffix(".png")
        frames[len(frames)//2].save(still)
        print("静图 -> %s" % still)
    return 0


def draw(body, target, image, ask, physical, row, args):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=90)
    for k, side in enumerate(("left", "right")):
        axis = axes[k]
        axis.imshow(image[k])
        height, width = image[k].shape[:2]
        axis.plot([width/2.], [height/2.], marker="+", color="#19d219", markersize=16,
                  markeredgewidth=1.6, linestyle="none")
        spot = row[side+"_spot"]
        if spot is not None:
            axis.plot([spot[0]], [spot[1]], marker="o", color="#ff2d2d", markersize=13,
                      markerfacecolor="none", markeredgewidth=1.8, linestyle="none")
        axis.set_title("%s eye  (want %+.1f deg, at %+.1f)"
                       % (side, np.degrees(ask[k]["yaw"]), np.degrees(physical[2*k])), fontsize=9)
        axis.set_xticks([]); axis.set_yticks([])
    axis = axes[2]
    origin, rotation = head_frame(body)
    ball = np.asarray(body.data.geom_xpos[target], dtype=float)
    axis.plot([ball[0]], [ball[1]], marker="o", color="#d81b1b", markersize=12)
    for k, side in enumerate(("left", "right")):
        eye = eye_world(body)[k]
        yaw = float(physical[2*k])
        gaze = rotation @ np.array([np.cos(yaw), np.sin(yaw), 0.])
        axis.plot([eye[0], eye[0] + gaze[0]*.6], [eye[1], eye[1] + gaze[1]*.6],
                  color="#1f77b4", linewidth=2)
        axis.plot([eye[0], ball[0]], [eye[1], ball[1]], color="#bbbbbb", linewidth=1,
                  linestyle="--")
        axis.plot([eye[0]], [eye[1]], marker="s", color="#1f77b4", markersize=7)
    axis.set_xlim(-.1, 1.3); axis.set_ylim(-.9, .9); axis.set_aspect("equal")
    axis.set_title("上面看：球 %.2f 米，两眼夹角 %+.1f 度（该 %+.1f 度）"
                   % (np.linalg.norm(ball[:2] - origin[:2]),
                      np.degrees(physical[0] - physical[2]),
                      np.degrees(ask[0]["yaw"] - ask[1]["yaw"])), fontsize=9)
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("红球 %+.1f 度 / %+.1f 度   t=%.2f 秒   偏离：左 %.1f 度 右 %.1f 度"
                    % (np.degrees(row["bearing"]), np.degrees(row["elevation"]), row["t"],
                       np.degrees(row["left_error"]), np.degrees(row["right_error"])), fontsize=10)
    figure.tight_layout()
    figure.canvas.draw()
    frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
    plt.close(figure)
    return frame


def run_strabismus(args):
    table = W.load_table(args.table)
    spec = W.spec_of(table, args.split, args.move_push, args.hold_push, np.radians(args.dead),
                     cell_time=args.cell_time, latch=args.latch, cross=args.cross,
                     motor_time=args.motor_time, gain=args.gain)
    ctx = tb.context({}, args.seed)
    body, brain, eyes, target = W.build(ctx, spec)
    environment = tb.blank_environment()
    print("%8s | %9s %9s | %9s %9s | %9s %9s | %8s"
          % ("距离 米", "左眼要", "左眼实际", "右眼要", "右眼实际", "左眼偏", "右眼偏", "画面列差"))
    rows = []
    for distance in args.distances:
        place(body, target, 0., 0., distance)
        for _ in range(args.steps):
            observation = body.observe()
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(brain.eye_command())
            body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        physical = np.asarray(body.eye_angles, dtype=float)
        ask = wanted(body, target)
        image = eyes.observe_raw()
        spots = [redness(image[0]), redness(image[1])]
        gap = None
        if spots[0] and spots[1]:
            gap = spots[0][0] - spots[1][0]
        row = dict(distance=distance,
                   want=[ask[0]["yaw"], ask[1]["yaw"]],
                   got=[float(physical[0]), float(physical[2])],
                   error=[float(physical[0]) - ask[0]["yaw"], float(physical[2]) - ask[1]["yaw"]],
                   spot=spots, gap=gap,
                   convergence=float(physical[0] - physical[2]),
                   want_convergence=float(ask[0]["yaw"] - ask[1]["yaw"]))
        rows.append(row)
        print("%8.2f | %+9.4f %+9.4f | %+9.4f %+9.4f | %+9.4f %+9.4f | %8s"
              % (distance, ask[0]["yaw"], physical[0], ask[1]["yaw"], physical[2],
                 row["error"][0], row["error"][1],
                 "—" if gap is None else "%.1f" % gap))
        print("%8s | 两眼夹角实际 %+.4f 弧度、该有 %+.4f 弧度（差 %+.4f）"
              % ("", row["convergence"], row["want_convergence"],
                 row["convergence"] - row["want_convergence"]))
    eyes.close()
    if args.out:
        Path(args.out).write_text(json.dumps(dict(options=vars(args), rows=rows),
                                             ensure_ascii=False, indent=1), encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("path", "strabismus"), default="path")
    ap.add_argument("--table", default=str(BIG))
    ap.add_argument("--split", type=float, default=.10)
    ap.add_argument("--gain", type=float, default=2.5)
    ap.add_argument("--move-push", type=float, default=.35)
    ap.add_argument("--hold-push", type=float, default=.10)
    ap.add_argument("--cell-time", type=float, default=.01)
    ap.add_argument("--motor-time", type=float, default=.20)
    ap.add_argument("--latch", type=float, default=0.)
    ap.add_argument("--cross", type=float, default=0.)
    ap.add_argument("--dead", type=float, default=3.)
    ap.add_argument("--distance", type=float, default=.90)
    ap.add_argument("--span", type=float, default=.28, help="球左右摆多大（弧度）")
    ap.add_argument("--lift", type=float, default=.16, help="球上下摆多大（弧度）")
    ap.add_argument("--seconds", type=float, default=2.40, help="一个来回多少秒")
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--frame-ms", type=int, default=70)
    ap.add_argument("--steps", type=int, default=140, help="静态时每个距离跑多少步")
    ap.add_argument("--distances", type=float, nargs="*", default=[.50, .70, .90, 1.20, 1.80, 3.00])
    ap.add_argument("--path", choices=("sweep", "circle"), default="sweep")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return run_strabismus(args) if args.mode == "strabismus" else run_path(args)


if __name__ == "__main__":
    raise SystemExit(main())