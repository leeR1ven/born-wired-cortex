# -*- coding: utf-8 -*-
"""把球放到视野里很多方位，看哪些神经元跟着亮。

    python tools/measure_red_ball_features.py --colour red --out artifacts/红球特征神经元.json

要教一只动物"看见红球就转眼睛把球送到画面正中"，第一步不是接线，是**先知道红球
在画面某个方位时，脑子里是哪些神经元在亮**。这个工具只做这一步：把球摆在很多
个方位上，每一步都读一遍视觉那几层细胞的读数，然后回答三个问题。

1. 球在画面左边 / 中间 / 右边时，亮起来的是第几列细胞（`retinal_opponent` 的红色
   通道，按列加起来），这个列的编号跟球真实方位是不是单调对得上。
2. 同上，上下方向对应的是第几行。
3. 18 格那层粗细胞（`retina`，两只眼 x 左中右 x 红绿蓝）红格亮了多少。

球放在离身体 `--distance` 米的地方，方位按真实几何算（`atan2`），不靠像素猜。
眼睛全程不动：这里读的是"给它看这个画面，细胞怎么答"，不是"它会怎么转眼睛"。
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.embodied import EmbodiedController      # noqa: E402
from born_wired.go2_body import Go2Body                 # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
SCENERY = ("red_pillar", "blue_box", "front_block", "curb", "low_step", "platform",
           "ramp", "passage_a", "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
DT = .01
COLOURS = {"red": [1., 0., 0., 1.], "green": [.12, .85, .15, 1.], "blue": [.1, .15, .95, 1.]}


def centroid(profile):
    """Where the activity sits along a 1-D profile: weighted mean, or -1 if dark."""
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    if total <= 1e-9:
        return -1.
    return float((profile * np.arange(profile.size)).sum() / total)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--colour", choices=sorted(COLOURS), default="red")
    ap.add_argument("--bearings", default="-0.5,-0.3,-0.15,0,0.15,0.3,0.5")
    ap.add_argument("--elevations", default="0,-0.25,0.25")
    ap.add_argument("--distance", type=float, default=.90)
    ap.add_argument("--size", type=float, default=.06)
    ap.add_argument("--settle", type=int, default=40, help="先跑多少步让它稳定")
    ap.add_argument("--read", type=int, default=20, help="再跑多少步取平均")
    ap.add_argument("--scenery", action="store_true", help="留着场地里的摆设（默认清空）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    body = Go2Body(model_path=ARENA)
    if not args.scenery:
        for name in SCENERY:
            geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [args.size] * 3
    body.model.geom_rgba[target] = COLOURS[args.colour]
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits))
    eyes = RawEyes(body)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    eye_height = float(body.data.xpos[base][2]) + .03

    shape = brain.eye_shape
    rows, columns = shape[1], shape[2]
    bearings = [float(v) for v in args.bearings.split(",")]
    elevations = [float(v) for v in args.elevations.split(",")]
    print("球色 %s，%g 米远，直径 %g 米，清空摆设 %s，视网膜 %d 行 x %d 列"
          % (args.colour, args.distance, args.size, not args.scenery, rows, columns))
    print("%9s %9s %7s %7s %7s %7s %9s %9s %9s" % (
        "方位(度)", "高低(度)", "左眼列", "右眼列", "左右差", "峰行", "左眼红量", "红像元", "18格红"))

    records = []
    for elevation in elevations:
        for bearing in bearings:
            x = args.distance * np.cos(elevation) * np.cos(bearing)
            y = args.distance * np.cos(elevation) * np.sin(bearing)
            z = eye_height + args.distance * np.sin(elevation)
            body.model.geom_pos[target] = [x, y, z]
            mujoco.mj_forward(body.model, body.data)
            offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
            rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
            true_bearing = float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0]))
            true_elevation = float(np.arctan2(offset @ rotation[:, 2],
                                              np.linalg.norm(offset @ rotation[:, :2])))
            red_total, rows_total, column_centroid, pixels = [], [], [], []
            retina_red = []
            for step in range(args.settle + args.read):
                activation = brain.step(observation, environment=environment,
                                        eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
                observation = body.step(np.asarray(body.home_angles), duration=DT,
                                        activation=activation)
                if step < args.settle:
                    continue
                rates = np.array(brain.network.activity, dtype=float)
                opponent = rates[brain.groups["retinal_opponent"]].reshape(shape)
                red = opponent[:, :, :, 0]
                red_total.append(red.sum(axis=(1, 2)))
                rows_total.append(red.sum(axis=2))
                column_centroid.append([centroid(red[eye].sum(axis=0)) for eye in range(2)])
                picture = rates[brain.groups["photoreceptors"]].reshape(shape)
                pixels.append(picture[0][:, :, :].reshape(-1, 3).max(axis=0))
                retina_red.append(rates[brain.groups["retina"]].reshape(2, 3, 3)[:, :, 0])
            red_total = np.mean(red_total, axis=0)
            rows_total = np.mean(rows_total, axis=0)
            column_centroid = np.mean(column_centroid, axis=0)
            retina_red = np.mean(retina_red, axis=0)
            row_centroid = [centroid(rows_total[eye]) for eye in range(2)]
            left_column, right_column = column_centroid
            record = dict(colour=args.colour, bearing=round(true_bearing, 4),
                          elevation=round(true_elevation, 4),
                          left_column=round(float(left_column), 3),
                          right_column=round(float(right_column), 3),
                          left_row=round(float(row_centroid[0]), 3),
                          right_row=round(float(row_centroid[1]), 3),
                          left_red=round(float(red_total[0]), 3),
                          right_red=round(float(red_total[1]), 3),
                          left_retina_red=[round(float(v), 4) for v in retina_red[0]],
                          left_pixel_max=[round(float(v), 3) for v in np.mean(pixels, axis=0)])
            records.append(record)
            print("%9.1f %9.1f %7.1f %7.1f %7.1f %7.1f %9.1f %9s %9s" % (
                np.degrees(true_bearing), np.degrees(true_elevation),
                left_column, right_column, right_column - left_column,
                row_centroid[0], red_total[0], record["left_pixel_max"], record["left_retina_red"]))
    eyes.close()

    usable = [r for r in records if r["left_column"] >= 0]
    if len(usable) >= 3:
        x = np.array([r["bearing"] for r in usable])
        y = np.array([r["left_column"] for r in usable])
        slope, intercept = np.polyfit(y, x, 1)
        residual = float(np.abs(np.polyval([slope, intercept], y) - x).max())
        print("\n列 -> 方位 的直线拟合（左眼）：方位 = %.6f * 列 + %.4f（最大偏差 %.4f 弧度，%d 个方位）"
              % (slope, intercept, residual, len(usable)))
        print("也就是说：第 %g 列大约在 %.1f 度，每列大约 %.2f 度"
              % (intercept / slope if slope else 0., np.degrees(intercept), np.degrees(slope)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(options=vars(args), records=records),
                                             ensure_ascii=False, indent=1), encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())