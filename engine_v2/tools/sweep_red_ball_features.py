# -*- coding: utf-8 -*-
"""红球扫过整个视野，逐点停留，把"红球在哪儿"的特征神经元全找出来。

    python tools/sweep_red_ball_features.py --colour red --out artifacts/红球扫场_红.json

一个圆球在视网膜上亮的是一**片方块**，不是一列。所以这里不按列汇总，而是：

1. 把球在视野里铺成一个格子（默认左右 31 个方位 x 上下 13 个高低 = 403 个位置），
   每个位置停 `--hold` 步，取后 `--read` 步的平均，读下面这些层的每一个细胞：
   `retinal_opponent`（每个像元一个红/绿/蓝通道细胞）、`retinal_contrast`、
   `binocular`（位置 x 眼 x 视差）、`retinal_memory`（4x4x3 的整块记忆）、`retina`（2x3x3 粗码）。
2. 对每个细胞算它自己的**位置野**：在 403 个位置里它最亮的那一个（最爱方位/高低）、
   以及选择性 z =（峰值 - 平均）/ 标准差。z 大 = 这个细胞只在球到某个位置时亮。
3. 打印：某一位置视网膜上亮起来的方块（ASCII 图，看它是块不是列）、
   视网膜格子的位置野是否整齐地铺满视野、以及选择性最强的一批细胞。

眼睛全程不动，读的是"给它看这个位置的红球，哪些细胞答话"。
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
LAYERS = ("retinal_opponent", "retinal_contrast", "binocular", "retinal_memory", "retina")
RAMP = " .:-=+*#%@"


def ascii_map(patch):
    peak = float(patch.max())
    lines = []
    for row in patch:
        if peak <= 1e-9:
            lines.append(" " * len(row))
        else:
            lines.append("".join(RAMP[min(9, int(9.999 * max(0., v) / peak))] for v in row))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--colour", choices=sorted(COLOURS), default="red")
    ap.add_argument("--bearings", default="-0.45,0.45,31", help="起,止,个数")
    ap.add_argument("--elevations", default="-0.36,0.36,13", help="起,止,个数")
    ap.add_argument("--distance", type=float, default=.90)
    ap.add_argument("--size", type=float, default=.06)
    ap.add_argument("--hold", type=int, default=12)
    ap.add_argument("--read", type=int, default=6)
    ap.add_argument("--scenery", action="store_true")
    ap.add_argument("--patch-at", default="0.0,0.0", help="打 ASCII 方块图的球位置：方位,高低（弧度）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    def spread(text):
        low, high, count = (float(v) for v in text.split(","))
        return np.linspace(low, high, int(count))

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

    ids = {name: np.asarray(brain.groups[name]) for name in LAYERS}
    sizes = {name: len(ids[name]) for name in LAYERS}
    bearings = spread(args.bearings)
    elevations = spread(args.elevations)
    places = [(b, e) for e in elevations for b in bearings]
    maps = {name: np.zeros((sizes[name], len(places)), np.float32) for name in LAYERS}
    truth = []
    print("球色 %s，%g 米远，直径 %g 米；扫 %d 个位置（左右 %d x 上下 %d），每个停 %d 步读 %d 步"
          % (args.colour, args.distance, args.size, len(places), len(bearings), len(elevations),
             args.hold, args.read))
    print("层的大小：" + "、".join("%s %d" % (n, sizes[n]) for n in LAYERS))

    def place(bearing, elevation):
        x = args.distance * np.cos(elevation) * np.cos(bearing)
        y = args.distance * np.cos(elevation) * np.sin(bearing)
        z = eye_height + args.distance * np.sin(elevation)
        body.model.geom_pos[target] = [x, y, z]
        mujoco.mj_forward(body.model, body.data)
        offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
        rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
        return (float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0])),
                float(np.arctan2(offset @ rotation[:, 2],
                                 np.linalg.norm(offset @ rotation[:, :2]))))

    place(0., 0.)
    for _ in range(30):
        observation = body.step(np.asarray(body.home_angles), duration=DT,
                                activation=brain.step(observation, environment=environment,
                                                      eye_pixels=eyes.observe_raw(), dt=DT,
                                                      learn=False)[1])
    patch_target = tuple(float(v) for v in args.patch_at.split(","))
    patch_best = None
    for index, (bearing, elevation) in enumerate(places):
        truth.append(place(bearing, elevation))
        for step in range(args.hold):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
            if step >= args.hold - args.read:
                rates = np.array(brain.network.activity, dtype=float)
                for name in LAYERS:
                    maps[name][:, index] += rates[ids[name]]
    eyes.close()
    for name in LAYERS:
        maps[name] /= float(args.read)
    truth = np.array(truth)

    red_maps = np.zeros((brain.eye_shape[1]*brain.eye_shape[2], len(places)), np.float32)
    report = dict(colour=args.colour, options=vars(args), places=truth.tolist(), layers={})
    print("\n=== 1. 视网膜上亮的是方块还是列？（左眼红通道，球在 %.2f, %.2f 弧度）===" % patch_target)
    index = int(np.argmin(((truth - np.array(patch_target)) ** 2).sum(axis=1)))
    opponent = maps["retinal_opponent"][:, index].reshape(brain.eye_shape)
    red_maps = maps["retinal_opponent"][0::3][:brain.eye_shape[1]*brain.eye_shape[2]]  # 每 3 个取 1 个 = 红色通道；前 36x48 个 = 左眼
    patch = opponent[0, :, :, 0]
    lines = ascii_map(patch)
    print("    " + "".join("%d" % (c % 10) for c in range(patch.shape[1])))
    for row, line in enumerate(lines):
        print("%3d %s" % (row, line))
    lit = patch > .5 * patch.max() if patch.max() > 1e-9 else np.zeros_like(patch, bool)
    rows_lit = np.where(lit.any(axis=1))[0]
    cols_lit = np.where(lit.any(axis=0))[0]
    print("方块范围：第 %d~%d 行、第 %d~%d 列，共 %d 个细胞（整幅图是 %d x %d = %d 个）"
          % (rows_lit.min(), rows_lit.max(), cols_lit.min(), cols_lit.max(), int(lit.sum()),
             patch.shape[0], patch.shape[1], patch.size))
    frac_row = (patch > .5 * patch.max()).sum(axis=1) if patch.max() > 1e-9 else None
    if frac_row is not None:
        print("最亮那一行有 %d 个细胞亮；最亮那一列有 %d 个细胞亮（所以是块，不是列）"
              % (int(frac_row.max()), int((patch > .5 * patch.max()).sum(axis=0).max())))
    report["patch"] = dict(bearing=truth[index].tolist(), rows=rows_lit.tolist(),
                           columns=cols_lit.tolist(), lit=int(lit.sum()),
                           map=[[round(float(v), 4) for v in row] for row in patch])

    print("\n=== 1b. 每一个位置上的亮块中心（左眼）===")
    print("高低\\方位" + "".join("%7.1f" % np.degrees(b) for b in bearings))
    blob_rows, blob_cols, blob_width, blob_height, seen = [], [], [], [], []
    for line, elevation in enumerate(elevations):
        centers_row, centers_col = [], []
        for column, bearing in enumerate(bearings):
            index = line * len(bearings) + column
            patch = red_maps[:, index].reshape(brain.eye_shape[1], brain.eye_shape[2])
            peak = float(patch.max())
            if peak <= .05:
                centers_row.append(float("nan"))
                centers_col.append(float("nan"))
                continue
            lit = patch > .5 * peak
            weight = np.where(lit, patch, 0.)
            center_row = float((weight.sum(axis=1) * np.arange(patch.shape[0])).sum() / weight.sum())
            center_col = float((weight.sum(axis=0) * np.arange(patch.shape[1])).sum() / weight.sum())
            seen.append(index)
            centers_row.append(center_row)
            centers_col.append(center_col)
            blob_rows.append(center_row)
            blob_cols.append(center_col)
            blob_width.append(float(lit.sum(axis=0).max()))
            blob_height.append(float(lit.sum(axis=1).max()))
        print("%7.1f" % np.degrees(elevation)
              + "".join("%7.1f" % v for v in centers_col))
        print("%7s" % ""
              + "".join("%7.1f" % v for v in centers_row))
    blob_rows = np.array(blob_rows)
    blob_cols = np.array(blob_cols)
    visible = np.array(seen, dtype=int)
    true_bearing = np.degrees(truth[visible][:, 0])
    true_elevation = np.degrees(truth[visible][:, 1])
    if len(blob_cols) >= 5:
        column_fit = np.polyfit(true_bearing, blob_cols, 1)
        column_bad = float(np.abs(np.polyval(column_fit, true_bearing) - blob_cols).max())
        row_fit = np.polyfit(true_elevation, blob_rows, 1)
        row_bad = float(np.abs(np.polyval(row_fit, true_elevation) - blob_rows).max())
        print("看得到的球 %d / %d 个位置；亮块平均 %0.1f x %0.1f 个细胞"
              % (len(blob_cols), len(places), np.mean(blob_width), np.mean(blob_height)))
        print("列中心 = %0.4f * 方位 + %0.2f（最大偏差 %0.1f 列 = %0.1f 度）"
              % (column_fit[0], column_fit[1], column_bad,
                 column_bad / abs(column_fit[0]) if column_fit[0] else float("nan")))
        print("行中心 = %0.4f * 高低 + %0.2f（最大偏差 %0.1f 行 = %0.1f 度）"
              % (row_fit[0], row_fit[1], row_bad,
                 row_bad / abs(row_fit[0]) if row_fit[0] else float("nan")))
        report["blob"] = dict(visible=int(len(blob_cols)), width=float(np.mean(blob_width)),
                              height=float(np.mean(blob_height)),
                              column_fit=[float(v) for v in column_fit], column_bad=column_bad,
                              row_fit=[float(v) for v in row_fit], row_bad=row_bad)

    print("\n=== 2. 视网膜格子的位置野（球在这个位置时，它最亮）===")
    print("%9s %9s %9s %9s %9s" % ("格子(行,列)", "最爱方位", "最爱高低", "选择性z", "峰值"))
    cells = maps["retinal_opponent"][:, :].reshape(brain.eye_shape[0], brain.eye_shape[1],
                                                  brain.eye_shape[2], brain.eye_shape[3],
                                                  len(places))
    red = cells[0, :, :, 0].reshape(-1, len(places))
    red_maps = maps["retinal_opponent"][0::3][:brain.eye_shape[1]*brain.eye_shape[2]]  # 每 3 个取 1 个 = 红色通道；前 36x48 个 = 左眼
    for row in (6, 12, 18, 24, 30):
        for col in (8, 16, 24, 32, 40):
            profile = red[row * brain.eye_shape[2] + col]
            if profile.max() <= 1e-6:
                continue
            best = int(np.argmax(profile))
            z = (profile.max() - profile.mean()) / (profile.std() + 1e-9)
            print("%9s %8.1f° %8.1f° %9.1f %9.3f"
                  % ("(%d,%d)" % (row, col), np.degrees(truth[best][0]),
                     np.degrees(truth[best][1]), z, profile.max()))

    print("\n=== 3. 各层里位置选择性最强的细胞 ===")
    print("%-20s %8s %8s %10s %9s %9s" % ("层", "细胞号(内)", "选择性z", "最爱方位", "最爱高低", "峰值"))
    for name in LAYERS:
        cell_map = maps[name]
        peak = cell_map.max(axis=1)
        mean = cell_map.mean(axis=1)
        z = (peak - mean) / (cell_map.std(axis=1) + 1e-9)
        order = np.argsort(-z)[:5]
        for cell in order:
            best = int(np.argmax(cell_map[cell]))
            print("%-20s %8d %8.1f %9.1f° %9.1f° %9.3f"
                  % (name, cell, z[cell], np.degrees(truth[best][0]),
                     np.degrees(truth[best][1]), peak[cell]))
        report["layers"][name] = dict(size=int(sizes[name]),
                                      selective=int((z > 6).sum()),
                                      best_z=float(z[order[0]]) if len(order) else 0.)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        print("\n读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())