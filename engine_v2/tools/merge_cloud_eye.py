# -*- coding: utf-8 -*-
"""把「线上演化学到的走路基因」和「红球接线（盯红球）」合成一只狗，当场检验两件事：

  1. 它还会不会走 —— 和「只有走路基因、不接红球」的对照比：走了多远、有没有摔；
  2. 走的时候眼睛有没有盯着红球 —— 用高分辨率画面量红球偏这只眼画面中心多少度。

两件事同时成立才算融合成功。红球一直摆在狗眼前（左右慢慢晃），所以「走路」和「看球」
互不干扰；对照那一趟球也照样摆着，差别只有红球接线有没有接上。
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco                                                       # noqa: E402
from born_wired.embodied import EmbodiedController                  # noqa: E402
from born_wired.reflex_senses import ReflexSenses                   # noqa: E402
from born_wired.stereo_senses import RawEyes                        # noqa: E402

import wire_red_gaze as W                                           # noqa: E402
import eye_geometry as G                                            # noqa: E402
from tools import taskbank as tb                                    # noqa: E402

DT = .01
CHAMPION = ROOT / "artifacts" / "云端" / "题1f_演化_云_冠军.jsonl"
HIGH, WIDE = 360, 480
BALL_EVERY = 20                     # 每 0.2 秒量一次红球在画面里的位置
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")
FAR = (60., 60., -8.)


def load_champion(path=CHAMPION):
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("冠军文件是空的：%s" % path)
    return json.loads(lines[0])


def offset_of_ball(renderer, per_pixel):
    image = np.asarray(renderer.render()).astype(int)
    mask = (image[:, :, 0] > 150) & (image[:, :, 1] < 90) & (image[:, :, 2] < 90)
    if mask.sum() < 20:
        return None
    rows, columns = np.nonzero(mask)
    return float((columns.mean() - WIDE/2.)*per_pixel), float((rows.mean() - HIGH/2.)*per_pixel)


def run(genome, seconds, use_eye, spec, seed=0, distance=1.15, label="", ball=True):
    ctx = tb.context(genome, seed)
    body = tb.clean_body(ctx["model_path"])
    # clean_body 只挪道具，四面墙还立在原地：云端那一趟是「空旷场地」，墙也是挪走的
    # （tools/evolve.py 的 open_props 同样挪这几面墙）。漏了这一步，狗走 3 米就撞墙——
    # 2026-09-30 实测：没挪墙时「不摆球」那一趟只走 3.40 米还翻了。
    for name in WALLS:
        wall = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if wall >= 0:
            body.model.geom_pos[wall] = list(FAR)
    geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    if geom < 0:
        raise SystemExit("场地里没有 green_target")
    body.model.geom_size[geom] = [.06]*3
    body.model.geom_rgba[geom] = [1., 0., 0., 1.]
    parameters = dict(ctx["parameters"])
    if use_eye:
        parameters["eye_red"] = spec
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               seed=seed,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    renderers = [mujoco.Renderer(body.model, height=HIGH, width=WIDE) for _ in range(2)]
    cameras = []
    for side in ("left", "right"):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_" + side)
        cameras.append(camera)
    half = np.radians(float(body.model.cam_fovy[cameras[0].fixedcamid]))/2.
    per_pixel = np.degrees(np.arctan(np.tan(half)*WIDE/HIGH))/WIDE

    body.reset(seed=seed, joint_noise=.01)
    if not ball:
        body.model.geom_pos[geom] = [60., 60., -8.]   # 球挪到天边：这就是云端那一趟的条件
        mujoco.mj_forward(body.model, body.data)
    senses = ReflexSenses(body)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    start = np.asarray(body.data.xpos[base], dtype=float).copy()
    previous = start.copy()
    observation = body.observe()
    environment = senses.observe()
    pixels = eyes.observe_raw()
    path = 0.0
    lowest_up = 1.0
    offsets = []
    started = time.perf_counter()
    for step in range(int(round(seconds/DT))):
        if ball:
            G.place_in_front(body.model, body.data, geom, .35*np.sin(step*DT*.8), .05, distance)
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
        if step % BALL_EVERY == 0:
            for i in range(2):
                renderers[i].update_scene(body.data, camera=cameras[i])
            offsets.append([offset_of_ball(renderers[i], per_pixel) for i in range(2)])
        place = np.asarray(body.data.xpos[base], dtype=float).copy()
        path += float(np.linalg.norm(place - previous))
        previous = place
        lowest_up = min(lowest_up, float(np.asarray(body.data.xmat[base]).reshape(3, 3)[2, 2]))
    end = np.asarray(body.data.xpos[base], dtype=float).copy()
    travelled = float(np.linalg.norm(end - start))
    for renderer in renderers:
        renderer.close()
    eyes.close()
    per_eye = []
    for i in (0, 1):
        side_offsets = [row[i] for row in offsets if row[i] is not None]
        seen = np.array([abs(value[0]) for value in side_offsets]) if side_offsets else np.array([np.nan])
        per_eye.append(dict(n=len(side_offsets), mean=float(np.nanmean(seen)),
                            worst=float(np.nanmax(seen)),
                            within6=float(np.mean(seen <= 6.)) if side_offsets else float("nan")))
    return dict(label=label, travelled_m=travelled, path_m=path,
                straightness=(travelled/path if path else 0.), lowest_up_z=lowest_up,
                eyes=per_eye, seconds=time.perf_counter() - started)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--champion", default=str(CHAMPION))
    ap.add_argument("--seconds", type=float, default=10.)
    ap.add_argument("--distance", type=float, default=1.15, help="红球摆在眼前多远")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "融合_云端走路+红球接线.json"))
    args = ap.parse_args(argv)

    row = load_champion(Path(args.champion))
    genome = row["genome"]
    print("冠军：第 %d 代 第 %d 只  线上读数 走了 %.2f 米、没摔=%s、直度 %.2f"
          % (row.get("gen", -1), row.get("index", -1), row.get("travelled_m", float("nan")),
             row.get("upright"), row.get("straightness", float("nan"))), flush=True)
    print("基因 %d 条" % len(genome), flush=True)
    spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))
    runs = []
    plans = ((True, True, "融合：走路基因 + 红球接线（球在眼前）"),
             (False, True, "对照一：只有走路基因（球也在眼前）"),
             (True, False, "对照二：走路基因 + 红球接线，但不摆球（和云端同一条件）"))
    for use_eye, ball, label in plans:
        print("\n%s ..." % label, flush=True)
        runs.append(run(genome, args.seconds, use_eye, spec, seed=args.seed,
                        distance=args.distance, label=label, ball=ball))
        got = runs[-1]
        print("  走了 %.2f 米（路程 %.2f、直度 %.2f）最低直立分量 %.3f  左眼偏心 %.1f 度、"
              "右眼偏心 %.1f 度  用时 %.1f 秒"
              % (got["travelled_m"], got["path_m"], got["straightness"], got["lowest_up_z"],
                 got["eyes"][0]["mean"], got["eyes"][1]["mean"], got["seconds"]), flush=True)
    Path(args.out).write_text(json.dumps(dict(champion=str(args.champion), seconds=args.seconds,
                                              distance=args.distance, runs=runs),
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n结论：", flush=True)
    merged, plain, cloud_like = runs[0], runs[1], runs[2]
    print("  走路：融合 %.2f 米 vs 对照 %.2f 米（差 %+.2f 米）"
          % (merged["travelled_m"], plain["travelled_m"],
             merged["travelled_m"] - plain["travelled_m"]), flush=True)
    print("  看球：融合 左 %.1f / 右 %.1f 度 vs 对照 左 %.1f / 右 %.1f 度"
          % (merged["eyes"][0]["mean"], merged["eyes"][1]["mean"],
             plain["eyes"][0]["mean"], plain["eyes"][1]["mean"]), flush=True)
    print("  存到 %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())