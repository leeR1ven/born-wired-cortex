# -*- coding: utf-8 -*-
"""追红球：眼睛看到球偏在哪边，身子就往哪边拐。

这只狗是两样东西拼起来的：
  - 线上 200 代演化出来的走路基因（artifacts/云端/题1f_演化_云_冠军.jsonl）；
  - 红球接线（眼睛盯球，tools/wire_red_gaze.py）+ 追球接线（eye_muscles 里的 chase）。

追球那条线的道理：球偏在左边时，眼肌「往左转」的神经元一直亮着（球没落到两眼正中间就一直
推），于是「往左拐」的转向细胞也一直亮着，狗就一直往左拐；球转到两眼正中间，眼肌那两个
神经元灭了，拐弯自己就停。不靠算角度，只靠这两个眼肌神经元谁亮着。

这脚本一次只跑一种接法，用来把「哪个转向细胞是往左拐」量出来。多个候选的比较在
tools/hunt_chase.py。

    python tools/chase_red_ball.py --left 0,3 --right 1,2 --bearing 0.5
"""
import argparse
import json
import math
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
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")
FAR = (60., 60., -8.)

# 红球逃跑的几个旋钮（用户 2026-10-01：球不是自己乱飘，而是**始终远离狗**，
#   只是跑的过程中是弯的；至少要够狗追着它绕场地一圈）。
WANDER_TAU = 1.0        # 弯的「记忆时间」（秒）：越小拐得越急
WANDER_SIGMA = 1.3      # 弯得有多厉害（弧度/√秒）
WANDER_MAX = 1.15       # 最多偏离「背对狗」多少弧度（约 66 度）—— 再大就不像「一直逃」了
TURN_RATE = 8.0         # 球每秒最多把朝向扳多少（弧度/秒）

TURN_NAMES = ("steering[0]", "steering[1]", "steering_inhibition[0]", "steering_inhibition[1]",
              "wall_turn[0]", "wall_turn[1]", "wall_turn_inhibition[0]", "wall_turn_inhibition[1]")


def load_champion(path=CHAMPION):
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("冠军文件是空的：%s" % path)
    return json.loads(lines[0])


def open_field(body, rgba=(1., 0., 0., 1.)):
    """空旷场地：四面墙挪走，中间那颗球改成红的小球（云端那一趟就是这么跑的）。"""
    for name in WALLS:
        wall = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if wall >= 0:
            body.model.geom_pos[wall] = list(FAR)
    geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    if geom < 0:
        raise SystemExit("场地里没有 green_target")
    body.model.geom_size[geom] = [.06]*3
    body.model.geom_rgba[geom] = list(rgba)
    return geom


def gaze_spec():
    table = W.load_table()
    return W.spec_of(table, .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))


def build(genome, chase, seed=0, eye=True, spec=None, ball_rgba=(1., 0., 0., 1.)):
    ctx = tb.context(genome, seed)
    body = tb.clean_body(ctx["model_path"])
    geom = open_field(body, ball_rgba)
    parameters = dict(ctx["parameters"])
    if eye:
        parameters["eye_red"] = gaze_spec() if spec is None else spec
    if chase is not None:
        parameters["chase_red"] = chase
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               seed=seed,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    return body, brain, eyes, geom


def snapshot(brain):
    """把这只狗脑子的细胞状态存一份（刚建出来那一刻的）。

    用户 2026-09-30：同一只狗要测两遍，球分别放左边和右边。两趟之间要把脑子复位，
    不然第二趟是从第一趟跑完的状态起跑，两趟就没法比了。
    """
    net = brain.network
    saved = {"voltage": np.array(net._voltage, copy=True),
             "adaptation": np.array(net._adaptation, copy=True)}
    device = getattr(net, "_device", None)
    if device is not None:
        saved["voltage_device"] = device.voltage.clone()
        saved["adaptation_device"] = device.adaptation.clone()
    return saved


def restore(brain, saved):
    """把脑子放回 snapshot 那一刻。"""
    net = brain.network
    net._voltage[:] = saved["voltage"]
    net._adaptation[:] = saved["adaptation"]
    device = getattr(net, "_device", None)
    if device is not None and "voltage_device" in saved:
        torch = device.torch
        with torch.no_grad():
            device.voltage.copy_(saved["voltage_device"])
            device.adaptation.copy_(saved["adaptation_device"])
        device.stale = False
    auditory = getattr(brain, "auditory", None)
    if auditory is not None and hasattr(auditory, "reset"):
        auditory.reset()


def measure(body, brain, eyes, geom, seconds, bearing, distance, elevation=0., seed=0,
            flee=0., escape=6., curve=0., ball=True):
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    left_eye = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
    right_eye = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
    body.reset(seed=seed, joint_noise=.01)
    G.place_in_front(body.model, body.data, geom, bearing, elevation, distance)
    if not ball:
        # 对照：球挪到天边。别的什么都不动，所以同一只狗、同一个种子下，
        # 两次的落点差多少，就是「球」这件事对它的行为有多大影响 —— 一点影响都没有
        # 说明它根本没在看球。
        body.model.geom_pos[geom] = list(FAR)
        mujoco.mj_forward(body.model, body.data)
    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()
    ball0 = ball.copy()
    wander_rng = np.random.default_rng(int(seed)*7919 + 13)
    wander = 0.0
    escape_yaw = None
    ball_track = []
    dog_track = []

    senses = ReflexSenses(body)
    observation = body.observe()
    environment = senses.observe()
    pixels = eyes.observe_raw()
    start = np.asarray(body.data.xpos[base], dtype=float).copy()
    previous = start.copy()
    path = 0.0
    lowest_up = 1.0
    headings, ranges = [], []
    started = time.perf_counter()
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
        if flee > 0.:
            # 红球躲着狗跑：**每一拍都往「背对狗」的方向挪**。球跑得比狗慢一点，
            # 所以会追的狗越追越近，不会追的狗眼看着它走远。
            # curve > 0 时，「背对狗」这个方向上再叠一个缓慢摆动的小偏角 —— 路线就是
            # 一条连续的弧线（用户 2026-10-01：不是乱飘，是始终远离狗、跑的是弯线）。
            # 偏角限死在 ±WANDER_MAX，所以它永远不会有朝狗跑的分量。
            # 地图是无限大的（用户 2026-10-01）：球不用被圈在场地里，也没有「绕一圈」
            # 这回事。它只管带着这个随机弯度背离狗跑。
            # 速度**恒定**（用户 2026-10-01）：一度试过「近了快跑、远了等它」，但那样狗
            # 永远追不上球。所以现在就是匀速逃，狗只要真会追就追得上。等迭代多了、
            # 狗确实会追了，再把 --flee 往上调给球提速。
            here = np.asarray(body.data.xpos[base], dtype=float)
            away = ball - here
            length = float(np.linalg.norm(away))
            if length > 1e-6 and (curve > 0. or length < escape):
                if curve > 0.:
                    wander += (-wander/WANDER_TAU)*DT + WANDER_SIGMA*curve*math.sqrt(DT)*float(wander_rng.normal())
                    wander = float(np.clip(wander, -WANDER_MAX, WANDER_MAX))
                    want = math.atan2(away[1], away[0]) + wander
                    if escape_yaw is None:
                        escape_yaw = want
                    step_yaw = math.atan2(math.sin(want - escape_yaw),
                                          math.cos(want - escape_yaw))
                    escape_yaw += step_yaw*min(1., TURN_RATE*DT)
                else:
                    escape_yaw = math.atan2(away[1], away[0])
                ball = ball + np.array([math.cos(escape_yaw), math.sin(escape_yaw), 0.])*flee*DT
                body.model.geom_pos[geom] = ball
                mujoco.mj_forward(body.model, body.data)
        if flee > 0.:
            ball_track.append([float(ball[0]), float(ball[1])])
            dog_track.append([float(body.data.xpos[base][0]),
                              float(body.data.xpos[base][1])])
        frame = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
        forward, leftward = frame[:, 0], frame[:, 1]
        middle = (np.asarray(body.data.xpos[left_eye], dtype=float)
                  + np.asarray(body.data.xpos[right_eye], dtype=float))/2.
        aim = ball - middle
        headings.append(float(np.arctan2(float(aim @ leftward), float(aim @ forward))))
        ranges.append(float(np.linalg.norm(aim)))
        place = np.asarray(body.data.xpos[base], dtype=float).copy()
        path += float(np.linalg.norm(place - previous))
        previous = place
        lowest_up = min(lowest_up, float(np.asarray(body.data.xmat[base]).reshape(3, 3)[2, 2]))
    end = np.asarray(body.data.xpos[base], dtype=float).copy()
    travelled = float(np.linalg.norm(end - start))
    settled = float(np.mean(ranges[-int(1./DT):]))
    facing = float(np.mean(np.abs(headings[-int(1./DT):])))
    return dict(settled=settled, facing=facing, end_xy=[float(end[0]), float(end[1])],
                start_xy=[float(start[0]), float(start[1])],
                ball0_xy=[float(ball0[0]), float(ball0[1])], heading0=headings[0], heading_end=headings[-1],
                heading_last=float(np.mean(headings[-int(1./DT):])),
                range0=ranges[0], range_end=ranges[-1], range_min=float(np.min(ranges)),
                travelled_m=travelled, path_m=path, straightness=(travelled/path if path else 0.),
                lowest_up_z=lowest_up, seconds=time.perf_counter() - started,
                headings=headings, ranges=ranges, ball_track=ball_track,
                dog_track=dog_track)


def parse_cells(text):
    if text in ("", "none", None):
        return []
    return [int(part) for part in str(text).replace(" ", "").split(",") if part != ""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--champion", default=str(CHAMPION))
    ap.add_argument("--seconds", type=float, default=8.)
    ap.add_argument("--bearing", type=float, default=0.5, help="球一开始摆在狗的左边多少弧度")
    ap.add_argument("--distance", type=float, default=1.5, help="球离两眼正中间多远")
    ap.add_argument("--elevation", type=float, default=0.)
    ap.add_argument("--turn-gain", type=float, default=None, help="拧髋关节的力，弧度；不给就不接追球")
    ap.add_argument("--gain", type=float, default=1., help="位置排到转向细胞的总增益")
    ap.add_argument("--source", default="position", choices=("position", "muscle"))
    ap.add_argument("--flip", action="store_true", help="把左右对调（用来看接反了没有）")
    ap.add_argument("--set", action="append", default=[], help="改基因：名字=数值")
    ap.add_argument("--no-eye", action="store_true", help="不接红球注视（对照用）")
    ap.add_argument("--ball-color", default="red", choices=("red", "green", "blue", "white"))
    ap.add_argument("--flee", type=float, default=0., help="球每拍往背对狗的方向挪多少米（球躲着狗跑）")
    ap.add_argument("--escape", type=float, default=6., help="球跑出这么远就不再挪了")
    ap.add_argument("--route", default="motor", choices=("motor", "orient"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    genome = dict(load_champion(Path(args.champion))["genome"])
    for item in args.set:
        name, _, value = item.partition("=")
        genome[name.strip()] = float(value)
    chase = None
    if args.turn_gain is not None:
        chase = {"turn_gain": args.turn_gain, "gain": args.gain,
                 "source": args.source, "flip": bool(args.flip), "route": args.route}
    print("追球接线：%s" % ("不接" if chase is None else
          "拧髋 %.3f 弧度、位置排总增益 %.1f、来源 %s%s"
          % (args.turn_gain, args.gain, args.source, "、左右对调" if args.flip else "")))
    print("球：左边 %.2f 弧度、%.2f 米" % (args.bearing, args.distance), flush=True)

    colors = {"red": (1., 0., 0., 1.), "green": (0., 1., 0., 1.),
              "blue": (0., 0., 1., 1.), "white": (1., 1., 1., 1.)}
    body, brain, eyes, geom = build(genome, chase, seed=args.seed, eye=not args.no_eye,
                                    ball_rgba=colors[args.ball_color])
    got = measure(body, brain, eyes, geom, args.seconds, args.bearing, args.distance,
                  args.elevation, seed=args.seed, flee=args.flee, escape=args.escape)
    eyes.close()
    if chase is not None:
        print("转向细胞收尾活动：%s" % np.round(
            [float(np.mean(brain.network.rates_at(brain.groups[name])))
             for name in ("chase_turn_left", "chase_turn_right")], 4))
    every = max(1, int(round(.5/DT)))
    print("  轨迹（每 0.5 秒）：球偏在 %s）" % " ".join(
        "%+.2f" % got["headings"][i] for i in range(0, len(got["headings"]), every)))
    print("\n球一开始在 %+.2f 弧度（狗的左前方），最后 %+.2f 弧度（%+.2f 到 %+.2f）"
          % (got["heading0"], got["heading_end"], got["heading0"], got["heading_end"]))
    print("离球：开头 %.2f 米、最后 %.2f 米、最近 %.2f 米" % (got["range0"], got["range_end"], got["range_min"]))
    print("走了 %.2f 米（路程 %.2f、直度 %.2f）最低直立分量 %.3f  用时 %.1f 秒"
          % (got["travelled_m"], got["path_m"], got["straightness"], got["lowest_up_z"], got["seconds"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(options=vars(args), result=got),
                                             ensure_ascii=False, indent=1), encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())