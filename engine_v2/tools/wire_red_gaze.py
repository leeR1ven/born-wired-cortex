# -*- coding: utf-8 -*-
"""眼睛第一关：红球注视。手动接线 —— 每根线都由实测的位置野表定死，不训练、不随机。

规则（全部来自 artifacts/红球位置野_红.json 这张逐细胞实测表）：

1. 起点是 retinal_opponent 的红色通道细胞（球是红的，这层只认红色）。细胞 i 的位置野
   说：球落在（方位 b_i，高低 e_i）时它最亮。实测表里列号到方位不是直线（中间密、
   边缘疏），所以用表，不用拟合直线。
2. 一个细胞答话的角度要按**这只眼自己**的坐标系算，不是头坐标系。表里记的是「球摆在头
   坐标系的哪个方位」，可这只眼要回答的是「球偏到我画面的哪一边」。同一个球摆在正前方
   0.9 米，左眼看到它偏右 0.181 弧度、右眼看到它偏左 0.181 弧度，两只眼都该往里转 ——
   这就是会聚。而头坐标系里这个球是 0 度，两只眼都按「球已经在正中了」处理，谁也不动，
   看着就是斜视（2026-09-30 实测：该转 0.173 的两只眼各只转了 0.010 和 0.023 弧度）。
   换法在 retina_angles：细胞在视网膜上的行列是它自己的坐标，拿实测表标定出「这个行列
   对应这只眼的多少角度」。行列到角度由眼球本身定死，所以标定跟球离多远无关：球越近，
   像落在越靠边的行列上，眼睛自然转得更多，会聚是这么来的，不是另写的规则。
3. 一个细胞出多大力 = 该方向这一档的「每弧度增益」* |它自己看到的角度|。球偏得越远，
   摊到的那片细胞越用力，所以「中央轻、越偏越重」是权重本身带出来的，不是另写的判断。
4. 方位为正（球在这只眼的左手边）的细胞接到 eye_red_*_left（+yaw 那一路），为负接
   right；高低为正（球在这只眼上方）接 up，为负接 down。+yaw 是往左看、+pitch 是往下
   看，这两个符号由 tools/probe_eye_axis.py 直接量出来（把眼球转 +0.30 弧度，红球亮块
   的行列往哪跑）。
5. 两种力度，而且各自按自己那一档的满量程归一：
   - 「保持」档：离正视方向近的那批细胞，推力小（默认 0.15 弧度），
     接线粗细定成「球摆在这一档最外边（默认 0.10 弧度）时刚好用满」；
   - 「移动」档：离得远的那批细胞，推力大（默认 0.60 弧度），
     接线粗细定成「球摆在最远处（默认 0.35 弧度）时刚好用满」。
   一档里细胞多少都不改变这一档的满力，只有球的位置决定用多少力。这样近处由小档接管、
   远处由大档接管，眼睛既不会在中央发抖，也不会在边上够不着。
   不归一的话（试过），远处那一档早就爆满，眼睛就会在两个极限之间来回撞。
6. 没有信号时两档都不放电，眼肌自己的静息电流把眼睛带回正前方（那条 tonic 线本来就接着）。
7. 离正视方向 3 度以内的细胞不接（球已经在这只眼的正中了不许再动）。

别的看东西的通路（先天那套按对比度看的、按明暗变化看的、声音、记忆）全部掐到约等于 0，
这一轮只看红色那一条的功劳。

    python tools/wire_red_gaze.py --split 0.10 --move-push 0.60 --hold-push 0.15
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

from born_wired.embodied import EmbodiedController                      # noqa: E402
from born_wired.stereo_senses import RawEyes                            # noqa: E402
from tools import taskbank as tb                                        # noqa: E402
from tools import eye_geometry as G                                     # noqa: E402

DT = .01
TABLE = ROOT / "artifacts" / "红球位置野_红.json"
# 补缺口用的表：扫得更宽（方位 ±0.62），把每只眼画面最边上那两列也标定到了。
# 只在主表是空的地方用它 —— 主表其他列的标定一个字都不动。
WIDE = ROOT / "artifacts" / "红球位置野_红_宽.json"
SILENT = 1e-9      # 把别的看东西的通路掐掉；它们各自要求正数，所以给约等于 0 的数
MUTED = ("eye_track_gain", "eye_pitch_gain", "eye_gaze_inhibition", "eye_change_gain",
         "eye_change_relay_gain", "eye_change_common", "eye_sound_gain", "eye_memory_gain",
         "eye_orient_gain", "eye_row_relay_gain", "eye_row_gain", "eye_row_common",
         "eye_band_gain", "eye_band_common", "eye_band_push", "eye_vergence_gain",
         "eye_vergence_steps", "eye_vergence_baseline", "eye_vergence_inhibition",
         "eye_near_gain", "eye_fusion_gain", "eye_wall_gain", "eye_loom_gain",
         "eye_stereo_gain", "eye_distance_slope")
# 红球 0.06 米、0.9 米远，在视网膜上亮的是一片约 5.6 x 6.2 个细胞的方块（实测，
# tools/measure_red_ball_features.py）。下面两个是半宽，单位是**细胞**，不是角度 ——
# 视网膜中央密、边上疏，同样 ±3.3 度在中央圈到七列、在边上只圈到两三列，归一化的分母
# 于是差两三倍，两只眼、两个方向都不一致（2026-09-30 实测：同一套线，左眼只到目标的
# 79%、右眼只到 56%；把推力上限抬到 0.60 后左眼到位 21/25 但抖动涨到 0.21 弧度，
# 根源都是分母在边上算小了）。球的像有这么大一团是球自己的事，跟它落在视网膜哪儿无关，
# 所以归一化按细胞圈，不按角度圈。
BLOB = {"yaw": 2.8, "pitch": 3.1}
BLOCK = {"yaw": np.radians(3.3), "pitch": np.radians(3.0)}   # 不分眼的老做法按角度圈


def load_table(path=TABLE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def red_cells(table, floor=.15, fill=None):
    """红色通道里值得接线的细胞。fill 那张表里扫亮过的，也算上。"""
    peak = np.asarray(table["cells"]["retinal_opponent"]["peak"], dtype=float)
    index = np.arange(len(peak))
    red = index[index % 3 == 0]                      # 红色通道：每 3 个取 1 个
    keep = peak[red] > floor                         # 球从没扫亮过的不接
    if fill is not None and fill is not table:
        more = np.asarray(fill["cells"]["retinal_opponent"]["peak"], dtype=float)
        if more.shape == peak.shape:
            keep = keep | (more[red] > floor)
    return red[keep]


def eye_of(index, table):
    """这些细胞属于哪只眼：按视网膜层的排布直接算，不用猜。"""
    shape = table["eye_shape"]
    stride = int(shape[1])*int(shape[2])*3
    return np.asarray(index, dtype=int)//stride


def cell_rows_columns(index, table):
    """细胞在视网膜上的行列：索引怎么排出来的就怎么拆回去。"""
    shape = table["eye_shape"]
    rows, columns, channels = int(shape[1]), int(shape[2]), int(shape[3])
    cell = np.asarray(index, dtype=int)//channels
    return (cell//columns) % rows, cell % columns


def retina_angles(table, floor=.15, fill=None):
    """把「球摆到哪儿这个细胞最亮」的表，换成「这个细胞答话时球偏在这只眼的哪儿」。

    返回两张标定表：列 -> 方位、行 -> 高低，两只眼各一张。同一列里各个细胞给出的角度
    几乎一样（实测同一列内最大标准差 0.0375 弧度、同一行内 0.0391），所以取中位。
    用表不用拟合直线：列到角度中间密、边缘疏（48 列走完 -0.65 ~ +0.56 弧度，
    中间每列约 0.02 弧度、边上约 0.05）。
    """
    shape = table["eye_shape"]
    eyes, rows, columns = int(shape[0]), int(shape[1]), int(shape[2])
    places = np.asarray(table["places"], dtype=float)
    best = np.asarray(table["cells"]["retinal_opponent"]["best_place"], dtype=int)
    red = red_cells(table, floor)
    eye = eye_of(red, table)
    row, column = cell_rows_columns(red, table)
    yaw = np.full((eyes, columns), np.nan)
    pitch = np.full((eyes, rows), np.nan)
    for which, side in ((0, "left"), (1, "right")):
        mine = np.nonzero(eye == which)[0]
        seen = G.seen_by_many(places[best[red[mine]]], side)
        for number in range(columns):
            taken = seen[column[mine] == number, 0]
            if len(taken):
                yaw[which, number] = float(np.median(taken))
        for number in range(rows):
            taken = seen[row[mine] == number, 1]
            if len(taken):
                pitch[which, number] = float(np.median(taken))
    if fill is not None and fill is not table:
        # 主表没标定到的列（左眼最左、右眼最右）才问补表。
        other_yaw, other_pitch = retina_angles(fill, floor)
        yaw = np.where(np.isfinite(yaw), yaw, other_yaw)
        pitch = np.where(np.isfinite(pitch), pitch, other_pitch)
    return yaw, pitch


def cell_angles(table, index, group, per_eye=True, floor=.15, fill=None):
    """每个细胞答话时，球偏在这只眼自己的哪个角度。

    不分眼的老做法（两只眼共用一套指令细胞）沿用表里记的头坐标系角度。
    """
    if not per_eye:
        places = np.asarray(table["places"], dtype=float)
        best = np.asarray(table["cells"]["retinal_opponent"]["best_place"], dtype=int)
        return places[best[np.asarray(index, dtype=int)]]
    yaw, pitch = retina_angles(table, floor, fill)
    row, column = cell_rows_columns(index, table)
    return np.stack([yaw[group, column], pitch[group, row]], axis=1)


def wires(table, split=.10, dead=np.radians(3.), move_top=.35, pitch_top=.28,
          hold_top=None, move_push=.60, hold_push=.15, gain=1., tiers=True,
          block=None, per_eye=True, only=None, fill=None):
    """算每一根线：从哪个细胞出发、接到哪个方向、哪一档、多粗。"""
    block = dict(BLOCK if block is None else block)
    hold_top = float(split if hold_top is None else hold_top)
    red = red_cells(table, fill=fill)
    group = eye_of(red, table)
    sight = cell_angles(table, red, group, per_eye, fill=fill)
    bearing, elevation = sight[:, 0], sight[:, 1]
    row, column = cell_rows_columns(red, table)
    yaw_map, pitch_map = retina_angles(table, fill=fill) if per_eye else (None, None)
    out = []
    eyes = (None,) if not per_eye else ((0, 1) if only is None else (int(only),))
    for eye in eyes:
        mine = np.ones(len(red), bool) if eye is None else (group == eye)
        for axis, seen, other, positive, negative, top in (
                ("yaw", bearing, elevation, "left", "right", move_top),
                ("pitch", elevation, bearing, "up", "down", pitch_top)):
            cross_axis = "pitch" if axis == "yaw" else "yaw"
            for tier, low, high, band_top, push in (
                    ("hold", dead, split, hold_top, hold_push),
                    ("move", split, np.inf, top, move_push)):
                if not tiers and tier == "hold":
                    continue
                # 球摆在「这一档最远处、正前方那个高度」时点亮的那片细胞。球在视网膜上
                # 亮的是一个团，不是一个竖列，所以两个方向都要卡：只按方位卡，会把所有
                # 高度的细胞都算进分母，接线就细了十倍。per_eye 时分母只算这一只眼的，
                # 而且按细胞圈（BLOB）：中央密、边上疏，按角度圈会让边缘的分母小两三倍。
                if per_eye:
                    centre = nearest(yaw_map[eye], band_top if axis == "yaw" else 0.)
                    row_here = nearest(pitch_map[eye], band_top if axis == "pitch" else 0.)
                    if centre is None or row_here is None:
                        continue
                    blob = (mine & (np.abs(column - centre) <= BLOB["yaw"])
                            & (np.abs(row - row_here) <= BLOB["pitch"]))
                else:
                    blob = (mine & (np.abs(seen - band_top) <= block[axis])
                            & (np.abs(other) <= block[cross_axis]))
                mass = float(np.abs(seen[blob]).sum())
                if mass <= 0:
                    continue
                thick = float(gain)/mass
                band = (np.abs(seen) >= low) & (np.abs(seen) < high)
                for direction, sign in ((positive, 1.), (negative, -1.)):
                    side = band & mine & (np.sign(seen) == sign)
                    for cell, angle in zip(red[side], seen[side]):
                        out.append([int(cell), direction, tier, thick*abs(float(angle))])
    return out


def nearest(values, target):
    """这一列（这一行）里，答话角度最接近 target 的那一个编号。"""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if not good.any():
        return None
    numbers = np.arange(len(values))[good]
    return int(numbers[np.argmin(np.abs(values[good] - target))])


def spec_of(table, split, move_push, hold_push, dead, **kwargs):
    gain = float(kwargs.pop("gain", 1.))
    cell_time = float(kwargs.pop("cell_time", .03))
    latch = float(kwargs.pop("latch", 0.))
    cross = float(kwargs.pop("cross", 0.))
    motor_time = float(kwargs.pop("motor_time", .015))
    per_eye = bool(kwargs.pop("per_eye", True))
    only = kwargs.pop("only", None)
    direct = bool(kwargs.pop("direct", True))
    fill = kwargs.pop("fill", None)
    speed = float(kwargs.pop("speed", 1.5))
    rest_time = float(kwargs.pop("rest_time", .6))
    rest_speed = float(kwargs.pop("rest_speed", .3))
    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,
                  dead=dead, per_eye=per_eye, only=only, gain=gain, fill=fill, **kwargs)
    if direct:
        # 用户 2026-09-30：眼肌神经元一被激活就发出**固定的力**，力气不随信号多少变化。
        # 所以每根红线都一样粗（1.0：一根就足够把眼肌神经元推满），不用再算什么参照、
        # 什么分母。球只要偏在那边、有任何一个细胞亮着，眼肌神经元就饱和，眼睛就用
        # 固定速度往那边转。分档这时候只剩「哪些细胞接线」的意思，粗细已经一样了。
        edges = [(cell, direction, tier, 1.0) for cell, direction, tier, weight in edges]
    return dict(gain=gain, edges=[tuple(edge) for edge in edges], move_push=float(move_push),
                hold_push=float(hold_push), cell_time=cell_time, latch=latch,
                cross=cross, motor_time=motor_time, per_eye=per_eye, only=only,
                direct=direct, speed=speed, rest_time=rest_time, rest_speed=rest_speed,
                push={side: (float(move_push), float(hold_push))
                      for side in ("left", "right", "down", "up")})


def build(ctx, spec, red=True, size=.06, eye_gain=None):
    body = tb.clean_body(ctx["model_path"])
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [size]*3
    body.model.geom_rgba[target] = [1., 0., 0., 1.] if red else [.12, .85, .15, 1.]
    parameters = dict(ctx["parameters"])
    if eye_gain is not None:
        parameters["eye_gain"] = float(eye_gain)
    for name in MUTED:
        # 无条件覆盖：只改「参数表里已有的那几项」是不够的 —— 2026-09-30 实测，
        # 先天的眼动通路（往哪看、会聚、条带反射）因为没被改到，一直在推眼肌，
        # 把红球那一路的读数全部淹没了。
        parameters[name] = SILENT
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               seed=ctx["seed"],
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               eye_red=spec, **parameters)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    return body, brain, eyes, target


def aimer(body, target, distance):
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def aim(bearing, elevation):
        G.place_ball(body.model, body.data, target, bearing, elevation, distance)
        offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
        rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
        return (float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0])),
                float(np.arctan2(offset @ rotation[:, 2],
                                 np.linalg.norm(offset @ rotation[:, :2]))))
    return aim


def view_half_angles(body, brain):
    """每只眼的画面有多宽（弧度），从模型里的相机读：竖直半角 fovy/2，水平半角
    atan(tan(竖直半角) * 宽/高)。

    2026-09-30 之前这里写死 0.52（竖直半角）去卡两个方向，于是 0.9 米外偏 0.35 弧度的
    球被记成「看不见」——其实它在水平方向（半角 0.656 弧度）里，只是那只眼转不过去。
    """
    camera = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_left")
    half = np.radians(float(body.model.cam_fovy[camera]))/2.
    wide = float(np.arctan(np.tan(half)*float(brain.eye_width)/float(brain.eye_height)))
    return wide, half


def eye_reach(body, which):
    """这只眼的两根轴各能转到哪儿（弧度，下限 上限）。"""
    return (float(body.eye_lower_limits[2*which]), float(body.eye_upper_limits[2*which]),
            float(body.eye_lower_limits[2*which + 1]), float(body.eye_upper_limits[2*which + 1]))


def mark(row, which):
    """这一格这只眼算不算数：看 / 盲（球不在它画面里）/ 够不着（转过去也到不了）/ 不管。"""
    if row.get("only") is not None and row["only"] != which:
        return "不管"
    if not row["visible"][which]:
        return "盲"
    if not row["reachable"][which]:
        return "够不着"
    return "看"


def required(body, target):
    """每只眼各自该转到哪儿才算正对球（肌肉的记法：+yaw 往左、+pitch 往下）。"""
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    rotation = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
    ball = np.asarray(body.data.geom_xpos[target], dtype=float)
    out = []
    for side in ("left", "right"):
        i = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_" + side)
        v = rotation.T @ (ball - np.asarray(body.data.xpos[i], dtype=float))
        out.append((float(np.arctan2(v[1], v[0])),
                    -float(np.arctan2(v[2], np.hypot(v[0], v[1])))))
    return out


def measure(brain, body, eyes, aim, target, places, steps, tail, only=None, view=None):
    """每只眼对着自己的目标量，不用两眼平均。

    球摆在一边时，离得远的那只眼本来就该看到更大的偏角。球不在它画面里（盲）、或者转到底也到不了（够不着）
    的那几个位置不判它 —— 拿一杆平均尺子会把歪的那只眼摊平到平均值里。
    """
    environment = tb.blank_environment()
    observation = body.observe()
    rows = []
    for bearing, elevation in places:
        true_bearing, true_elevation = aim(bearing, elevation)
        ask = required(body, target)
        visible = [True, True] if view is None else [abs(a[0]) <= view[0] and abs(a[1]) <= view[1]
                                                    for a in ask]
        reachable = []
        for which in (0, 1):
            low_y, high_y, low_p, high_p = eye_reach(body, which)
            reachable.append(low_y <= ask[which][0] <= high_y and low_p <= ask[which][1] <= high_p)
        asked = [see and can for see, can in zip(visible, reachable)]
        if only is not None:
            asked = [want and which == int(only) for which, want in enumerate(asked)]
        yaws, pitches = [], []
        for _ in range(int(steps)):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(brain.eye_command())
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
            command = brain.eye_command()
            yaws.append([float(command[0]) - ask[0][0], float(command[2]) - ask[1][0]])
            pitches.append([float(command[1]) - ask[0][1], float(command[3]) - ask[1][1]])
        closes, trims = yaws[-int(tail):], pitches[-int(tail):]
        rows.append(dict(bearing=true_bearing, elevation=true_elevation, asked=asked,
                         visible=visible, reachable=reachable,
                         only=None if only is None else int(only),
                         yaw_error=[float(np.mean([one[k] for one in closes])) for k in (0, 1)],
                         pitch_error=[float(np.mean([one[k] for one in trims])) for k in (0, 1)],
                         yaw_wander=[float(np.std([one[k] for one in closes])) for k in (0, 1)],
                         pitch_wander=[float(np.std([one[k] for one in trims])) for k in (0, 1)]))
    for row in rows:
        row["error"] = max([max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k]))
                            for k in (0, 1) if row["asked"][k]] or [0.])
        row["wander"] = max([max(row["yaw_wander"][k], row["pitch_wander"][k])
                             for k in (0, 1) if row["asked"][k]] or [0.])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=float, default=.10, help="两档的分界（弧度）")
    ap.add_argument("--move-top", type=float, default=.35, help="移动档的满量程（弧度）")
    ap.add_argument("--pitch-top", type=float, default=.28, help="高低方向的满量程（弧度）")
    ap.add_argument("--hold-top", type=float, default=None, help="保持档的满量程，不给就同 --split")
    ap.add_argument("--move-push", type=float, default=.60, help="移动档推多少弧度")
    ap.add_argument("--hold-push", type=float, default=.15, help="保持档推多少弧度")
    ap.add_argument("--gain", type=float, default=1., help="所有接线的总缩放（1 = 按满量程归一）")
    ap.add_argument("--dead", type=float, default=3., help="死区（度）：离正中这么近的细胞不接")
    ap.add_argument("--cell-time", type=float, default=.03, help="红色指令细胞的时间常数（秒）")
    ap.add_argument("--latch", type=float, default=0., help="指令细胞自己激自己多重（勾住，去掉回落）")
    ap.add_argument("--cross", type=float, default=0., help="相反方向的指令细胞互相压多重")
    ap.add_argument("--motor-time", type=float, default=.015, help="眼肌自己的时间常数（秒）")
    ap.add_argument("--per-eye", action=argparse.BooleanOptionalAction, default=True,
                    help="每只眼只被自己那只眼的红细胞驱动（默认开）")
    ap.add_argument("--eye", choices=("both", "left", "right"), default="both",
                    help="这一轮只接、只看哪只眼（默认两只都做）")
    ap.add_argument("--direct", action=argparse.BooleanOptionalAction, default=True,
                    help="红色特征细胞直接接到眼肌神经元（默认开）")
    ap.add_argument("--speed", type=float, default=1.5,
                    help="眼肌神经元被激活时眼球每秒转多少弧度（固定力）")
    ap.add_argument("--rest-time", type=float, default=.6,
                    help="回正的时间常数（秒）；只有旧的分级模式用，直接接线看 --rest-speed")
    ap.add_argument("--rest-speed", type=float, default=.3,
                    help="回正力多大：没有信号时眼球每秒回正多少弧度（固定力）")
    ap.add_argument("--eye-gain", type=float, default=None,
                    help="眼肌自己的增益（不给就用模型里的值；调它能让静止时两端的眼肌神经元不顶死）")
    ap.add_argument("--no-tiers", action="store_true", help="对照：不分两档，全走移动档")
    ap.add_argument("--green-ball", action="store_true", help="对照：球是绿的，红色那一路不该有反应")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--tail", type=int, default=25)
    ap.add_argument("--tolerance", type=float, default=.10)
    ap.add_argument("--distance", type=float, default=.90)
    ap.add_argument("--spread", type=float, default=.35, help="最远方位（弧度）")
    ap.add_argument("--lift", type=float, default=.28, help="最高/最低（弧度）")
    ap.add_argument("--grid", type=int, default=5, help="每边几个位置")
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--fill", default=str(WIDE),
                    help="补缺口用的表：主表没标定到的列（左眼最左、右眼最右各 2 列）用它补上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    only = None if args.eye == "both" else (0 if args.eye == "left" else 1)
    table = load_table(args.table)
    spec = spec_of(table, args.split, args.move_push, args.hold_push, np.radians(args.dead),
                   move_top=args.move_top, pitch_top=args.pitch_top, hold_top=args.hold_top,
                   gain=args.gain, tiers=not args.no_tiers, cell_time=args.cell_time,
                   latch=args.latch, cross=args.cross, motor_time=args.motor_time,
                   per_eye=args.per_eye, only=only, direct=args.direct,
                   speed=args.speed, rest_time=args.rest_time,
                   rest_speed=args.rest_speed,
                   fill=None if args.fill in ("", "none") else load_table(args.fill))
    moves = len([e for e in spec["edges"] if e[2] == "move"])
    holds = len(spec["edges"]) - moves
    print("接线 %d 根（移动档 %d、保持档 %d）：分界 %.2f、移动档满量程 %.2f 推 %.2f 弧度、"
          "保持档满量程 %.2f 推 %.2f 弧度、死区 %.1f 度、总缩放 %.2f、自激勾住 %.2f、互压 %.2f、"
          "眼肌时间 %.3f、每眼独立 %s%s"
          % (len(spec["edges"]), moves, holds, args.split, args.move_top, args.move_push,
             args.split if args.hold_top is None else args.hold_top, args.hold_push,
             args.dead, args.gain, args.latch, args.cross, args.motor_time,
             args.per_eye, "" if only is None else "、只做%s眼" % args.eye))
    print("眼肌：%s" % ("一只眼 4 个神经元：四个方向各一个，被激活就发出固定的力"
                      "（每秒 %.2f 弧度往那边转）；没有信号时被固定的回正力"
                      "（每秒 %.2f 弧度）拉回正前方" % (args.speed, args.rest_speed)
                      if args.direct else "24 级阶梯（旧：静息电流顶在中间）"))

    ctx = tb.context({}, args.seed)
    body, brain, eyes, target = build(ctx, spec, red=not args.green_ball,
                                      eye_gain=args.eye_gain)
    aim = aimer(body, target, args.distance)
    count = max(2, int(args.grid))
    places = [(b, e) for e in np.linspace(-args.lift, args.lift, count)
              for b in np.linspace(-args.spread, args.spread, count)]
    view = view_half_angles(body, brain)
    print("每只眼的画面：水平半角 %.3f、竖直半角 %.3f 弧度；左眼行程 %s"
          % (view[0], view[1], np.round(np.asarray(eye_reach(body, 0)), 2)))
    print("静止时给出的角度（球还没摆）：%s" % np.round(brain.eye_command(), 4))
    rows = measure(brain, body, eyes, aim, target, places, args.steps, args.tail, only, view)
    eyes.close()
    print("\n%8s %8s | %13s | %13s | %8s" % ("球方位", "球高低", "左眼 yaw/pitch",
                                              "右眼 yaw/pitch", "后段抖动"))
    for row in rows:
        print("%8.3f %8.3f | %+5.3f/%+5.3f %s | %+5.3f/%+5.3f %s | %5.3f"
              % (row["bearing"], row["elevation"],
                 row["yaw_error"][0], row["pitch_error"][0],
                 mark(row, 0),
                 row["yaw_error"][1], row["pitch_error"][1],
                 mark(row, 1), row["wander"]))
    judged = [(row, k) for row in rows for k in (0, 1) if row["asked"][k]]
    hits = [(row, k) for row, k in judged
            if max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k])) <= args.tolerance]
    errors = [max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k])) for row, k in judged]
    print("\n命中 %d/%d = %.2f（只算球能看见、又转得到的那几格）"
          % (len(hits), len(judged), len(hits)/float(max(1, len(judged)))))
    print("误差：中位 %.3f、平均 %.3f、最差 %.3f 弧度（容差 %.2f）"
          % (float(np.median(errors)), float(np.mean(errors)), float(np.max(errors)),
             args.tolerance))
    print("后段抖动：最大 %.4f 弧度（抖得大 = 眼睛在来回撞，不是稳在一个角度）"
          % float(np.max([row["wander"] for row in rows])))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(options=vars(args), edges=len(spec["edges"]), places=rows,
                 accuracy=float(len(hits)/float(max(1, len(judged)))),
                 mean_error=float(np.mean(errors)), worst_error=float(np.max(errors))),
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("读数 -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())