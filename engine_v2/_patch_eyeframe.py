# -*- coding: utf-8 -*-
"""一次性改动：接线改用每只眼自己的角度；球的摆放统一到 tools/eye_geometry.py。"""
from pathlib import Path

def edit(path, pairs, encoding="utf-8"):
    p = Path(path)
    text = p.read_text(encoding=encoding, newline="")
    for old, new in pairs:
        assert old in text, "没找到（%s）：%r" % (path, old[:60])
        assert text.count(old) == 1, "出现多次（%s）：%r" % (path, old[:60])
        text = text.replace(old, new)
    p.write_text(text, encoding=encoding, newline="")
    print("改完 %s（%d 处）" % (path, len(pairs)))

# ---------- 扫场：球的摆放交给共用几何；行为不变（还是 base 高度 + .03） ----------
edit("tools/sweep_red_ball_features.py", [
 ("from born_wired.stereo_senses import RawEyes            # noqa: E402\n",
  "from born_wired.stereo_senses import RawEyes            # noqa: E402\nfrom tools import eye_geometry as G                     # noqa: E402\n"),
 ("""    def place(bearing, elevation):
        x = args.distance * np.cos(elevation) * np.cos(bearing)
        y = args.distance * np.cos(elevation) * np.sin(bearing)
        z = eye_height + args.distance * np.sin(elevation)
        body.model.geom_pos[target] = [x, y, z]
        mujoco.mj_forward(body.model, body.data)
""",
  """    def place(bearing, elevation):
        G.place_ball(body.model, body.data, target, bearing, elevation, args.distance,
                     ball_z=eye_height)
"""),
])

# ---------- 接线工具 ----------
edit("tools/wire_red_gaze.py", [
 ("from tools import taskbank as tb                                        # noqa: E402\n",
  "from tools import taskbank as tb                                        # noqa: E402\nfrom tools import eye_geometry as G                                     # noqa: E402\n"),
 ("""1. 起点是 retinal_opponent 的红色通道细胞（球是红的，这层只认红色）。细胞 i 的位置野
   说：球落在（方位 b_i，高低 e_i）时它最亮。实测表里列号到方位不是直线（中间密、
   边缘疏），所以用表，不用拟合直线。
2. 一个细胞出多大力 = 该方向这一档的「每弧度增益」* |它自己看到的角度|。球偏得越远，
   摊到的那片细胞越用力，所以「中央轻、越偏越重」是权重本身带出来的，不是另写的判断。
3. 方位为正（机器人左侧）的细胞接到 eye_red_*_left（+yaw 那一路），为负接 right；
   高低为正（上方）接 up，为负接 down。+yaw 是往左看、+pitch 是往下看，这两个符号由
   tools/probe_eye_axis.py 直接量出来（把眼球转 +0.30 弧度，红球亮块的行列往哪跑）。
4. 两种力度""",
  """1. 起点是 retinal_opponent 的红色通道细胞（球是红的，这层只认红色）。细胞 i 的位置野
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
5. 两种力度"""),
 ("""5. 没有信号时两档都不放电，眼肌自己的静息电流把眼睛带回正前方（那条 tonic 线本来就接着）。
6. 离正视方向 3 度以内的细胞不接（球已经在正中了不许再动）。""",
  """6. 没有信号时两档都不放电，眼肌自己的静息电流把眼睛带回正前方（那条 tonic 线本来就接着）。
7. 离正视方向 3 度以内的细胞不接（球已经在这只眼的正中了不许再动）。"""),
 ("""def red_cells(table):
    \"\"\"红色通道里值得接线的细胞，以及它们各自看到的角度。\"\"\"
    angles = np.asarray(table["places"], dtype=float)
    layer = table["cells"]["retinal_opponent"]
    best = np.asarray(layer["best_place"], dtype=int)
    peak = np.asarray(layer["peak"], dtype=float)
    index = np.arange(len(best))
    red = index[index % 3 == 0]                      # 红色通道：每 3 个取 1 个
    red = red[peak[red] > .15]                       # 球从没扫亮过的不接
    return red, angles[best[red]]


def eye_of(index, table):
    \"\"\"这些细胞属于哪只眼：按视网膜层的排布直接算，不用猜。\"\"\"
    shape = table["eye_shape"]
    stride = int(shape[1])*int(shape[2])*3
    return np.asarray(index, dtype=int)//stride
""",
  """def red_cells(table, floor=.15):
    \"\"\"红色通道里值得接线的细胞。\"\"\"
    peak = np.asarray(table["cells"]["retinal_opponent"]["peak"], dtype=float)
    index = np.arange(len(peak))
    red = index[index % 3 == 0]                      # 红色通道：每 3 个取 1 个
    return red[peak[red] > floor]                    # 球从没扫亮过的不接


def eye_of(index, table):
    \"\"\"这些细胞属于哪只眼：按视网膜层的排布直接算，不用猜。\"\"\"
    shape = table["eye_shape"]
    stride = int(shape[1])*int(shape[2])*3
    return np.asarray(index, dtype=int)//stride


def cell_rows_columns(index, table):
    \"\"\"细胞在视网膜上的行列：索引怎么排出来的就怎么拆回去。\"\"\"
    shape = table["eye_shape"]
    rows, columns, channels = int(shape[1]), int(shape[2]), int(shape[3])
    cell = np.asarray(index, dtype=int)//channels
    return (cell//columns) % rows, cell % columns


def retina_angles(table, floor=.15):
    \"\"\"把「球摆到哪儿这个细胞最亮」的表，换成「这个细胞答话时球偏在这只眼的哪儿」。

    返回两张标定表：列 -> 方位、行 -> 高低，两只眼各一张。同一列里各个细胞给出的角度
    几乎一样（实测同一列内最大标准差 0.0375 弧度、同一行内 0.0391），所以取中位。
    用表不用拟合直线：列到角度中间密、边缘疏（48 列走完 -0.65 ~ +0.56 弧度，
    中间每列约 0.02 弧度、边上约 0.05）。
    \"\"\"
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
    return yaw, pitch


def cell_angles(table, index, group, per_eye=True, floor=.15):
    \"\"\"每个细胞答话时，球偏在这只眼自己的哪个角度。

    不分眼的老做法（两只眼共用一套指令细胞）沿用表里记的头坐标系角度。
    \"\"\"
    if not per_eye:
        places = np.asarray(table["places"], dtype=float)
        best = np.asarray(table["cells"]["retinal_opponent"]["best_place"], dtype=int)
        return places[best[np.asarray(index, dtype=int)]]
    yaw, pitch = retina_angles(table, floor)
    row, column = cell_rows_columns(index, table)
    return np.stack([yaw[group, column], pitch[group, row]], axis=1)
"""),
 ("""    red, angles = red_cells(table)
    bearing, elevation = angles[:, 0], angles[:, 1]
    group = eye_of(red, table)""",
  """    red = red_cells(table)
    group = eye_of(red, table)
    sight = cell_angles(table, red, group, per_eye)
    bearing, elevation = sight[:, 0], sight[:, 1]"""),
 ("""    def aim(bearing, elevation):
        body.model.geom_pos[target] = [distance*np.cos(elevation)*np.cos(bearing),
                                       distance*np.cos(elevation)*np.sin(bearing),
                                       .32 + distance*np.sin(elevation)]
        mujoco.mj_forward(body.model, body.data)
""",
  """    def aim(bearing, elevation):
        G.place_ball(body.model, body.data, target, bearing, elevation, distance)
"""),
])

# ---------- 跟踪工具 ----------
edit("tools/track_red_gaze.py", [
 ("from tools import taskbank as tb                                   # noqa: E402\n",
  "from tools import taskbank as tb                                   # noqa: E402\nfrom tools import eye_geometry as G                                # noqa: E402\n"),
 ("""def place(body, target, bearing, elevation, distance):
    body.model.geom_pos[target] = [distance*np.cos(elevation)*np.cos(bearing),
                                   distance*np.cos(elevation)*np.sin(bearing),
                                   .32 + distance*np.sin(elevation)]
    mujoco.mj_forward(body.model, body.data)
""",
  """def place(body, target, bearing, elevation, distance):
    G.place_ball(body.model, body.data, target, bearing, elevation, distance)
"""),
 ('''    """每只眼要转到多少弧度才算正对球（头坐标系里算，不是猜）。"""''',
  '''    """每只眼要转到多少弧度才算正对球 —— 这只眼自己的坐标系，不是头坐标系。"""'''),
])
print("全部改完")