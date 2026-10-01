# -*- coding: utf-8 -*-
"""给 chase_red_ball.measure 加上「球沿随机曲线逃」的选项（临时补丁脚本）。"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

def swap(old, new, tag):
    global s
    assert old in s, "没找到：" + tag
    s = s.replace(old, new, 1)
    print("改好", tag)

swap("import argparse\nimport json\nimport sys\n",
     "import argparse\nimport json\nimport math\nimport sys\n",
     "import math")

swap('FAR = (60., 60., -8.)\n',
     'FAR = (60., 60., -8.)\n\n'
     '# 球「沿随机曲线逃」的几个旋钮（用户 2026-10-01）：\n'
     '#   朝向基本上还是「背对狗」，但会平滑地随机摆一个角度，所以路线是弯的、连续的。\n'
     'WANDER_TAU = 1.2        # 摆角的记忆时间（秒）：越小摆得越急\n'
     'WANDER_SIGMA = 1.6      # 摆角抖多厉害（弧度/√秒）\n'
     'WANDER_MAX = 1.1        # 最多偏离「背对狗」多少弧度（约 63 度）\n'
     'TURN_RATE = 3.0         # 球每秒最多把朝向扳多少（弧度/秒），保证路线是弧线不是折线\n'
     'WANDER_FIELD = 6.0      # 球离它的出发点超过这么远，就往回拐，不跑出视野之外\n',
     "逃跑常数")

swap("def measure(body, brain, eyes, geom, seconds, bearing, distance, elevation=0., seed=0,\n"
     "            flee=0., escape=6., ball=True):",
     "def measure(body, brain, eyes, geom, seconds, bearing, distance, elevation=0., seed=0,\n"
     "            flee=0., escape=6., curve=0., ball=True):",
     "measure 签名")

swap("    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()\n"
     "    ball0 = ball.copy()\n",
     "    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()\n"
     "    ball0 = ball.copy()\n"
     "    origin = np.asarray(ball, dtype=float).copy()\n"
     "    wander_rng = np.random.default_rng(int(seed)*7919 + 13)\n"
     "    wander = 0.0\n"
     "    escape_yaw = None\n"
     "    ball_track = []\n",
     "记球的起点和随机数")

old_flee = (
    "        if flee > 0.:\n"
    "            # 红球躲着狗跑：每一拍都往「背对狗」的方向挪一点。球跑得比狗慢一点，\n"
    "            # 所以会追的狗越追越近，不会追的狗眼看着它走远。跑到 escape 米就不再\n"
    "            # 挪了（不然一只不会追的狗会被扣到无穷远，读数没有上界）。\n"
    "            here = np.asarray(body.data.xpos[base], dtype=float)\n"
    "            away = ball - here\n"
    "            length = float(np.linalg.norm(away))\n"
    "            if length > 1e-6 and length < escape:\n"
    "                ball = ball + away/length*flee*DT\n"
    "                body.model.geom_pos[geom] = ball\n"
    "                mujoco.mj_forward(body.model, body.data)\n")
new_flee = (
    "        if flee > 0.:\n"
    "            # 红球躲着狗跑：每一拍都往「背对狗」的方向挪一点。球跑得比狗慢一点，\n"
    "            # 所以会追的狗越追越近，不会追的狗眼看着它走远。\n"
    "            # curve > 0 时，朝向在「背对狗」的基础上平滑地随机摆动（用户 2026-10-01：\n"
    "            # 让红球沿着随机曲线逃跑），所以路线是弯的、连续的，不是直线也不是折线。\n"
    "            here = np.asarray(body.data.xpos[base], dtype=float)\n"
    "            away = ball - here\n"
    "            length = float(np.linalg.norm(away))\n"
    "            if length > 1e-6 and (curve > 0. or length < escape):\n"
    "                if curve > 0.:\n"
    "                    wander += (-wander/WANDER_TAU)*DT + WANDER_SIGMA*curve*math.sqrt(DT)*float(wander_rng.normal())\n"
    "                    wander = float(np.clip(wander, -WANDER_MAX, WANDER_MAX))\n"
    "                    want = math.atan2(away[1], away[0]) + wander\n"
    "                    drift = float(np.linalg.norm(ball[:2] - origin[:2]))\n"
    "                    if drift > WANDER_FIELD:\n"
    "                        want = math.atan2(origin[1] - ball[1], origin[0] - ball[0])\n"
    "                    if escape_yaw is None:\n"
    "                        escape_yaw = want\n"
    "                    step_yaw = math.atan2(math.sin(want - escape_yaw),\n"
    "                                          math.cos(want - escape_yaw))\n"
    "                    escape_yaw += step_yaw*min(1., TURN_RATE*DT)\n"
    "                else:\n"
    "                    escape_yaw = math.atan2(away[1], away[0])\n"
    "                ball = ball + np.array([math.cos(escape_yaw), math.sin(escape_yaw), 0.])*flee*DT\n"
    "                body.model.geom_pos[geom] = ball\n"
    "                mujoco.mj_forward(body.model, body.data)\n"
    "        if flee > 0.:\n"
    "            ball_track.append([float(ball[0]), float(ball[1])])\n")
swap(old_flee, new_flee, "逃跑那一拍")

swap("                lowest_up_z=lowest_up, seconds=time.perf_counter() - started,\n"
     "                headings=headings, ranges=ranges)",
     "                lowest_up_z=lowest_up, seconds=time.perf_counter() - started,\n"
     "                headings=headings, ranges=ranges, ball_track=ball_track)",
     "返回值加上球的路线")

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("写回", p)