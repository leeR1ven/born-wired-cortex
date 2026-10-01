# -*- coding: utf-8 -*-
"""实时窗口：红球一直在机器狗眼前随机动，直接看两只眼睛怎么转。

窗口里：
  顶部两条黄框画面 = 左眼、右眼各自看到的画面（按格子真实位置还原过，不是扭曲的格子图）
  中间 = 3D 场景（球一直在狗头前面）
  底部 = 球的位置 / 模式，和两只眼的角度（实际 vs 该到）

键盘：
  A       球随机漫游（默认）：一直在眼前 ±26 度、±16 度以内来回随机动
  F       球走老路线（左右扫 → 上下扫 → 斜着来回 → 绕圈 → 靠近）
  S       球停在原地
  0       球回到正前方
  1~8     把球直接摆到 8 个固定位置（1 左 2 右 3 上 4 下 5~8 四个角）
  方向键  手动把球往左/右/上/下挪一点（每次 0.06 弧度）
  Z / X   球近一点 / 远一点
  P       暂停 / 继续
  R       复位（眼球回正、球回正前方、暂停解除）
  ESC     关窗退出
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco                                                       # noqa: E402
from mujoco import viewer as mjviewer                               # noqa: E402

import wire_red_gaze as W                                           # noqa: E402
import eye_geometry as G
import eye_ball_movie as M                                          # noqa: E402
import retina_view as R                                             # noqa: E402
from tools import taskbank as tb                                    # noqa: E402

DT = .01
SCALE = 2                       # 还原后的画面再放大多少倍做显示（144x108 -> 288x216）
LIMIT_SIDE = .42                # 球最多偏左/偏右这么多弧度（24 度）
LIMIT_UP = .28                  # 球最多偏上/偏下这么多弧度（16 度）
NEAR, FAR = .70, 1.05           # 球最近 / 最远（太近时眼球会转不过来）
PRESETS = {ord("0"): (0., 0.), ord("1"): (.45, 0.), ord("2"): (-.45, 0.),
           ord("3"): (0., .28), ord("4"): (0., -.28), ord("5"): (.40, .24),
           ord("6"): (.40, -.24), ord("7"): (-.40, .24), ord("8"): (-.40, -.24)}
ARROWS = {263: (-.06, 0.), 262: (.06, 0.), 265: (0., .06), 264: (0., -.06)}
LEGEND = "A随机 F老路 S停 0正前 1~8位置 ←→挪球 Z/X远近 P暂停 R复位"
MODE_NAME = {"random": "随机漫游", "route": "老路线", "hold": "停住"}


place_in_front = G.place_in_front    # 摆球的规矩只有一份，在 tools/eye_geometry.py


def wander(state, clock, dt):
    """眼前随机漫游：走一小段就换个方向，撞到边界就弹回来。"""
    state["dwell"] -= dt
    if state["dwell"] <= 0:
        state["v_side"] = float(np.random.uniform(-.45, .45))
        state["v_up"] = float(np.random.uniform(-.30, .30))
        state["dwell"] = float(np.random.uniform(.25, .85))
    state["bearing"] += state["v_side"]*dt
    state["elevation"] += state["v_up"]*dt
    if abs(state["bearing"]) > LIMIT_SIDE:
        state["bearing"] = float(np.clip(state["bearing"], -LIMIT_SIDE, LIMIT_SIDE))
        state["v_side"] = -state["v_side"]
    if abs(state["elevation"]) > LIMIT_UP:
        state["elevation"] = float(np.clip(state["elevation"], -LIMIT_UP, LIMIT_UP))
        state["v_up"] = -state["v_up"]
    state["distance"] = (NEAR + FAR)/2 + (FAR - NEAR)/2*np.sin(clock*.6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speed", type=float, default=1.5, help="眼球转多快（弧度/秒）")
    ap.add_argument("--rest-speed", type=float, default=.6, help="回正力多大（弧度/秒）")
    ap.add_argument("--run-for", type=float, default=0., help="只跑这么多秒就退出（自检用，0 = 一直跑）")
    args = ap.parse_args(argv)

    spec = W.spec_of(W.load_table(), .10, .35, .10, np.radians(1.5), rest_speed=args.rest_speed,
                     fill=W.load_table(W.WIDE))
    if args.speed != 1.5:
        spec["speed"] = float(args.speed)
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()

    state = dict(mode="random", bearing=.0, elevation=.0, distance=.90, t=0., paused=False,
                 v_side=.2, v_up=.1, dwell=.5)
    quit_now = dict(done=False, reason="")

    def on_key(code):
        if code in PRESETS:
            state["mode"] = "hold"
            state["bearing"], state["elevation"] = PRESETS[code]
        elif code in ARROWS:
            state["mode"] = "hold"
            state["bearing"] = float(np.clip(state["bearing"] + ARROWS[code][0],
                                             -LIMIT_SIDE, LIMIT_SIDE))
            state["elevation"] = float(np.clip(state["elevation"] + ARROWS[code][1],
                                               -LIMIT_UP, LIMIT_UP))
        elif code == ord("A"):
            state["mode"] = "random"
        elif code == ord("F"):
            state["mode"] = "route"
            state["t"] = 0.
        elif code == ord("S"):
            state["mode"] = "hold"
        elif code == ord("Z"):
            state["distance"] = max(.45, state["distance"] - .1)
        elif code == ord("X"):
            state["distance"] = min(1.3, state["distance"] + .1)
        elif code == ord("P"):
            state["paused"] = not state["paused"]
        elif code == ord("R"):
            state.update(mode="random", bearing=.0, elevation=.0, distance=.90, t=0.,
                         paused=False, v_side=.2, v_up=.1, dwell=.5)
            body.data.qpos[:] = body.model.qpos0
            mujoco.mj_forward(body.model, body.data)
        elif code == 256:
            quit_now["done"] = True
            quit_now["reason"] = "收到 ESC"

    viewer = mjviewer.launch_passive(body.model, body.data, key_callback=on_key)
    with viewer.lock():
        viewer.cam.lookat[:] = [0.40, 0.0, 0.22]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 118.0
        viewer.cam.elevation = -14.0
    print("窗口已开：%s" % LEGEND, flush=True)

    started = time.time()
    next_report = started + 1.0
    frames = 0
    board = None
    shown = False
    try:
        while viewer.is_running() and not quit_now["done"]:
            tick = time.time()
            if not state["paused"]:
                if state["mode"] == "random":
                    wander(state, time.time() - started, DT)
                elif state["mode"] == "route":
                    state["bearing"], state["elevation"], state["distance"] = M.path_at(state["t"])
                    state["t"] += DT
                place_in_front(body, target, state["bearing"], state["elevation"],
                               state["distance"])
                command = brain.eye_command()
                body.command_eyes(command)
                obs = body.observe()
                raw = eyes.observe_raw()
                activation = brain.step(obs, environment=env, eye_pixels=raw, dt=DT,
                                        learn=False)[1]
                body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
                frames += 1
                board = [R.picture(eyes, raw[i], SCALE) for i in (0, 1)]
            command = brain.eye_command()
            ask = W.required(body, target)
            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "球 方位%+.2f 高低%+.2f %.2f米  %s%s"
                 % (state["bearing"], state["elevation"], state["distance"],
                    MODE_NAME[state["mode"]], " 已暂停" if state["paused"] else ""),
                 "左眼 %+5.1f 度（该到 %+5.1f）"
                 % (np.degrees(command[0]), np.degrees(ask[0][0]))),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 "右眼 %+5.1f 度（该到 %+5.1f）"
                 % (np.degrees(command[2]), np.degrees(ask[1][0])),
                 LEGEND),
            ])
            if board is not None:
                view = viewer.viewport
                high, wide = int(board[0].shape[0]), int(board[0].shape[1])
                top = view.bottom + view.height - high - 8
                if not shown:
                    print("贴图：视口 left=%d width=%d height=%d，图 %dx%d，放 x=%d/%d y=%d"
                          % (view.left, view.width, view.height, wide, high,
                             view.left + 8, view.left + view.width - wide - 8, top), flush=True)
                    shown = True
                viewer.set_images([
                    (mujoco.MjrRect(view.left + 8, top, wide, high), board[0]),
                    (mujoco.MjrRect(view.left + view.width - wide - 8, top, wide, high), board[1]),
                ])
            viewer.sync()
            if time.time() > next_report:
                print("t=%5.1f 秒  球 方位%+.2f 高低%+.2f 距离%.2f  左眼%+5.1f(该到%+5.1f)  "
                      "右眼%+5.1f(该到%+5.1f)  %s"
                      % (time.time() - started, state["bearing"], state["elevation"],
                         state["distance"], np.degrees(command[0]), np.degrees(ask[0][0]),
                         np.degrees(command[2]), np.degrees(ask[1][0]),
                         "已暂停" if state["paused"] else MODE_NAME[state["mode"]]), flush=True)
                next_report = time.time() + 1.0
            left = DT - (time.time() - tick)
            if left > 0:
                time.sleep(left)
            if args.run_for and time.time() - started > args.run_for:
                break
    finally:
        viewer.close()
        eyes.close()
    print("跑了 %.1f 秒、%d 帧，窗口已关%s"
          % (time.time() - started, frames,
             "（%s）" % quit_now["reason"] if quit_now["reason"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())