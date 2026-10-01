# -*- coding: utf-8 -*-
"""一段动画：红球在眼前各处移动，看眼睛怎么动。

画面分三块 + 一条曲线：
  第三人称（球在头前怎么走） | 左眼看到的 | 右眼看到的
  下面：左右眼球角度随时间（虚线 = 球相对这只眼的角度）
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco                                                    # noqa: E402
from PIL import Image, ImageDraw, ImageFont                      # noqa: E402

import wire_red_gaze as W                                        # noqa: E402
import eye_geometry as G                                         # noqa: E402
from tools import taskbank as tb                                 # noqa: E402

DT = .01
STRIDE = 5                      # 每 5 步（0.05 秒）出一帧 = 20 帧/秒的素材，导出时抽成 10 帧/秒
PANEL = (240, 180)              # 每只眼的画面放大到多大
THIRD = (320, 240)              # 第三人称画面
PLOT = (800, 150)
FPS = 10


def path_at(t):
    """球的轨迹：左右扫 -> 上下扫 -> 斜着来回 -> 小圈 -> 靠近。返回（方位, 高低, 距离）。"""
    if t < 4.0:                                   # 左右横扫
        f = t/4.0
        return -0.55 + 1.10*f, 0.0, 0.90
    if t < 6.0:                                   # 上下扫
        f = (t - 4.0)/2.0
        return 0.0, -0.40 + 0.80*f, 0.90
    if t < 8.5:                                   # 斜着来回
        f = (t - 6.0)/2.5
        wobble = np.sin(f*np.pi*3.)
        return 0.45*wobble, -0.30 + 0.60*f, 0.90
    if t < 11.0:                                  # 画小圈
        f = (t - 8.5)/2.5
        a = 2*np.pi*f
        return 0.20 + 0.25*np.cos(a), 0.25*np.sin(a), 0.90
    f = (t - 11.0)/2.0                            # 靠近
    return 0.15, 0.0, 1.10 - 0.55*f


def font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    table = W.load_table()
    spec = W.spec_of(table, .10, .35, .10, np.radians(1.5), rest_speed=.6,
                     fill=W.load_table(W.WIDE))
    ctx = tb.context({}, 0)
    body, brain, eyes, target = W.build(ctx, spec)
    env = tb.blank_environment()
    third = mujoco.Renderer(body.model, height=THIRD[1], width=THIRD[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.55, 0.0, 0.30]
    cam.distance = 1.9
    cam.azimuth = 78.0
    cam.elevation = -18.0
    big = font(15)
    small = font(13)
    total = int(13.0/DT)
    frames, times, lefts, rights, wants_l, wants_r = [], [], [], [], [], []
    for step in range(total):
        t = step*DT
        bearing, elevation, distance = path_at(t)
        G.place_ball(body.model, body.data, target, bearing, elevation, distance)
        obs = body.observe()
        raw = eyes.observe_raw()
        activation = brain.step(obs, environment=env, eye_pixels=raw, dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        obs = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
        command = brain.eye_command()
        ask = W.required(body, target)
        times.append(t)
        lefts.append(float(command[0]))
        rights.append(float(command[2]))
        wants_l.append(ask[0][0])
        wants_r.append(ask[1][0])
        if step % STRIDE:
            continue
        fresh = eyes.observe_raw()
        left_image = Image.fromarray(np.asarray(fresh[0]), "RGB").resize(PANEL, Image.NEAREST)
        right_image = Image.fromarray(np.asarray(fresh[1]), "RGB").resize(PANEL, Image.NEAREST)
        third.update_scene(body.data, camera=cam)
        seen = Image.fromarray(third.render()).resize(PANEL, Image.LANCZOS)
        canvas = Image.new("RGB", (PANEL[0]*3, PANEL[1] + PLOT[1]), (24, 24, 28))
        canvas.paste(third_image := seen, (0, 0))
        canvas.paste(left_image, (PANEL[0], 0))
        canvas.paste(right_image, (PANEL[0]*2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), "第三人称", font=big, fill=(255, 255, 0))
        draw.text((PANEL[0] + 6, 4), "左眼看到的", font=big, fill=(255, 255, 0))
        draw.text((PANEL[0]*2 + 6, 4), "右眼看到的", font=big, fill=(255, 255, 0))
        # 曲线
        top = PANEL[1]
        draw.rectangle([0, top, canvas.width, canvas.height], fill=(16, 16, 20))
        span = .7
        mid = top + PLOT[1]//2

        def y(value):
            return mid - value/span*(PLOT[1]//2 - 12)
        draw.line([0, mid, canvas.width, mid], fill=(70, 70, 80))
        for index in range(1, len(times)):
            x0 = int((index - 1)*canvas.width/len(times))
            x1 = int(index*canvas.width/len(times))
            draw.line([x0, y(lefts[index - 1]), x1, y(lefts[index])], fill=(90, 170, 255))
            draw.line([x0, y(rights[index - 1]), x1, y(rights[index])], fill=(255, 160, 60))
            draw.line([x0, y(wants_l[index - 1]), x1, y(wants_l[index])], fill=(255, 90, 90))
            draw.line([x0, y(wants_r[index - 1]), x1, y(wants_r[index])], fill=(90, 230, 130))
        now = int(len(times)*canvas.width/len(times)) - 1
        draw.line([now, top, now, canvas.height], fill=(240, 240, 240))
        draw.text((6, top + 4), "眼球左右转（度） 蓝=左眼 橙=右眼 红虚线=左眼该到 绿虚线=右眼该到",
                  font=small, fill=(220, 220, 220))
        draw.text((6, canvas.height - 22),
                  "t=%4.1f秒  球：方位%+.2f 高低%+.2f 距离%.2f米   左眼%+5.1f度 右眼%+5.1f度"
                  % (t, bearing, elevation, distance, np.degrees(lefts[-1]),
                     np.degrees(rights[-1])), font=small, fill=(255, 255, 0))
        frames.append(canvas)
    out = ROOT / "artifacts" / "眼睛追球.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000*STRIDE*DT),
                   loop=0, optimize=True)
    print("%s  共 %d 帧，%.1f 秒" % (out, len(frames), len(frames)*STRIDE*DT))
    print("文件大小 %.1f MB" % (out.stat().st_size/1e6))
    eyes.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())