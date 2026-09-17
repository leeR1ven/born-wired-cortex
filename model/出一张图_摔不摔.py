# -*- coding: utf-8 -*-
"""出一张图_摔不摔.png —— 同一个动作（前进0.8），三种下令方式并排看。

  第一行：老师闭环（每一拍都看着身体修正）
  第二行：改之前 —— 抄的时候用老师的肌肉参数，回放用我们的（库里原来就是这样）
  第三行：改之后 —— 抄和回放用同一套参数

每一格是一个时刻的真实姿态，格子上写着当时的机身高度。
命令：python 出一张图_摔不摔.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import 身体_go2 as 身体
from 取动作_从训练好的模型 import _读策略, 默认角, 物理步长, 控制间隔

根 = pathlib.Path(__file__).resolve().parent
帧秒 = 1.0 / 50.0
策略 = _读策略()
库 = json.loads((根 / "动作库.json").read_text(encoding="utf-8"))
旧库 = json.loads((根 / "动作库_训练好的.json.bak_老师参数").read_text(encoding="utf-8"))
名 = "前进0.8"
指令 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))[名]["指令"]

宽, 高 = 420, 300
时刻 = [0.2, 1.0, 2.0, 4.0]


def 字体(大小=16):
    for 路径 in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\simsun.ttc"):
        try:
            return ImageFont.truetype(路径, 大小)
        except Exception:
            pass
    return ImageFont.load_default()


def 老师闭环快照():
    身 = 身体.身体()
    身.摆成站姿()
    上次 = np.zeros(12)
    图, 记录 = [], []
    for k in range(int(4.0 / 帧秒)):
        R = 身.数据.xmat[1].reshape(3, 3)
        观察 = np.concatenate([身.数据.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]),
                              np.asarray(指令, dtype=float), 身.数据.qpos[7:] - 默认角,
                              身.数据.qvel[6:], 上次])
        上次 = 策略(观察)
        身.步进(身体.角度转发力(np.degrees(默认角 + 0.5 * 上次)))
        记录.append((身.数据.qpos.copy(), 身.机身高度(), 身.机身歪了()))
    return 记录


def 开环快照(拍):
    身 = 身体.身体()
    身.摆成站姿()
    记录 = []
    for k in range(int(4.0 / 帧秒)):
        身.步进(np.clip(np.asarray(拍, dtype=float)[k % len(拍)], 0, 1))
        记录.append((身.数据.qpos.copy(), 身.机身高度(), 身.机身歪了()))
    return 记录


def 取一格(记录, 秒):
    i = min(int(秒 / 帧秒), len(记录) - 1)
    return 记录[i]


print("跑三遍 ...")
行 = [("老师闭环（边看边修正）", 老师闭环快照()),
      ("改之前：抄和回放用的不是同一套肌肉参数",
       开环快照(np.array(旧库[名]["发力"], dtype=float))),
      ("改之后：两边用同一套（我们自己的）",
       开环快照(np.array(库[名]["拍"], dtype=float)))]

画布 = Image.new("RGB", (宽 * len(时刻), 高 * len(行) + 26 * len(行)), (20, 20, 24))
画 = ImageDraw.Draw(画布)
f = 字体(15)
身 = 身体.身体()
for r, (标签, 记录) in enumerate(行):
    y0 = r * (高 + 26)
    画.text((8, y0 + 5), 标签, fill=(255, 220, 120), font=f)
    for c, 秒 in enumerate(时刻):
        q, 高米, 歪 = 取一格(记录, 秒)
        身.数据.qpos[:] = q
        身.mujoco.mj_forward(身.模型, 身.数据)
        图 = 身.画(相机=88.0, 俯角=-16.0, 距离=1.35,
                   看向=(float(q[0]), float(q[1]), 0.24))
        图 = np.asarray(图, dtype=np.uint8)
        图 = Image.fromarray(图).resize((宽, 高))
        d = ImageDraw.Draw(图)
        d.text((6, 4), "%.1f 秒   机身 %.2f 米   歪 %.0f 度" % (秒, 高米, 歪),
               fill=(255, 255, 255), font=f)
        画布.paste(图, (c * 宽, y0 + 26))
画布.save(根 / "出一张图_摔不摔.png")
print("存好了：", 根 / "出一张图_摔不摔.png")