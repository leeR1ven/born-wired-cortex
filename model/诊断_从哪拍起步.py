# -*- coding: utf-8 -*-
"""诊断_从哪拍起步.py —— 同一串步态，从周期的哪一拍开始回放，摔不摔？

现在切周期是「在相位信号的谷底里，随便取最靠前的那一个」。身体是从站姿开始的，
如果那一拍的姿势离站姿很远，一上来就崴，跟动作好不好无关。

这个脚本对几个「摔得早」的动作，把周期的每一个起点都试一遍，看撑住秒数的分布。
如果换个起点就能走满 4 秒，那修法很简单：**挑离站姿最近的那一拍当起点**。

命令：python 诊断_从哪拍起步.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import 身体_go2 as 身体
from 取动作_从训练好的模型 import _读策略, 默认角, 找周期, 物理步长, 控制间隔

根 = pathlib.Path(__file__).resolve().parent
帧秒 = 1.0 / 50.0
策略 = _读策略()
验收 = json.loads((根 / "验收_动作库.json").read_text(encoding="utf-8"))
训练 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))

手编 = {"站姿", "趴姿", "坐", "蹲", "站高", "抬左前腿", "抬右前腿", "趴下", "趴姿低",
       "蹲坐", "站起来", "趴下去", "坐下", "抬腿", "低头"}
摔得早 = [k for k in 验收 if k not in 手编 and 验收[k]["撑住秒"] < 1.5 and k in 训练]
print("摔得早的有 %d 个，挑前 6 个来看" % len(摔得早))
挑 = 摔得早[:6]


def 老师走(指令, 秒=5.0):
    import mujoco
    模型 = mujoco.MjModel.from_xml_path(str(身体.保证模型存在() / "scene.xml"))
    数据 = mujoco.MjData(模型)
    mujoco.mj_resetDataKeyframe(模型, 数据, 0)
    数据.qpos[7:] = 默认角
    mujoco.mj_forward(模型, 数据)
    上次 = np.zeros(12)
    A = []
    for _ in range(int(秒 / (控制间隔 * 物理步长))):
        R = 数据.xmat[1].reshape(3, 3)
        观察 = np.concatenate([数据.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]),
                              np.asarray(指令, dtype=float), 数据.qpos[7:] - 默认角,
                              数据.qvel[6:], 上次])
        上次 = 策略(观察)
        目标 = 默认角 + 0.5 * 上次
        for _ in range(控制间隔):
            数据.ctrl[:] = np.clip(身体.刚度 * (目标 - 数据.qpos[7:])
                                   - 身体.阻尼 * 数据.qvel[6:], -身体.最大力矩, 身体.最大力矩)
            mujoco.mj_step(模型, 数据)
        A.append(目标.copy())
    return np.array(A)


def 回放(拍, 秒=4.0):
    身 = 身体.身体()
    身.摆成站姿()
    for i in range(int(秒 / 帧秒)):
        身.步进(拍[i % len(拍)])
        if 身.机身歪了() > 75:
            break
    return i * 帧秒


片长 = np.radians(身体.角度范围[:, 1] - 身体.角度范围[:, 0])
stand = np.radians(身体.站姿角)


def 离站姿多远(角):
    return float(np.abs((角 - stand) / 片长).max())


for 名 in 挑:
    A = 老师走(训练[名]["指令"])
    信号 = A[:, 1] - A[:, 4]
    信号 = 信号 - 信号.mean()
    后半 = 信号[int(2.0 / 帧秒):]
    n = 找周期(后半, 帧秒)
    候选 = []
    for i in range(1, min(len(后半) - n - 1, 3 * n)):
        if 后半[i] <= 后半[i - 1] and 后半[i] <= 后半[i + 1]:
            候选.append(i)
    if not 候选:
        候选 = [int(np.argmin(后半))]
    成绩 = []
    for c in 候选:
        起 = int(2.0 / 帧秒) + c
        拍 = 身体.角度转发力(np.degrees(A[起:起 + n]))
        成绩.append((回放(拍), 离站姿多远(A[起])))
    现在 = int(np.argmin(后半[:max(n, 1) * 3]))
    最好 = max(成绩)[0]
    按距离 = 成绩[int(np.argmin([x[1] for x in 成绩]))][0]
    print("%-22s 周期%2d拍 候选起点%2d个  现在这个起点撑%.1f秒  离站姿最近那个撑%.1f秒  最好%.1f秒"
          % (名, n, len(候选), 回放(身体.角度转发力(np.degrees(A[int(2.0/帧秒)+现在:int(2.0/帧秒)+现在+n]))), 按距离, 最好))