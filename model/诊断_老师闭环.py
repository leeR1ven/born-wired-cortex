# -*- coding: utf-8 -*-
"""诊断_老师闭环.py —— 摔，是「身体/肌肉参数不行」，还是「没有反馈不行」？

同一个身体、同一套节奏（50 Hz），只换「谁在下令」：

  A 老师闭环（我们这套肌肉参数）：老师每一拍都看身体状态，边看边修正
  B 老师闭环（老师自己的肌肉参数）：同上，但用老师训练时的那套 PD
  C 我们开环回放：照抄下来的动作，每拍照本宣科，完全不看身体状态

判读：
  A 稳、C 摔  ->  摔的原因是「没有反馈」，不是身体或参数
  A 也摔      ->  是身体/肌肉参数不匹配，抄得再准也没用

命令：python 诊断_老师闭环.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import 身体_go2 as 身体
from 取动作_从训练好的模型 import _读策略, 默认角

根 = pathlib.Path(__file__).resolve().parent
帧秒 = 1.0 / 50.0
策略 = _读策略()

库 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))
挑 = [n for n in ["前进0.8", "前进1.6", "后退0.4", "右移0.4", "左转0.75",
                 "前进0.8左转0.75"] if n in 库]
参数组 = [
    ("A 老师闭环  用我们的 k80 d4", dict(刚度=80.0, 阻尼=4.0, 力矩=25.0)),
    ("B 老师闭环  用老师的 k20/40 d1/2", None),
    ("C 我们开环  用我们的 k80 d4", dict(刚度=80.0, 阻尼=4.0, 力矩=25.0)),
]


def _摆参数(条):
    if 条 is None:
        身体.刚度 = np.array([20.0, 20.0, 40.0] * 4)
        身体.阻尼 = np.array([1.0, 1.0, 2.0] * 4)
        身体.最大力矩 = np.array([23.7, 23.7, 45.43] * 4)
    else:
        身体.刚度 = np.full(12, 条["刚度"])
        身体.阻尼 = np.full(12, 条["阻尼"])
        身体.最大力矩 = np.full(12, 条["力矩"])


def 老师闭环(指令, 秒, 条):
    _摆参数(条)
    身 = 身体.身体()
    身.摆成站姿()
    起 = 身.数据.qpos[:2].copy()
    上次 = np.zeros(12)
    撑 = 0.0
    for k in range(int(秒 / 帧秒)):
        R = 身.数据.xmat[1].reshape(3, 3)
        观察 = np.concatenate([身.数据.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]),
                              np.asarray(指令, dtype=float), 身.数据.qpos[7:] - 默认角,
                              身.数据.qvel[6:], 上次])
        上次 = 策略(观察)
        身.步进(身体.角度转发力(np.degrees(默认角 + 0.5 * 上次)))
        撑 = k * 帧秒
        if 身.机身歪了() > 75:
            break
    p = 身.数据.qpos
    return 撑, float(np.hypot(p[0] - 起[0], p[1] - 起[1]))


def 我们开环(名, 秒, 条):
    _摆参数(条)
    拍 = np.array(库[名]["发力"], dtype=float)
    身 = 身体.身体()
    身.摆成站姿()
    起 = 身.数据.qpos[:2].copy()
    撑 = 0.0
    for k in range(int(秒 / 帧秒)):
        身.步进(拍[k % len(拍)])
        撑 = k * 帧秒
        if 身.机身歪了() > 75:
            break
    p = 身.数据.qpos
    return 撑, float(np.hypot(p[0] - 起[0], p[1] - 起[1]))


秒 = 5.0
print("每个格子里是「撑住秒 / 走了米」，5.0 秒是满分（没摔）")
print("%-32s" % "谁在下令", " ".join("%13s" % n for n in 挑))
for 标签, 条 in 参数组:
    行 = []
    for 名 in 挑:
        if 标签.startswith("C"):
            t, m = 我们开环(名, 秒, 条)
        else:
            t, m = 老师闭环(库[名]["指令"], 秒, 条)
        行.append("%5.1f/%.2f" % (t, m))
    print("%-32s" % 标签, " ".join("%13s" % x for x in 行))