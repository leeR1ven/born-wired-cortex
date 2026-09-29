# -*- coding: utf-8 -*-
"""诊断_参数配对.py —— 「抄的时候用什么参数」和「回放的时候用什么参数」配成四组。

        回放=我们 k80 d4      回放=老师 k20/40 d1/2
 抄=我们      ??                     ??
 抄=老师      ??（现在库里就是这格）      ??

只有对角线上那两格才是「参数一致」，才可能站得住。

命令：python 诊断_参数配对.py
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
库 = json.loads((根 / "动作库.json").read_text(encoding="utf-8"))
训练 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))

组 = {
    "我们": (np.full(12, 80.0), np.full(12, 4.0), np.full(12, 25.0)),
    "老师": (np.array([20.0, 20.0, 40.0] * 4), np.array([1.0, 1.0, 2.0] * 4),
             np.array([23.7, 23.7, 45.43] * 4)),
}


def 老师走(指令, 秒, 条):
    import mujoco
    k, d, τ = 条
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
            数据.ctrl[:] = np.clip(k * (目标 - 数据.qpos[7:]) - d * 数据.qvel[6:], -τ, τ)
            mujoco.mj_step(模型, 数据)
        A.append(目标.copy())
    return np.array(A)


def 切周期(A):
    信号 = A[:, 1] - A[:, 4]
    信号 = 信号 - 信号.mean()
    后半 = 信号[int(2.0 / 帧秒):]
    n = 找周期(后半, 帧秒)
    if n is None:
        return None, 0
    return int(2.0 / 帧秒) + int(np.argmin(后半[:max(n, 1) * 3])), n


def 回放(拍, 条, 秒=5.0):
    k, d, τ = 条
    身体.刚度, 身体.阻尼, 身体.最大力矩 = k, d, τ
    身 = 身体.身体()
    身.摆成站姿()
    起 = 身.数据.qpos[:2].copy()
    撑 = 0.0
    for i in range(int(秒 / 帧秒)):
        身.步进(拍[i % len(拍)])
        撑 = i * 帧秒
        if 身.机身歪了() > 75:
            break
    p = 身.数据.qpos
    return 撑, float(np.hypot(p[0] - 起[0], p[1] - 起[1]))


挑 = [n for n in ["前进0.8", "前进1.6", "后退0.4", "右移0.4", "左转0.75",
                 "前进1.2", "右移0.8"] if n in 库]
print("每格「撑住秒 / 走了米」，5.0 是满分。偷懒版：库里存的那串拍直接回放")
print("%-26s" % "", " ".join("%12s" % n for n in 挑))
print("%-26s" % "回放=我们 k80 d4", " ".join(
    "%12s" % ("%5.1f/%.2f" % 回放(np.array(库[n]["拍"], dtype=float), 组["我们"])) for n in 挑))
print("%-26s" % "回放=老师 k20/40 d1/2", " ".join(
    "%12s" % ("%5.1f/%.2f" % 回放(np.array(库[n]["拍"], dtype=float), 组["老师"])) for n in 挑))