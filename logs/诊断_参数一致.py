# -*- coding: utf-8 -*-
"""诊断_参数一致.py —— 抄动作的时候用的肌肉参数，和回放的时候不是同一套？

  库里的动作是「老师用**它自己的** PD（k20/40 d1/2）走出来的那一串角」。
  我们现在回放用的是**我们的** PD（k80 d4）。
  身体软硬不一样 → 抄下来的那串姿势对不上我们的身体 → 摔。

  对照组：
    甲：抄的时候用老师的 PD，回放用我们的 PD      （= 现在库里的情况）
    乙：抄的时候用我们的 PD，回放用我们的 PD      （参数一致）
    丙：老师闭环参考

命令：python 诊断_参数一致.py
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

老师k = np.array([20.0, 20.0, 40.0] * 4)
老师d = np.array([1.0, 1.0, 2.0] * 4)
老师τ = np.array([23.7, 23.7, 45.43] * 4)
我们k = np.full(12, 80.0)
我们d = np.full(12, 4.0)
我们τ = np.full(12, 25.0)


def 老师走(指令, 秒, k, d, τ):
    import mujoco
    模型 = mujoco.MjModel.from_xml_path(str(身体.保证模型存在() / "scene.xml"))
    数据 = mujoco.MjData(模型)
    mujoco.mj_resetDataKeyframe(模型, 数据, 0)
    数据.qpos[7:] = 默认角
    mujoco.mj_forward(模型, 数据)
    上次 = np.zeros(12)
    A, Q, V = [], [], []
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
        A.append(目标.copy()); Q.append(数据.qpos.copy()); V.append(数据.qvel.copy())
    return np.array(A), np.array(Q), np.array(V)


def 切周期(A):
    """照 取动作 里的算法切一个周期，返回（起拍下标, 拍数）。"""
    信号 = A[:, 1] - A[:, 4]
    信号 = 信号 - 信号.mean()
    后半 = 信号[int(2.0 / 帧秒):]
    n = 找周期(后半, 帧秒)
    if n is None:
        return None, 0
    起 = int(2.0 / 帧秒) + int(np.argmin(后半[:max(n, 1) * 3]))
    return 起, n


def 回放(拍, 秒):
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
print("每格「撑住秒 / 走了米」，5.0 是满分")
print("%-40s" % "", " ".join("%12s" % n for n in 挑))
甲, 乙, 丙, 丁 = [], [], [], []
for 名 in 挑:
    指令 = 训练[名]["指令"]
    A1, _, _ = 老师走(指令, 5.0, 老师k, 老师d, 老师τ)
    起1, n1 = 切周期(A1)
    甲.append(回放(身体.角度转发力(np.degrees(A1[起1:起1 + n1])), 5.0))
    A2, _, _ = 老师走(指令, 5.0, 我们k, 我们d, 我们τ)
    起2, n2 = 切周期(A2)
    乙.append(回放(身体.角度转发力(np.degrees(A2[起2:起2 + n2])), 5.0))
    丁.append(回放(np.array(库[名]["拍"], dtype=float), 5.0))
    丙.append((5.0, float(np.hypot(A2[-1, 0] - A2[0, 0], A2[-1, 1] - A2[0, 1]))))
for 标签, 数据 in [("甲 抄=老师PD / 回放=我们PD（≈现在库里）", 甲),
                  ("乙 抄=我们PD / 回放=我们PD（参数一致）", 乙),
                  ("丁 直接回放库里存的那串拍", 丁),
                  ("丙 老师闭环（参考）", 丙)]:
    print("%-40s" % 标签, " ".join("%12s" % ("%5.1f/%.2f" % t) for t in 数据))