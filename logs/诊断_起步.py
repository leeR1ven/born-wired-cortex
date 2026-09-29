# -*- coding: utf-8 -*-
"""诊断_起步.py —— 开环回放一上来就摔，会不会只是「起步那一拍的姿势不对」？

对照三组，动作都一样（照抄的一个步态周期，50 Hz k80 d4）：
  (1) 从我们自己的站姿起步，直接回放            （现在 验收_动作库.py 的做法）
  (2) 从老师切周期那一刻的真实姿势/速度起步，再回放
  (3) 老师闭环，当作满分参考

如果 (2) 明显比 (1) 好，说明摔的一大半原因是「起步对不上」，
而不是动作本身有问题。

命令：python 诊断_起步.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import 身体_go2 as 身体
from 取动作_从训练好的模型 import _读策略, 默认角, 切一拍, 物理步长, 控制间隔

根 = pathlib.Path(__file__).resolve().parent
帧秒 = 1.0 / 50.0
策略 = _读策略()
库 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))


def 老师走(指令, 秒=5.0):
    """老师闭环走一遍，把每拍的（目标角 / 整条 qpos / 整条 qvel）都记下来。"""
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
            数据.ctrl[:] = np.clip(身体.刚度 * (目标 - 数据.qpos[7:])
                                   - 身体.阻尼 * 数据.qvel[6:], -25.0, 25.0)
            mujoco.mj_step(模型, 数据)
        A.append(目标.copy()); Q.append(数据.qpos.copy()); V.append(数据.qvel.copy())
    return np.array(A), np.array(Q), np.array(V)


def 切点(角历史):
    """跟 取动作 里一样，找出「一个完整周期从哪一拍开始」。"""
    信号 = 角历史[:, 1] - 角历史[:, 4]
    信号 = 信号 - 信号.mean()
    后半 = 信号[int(2.0 / 帧秒):]
    from 取动作_从训练好的模型 import 找周期
    n = 找周期(后半, 帧秒)
    if n is None:
        return None, 0
    起 = int(np.argmin(后半[:max(n, 1) * 3]))
    return n, 起


def 开环(拍, 秒, 起手=None):
    """起手=None 就从我们自己的站姿起步；否则从给的那套 qpos/qvel 起步。"""
    身 = 身体.身体()
    if 起手 is None:
        身.摆成站姿()
    else:
        q, v = 起手
        身.数据.qpos[:] = q
        身.数据.qvel[:] = v
        身.mujoco.mj_forward(身.模型, 身.数据)
    起 = 身.数据.qpos[:2].copy()
    撑 = 0.0
    for k in range(int(秒 / 帧秒)):
        身.步进(拍[k % len(拍)])
        撑 = k * 帧秒
        if 身.机身歪了() > 75:
            break
    p = 身.数据.qpos
    return 撑, float(np.hypot(p[0] - 起[0], p[1] - 起[1]))


def 老师闭环(指令, 秒):
    A, Q, V = 老师走(指令, 秒)
    起 = Q[0, :2].copy()
    歪 = []
    for k in range(len(Q)):
        w, x, y, z = Q[k, 3:7]
        上 = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
        歪.append(np.degrees(np.arccos(np.clip(上[2], -1, 1))))
        撑 = k * 帧秒
        if 歪[-1] > 75:
            break
    p = Q[k]
    return 撑, float(np.hypot(p[0] - 起[0], p[1] - 起[1]))


挑 = [n for n in ["前进0.8", "前进1.6", "后退0.4", "右移0.4", "左转0.75"] if n in 库]
print("每格「撑住秒 / 走了米」，5.0 是满分")
print("%-34s" % "", " ".join("%12s" % n for n in 挑))
格 = {1: [], 2: [], 3: []}
for 名 in 挑:
    指令 = 库[名]["指令"]
    A, Q, V = 老师走(指令, 5.0)
    n, 起 = 切点(A)
    拍 = 身体.角度转发力(np.degrees(A[起:起 + n]))
    t1, m1 = 开环(拍, 5.0)
    t2, m2 = 开环(拍, 5.0, 起手=(Q[起], V[起]))
    t3, m3 = 老师闭环(指令, 5.0)
    for i, (t, m) in enumerate([(t1, m1), (t2, m2), (t3, m3)], 1):
        格[i].append("%5.1f/%.2f" % (t, m))
for i, 标签 in [(1, "(1) 从我们的站姿起步，开环回放"),
                (2, "(2) 从老师切周期那拍的姿势起步，开环"),
                (3, "(3) 老师闭环（参考）")]:
    print("%-34s" % 标签, " ".join("%12s" % x for x in 格[i]))