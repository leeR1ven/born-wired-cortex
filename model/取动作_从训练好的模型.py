# -*- coding: utf-8 -*-
"""取动作_从训练好的模型.py —— 把别人训练好的四足模型能做的动作，**尽量全**解析出来。

★ 在做什么

  我们自己没能力训练一个会走路的四条腿模型，但网上有现成的（见"来源"）。
  这个脚本把它当老师：让它按**一大片速度指令**（前后 / 左右 / 转圈的各种组合）
  一个个走一遍，把它每一刻**命令**的关节角抄下来，切成"一个完整步态周期"，
  再换算成我们模型要的 12 块肌肉 × 每一拍的发力（0~1）。
  最后把重复的合并掉（不同的指令常常给出一模一样的步态）。

  为什么要抄：我们的架构里运动皮层是"肌群神经元"，亮几个就是出多大力。
  把角度换成发力，就能写进皮层名字 / 本能表，变成"天生就会的一串动作"。

★ 来源

  宇树 Go2 的走路策略：HuggingFace 上的 diasAiMaster/unitree-go2-velocity-flat
  （MuJoCo + PPO 训的，4.5MB，会站立/各种速度走/横移/转圈）
  下载到 C:\\go2_policy\\vel\\model_500.pt

★ 节奏要对上（2026-09-14 晚）

  老师是 50 Hz（每 20 毫秒下一次令）。我们大脑一拍也设成 1/50 秒，
  抄下来的动作才不会被"兑稀"。实测：30 Hz 时同一批动作只有 13 个能走，
  50 Hz + 肌肉刚度 80/阻尼 4 时 25 个都能自己做完。

★ 输出

  动作库_训练好的.json —— 每个动作：指令、一个完整步态周期的发力序列

命令：
    python 取动作_从训练好的模型.py           # 重新解析一遍（约 2 分钟）
    python 取动作_从训练好的模型.py 看         # 只看已有结果
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import 身体_go2 as 身体       # 肌肉参数只有这一个来源：身体说了算

根 = pathlib.Path(__file__).resolve().parent
策略文件 = pathlib.Path(r"C:\go2_policy\vel\model_500.pt")
输出文件 = 根 / "动作库_训练好的.json"

物理步长 = 0.002
控制间隔 = 10                 # 每 10 个物理步下一次令 = 50 Hz
默认角 = np.array([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8])
# 抄的时候用的肌肉参数。★ 必须和回放时（身体_go2.py）是同一套 ★
#
# 2026-09-14 晚实测（诊断_参数配对.py）：原来这里写的是老师训练时的那套
# （k20/20/40 d1/1/2），而回放用的是我们自己的 k80 d4 —— 两边不一致，
# 抄下来的那串姿势对不上我们的身体，7 个动作回放全摔（0.9~2.9 秒）。
# 改成跟身体用同一套之后，同样 7 个动作里有 5 个能自己走满 5 秒。
# 所以这里不再自己写死，直接跟着 身体_go2 走，免得以后又对不上。
刚度 = np.full(12, float(身体.刚度))
阻尼 = np.full(12, float(身体.阻尼))
力矩上限 = np.full(12, float(身体.最大力矩))
大脑帧秒 = 1.0 / 50.0          # 我们大脑一拍 = 1/50 秒（和老师一样）
模拟秒 = 5.0                   # 每个指令走多久（前 2 秒用来站稳，后面才拿来切周期）

# ================= 速度指令的采样网格 =================
# 老师的指令是连续的三个数（前后速度、左右速度、转圈速度），能取的值是无穷多。
# 我们把每一维分成几档，取"每一档 × 每一档 × 每一档"，就是一张地毯。
# 相邻两档给出的步态往往一模一样，后面会去重（见 合并重复）。
前后档 = [-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6]
左右档 = [-0.8, -0.4, 0.0, 0.4, 0.8]
转档 = [-1.5, -0.75, 0.0, 0.75, 1.5]
去重阈值 = 0.05                # 两个步态逐拍差都不超过这个数，就算同一个动作


def 动作名(指令):
    """把三个速度数翻译成一个中文名字。"""
    前, 左, 转 = float(指令[0]), float(指令[1]), float(指令[2])
    段 = []
    if abs(前) >= 1e-9:
        段.append("%s%g" % ("前进" if 前 > 0 else "后退", abs(前)))
    if abs(左) >= 1e-9:
        段.append("%s%g" % ("左移" if 左 > 0 else "右移", abs(左)))
    if abs(转) >= 1e-9:
        段.append("%s%g" % ("左转" if 转 > 0 else "右转", abs(转)))
    return "".join(段) or "站"


def 网格():
    """把三档拼成一片指令，返回 [(名字, (前,左,转)), ...]。"""
    出 = []
    见过 = set()
    for 前 in 前后档:
        for 左 in 左右档:
            for 转 in 转档:
                if 前 == 0.0 and 左 == 0.0 and 转 == 0.0:
                    continue                       # 原地不动 = 站着，用手编的站姿
                名 = 动作名((前, 左, 转))
                if 名 in 见过:
                    continue
                见过.add(名)
                出.append((名, (前, 左, 转)))
    return 出


# ================= 把训练好的模型跑起来 =================
def _读策略():
    import torch
    sd = torch.load(策略文件, map_location="cpu", weights_only=False)["model_state_dict"]
    权 = [sd[f"actor.{i}.weight"].numpy() for i in (0, 2, 4, 6)]
    偏 = [sd[f"actor.{i}.bias"].numpy() for i in (0, 2, 4, 6)]
    均 = sd["actor_obs_normalizer._mean"].numpy().ravel()
    方 = sd["actor_obs_normalizer._var"].numpy().ravel()

    def 前向(obs):
        x = (obs - 均) / np.sqrt(方 + 1e-8)
        for k in range(3):
            x = 权[k] @ x + 偏[k]
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 50.0)))     # ELU
        return 权[3] @ x + 偏[3]
    return 前向


def 走一遍(指令, 秒=None, 前向=None):
    """让训练好的模型带着身体走这么久，返回（每一拍的目标角, 位置历史, 控制间隔秒）"""
    import mujoco
    import 身体_go2 as 身体
    秒 = 模拟秒 if 秒 is None else 秒
    前向 = 前向 or _读策略()
    模型 = mujoco.MjModel.from_xml_path(str(身体.保证模型存在() / "scene.xml"))
    数据 = mujoco.MjData(模型)
    mujoco.mj_resetDataKeyframe(模型, 数据, 0)
    数据.qpos[7:] = 默认角
    mujoco.mj_forward(模型, 数据)
    上次 = np.zeros(12)
    角历史, 位历史 = [], []
    for _ in range(int(秒 / (控制间隔 * 物理步长))):
        R = 数据.xmat[1].reshape(3, 3)
        观察 = np.concatenate([数据.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]), 指令,
                              数据.qpos[7:] - 默认角, 数据.qvel[6:], 上次])
        上次 = 前向(观察)
        目标 = 默认角 + 0.5 * 上次
        for _ in range(控制间隔):
            数据.ctrl[:] = np.clip(刚度 * (目标 - 数据.qpos[7:]) - 阻尼 * 数据.qvel[6:],
                                   -力矩上限, 力矩上限)
            mujoco.mj_step(模型, 数据)
        w, x, y, z = 数据.qpos[3:7]
        角历史.append(目标.copy())        # 记的是「大脑发出的目标角」，不是实际角度
        位历史.append((float(数据.qpos[0]), float(数据.qpos[1]), float(数据.qpos[2]),
                      float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))))
    return np.array(角历史), np.array(位历史), 控制间隔 * 物理步长


# ================= 从一个长序列里切出一个完整周期 =================
def 找周期(信号, 帧秒, 最短=0.22, 最长=1.3):
    """找一个完整步态周期有多长。用自相关：r(P) 在周期的整数倍上都有峰，
    取**最靠前的那个大峰**才是真周期（不然会退到 2 倍、3 倍上去）。"""
    信号 = 信号 - 信号.mean()
    能量 = float(np.dot(信号, 信号))
    if 能量 <= 0:
        return None
    候选 = []
    for P in np.arange(最短, 最长, 0.005):
        n = int(round(P / 帧秒))
        if n < 2 or n >= 信号.size // 2:
            continue
        a = 信号[:-n]
        b = 信号[n:]
        r = float(np.dot(a, b)) / np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        候选.append((P, n, r))
    if not 候选:
        return None
    最高 = max(r for _, _, r in 候选)
    for i in range(1, len(候选) - 1):
        P, n, r = 候选[i]
        if r >= 0.75 * 最高 and r >= 候选[i - 1][2] and r >= 候选[i + 1][2]:
            return n
    return max(候选, key=lambda x: x[2])[1]


def 切一拍(角历史, 帧秒):
    """返回（一拍的角度序列, 周期秒, 拍数）。用两条对角前腿的大腿角之差当相位信号。"""
    信号 = 角历史[:, 1] - 角历史[:, 4]
    信号 = 信号 - 信号.mean()
    后半 = 信号[int(2.0 / 帧秒):]
    n = 找周期(后半, 帧秒)
    if n is None:
        return None, None, 0
    起 = int(np.argmin(后半[:max(n, 1) * 3]))
    一段 = 角历史[int(2.0 / 帧秒) + 起: int(2.0 / 帧秒) + 起 + n + 1]
    if 一段.shape[0] < n + 1:
        return None, None, 0
    拍数 = max(4, int(round(n * 帧秒 / 大脑帧秒)))
    原t = np.linspace(0.0, 1.0, 一段.shape[0])
    新t = np.linspace(0.0, 1.0, 拍数 + 1)[:-1]
    重采 = np.stack([np.interp(新t, 原t, 一段[:, j]) for j in range(12)], axis=1)
    return 重采, n * 帧秒, 拍数


# ================= 去重 =================
def 是同一个(甲, 乙):
    """两个步态的拍数一样、逐拍逐肌肉都差不多，就算同一个动作。"""
    if len(甲) != len(乙):
        return False
    return float(np.abs(np.asarray(甲) - np.asarray(乙)).max()) <= 去重阈值


def 合并重复(动作, 说=True):
    留下 = {}
    合并数 = 0
    for 名, v in 动作.items():
        for 已有名, 已 in 留下.items():
            if 是同一个(v["发力"], 已["发力"]):
                已["别名"].append(名)
                合并数 += 1
                break
        else:
            留下[名] = v
            留下[名].setdefault("别名", [])
    if 说:
        print("去重：采样 %d 条 -> 去掉 %d 条重复 -> 留下 %d 个不同的动作"
              % (len(动作), 合并数, len(留下)))
    return 留下


def 解析(说=True):
    import 身体_go2 as 身体
    前 = _读策略()
    采样 = 网格()
    if 说:
        print("采 %d 条速度指令（前后 %d 档 × 左右 %d 档 × 转 %d 档），每条走 %.0f 秒"
              % (len(采样), len(前后档), len(左右档), len(转档), 模拟秒))
    动作 = {}
    摔 = 0
    for i, (名, 指令) in enumerate(采样, 1):
        角历史, 位历史, 帧秒 = 走一遍(np.array(指令, dtype=float), 前向=前)
        静止 = 位历史[int(2.0 / 帧秒):]
        位移 = float(np.hypot(静止[-1, 0] - 静止[0, 0], 静止[-1, 1] - 静止[0, 1]))
        转角 = float(np.degrees(静止[-1, 3] - 静止[0, 3]))
        最低 = float(静止[:, 2].min())
        一拍, 周期, 拍数 = 切一拍(角历史, 帧秒)
        if 一拍 is None:
            一拍 = 角历史[int(2.0 / 帧秒): int(2.0 / 帧秒) + 1]; 周期 = 0.0; 拍数 = 1
        发力 = 身体.角度转发力(np.degrees(一拍))
        动作[名] = dict(指令=list(指令), 一拍个数=拍数, 周期秒=round(float(周期), 3),
                        发力=[[round(float(v), 3) for v in 行] for 行 in 发力],
                        走动米=round(位移, 2), 转角度=round(转角, 1),
                        最低机身=round(最低, 3), 别名=[])
        if 最低 <= 0.18:
            摔 += 1
        if 说 and (i % 25 == 0 or i == len(采样)):
            print("   进度 %d/%d ..." % (i, len(采样)))
    动作 = 合并重复(动作, 说=说)
    输出文件.write_text(json.dumps(动作, ensure_ascii=False, indent=1), encoding="utf-8")
    if 说:
        拍总 = sum(v["一拍个数"] for v in 动作.values())
        print("存到 %s：%d 个动作，一共 %d 拍" % (输出文件.name, len(动作), 拍总))
        print("（照抄的时候老师自己摔了 %d 次，这些也留着了，用之前看 最低机身）" % 摔)
    return 动作


def 看():
    动作 = json.loads(输出文件.read_text(encoding="utf-8"))
    print("共 %d 个动作" % len(动作))
    for 名, v in 动作.items():
        print("  %-22s 指令%-20s %3d 拍  周期 %5.2f 秒  走 %4.1f 米  转 %6.1f 度%s"
              % (名, str(v["指令"]), v["一拍个数"], v["周期秒"], v["走动米"], v["转角度"],
                 "   （含别名 " + "、".join(v.get("别名", [])) + "）" if v.get("别名") else ""))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "看":
        看()
    else:
        print("让训练好的模型带着身体把整片速度指令走一遍，把动作抄下来：")
        解析()