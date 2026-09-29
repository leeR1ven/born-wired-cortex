# -*- coding: utf-8 -*-
"""对照_RL_训练.py —— 同一个身体、同一个任务、同一个判据，强化学习要多少样本。

为什么要这个（用户 2026-09-16："同一个身体、同一个任务，强化学习要多少样本/时间"）：
  我们那个系统做这些事**训练步数 = 0**（本能表是人手写的，见论文 R1）。
  这是审稿人一定会问的对照。所以这个文件老老实实从头训一个 PPO，
  每 2 万步用**论文里同一套判据**考一次，记下"多少步学会"。

命令：
    python 对照_RL_训练.py 看见红                     # 默认最多 400 万步
    python 对照_RL_训练.py 站起来 300000
    python 对照_RL_训练.py 看见红 4000000 7           # 换随机种子

写出来的东西：
    日志_RL对照_<任务>.log     每一步数一行（成功率、走了多远）
    RL对照_<任务>.pt           策略网络的权重 + 观测归一化的均值方差
"""
from __future__ import annotations

import io
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import 对照_RL_环境 as 环境

torch.set_num_threads(4)
根 = pathlib.Path(__file__).resolve().parent


# ================= 策略网络 =================
class 策略(nn.Module):
    """小网络：观测 -> 12 块肌肉的发力。拿到的信息、交出的动作，和大脑是同一份。"""

    def __init__(self, 观测维, 动作数, 隐含=64):
        super().__init__()
        self.躯干 = nn.Sequential(
            nn.Linear(观测维, 隐含), nn.Tanh(),
            nn.Linear(隐含, 隐含), nn.Tanh(),
        )
        self.均值层 = nn.Linear(隐含, 动作数)
        self.价值层 = nn.Linear(隐含, 1)
        self.对数标 = nn.Parameter(torch.full((动作数,), -0.5))   # 一开始别抖得太厉害（σ≈0.6）

    def 前向(self, x):
        h = self.躯干(x)
        标 = self.对数标.clamp(-3.0, 0.0).exp()
        return self.均值层(h), 标, self.价值层(h).squeeze(-1)


def 对数概率(原, 均, 标):
    """tanh 压过之后的对数概率 —— 要减掉雅可比项，少了这一步 PPO 是错的。"""
    分 = torch.distributions.Normal(均, 标)
    return 分.log_prob(原).sum(-1) - torch.log(1.0 - torch.tanh(原) ** 2 + 1e-6).sum(-1)


def 原到发力(原):
    return ((torch.tanh(原) + 1.0) * 0.5).numpy()


# ================= 观测归一化 =================
class 归一化:
    """跑动均值/方差。不归一化的话，这种量级不齐的观测 PPO 基本学不动。"""

    def __init__(self, 维, 上限=10.0):
        self.均 = np.zeros(维)
        self.方 = np.ones(维)
        self.个数 = 1e-4
        self.上限 = 上限

    def 更新(self, x):
        self.个数 += 1.0
        差 = x - self.均
        self.均 += 差 / self.个数
        self.方 += 差 * (x - self.均)

    def 用(self, x):
        return np.clip((x - self.均) / np.sqrt(self.方 / self.个数 + 1e-8),
                       -self.上限, self.上限).astype(np.float32)


# ================= 考试 =================
def 考试(环, 净, 归, 局数=20, 种子=0):
    成功纸, 成功我们, 走远们, 站高们, 歪们 = 0, 0, [], [], []
    rng = np.random.default_rng(种子)
    for _ in range(局数):
        o = 环.复位()
        if 环.名 == "站起来" and rng.random() < 0.5:
            # 一半的考试给一点随机扰动：别只会从"一模一样的那一下"站起来
            环.身.数据.qpos[:2] += rng.normal(0.0, 0.02, 2)
            环.身.数据.qvel[:6] += rng.normal(0.0, 0.05, 6)
            环.身.mujoco.mj_forward(环.身.模型, 环.身.数据)
            o = 环.观测()
        信息 = None
        while True:
            with torch.no_grad():
                x = torch.from_numpy(归.用(o)).unsqueeze(0)
                均, _, _ = 净.前向(x)
                发力 = 原到发力(均)[0]
            o, _, 结束, 信息 = 环.步进(发力)
            if 结束:
                break
        走 = 信息["走了"]
        站着 = 信息["歪"] < 40.0 and 信息["机身高度"] >= 0.20
        # 「论文门槛」= 论文 R1 自己那条线（4 秒前进 > 0.30 米）
        # 「我们的水平」= 我们那个系统实际走到的（中位 1.73 米，最差 1.48 米）取 1.0 米
        if 环.名 == "站起来":
            纸 = 环.成功(信息)
            我们 = 纸
        else:
            纸 = 走 >= 0.30 and 站着
            我们 = 走 >= 1.00 and 站着
        成功纸 += int(纸)
        成功我们 += int(我们)
        走远们.append(走)
        站高们.append(信息["机身高度"])
        歪们.append(信息["歪"])
    return {"成功": 成功纸 / 局数, "强": 成功我们 / 局数, "走了": float(np.mean(走远们)),
            "高度": float(np.mean(站高们)), "歪": float(np.mean(歪们))}


# ================= 训练 =================
def 训练(任务名="看见红", 最大步=4_000_000, 种子=0, 考试间隔=20_000, 达标=0.95):
    np.random.seed(种子)
    torch.manual_seed(种子)
    环 = 环境.任务(任务名)
    o = 环.复位()
    观测维, 动作数 = int(o.size), 环境.肌肉数
    净 = 策略(观测维, 动作数)
    归 = 归一化(观测维)
    优化 = torch.optim.Adam(净.parameters(), lr=2e-4, eps=1e-5)

    每轮拍 = 2048
    轮次 = 10
    小批 = 256
    γ, λ, 裁剪 = 0.99, 0.95, 0.2
    熵系数, 价值系数 = 0.001, 0.5
    # ★ 奖励要整体放大（2026-09-16 踩的坑）：势函数塑形以后，**每一拍的奖励只有 0.01 上下**，
    #   而熵项是「系数 × 12 个维度」= 0.036 一拍 —— 熵比奖励还大 3 倍，策略就被推着
    #   发散（实测 20 万步在 0% 和 85% 之间来回跳，而**没训练过的策略本来是 100%**）。
    #   放大 50 倍以后，每拍奖励 ~2.5、熵项 0.012，比例才是正常的。
    奖励缩放 = 50.0

    日志 = []
    日志.append("任务「%s」，种子 %d；观测 %d 维、动作 %d 维；最多 %s 步"
              % (任务名, 种子, 观测维, 动作数, format(最大步, ",")))
    日志.append("%11s | %10s | %10s | %9s | %9s | %8s | %9s | %s"
              % ("步数", "论文门槛", "我们的水平", "走了多远", "末了机身高", "每秒步数", "累计耗时", "说明"))
    日志.append("-" * 106)

    全体步 = 0
    达标步 = None
    达标步强 = None
    下次考试 = 考试间隔
    起时刻 = time.time()
    o = 环.复位()

    while 全体步 < 最大步:
        观测 = np.zeros((每轮拍, 观测维), dtype=np.float32)
        原存 = np.zeros((每轮拍, 动作数), dtype=np.float32)
        对数存 = np.zeros(每轮拍, dtype=np.float32)
        价值存 = np.zeros(每轮拍, dtype=np.float32)
        奖励存 = np.zeros(每轮拍, dtype=np.float32)
        末值存 = np.zeros(每轮拍, dtype=np.float32)
        终值存 = np.zeros(每轮拍, dtype=np.float32)

        for t in range(每轮拍):
            归.更新(o.astype(np.float64))
            x = torch.from_numpy(归.用(o)).unsqueeze(0)
            with torch.no_grad():
                均, 标, 价值 = 净.前向(x)
                原 = torch.distributions.Normal(均, 标).sample()
                lp = 对数概率(原, 均, 标)
            观测[t] = 归.用(o)
            原存[t] = 原.numpy()[0]
            对数存[t] = float(lp)
            价值存[t] = float(价值)
            o, r, 结束, 信息 = 环.步进(原到发力(原)[0])
            奖励存[t] = r * 奖励缩放
            全体步 += 1
            # 摔倒结束的，后面那些"本来还能拿到的分"不该算进来（终止状态不 bootstrap）
            终值存[t] = 1.0 if 信息.get("掉倒") else 0.0
            if 结束:
                o = 环.复位()
            with torch.no_grad():
                末值存[t] = float(净.前向(torch.from_numpy(归.用(o)).unsqueeze(0))[2])
            if 全体步 >= 下次考试:
                break

        # ---- GAE ----
        优势 = np.zeros(每轮拍, dtype=np.float32)
        累积 = 0.0
        for t in range(每轮拍 - 1, -1, -1):
            活 = 1.0 - 终值存[t]
            δ = 奖励存[t] + γ * 末值存[t] * 活 - 价值存[t]
            累积 = δ + γ * λ * 活 * 累积
            优势[t] = 累积
        回报 = 优势 + 价值存
        优势 = (优势 - 优势.mean()) / (优势.std() + 1e-8)

        # ---- PPO 更新 ----
        张观 = torch.from_numpy(观测)
        张原 = torch.from_numpy(原存)
        张对 = torch.from_numpy(对数存)
        张优 = torch.from_numpy(优势)
        张回 = torch.from_numpy(回报)
        进度 = 全体步 / float(最大步)
        for g in 优化.param_groups:
            g["lr"] = 2e-4 * max(0.1, 1.0 - 进度)
        for _ in range(轮次):
            序 = np.random.permutation(每轮拍)
            for 起 in range(0, 每轮拍, 小批):
                批 = 序[起:起 + 小批]
                均, 标, 价值 = 净.前向(张观[批])
                新对 = 对数概率(张原[批], 均, 标)
                比 = torch.exp(新对 - 张对[批])
                一 = 比 * 张优[批]
                二 = torch.clamp(比, 1 - 裁剪, 1 + 裁剪) * 张优[批]
                熵 = torch.distributions.Normal(均, 标).entropy().sum(-1).mean()
                损失 = -torch.min(一, 二).mean() + 价值系数 * ((价值 - 张回[批]) ** 2).mean() \
                     - 熵系数 * 熵
                优化.zero_grad()
                损失.backward()
                nn.utils.clip_grad_norm_(净.parameters(), 0.5)
                优化.step()

        # ---- 考试 ----
        if 全体步 >= 下次考试:
            下次考试 += 考试间隔
            好 = 考试(环, 净, 归, 局数=20, 种子=种子)
            用 = time.time() - 起时刻
            说 = ""
            if 好["成功"] >= 达标 and 达标步 is None:
                达标步 = 全体步
                说 = "★ 论文门槛达标"
            if 好["强"] >= 达标 and 达标步强 is None:
                达标步强 = 全体步
                说 += " ★★ 达到我们那个系统的水平"
            行 = ("%11s | %9.0f%% | %9.0f%% | %9.3f | %9.3f | %8.0f | %8.1f 分 | %s"
                 % (format(全体步, ","), 好["成功"] * 100, 好["强"] * 100, 好["走了"],
                    好["高度"], 全体步 / max(用, 1e-6), 用 / 60.0, 说))
            日志.append(行)
            print(行, flush=True)
            if 达标步强 is not None and 全体步 - 达标步强 >= 考试间隔 * 3:
                break

    文本 = "\n".join(日志) + "\n"
    文本 += "\n结论：任务「%s」种子 %d（考试 20 局，判据和论文一样）\n" % (任务名, 种子)
    文本 += "   论文门槛（R1 那条线）：%s\n" % (
        ("%s 步达标" % format(达标步, ",")) if 达标步 else
        ("跑到 %s 步也没达标" % format(最大步, ",")))
    文本 += "   我们那个系统的水平：%s\n" % (
        ("%s 步达标" % format(达标步强, ",")) if 达标步强 else
        ("跑到 %s 步也没达标" % format(最大步, ",")))
    文本 += "   （对照组：我们那个系统是 0 步 —— 本能表是人手写的。）\n"
    (pathlib.Path(__file__).resolve().parent.parent / "logs" / ("日志_RL对照_%s.log" % 任务名)).write_text(文本, encoding="utf-8", newline="\n")
    torch.save({"策略": 净.state_dict(), "归一化": {"均": 归.均, "方": 归.方, "个数": 归.个数},
                "观测维": 观测维, "动作数": 动作数, "任务": 任务名, "种子": 种子,
                "达标步": 达标步, "达标步强": 达标步强, "全体步": 全体步},
               根 / ("RL对照_%s.pt" % 任务名))
    print(文本.splitlines()[-1])
    return 达标步


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)
    任务 = sys.argv[1]
    上限 = int(float(sys.argv[2])) if len(sys.argv) > 2 else 4_000_000
    种子 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    训练(任务, 上限, 种子)