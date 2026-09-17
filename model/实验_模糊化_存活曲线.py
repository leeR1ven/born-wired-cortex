# -*- coding: utf-8 -*-
"""实验_模糊化_存活曲线.py —— 先天连线"模糊"到什么程度，行为才散架？

★ 为什么要重做（2026-09-15）。
  原来那版（实验_本能模糊化.py）每一档 p 都重新抽一次随机数，各档**不是嵌套的**
  （20% 换掉的那些，到了 35% 不一定也换掉），所以曲线会上下跳 —— 20% 断了、
  35% 又活了，这不是"耐受度"，这是抽样噪声。

★ 这一版改成正规的剂量-反应曲线：
    · 同一个"抽"里，先抽一个顺序，p 越大换掉的就是**前面那些条目的前缀** ——
      天然嵌套，p 一大，坏的一定包含 p 小时候坏的。
    · 每种换法抽 3 次（3 个独立顺序），报"3 抽里有几抽还活着"。
    · 两种换法：随机乱接（方向也没了）/ 挪一格（方向还在，只是糊了）。
    · 三条通路：红→走路 / 响→踏步 / 小球横飘→眼睛跟住。

命令：python 实验_模糊化_存活曲线.py
      python 实验_模糊化_存活曲线.py 抽数 方式     例：python 实验_模糊化_存活曲线.py 3 随机乱接
"""
from __future__ import annotations

import sys

import numpy as np

import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆
import 主循环_完整的一拍 as 主循环
import 实验_眼睛跟随 as 跟随
import 实验_本能模糊化 as 旧

真读本能表 = 本能.读本能表
百分比们 = [10, 25, 50, 75, 100]
走路够 = 6      # 8 拍里点到 6 拍算"这条路还活着"
踏步够 = 4
眼睛够 = 12.0   # 平均偏离 <= 12 度算"眼睛还跟得住"（接对是 5.6 度，完全不接 15.3 度）


def 造新表(表, 名字, p, rng, 方式):
    可换 = [i for i, 条 in enumerate(表) if ":" in 条[0] and 条[0].split(":", 1)[0].strip() in 名字]
    顺序 = rng.permutation(len(可换))
    要换 = set(可换[int(j)] for j in 顺序[:int(round(p / 100.0 * len(可换)))])
    出 = []
    换了几条 = 0
    for i, (源块, 目标块, 权重, 固化, 行号) in enumerate(表):
        if i not in 要换:
            出.append((源块, 目标块, 权重, 固化, 行号))
            continue
        区, 名 = 源块.split(":", 1)
        区, 名 = 区.strip(), 名.strip()
        新 = 旧.挪名字(区, 名, rng) if 方式 == "挪一格" else None
        if 新 is None:
            候选 = [n for n in 名字.get(区, {}) if n != 名 and len(名字[区][n])]
            if not 候选:
                出.append((源块, 目标块, 权重, 固化, 行号))
                continue
            新 = 候选[int(rng.integers(len(候选)))]
        出.append(("%s:%s" % (区, 新), 目标块, 权重, 固化, 行号))
        换了几条 += 1
    return 出, 换了几条


def 两半还在吗(表):
    n = 0
    for 源块, 目标块, 权重, 固化, 行号 in 表:
        if 目标块.strip() == "运动记忆:前进1.2" and 源块.strip() in ("视觉:中有红", "前额叶:看着红"):
            n += 1
    return n


def 探(脑, 拍数=8):
    运记起 = 脑.网.区起点["运动记忆"]
    走 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
    踏 = 运记起 + np.asarray(运动记忆.时刻们("原地踏步"), dtype=np.int64)
    红 = 主循环.红图("中")
    谱 = 本能.听觉例子["响"]()
    出 = {}
    for 名, 图, 声, 时刻 in (("走红", 红, None, 走), ("听响", None, 谱, 踏)):
        脑.亮[:] = False
        脑.上一拍[:] = False
        中 = 0
        for _ in range(拍数):
            脑.一拍(图=图, 声=声)
            if 脑.上一拍[时刻].any():
                中 += 1
        出[名] = 中
    return 出


def 跑一个(表, 名字, p, rng, 方式):
    新表, 换了几条 = 造新表(表, 名字, p, rng, 方式)
    本能.读本能表 = (lambda 表=新表: 表)
    try:
        脑 = 主循环.脑(学=False, 说=False)
        出 = 探(脑)
        记 = 跟随.跑("p=%d%%" % p, 球方位角=-25.0, 球半径=0.05, 球横速=0.5,
                    秒数=4.0, 说=False)
    finally:
        本能.读本能表 = 真读本能表
    偏 = [max(abs(r[1]), abs(r[2])) for r in 记 if abs(r[1]) <= 45 and abs(r[2]) <= 35]
    平均 = sum(偏) / len(偏) if 偏 else float("nan")
    return {"p": p, "换了几条": 换了几条, "条数": len(表),
            "走红": 出["走红"], "听响": 出["听响"], "平均偏": 平均,
            "两半": 两半还在吗(新表),
            "走红活": 出["走红"] >= 走路够, "听响活": 出["听响"] >= 踏步够,
            "眼睛活": 平均 <= 眼睛够}


def main():
    参 = sys.argv[1:]
    抽数 = int(参[0]) if 参 else 3
    方式们 = [参[1]] if len(参) > 1 else ["随机乱接", "挪一格"]
    print("=" * 100)
    print("先天连线模糊化：剂量-反应曲线（每一档嵌套，抽 %d 个独立顺序）" % 抽数)
    print("判活：红→走路 >= %d/8 拍；响→踏步 >= %d/8 拍；眼睛平均偏离 <= %.1f 度"
          % (走路够, 踏步够, 眼睛够))
    print("=" * 100, flush=True)
    名字 = 本能.读名字()
    表 = 真读本能表()
    全部 = []
    for 方式 in 方式们:
        for 抽 in range(1, 抽数 + 1):
            rng = np.random.default_rng(900000 + 抽 * 97)
            参 = []
            # p=0 只跑一次（三抽结果一样）
            if 抽 == 1:
                r = 跑一个(表, 名字, 0, rng, 方式)
                r["方式"], r["抽"] = 方式, 0
                全部.append(r)
                print("[%s 基准] 换掉 0 条：走红 %d/8、听响 %d/8、眼睛 %.1f 度"
                      % (方式, r["走红"], r["听响"], r["平均偏"]), flush=True)
            for p in 百分比们:
                r = 跑一个(表, 名字, p, rng, 方式)
                r["方式"], r["抽"] = 方式, 抽
                全部.append(r)
                print("[%s 第%d抽 %3d%%] 换掉 %3d/%d 条，走路两半剩 %d/2：走红 %d/8、听响 %d/8、眼睛 %.1f 度"
                      % (方式, 抽, p, r["换了几条"], r["条数"], r["两半"],
                         r["走红"], r["听响"], r["平均偏"]), flush=True)
    print("\n" + "=" * 100)
    print("【存活率总表】每一格 = 几抽里还活着 / 共几抽")
    print("%-10s | %-7s | %-14s | %-14s | %-14s" % ("换法", "换掉", "红→走路", "响→踏步", "小球→眼睛"))
    print("-" * 100)
    for 方式 in 方式们:
        for p in [0] + 百分比们:
            行 = [r for r in 全部 if r["方式"] == 方式 and r["p"] == p]
            if not 行:
                continue
            n = len(行)
            print("%-10s | %5d%% | %-14s | %-14s | %-14s"
                  % (方式, p,
                     "%d/%d" % (sum(1 for r in 行 if r["走红活"]), n),
                     "%d/%d" % (sum(1 for r in 行 if r["听响活"]), n),
                     "%d/%d" % (sum(1 for r in 行 if r["眼睛活"]), n)))
    print("\n【原始数字】")
    for 方式 in 方式们:
        for p in [0] + 百分比们:
            行 = [r for r in 全部 if r["方式"] == 方式 and r["p"] == p]
            if not 行:
                continue
            print("  %-8s %5d%%  走红 %s 拍 | 听响 %s 拍 | 眼睛偏离 %s 度"
                  % (方式, p,
                     "、".join(str(r["走红"]) for r in 行),
                     "、".join(str(r["听响"]) for r in 行),
                     "、".join("%.1f" % r["平均偏"] for r in 行)))
    return 全部


if __name__ == "__main__":
    全部 = main()