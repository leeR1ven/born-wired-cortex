# -*- coding: utf-8 -*-
"""诊断_发育_新连接长在哪.py —— 一边生活，一边看赫布学习把新连接长在哪儿

★ 为什么要这个：
  诊断_发育_电流在长吗.py 量到「声音 -> 走路第一拍」的电流一直是 0。要么是
  学习太慢，要么是它根本没往那条路上长。得看清楚：每一拍到底有多少细胞亮、
  「上一拍亮 × 这一拍新亮」有多少对、候选表多大，新长出来的连接又落在哪两个区之间。

命令： python 诊断_发育_新连接长在哪.py [种子] [生活秒数] [每多少拍量一次]
"""
from __future__ import annotations

import sys

import numpy as np

import 本能工具_instincts as 本能
import 听觉前处理_auditory_preprocess as 听觉
import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体
import 运动记忆区_motor_memory as 运动记忆

图高, 图宽 = 主循环.图高, 主循环.图宽
帧秒 = 主循环.帧秒


def 红图():
    图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
    图[:, 图宽 // 3: 2 * 图宽 // 3, 0] = 255
    return 图


def 量(脑, 声, 走路时刻):
    网 = 脑.网
    听出 = np.asarray(听觉.网.前向传播(听觉.频谱转信号(声)), dtype=bool)
    听号 = 网.区名.index("听觉")
    运记号 = 网.区名.index("运动记忆")
    待 = 网._待出
    # 临时表里有几条边、落在哪两个区
    区对 = {}
    听运 = 0
    听运权 = 0.0
    走时刻集合 = set(int(x) for x in 走路时刻)
    听走 = 0
    听走权 = 0.0
    听起, 听宽 = 脑.听起, 脑.听宽
    for s, d in 待.items():
        s = int(s)
        s区 = 网.区号[s]
        s是听 = 听起 <= s < 听起 + 听宽
        for t, w in d.items():
            t = int(t)
            t区 = 网.区号[t]
            区对[(s区, t区)] = 区对.get((s区, t区), 0) + 1
            if s是听 and t区 == 运记号:
                听运 += 1
                听运权 += float(w)
                if t in 走时刻集合:
                    听走 += 1
                    听走权 += float(w)
    总数 = sum(区对.values())
    排 = sorted(区对.items(), key=lambda kv: -kv[1])[:4]
    名 = {i: n for i, n in enumerate(网.区名)}
    return {"听出": int(听出.sum()), "候选": len(网._候选), "临时边": 总数,
            "听运": 听运, "听运权": 听运权, "听走": 听走, "听走权": 听走权,
            "前四": [("%s->%s" % (名[a], 名[b]), c) for (a, b), c in 排],
            "学出总数": int(网.统计.get("学出的兴奋直连", 0))}


def main():
    种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260914
    生活秒 = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    每多少拍 = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=True, 说=False)
    身 = 身体.身体()
    身.摆成站姿()
    身.走(0.5, None)
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    红 = 红图()
    走路时刻 = 脑.网.区起点["运动记忆"] + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
    拍数 = int(round(生活秒 / 帧秒))
    print("=" * 118)
    print("诊断：生活时赫布学习把新连接长在哪（种子 %d，生活 %.0f 秒 = %d 拍）" % (种子, 生活秒, 拍数))
    print("  走路那串时刻共 %d 个：%s" % (len(走路时刻), 走路时刻[:4].tolist()))
    print("=" * 118)
    print("  拍   亮数  新亮  候选表   临时边  听->运  听->运权和  听->走路时刻 权和  学出总数  新边集中在前四")
    print("-" * 118)
    for k in range(拍数 + 1):
        if k % 每多少拍 == 0:
            r = 量(脑, 声, 走路时刻)
            亮 = np.nonzero(脑.亮)[0]
            上 = np.nonzero(脑.上一拍)[0]
            新 = np.setdiff1d(亮, 上, assume_unique=False).size
            print("  %5d %6d %5d %7d %8d %7d %10.3f %14d %9.3f %9d  %s"
                  % (k, 亮.size, 新, r["候选"], r["临时边"], r["听运"], r["听运权"],
                     r["听走"], r["听走权"], r["学出总数"],
                     "  ".join("%s(%d)" % t for t in r["前四"])), flush=True)
        if k >= 拍数:
            break
        脑.一拍(图=红, 声=声, 身=身)
        身.步进(脑.肌肉发力())
    print()
    print("【怎么读】")
    print("  亮数/新亮  = 这一拍多少细胞亮、其中多少是**刚亮**的（配对的原料）")
    print("  候选表     = 「同一对反复见过几次」那张表；要见够 4 次才算真长出一根")
    print("  听->运     = 临时表里 听觉->运动记忆 的新边；听->走路时刻 = 其中真正落在走路那串上的")
    print("  权和       = 这些新边的权重合计（就是它们能送出的电流，阈值 1.0）")
    return None


if __name__ == "__main__":
    main()