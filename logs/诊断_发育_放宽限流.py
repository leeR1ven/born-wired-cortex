# -*- coding: utf-8 -*-
"""诊断_发育_放宽限流.py —— 学习长不出新路，到底是「机制不行」还是「限流太紧」？

★ 背景：诊断_发育_新连接长在哪.py 量到，生活 400 拍里「听觉 -> 走路时刻」的
  新连接是 **0 条**（临时表整个是空的）。原因看样子是这几条限流：
    · 一次最多配对 = 4000   —— 每拍 750（上一拍亮）× 400（这一拍刚亮）≈ 30 万对，
                                只随机抽 63×63 来配，一对被抽中的机会约 1.3%
    · 需要共激活次数 = 4    —— 同一对要见过 4 次才真长出一根
    · 候选上限 = 50000      —— 候选表一满就丢掉一半，没见够 4 次的对被清掉
    · 每步最多新建 = 200
  三者一凑，一对要"被抽中 4 次"的概率极低，而候选表还在不停地丢。

★ 这个脚本只做一件事：**临时**把这三个数放宽（不改文件，只在这个进程里生效），
  看新连接到底长不长得出来、长出来之后电流多久爬到 1.0。
  这样就分得清是"规则根本不行"，还是"限流把规则卡死了"。

命令： python 诊断_发育_放宽限流.py [种子] [生活秒数] [每多少拍量一次] [配对上限] [需要次数] [每步最多新建]
"""
from __future__ import annotations

import sys

import numpy as np

import 皮层连接_cortex_links as 皮层连接
import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体
import 运动记忆区_motor_memory as 运动记忆

图高, 图宽 = 主循环.图高, 主循环.图宽
帧秒 = 主循环.帧秒


def 红图():
    图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
    图[:, 图宽 // 3: 2 * 图宽 // 3, 0] = 255
    return 图


def 量(脑):
    网 = 脑.网
    听起, 听宽 = 脑.听起, 脑.听宽
    运记号 = 网.区名.index("运动记忆")
    听运 = 0
    听运权 = 0.0
    听走 = 0
    听走权 = 0.0
    走路时刻 = set(int(x) for x in (网.区起点["运动记忆"]
                                 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)))
    for s, d in 网._待出.items():
        s = int(s)
        if not (听起 <= s < 听起 + 听宽):
            continue
        for t, w in d.items():
            t = int(t)
            if 网.区号[t] != 运记号:
                continue
            听运 += 1
            听运权 += float(w)
            if t in 走路时刻:
                听走 += 1
                听走权 += float(w)
    return 听运, 听运权, 听走, 听走权


def 电流进走路(脑, 声):
    """只放这个声音（不给身子、不给图）时，走路那 15 个时刻各自收到多少电流"""
    import 听觉前处理_auditory_preprocess as 听觉
    网 = 脑.网
    亮 = np.zeros(网.总数, dtype=bool)
    亮[脑.听起: 脑.听起 + 脑.听宽] = np.asarray(
        听觉.网.前向传播(听觉.频谱转信号(声)), dtype=bool)
    电流 = 网._兴奋电流(亮) - 网._抑制电流(亮)
    时 = 网.区起点["运动记忆"] + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
    return 电流[时]


def main():
    种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260914
    生活秒 = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    每多少拍 = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    配对上 = int(sys.argv[4]) if len(sys.argv) > 4 else 400000
    需要次 = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    每步新 = int(sys.argv[6]) if len(sys.argv) > 6 else 20000
    皮层连接.一次最多配对 = 配对上
    皮层连接.需要共激活次数 = 需要次
    皮层连接.每步最多新建 = 每步新
    皮层连接.候选上限 = 2000000
    print("=" * 110)
    print("★ 本进程临时放宽限流：一次最多配对=%d，需要共激活次数=%d，每步最多新建=%d，候选上限=%d"
          % (配对上, 需要次, 每步新, 皮层连接.候选上限))
    print("  种子 %d，生活 %.0f 秒（%d 拍）" % (种子, 生活秒, int(生活秒 / 帧秒)))
    print("=" * 110)
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=True, 说=False)
    身 = 身体.身体()
    身.摆成站姿()
    身.走(0.5, None)
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    红 = 红图()
    拍数 = int(round(生活秒 / 帧秒))
    print("  拍   听->运边  听->运权和  听->走路时刻边  权和   走路15个时刻收到的电流(最大/第1个)")
    print("-" * 110)
    for k in range(拍数 + 1):
        if k % 每多少拍 == 0:
            a, b, c, d = 量(脑)
            电 = 电流进走路(脑, 声)
            print("  %5d %9d %11.3f %13d %8.3f %14.3f / %.3f"
                  % (k, a, b, c, d, float(电.max()), float(电[0])), flush=True)
        if k >= 拍数:
            break
        脑.一拍(图=红, 声=声, 身=身)
        身.步进(脑.肌肉发力())
    print()
    print("【怎么读】走路那 15 个时刻收到的电流过 1.0，这个声音就能自己把走路点起来。")
    print("          本能给的是 0.80（视觉:中有红那一半）。")


if __name__ == "__main__":
    main()