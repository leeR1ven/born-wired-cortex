# -*- coding: utf-8 -*-
"""实验_闭环前提_多种子.py —— 把 闭环_前提.py 那一套（站起来 / 朝红走 / 瘫掉）在多个种子上重复。

★ 为什么要有这个：论文里不能只有一个种子跑一遍。换一批随机的背景连接（建网种子），
  看这三件事是不是每次都对。

命令：python 实验_闭环前提_多种子.py [种子,种子,...]
"""
from __future__ import annotations

import sys

import numpy as np

import 本能工具_instincts as 本能
import 闭环_前提 as 前提
import 身体_go2 as 身体

默认种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]


def 跑一趟(种子):
    本能.建网种子 = 种子
    网, 条 = 本能.装(说=False)
    身 = 身体.身体()
    # [1] 趴着，什么都不给
    身.摆成趴姿()
    身.走(0.5, None)
    起高 = 身.机身高度()
    r1 = 前提.一张段(网, 身, 3.0, 说=False)
    # [2] 站着，眼前一片红
    身.摆成站姿()
    身.走(0.5, None)
    红激活 = np.asarray(前提.视觉.网.前向传播(前提.视觉.图像转信号(前提.红图("中"))), dtype=bool)
    起 = 身.数据.qpos[:2].copy()
    r2 = 前提.一张段(网, 身, 4.0, 视觉激活=红激活, 说=False)
    前 = float(身.数据.qpos[0] - 起[0])
    # [3] 什么都不给（连感觉都不给）
    身.摆成站姿()
    r3 = 前提.一张段(网, 身, 1.5, 用感觉=False, 说=False)
    return {"种子": 种子, "本能条数": 条, "起手高": 起高,
            "站起来末高": r1["末高"], "站起来最高": r1["最高"],
            "朝红走了": 前, "瘫掉末高": r3["末高"], "瘫掉歪了": r3["歪"]}


def main():
    种子们 = 默认种子们
    if len(sys.argv) > 1 and sys.argv[1].strip():
        种子们 = [int(a) for a in sys.argv[1].split(",")]
    print("=" * 96)
    print("闭环前提：摆成趴姿自己站起来 / 站着朝红走 / 什么都不给就瘫掉 —— 换 %d 个种子各跑一遍" % len(种子们))
    print("=" * 96)
    出 = []
    for s in 种子们:
        r = 跑一趟(s)
        出.append(r)
        print("种子 %d：起手 %.3f 米 → 站起来末高 %.3f（最高 %.3f）；朝红走了 %+.2f 米；"
              "什么都不给 → 末高 %.3f、歪 %.0f 度"
              % (r["种子"], r["起手高"], r["站起来末高"], r["站起来最高"],
                 r["朝红走了"], r["瘫掉末高"], r["瘫掉歪了"]), flush=True)
    print("\n" + "=" * 96)
    print("【汇总】")
    a = np.array([r["站起来末高"] for r in 出])
    b = np.array([r["朝红走了"] for r in 出])
    c = np.array([r["瘫掉末高"] for r in 出])
    print("  自己站起来（末高 > 0.20 算成功）： %d/%d，末高 中位 %.3f、最低 %.3f"
          % (int((a > 0.20).sum()), len(a), float(np.median(a)), float(a.min())))
    print("  朝红走（> 0.30 米算成功）：        %d/%d，走了 中位 %+.2f 米、最少 %+.2f"
          % (int((b > 0.30).sum()), len(b), float(np.median(b)), float(b.min())))
    print("  什么都不给（末高 < 0.20 算瘫掉）：  %d/%d，末高 中位 %.3f、最高 %.3f"
          % (int((c < 0.20).sum()), len(c), float(np.median(c)), float(c.max())))
    return 出


if __name__ == "__main__":
    出 = main()