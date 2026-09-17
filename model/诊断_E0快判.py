import pathlib
# -*- coding: utf-8 -*-
"""快速判定：站着 + 红 + 声音，不学（论文 E0 那一档），到底是哪一路把它拱翻的。
同一颗脑（同种子），一组开关各跑 200 拍，每 10 拍报一次。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
动作们 = ["前进1.2", "原地踏步", "站姿", "站起来"]

配置们 = [
    ("① 原样（带环）",        dict(),                       红图),
    ("② 环关掉（每拍重算）",  dict(前额叶不保留=True),      红图),
    ("③ 前额叶按灭",          dict(前额叶按灭=True),         红图),
    ("④ 前额叶自压 0.002",    dict(自压="0.002"),            红图),
    ("⑤ 黑屏（不红）",        dict(),                       黑图),
    ("⑥ 环关+黑屏",           dict(前额叶不保留=True),      黑图),
]

def 跑一遍(名, 开关, 图):
    for k in ("前额叶不保留", "前额叶按灭", "前额叶自压"):
        os.environ.pop(k, None)
    if 开关.get("自压"):
        os.environ["前额叶自压"] = 开关["自压"]
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False,
                  前额叶不保留=开关.get("前额叶不保留"),
                  前额叶按灭=开关.get("前额叶按灭", False))
    身 = 身体.身体()
    网 = 脑.网
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    运记起 = 网.区起点["运动记忆"]
    时刻 = {a: 运记起 + np.asarray(运动记忆.时刻们(a), dtype=np.int64) for a in 动作们}
    区名们 = ["视觉", "听觉", "前额叶", "运动", "运动记忆", "本体感觉", "平衡感觉", "机身状态", "视觉运动"]
    段 = {n: 网.区段(n) for n in 区名们}

    身.摆成站姿(); 身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    print("### %s（种子 %d）" % (名, 种子), flush=True)
    print("  拍  高     歪  路程  力   " + " ".join("%s" % n[:4] for n in 区名们) + "  在跑什么", flush=True)
    上一 = 身.数据.qpos[:2].copy(); 路程 = 0.0; 翻在 = None
    for k in range(1, 201):
        脑.一拍(图=图, 声=声, 身=身)
        力 = np.asarray(脑.肌肉发力(), dtype=float)
        身.步进(力)
        p = 身.数据.qpos[:2]
        路程 += float(np.hypot(p[0] - 上一[0], p[1] - 上一[1])); 上一 = p.copy()
        歪 = 身.机身歪了()
        if 翻在 is None and 歪 > 75:
            翻在 = k
        if k % 10 == 0:
            在跑 = " ".join("%s%d" % (a, int(脑.亮[时刻[a]].sum())) for a in 动作们
                            if 脑.亮[时刻[a]].any())
            print("  %3d %5.3f %4.0f %5.2f %5.2f  " % (k, 身.机身高度(), 歪, 路程, 力.sum())
                  + " ".join("%5d" % int(脑.亮[段[n]].sum()) for n in 区名们) + "  " + 在跑, flush=True)
    print("  -> 挪 %.2f 米，%s，末高 %.3f，末歪 %.0f\n"
          % (路程, ("第 %d 拍翻倒" % 翻在) if 翻在 else "一次没翻", 身.机身高度(), 身.机身歪了()), flush=True)
    return 翻在, 路程

结果 = []
for 名, 开关, 图 in 配置们:
    翻在, 路程 = 跑一遍(名, 开关, 图)
    结果.append((名, 翻在, 路程))
print("======== 汇总（种子 %d）========" % 种子, flush=True)
for 名, 翻在, 路程 in 结果:
    print("  %-24s %s  挪 %.2f 米" % (名, ("第 %d 拍翻" % 翻在) if 翻在 else "没翻", 路程), flush=True)