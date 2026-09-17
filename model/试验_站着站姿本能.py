import pathlib
# -*- coding: utf-8 -*-
"""试验：给"站着"补一条本能，看能不能站住、又不耽误走路。全程在内存里改，不动任何文件。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ["前额叶不保留"] = "1"

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

原读表 = 本能.读本能表
黑图 = np.zeros((主循环.图高, 主循环.图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
帧秒 = 主循环.帧秒

def 跑一遍(加规则们, 名字):
    本能.读本能表 = (lambda: 原读表() + list(加规则们)) if 加规则们 else 原读表
    本能.建网种子 = 20260914
    脑 = 主循环.脑(学=True, 说=False)
    网 = 脑.网
    身 = 身体.身体()
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
    站块 = 运记起 + np.asarray(运动记忆.时刻们("站起来"), dtype=np.int64)
    结果 = {}
    for 幕, 图, 秒 in (("站着黑屏", 黑图, 3.0), ("站着看红", 红图, 4.0)):
        身.摆成站姿(); 身.走(0.5, None)
        脑.亮[:] = False; 脑.上一拍[:] = False
        起 = 身.数据.qpos[:2].copy()
        走点火 = 0; 站点火 = 0; 力0 = 0
        for k in range(int(秒 / 帧秒)):
            出 = 脑.一拍(图=图, 声=None, 身=身)
            力 = 脑.肌肉发力()
            if k < 6 and float(np.max(np.abs(力))) > 0.01:
                力0 += 1
            if np.count_nonzero(出[走块]): 走点火 += 1
            if np.count_nonzero(出[站块]): 站点火 += 1
            身.步进(力)
        结果[幕] = dict(末高=身.机身高度(), 歪=身.机身歪了(),
                        挪=float(np.hypot(身.数据.qpos[0]-起[0], 身.数据.qpos[1]-起[1])),
                        走点火=走点火, 站点火=站点火)
    本能.读本能表 = 原读表
    print("[%s] 本能 %d 条" % (名字, 脑.本能条数))
    for 幕 in 结果:
        r = 结果[幕]
        print("    %s：末高 %.3f 米、歪 %3.0f 度、挪了 %.2f 米、走路点火 %d 拍、站起来链点火 %d 拍"
              % (幕, r["末高"], r["歪"], r["挪"], r["走点火"], r["站点火"]))
    return 结果

跑一遍([], "V0 基准（不加）")
跑一遍([("机身状态:站得高", "运动:站姿", 2.0, True, "试验")], "V1 站得高→运动:站姿 2.0")
跑一遍([("机身状态:站得高", "运动记忆:站姿", 2.0, True, "试验")], "V2 站得高→运动记忆:站姿 2.0")
跑一遍([("机身状态:站得高", "运动:站姿", 2.0, True, "试验"),
        ("机身状态:站得高", "运动记忆:站姿", 2.0, True, "试验")], "V3 两条都加")