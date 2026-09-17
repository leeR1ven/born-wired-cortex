import pathlib
# -*- coding: utf-8 -*-
"""试两种论文里已经有声明的 C 幕做法：
   ① 趴着站起来，站定之后一直不给红 → 它走不走？（对照 A/B，起点公平）
   ② 站着看到红 0.5 秒后把红撤掉 → 它还走不走？（R8：想法还在吗）
"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ["前额叶不保留"] = "1"

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

本能.建网种子 = 20260914
脑 = 主循环.脑(学=True, 说=False)
网 = 脑.网
身 = 身体.身体()
运记起 = 网.区起点["运动记忆"]
走块 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
黑图 = np.zeros((主循环.图高, 主循环.图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
帧秒 = 主循环.帧秒

def 跑(名, 起点, 安排, 秒):
    print("=== %s ===" % 名, flush=True)
    身.摆成趴姿() if 起点 == "趴" else 身.摆成站姿()
    身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    起 = 身.数据.qpos[:2].copy()
    for k in range(int(秒 / 帧秒)):
        图 = 安排(k * 帧秒)
        出 = 脑.一拍(图=图, 声=None, 身=身)
        身.步进(脑.肌肉发力())
        if k % 20 == 0:
            print("   %5.2f 秒  末高 %.3f  歪 %3.0f  挪 %5.2f 米  走路点火 %d"
                  % (k * 帧秒, 身.机身高度(), 身.机身歪了(),
                     float(np.hypot(身.数据.qpos[0]-起[0], 身.数据.qpos[1]-起[1])),
                     int(np.count_nonzero(出[走块]))), flush=True)
    print("   --- 末了：末高 %.3f 米、歪 %.0f 度、挪了 %.2f 米"
          % (身.机身高度(), 身.机身歪了(),
             float(np.hypot(身.数据.qpos[0]-起[0], 身.数据.qpos[1]-起[1]))), flush=True)

跑("① 趴着站起来，之后一直不给红", "趴", lambda t: 黑图, 6.0)
跑("② 站着看红 0.5 秒，然后撤掉红", "站", lambda t: (红图 if t < 0.5 else 黑图), 5.0)