import pathlib
# -*- coding: utf-8 -*-
"""诊断：种子里 20260918 的考试段，哪几根连接长得最快（每 5 拍看一次）。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260918

本能.建网种子 = 种子
脑 = 主循环.脑(学=True, 说=False)
身 = 身体.身体()
网 = 脑.网
运记起 = 网.区起点["运动记忆"]
走块 = 运记起 + 运动记忆.时刻们("前进1.2")
声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)

区名 = {k: (起, 起 + 网.区宽[k]) for k, 起 in 网.区起点.items()}
def 谁(i):
    for k, (a, b) in 区名.items():
        if a <= i < b:
            return k
    return "?"

def 段(拍数, 图, 有声, 多巴胺=False, 看=False):
    脑.多巴胺亮 = 多巴胺
    起 = 身.数据.qpos[:2].copy()
    w0 = None
    for k in range(1, 拍数 + 1):
        脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
        身.步进(脑.肌肉发力())
        if k == 1:
            w0 = 网._兴奋权.copy()
        if 看 and (k % 5 == 0 or k == 拍数):
            非固 = ~网._兴奋固化
            差 = (网._兴奋权.astype(np.float64) - w0.astype(np.float64))
            差[~非固] = 0.0
            idx = np.argsort(差)[::-1][:6]
            print("  拍%3d 末高%.3f 歪%3.0f 走链%d | 总增 %.1f 最大增 %.3f"
                  % (k, 身.机身高度(), 身.机身歪了(), int(脑.亮[走块].sum()),
                     float(差.sum()), float(差[idx[0]])), flush=True)
            for i in idx:
                if 差[i] <= 0.004:
                    break
                print("       +%.3f  %s → %s（现在 %.3f）"
                      % (差[i], 谁(int(网._兴奋源[i])), 谁(int(网._兴奋目[i])),
                         float(网._兴奋权[i])), flush=True)

身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
段(30, 黑图, True)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段(30, 红图, False)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段(200, 红图, True, 多巴胺=True)
身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
print("=== 考试（黑屏 + 只放声音）===", flush=True)
段(120, 黑图, True, 看=True)