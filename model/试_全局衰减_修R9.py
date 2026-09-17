import pathlib
# -*- coding: utf-8 -*-
"""试_全局衰减_修R9.py —— 打开「整体降权」，跑 R9 四段，看摔倒能不能修好。

用法：python 试_全局衰减_修R9.py 全局衰减率 种子1 种子2 ...
例：  python 试_全局衰减_修R9.py 0.004 20260914 20260917

只改一个数（网.全局衰减率），其余和论文 R9 完全一样。
R9 是"带环"那档，所以要 pop 掉 前额叶不保留。
"""
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

衰减 = float(sys.argv[1])
种子表 = [int(x) for x in sys.argv[2:]] or [20260914, 20260917]

def 跑一个种子(种子):
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=True, 说=False)
    身 = 身体.身体()
    网 = 脑.网
    网.全局衰减率 = 衰减          # ★ 只改这一个数
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + 运动记忆.时刻们("前进1.2")
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    总重 = lambda: float(网._兴奋权[~网._兴奋固化].astype(np.float64).sum())

    def 段(拍数, 图, 有声, 多巴胺=False):
        脑.多巴胺亮 = 多巴胺
        w0 = 总重()
        起 = 身.数据.qpos[:2].copy(); 路程 = 0.0; 走拍 = 0; 翻在 = None
        for k in range(1, 拍数 + 1):
            脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
            身.步进(脑.肌肉发力())
            if 脑.亮[走块].any():
                走拍 += 1
            p = 身.数据.qpos[:2]
            路程 += float(np.hypot(p[0] - 起[0], p[1] - 起[1]))
            if 翻在 is None and 身.机身歪了() > 75:
                翻在 = k
        w1 = 总重()
        return ("走链 %3d/%d、挪 %6.3f、%s、末高 %.3f、歪 %4.0f、可塑总重 %s"
                % (走拍, 拍数, 路程,
                   ("第 %d 拍翻倒" % 翻在) if 翻在 else "一次没翻",
                   身.机身高度(), 身.机身歪了(),
                   ("%+.1f%%" % (100.0 * (w1 - w0) / max(w0, 1e-9))) if w0 > 0 else "n/a"))

    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    a = 段(30, 黑图, True)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    b = 段(30, 红图, False)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    c = 段(200, 红图, True, 多巴胺=True)
    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    d = 段(200, 黑图, True)
    print("  种子 %d" % 种子, flush=True)
    for 名, v in (("出生：", a), ("天生：", b), ("教学：", c), ("考试：", d)):
        print("    %s %s" % (名, v), flush=True)

print("============ 全局衰减率 = %s ============" % 衰减, flush=True)
for s in 种子表:
    跑一个种子(s)
print("完", flush=True)