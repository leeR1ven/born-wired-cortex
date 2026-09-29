import pathlib
# -*- coding: utf-8 -*-
"""诊断：R9 考试段，把 12 块肌肉的发力按"步态相位"折叠，比早段 vs 晚段。
另外量一量：运动区收到的电流，各来源区各占多少（谁在一直推它）。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)
os.environ.pop("前额叶自压", None)
os.environ.pop("用不上衰减率", None)

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体
import 运动输出区_motor_output as 运动

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
肌肉名 = list(身体.肌肉名)

种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
本能.建网种子 = 种子
脑 = 主循环.脑(学=True, 说=False)
身 = 身体.身体()
网 = 脑.网
运记起 = 网.区起点["运动记忆"]
走块 = 运记起 + 运动记忆.时刻们("前进1.2")
声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
运起, 运宽 = 网.区起点["运动"], 网.区宽["运动"]
终区 = 运起 + 运动.腿区宽度

def 运动电流来源(亮拍):
    """运动区（腿那 120 个）收到的兴奋电流，按来源区分解。"""
    out = {}
    亮 = np.nonzero(亮拍)[0]
    if not 亮.size:
        return out
    边, _ = 网._出边切片(亮)
    if not 边.size:
        return out
    目 = 网._兴奋目[边]
    m = (目 >= 运起) & (目 < 终区)
    if not m.any():
        return out
    源 = 网._兴奋源[边][m]
    权 = 网._兴奋权[边][m].astype(np.float64)
    号 = 网.区号[源]
    s = np.bincount(号, weights=权, minlength=len(网.区名))
    for i, 名 in enumerate(网.区名):
        if s[i] > 1e-6:
            out[名] = float(s[i])
    return out

def 段(拍数, 图, 有声, 多巴胺=False):
    脑.多巴胺亮 = 多巴胺
    for _ in range(拍数):
        脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
        身.步进(脑.肌肉发力())

身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
段(30, 黑图, True)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段(30, 红图, False)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段(200, 红图, True, 多巴胺=True)
身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False

记录 = []
for k in range(1, 201):
    源分布 = 运动电流来源(脑.上一拍)
    脑.一拍(图=黑图, 声=声, 身=身)
    力 = 脑.肌肉发力()
    身.步进(力)
    首位 = int(np.nonzero(脑.亮[走块])[0][0]) if 脑.亮[走块].any() else -1
    记录.append((k, 身.机身高度(), 身.机身歪了(), 首位, np.asarray(力, dtype=float), 源分布,
                 int(脑.亮[运起:运起 + 运宽].sum())))

print("### 种子 %d" % 种子, flush=True)
print("肌肉顺序：%s" % "、".join(肌肉名), flush=True)
def 折叠(起, 止):
    a = {}
    for k, 高, 歪, 首, 力, 源, 运 in 记录:
        if 起 <= k <= 止 and 首 >= 0:
            a.setdefault(首, []).append(力)
    return {p: np.mean(v, axis=0) for p, v in a.items()}

早 = 折叠(20, 50); 晚 = 折叠(90, 125)
print("\n按步态相位折叠的平均发力（早段 20~50 拍 vs 晚段 90~125 拍）：", flush=True)
print("相位 " + " ".join("%7s" % n for n in 肌肉名), flush=True)
for p in range(15):
    if p in 早 and p in 晚:
        差 = 晚[p] - 早[p]
        print(" %2d  " % p + " ".join("%7.2f" % v for v in 早[p]), flush=True)
        print("     " + " ".join("%7.2f" % v for v in 差) + "   <- 晚段减早段", flush=True)

print("\n每 10 拍照看：运动区亮、来源区推它的电流", flush=True)
for k, 高, 歪, 首, 力, 源, 运 in 记录:
    if k % 10 == 0:
        s = "、".join("%s %.1f" % (a, b) for a, b in sorted(源.items(), key=lambda kv: -kv[1])[:5])
        print(" %3d 拍  高 %.3f 歪 %3.0f 相位 %2d 运动亮 %3d  电流来源：%s"
              % (k, 高, 歪, 首, 运, s), flush=True)