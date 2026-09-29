import pathlib
# -*- coding: utf-8 -*-
"""验证_学抑制_逐拍对照.py —— 同一颗脑子、同一批状态，新写法 vs 旧写法，逐拍比抑制权。

★ 注意（第一版这里写错过）：要在**同一份 _迹** 上比。_迹 在 学习() 的最后一步才更新，
  所以必须先把 _迹 存下来，让旧写法也在"更新前"的那份 _迹 上算，否则比的是两回事。
"""
import os, sys, time
import numpy as np
os.environ.setdefault("前额叶不保留", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); os.chdir(str(pathlib.Path(__file__).resolve().parent))
import 皮层连接_cortex_links as C
import 主循环_完整的一拍 as 主
import 身体_go2 as 身体

抑制上限 = C.抑制上限; 抑制学习率 = C.抑制学习率; 痕迹门限 = C.痕迹门限; 去抑学习率 = C.去抑学习率


def 旧学抑制(网, 上, 这):
    """改动之前那一版（照原文抄下来，注释去掉）。"""
    窗口 = 上 | 这
    共 = 上 & 这
    有迹 = 网._迹 > 痕迹门限
    碰 = 窗口[网._抑制源] | 窗口[网._抑制目]
    相关 = 有迹[网._抑制源] & 有迹[网._抑制目]
    一个亮 = 碰 & ~相关
    两个亮 = 碰 & 相关
    慢 = (网._慢区[网._抑制源] & 网._慢区[网._抑制目]) | 网._终区[网._抑制目]
    一个亮 &= ~网._抑制固化 & ~慢
    两个亮 &= ~网._抑制固化 & ~慢
    if 一个亮.any():
        网._抑制权[一个亮] = np.minimum(
            抑制上限, 网._抑制权[一个亮].astype(np.float64) + 抑制学习率).astype(np.float32)
    if 两个亮.any():
        网._抑制权[两个亮] = np.maximum(
            0.0, 网._抑制权[两个亮].astype(np.float64) - 抑制学习率).astype(np.float32)
    伙 = 网.去抑伙伴
    有 = 伙 >= 0
    双方共亮 = np.zeros(网.总数, dtype=bool)
    双方共亮[有] = 共[伙[有]]
    双方共亮 &= 共
    if 双方共亮.any():
        idx = np.nonzero(双方共亮)[0]
        网.去抑权重[idx] = np.minimum(
            1.0, 网.去抑权重[idx].astype(np.float64) + 去抑学习率).astype(np.float32)


脑 = 主.脑(学=True, 说=False)
网 = 脑.网
身 = 身体.身体(); 身.摆成站姿()
图 = 主.红图("中")
最差 = 0.0; 不同数 = 0; 走过 = 0; 比过的边 = 0
原学习 = 网.学习


def 包(上, 这):
    global 最差, 不同数, 走过, 比过的边
    if 走过 >= 100:
        原学习(上, 这); return
    迹0 = 网._迹.copy()
    K0 = 网._抑制权.copy(); D0 = 网.去抑权重.copy()
    原学习(上, 这)                      # ← 新写法
    新权 = 网._抑制权.copy(); 新去 = 网.去抑权重.copy(); 新迹 = 网._迹.copy()
    网._迹[:] = 迹0                     # ← 关键：退回到"更新之前"的那份痕迹
    网._抑制权[:] = K0; 网.去抑权重[:] = D0
    旧学抑制(网, 上, 这)                 # ← 旧写法
    差 = float(np.abs(网._抑制权.astype(np.float64) - 新权.astype(np.float64)).max())
    差去 = float(np.abs(网.去抑权重.astype(np.float64) - 新去.astype(np.float64)).max())
    最差 = max(最差, 差, 差去)
    不同数 += int((网._抑制权 != 新权).sum())
    比过的边 = 网._抑制权.size
    网._抑制权[:] = 新权; 网.去抑权重[:] = 新去; 网._迹[:] = 新迹
    走过 += 1


网.学习 = 包
t0 = time.perf_counter()
for k in range(120):
    脑.一拍(图=图, 身=身)
    身.步进(脑.肌肉发力())
print("对照 %d 拍：新写法 vs 旧写法（同一份 _迹、同一份权重）" % 走过)
print("  抑制权最大差 %.10f；不一样的条数 %d（每次比 %d 条）" % (最差, 不同数, 比过的边))
print("  逐位相同？%s" % ("是" if (最差 == 0.0 and 不同数 == 0) else "不是"))
print("  （每拍 %.1f 毫秒，这里面两套算法各跑了一遍）" % ((time.perf_counter() - t0) / 120 * 1000))