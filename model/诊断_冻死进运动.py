import pathlib
# -*- coding: utf-8 -*-
"""诊断：R9 考试段摔，是不是"进运动区的连接边跑边变粗"造成的。
对照：① 原样  ② 把"进运动区的边"冻死（学习倍率 0）。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)
os.environ.pop("前额叶自压", None)
os.environ.pop("用不上衰减率", None)

import 皮层连接_cortex_links as 皮层
import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

原学多快 = 皮层.皮层连接网.这对学多快
冻结进运动 = [False]

def 替代(self, 源, 目):
    r = np.where((self._慢区[源] & self._慢区[目]) | self._终区[目],
                 皮层.慢区可塑倍率, 1.0)
    if 冻结进运动[0]:
        r = np.where(self._终区[目], 0.0, r)
    return r
皮层.皮层连接网.这对学多快 = 替代

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915

def 跑一遍(冻结):
    冻结进运动[0] = 冻结
    os.environ.pop("前额叶不保留", None)
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=True, 说=False)
    身 = 身体.身体()
    网 = 脑.网
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + 运动记忆.时刻们("前进1.2")
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    运起, 运宽 = 网.区起点["运动"], 网.区宽["运动"]

    def 进运动区总重():
        m = (网._兴奋目 >= 运起) & (网._兴奋目 < 运起 + 运宽)
        return float(网._兴奋权[m].astype(np.float64).sum()), int(m.sum())

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

    脑.多巴胺亮 = False
    起重, 起边 = 进运动区总重()
    print("  [%s] 考试起步：进运动区 %d 根，合计重 %.0f" % ("冻" if 冻结 else "原样", 起边, 起重), flush=True)
    上一 = 身.数据.qpos[:2].copy(); 路程 = 0.0; 翻在 = None; 走拍 = 0
    for k in range(1, 201):
        脑.一拍(图=黑图, 声=声, 身=身)
        身.步进(脑.肌肉发力())
        p = 身.数据.qpos[:2]
        路程 += float(np.hypot(p[0] - 上一[0], p[1] - 上一[1])); 上一 = p.copy()
        歪 = 身.机身歪了()
        if 翻在 is None and 歪 > 75:
            翻在 = k
        if 脑.亮[走块].any():
            走拍 += 1
        if k % 5 == 0:
            重, 边 = 进运动区总重()
            力 = np.asarray(脑.肌肉发力(), dtype=float)
            print("    %3d 拍 高 %.3f 歪 %3.0f 路程 %5.2f 走拍 %3d 运记亮 %d 运动亮 %d "
                  "力%5.2f 进运动重 %6.0f(+%4.0f) 边 %d"
                  % (k, 身.机身高度(), 歪, 路程, 走拍,
                     int(脑.亮[运记起:运记起 + 网.区宽["运动记忆"]].sum()),
                     int(脑.亮[运起:运起 + 运宽].sum()),
                     力.sum(), 重, 重 - 起重, 边), flush=True)
    末重, 末边 = 进运动区总重()
    return dict(翻在=翻在, 路程=路程, 高=身.机身高度(), 歪=身.机身歪了(),
                走拍=走拍, 重0=起重, 重1=末重, 边0=起边, 边1=末边)

for 冻结 in (False, True):
    r = 跑一遍(冻结)
    print("### 种子 %d %s：挪 %.2f 米、%s、末高 %.3f、末歪 %.0f、走链 %d/200；"
          "进运动区重 %.0f→%.0f、边 %d→%d"
          % (种子, "冻死进运动区的边" if 冻结 else "原样",
             r["路程"], ("第 %d 拍翻倒" % r["翻在"]) if r["翻在"] else "一次没翻",
             r["高"], r["歪"], r["走拍"], r["重0"], r["重1"], r["边0"], r["边1"]), flush=True)