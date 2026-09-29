# -*- coding: utf-8 -*-
"""实验_专注_无关信号扩不出去.py

用户 2026-09-29 的说法：
「原本很多神经元就是被抑制的，只有发送更多信号的神经元更容易被激活；深度思考时是**太多与
思考内容相关的神经元被激活**，外界无关的神经元虽然也会被激活，但**很难再激活出更多其他有关
的神经元**，并不是完全被抑制。就像文字网络里有些文字可能始终被激活，但也不会对整体网络的
思考范围产生太多影响。」

怎么测才不混：
  只钉住「在想的事」P（前额叶前 k 个）           -> 视觉区亮 X 个（这是想的内容自己带亮的）
  再额外钉住「外界无关信号」A（视觉前 2000 个）  -> 视觉区亮 Y 个
  **信号额外带亮 = Y - X**   <- 这才是「无关信号能扩出去多少」
（A 自己那 2000 个细胞一直是被钉住的，所以它『一定会亮』，这一点不受影响。）

跑两遍：一遍用仓库现在的接线，一遍用「每个细胞 50 就近 + 50 全脑随机」的修正接线
（只在内存里替换 _生抑制，不改任何源文件）。

用法：cd model; python 实验_专注_无关信号扩不出去.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import 皮层连接_cortex_links as C


def _生抑制修正(self):
    rng = self.rng
    P = self.参数
    N = self.总数
    R = P["抑制半径"]
    条 = P["抑制条数"]
    就近条 = int(round(条 * P["抑制就近比例"]))
    随机条 = 条 - 就近条
    源 = np.repeat(np.arange(N, dtype=np.int64), 条)
    目 = np.empty(N * 条, dtype=np.int64)
    B = 目.reshape(N, 条)
    S = np.arange(N, dtype=np.int64).reshape(N, 1)
    if 就近条:
        B[:, :就近条] = S + rng.integers(-R, R + 1, size=(N, 就近条))
    if 随机条:
        B[:, 就近条:] = rng.integers(0, N, size=(N, 随机条))
    好 = (目 >= 0) & (目 < N) & (目 != 源) & ~self._不许冒出[源]
    源, 目 = 源[好], 目[好]
    键 = np.unique(源 * N + 目)
    self._抑制键 = 键
    self._抑制源 = (键 // N).astype(np.int32)
    self._抑制目 = (键 % N).astype(np.int32)
    self._抑制权 = rng.uniform(0.0, P["抑制出生权重"], size=键.size).astype(np.float32)
    self._抑制出生 = self._抑制权.copy()
    self._抑制固化 = np.zeros(键.size, dtype=bool)
    self._抑制起 = np.searchsorted(self._抑制源, np.arange(N + 1, dtype=np.int32)).astype(np.int64)
    self.抑制权重 = self._抑制权
    self.去抑伙伴 = np.where(rng.random(N) < P["带去抑伙伴比例"],
                            rng.integers(0, N, size=N), -1).astype(np.int32)
    self.去抑权重 = np.zeros(N, dtype=np.float32)
    self._备抑制索引()


def 跑(网, 想, 信号, 拍数=25):
    上 = np.zeros(网.总数, dtype=bool)
    for _ in range(拍数):
        这 = 网.步进(上)
        if 信号 is not None:
            这[信号] = True
        if 想.size:
            这[想] = True
        上 = 这
    return 上


def 扫一遍(标签, 想建网=None):
    if 想建网 is not None:
        C.皮层连接网._生抑制 = 想建网
    import 本能工具_instincts as 本能
    import importlib
    importlib.reload(本能)
    网, 条 = 本能.装(说=False)
    v起, v宽 = 网.区起点["视觉"], 网.区宽["视觉"]
    p起 = 网.区起点["前额叶"]
    信号 = np.arange(v起, v起 + 2000)
    print("")
    print("=" * 104)
    print(f"【{标签}】真模型 {网.总数:,} 个细胞、本能 {条} 条；外界无关信号 = 视觉前 2000 个细胞")
    print("=" * 104)
    print(f"{'在想的事(前额叶亮)':>18} | {'只想时视觉区亮':>14} | {'加信号后':>10} | {'信号额外带亮':>12} | {'全脑总激活':>12}")
    print("-" * 104)
    行 = []
    for k in (0, 500, 2000, 8000, 20000):
        想 = np.arange(p起, p起 + k)
        甲 = 跑(网, 想, None)
        乙 = 跑(网, 想, 信号)
        甲视 = int(甲[v起:v起 + v宽].sum())
        乙视 = int(乙[v起:v起 + v宽].sum())
        加 = 乙视 - 甲视
        print(f"{k:>18,} | {甲视:>14,} | {乙视:>10,} | {加:>12,} | {int(乙.sum()):>12,}")
        行.append((k, 甲视, 乙视, 加))
    return 行


原始 = C.皮层连接网._生抑制
扫一遍("现在仓库的接线：只有下标前半段的细胞发抑制")
扫一遍("修正接线：每个细胞 50 就近 + 50 全脑随机", _生抑制修正)
C.皮层连接网._生抑制 = 原始
print("")
print("看最后一列『信号额外带亮』：前额叶亮得越多，这个数应该越小（无关信号扩不出去），")
print("但前两列说明那 2000 个信号细胞自己照样亮着 —— 不是被完全压住。")