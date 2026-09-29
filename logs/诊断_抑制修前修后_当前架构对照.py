# -*- coding: utf-8 -*-
"""诊断_抑制修前修后_当前架构对照.py

论文 R1 引用的日志是**旧架构**（前额叶 53,200、共 143,796 个细胞）跑出来的；
现在的代码是 148,032 个细胞。所以「R1 数字变了多少」里混了两件事。
本脚本在当前架构上，把**修之前的抑制构造**临时装回内存，跑同一遍 R1，做干净归因。

用法：cd model; python -X utf8 诊断_抑制修前修后_当前架构对照.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import 皮层连接_cortex_links as C


def _生抑制_修前(self):
    """照抄修之前那一段（只有下标前半段发得出来，且就近那半被覆盖掉）。"""
    rng = self.rng
    P = self.参数
    N = self.总数
    R = P["抑制半径"]
    条 = P["抑制条数"]
    就近条 = int(round(条 * P["抑制就近比例"]))
    随机条 = 条 - 就近条
    源 = np.repeat(np.arange(N, dtype=np.int64), 条)
    偏 = np.concatenate([
        rng.integers(-R, R + 1, size=N * 就近条).astype(np.int64),
        np.zeros(N * 随机条, dtype=np.int64),
    ])
    目 = 源 + 偏
    if 随机条:
        目[:N * 随机条] = rng.integers(0, N, size=N * 随机条, dtype=np.int64)
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


修后 = C.皮层连接网._生抑制
种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]
import 实验_闭环前提_多种子 as E

for 标签, 方法 in (("FIXED 修后", 修后), ("BUGGY 修前", _生抑制_修前)):
    C.皮层连接网._生抑制 = 方法
    站 = []; 走 = []; 末 = []
    for s in 种子们:
        r = E.跑一趟(s)
        站.append(r["站起来末高"]); 走.append(r["朝红走了"]); 末.append(r["瘫掉末高"])
    站 = np.array(站); 走 = np.array(走); 末 = np.array(末)
    print(f"{标签} | stand {int((站 > 0.20).sum())}/{len(站)} med {np.median(站):.3f} min {站.min():.3f}"
          f" | walk {int((走 > 0.30).sum())}/{len(走)} med {np.median(走):+.2f} min {走.min():+.2f}"
          f" | limp {int((末 < 0.20).sum())}/{len(末)}", flush=True)
C.皮层连接网._生抑制 = 修后