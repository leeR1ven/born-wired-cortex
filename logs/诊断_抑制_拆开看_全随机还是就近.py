# -*- coding: utf-8 -*-
"""诊断_抑制_拆开看_全随机还是就近.py

三个版本跑同一遍 R1，拆清楚「修完之后本能变弱」是哪一半造成的：
  (甲) 修前：只有前半段细胞发，全是全脑随机
  (乙) 每个细胞都发，但**全都撒全脑**（去掉就近那一半）
  (丙) 每个细胞都发，50 就近 + 50 全脑随机（= 现在的修复版）
如果乙 ≈ 甲、丙 变弱，那问题在「就近」；如果乙 也变弱，那问题在「多了一半细胞」。

用法：cd model; python -X utf8 诊断_抑制_拆开看_全随机还是就近.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import 皮层连接_cortex_links as C


def _装表(self, 源, 目, rng):
    N = self.总数
    好 = (目 >= 0) & (目 < N) & (目 != 源) & ~self._不许冒出[源]
    源, 目 = 源[好], 目[好]
    键 = np.unique(源 * N + 目)
    self._抑制键 = 键
    self._抑制源 = (键 // N).astype(np.int32)
    self._抑制目 = (键 % N).astype(np.int32)
    self._抑制权 = rng.uniform(0.0, self.参数["抑制出生权重"], size=键.size).astype(np.float32)
    self._抑制出生 = self._抑制权.copy()
    self._抑制固化 = np.zeros(键.size, dtype=bool)
    self._抑制起 = np.searchsorted(self._抑制源, np.arange(N + 1, dtype=np.int32)).astype(np.int64)
    self.抑制权重 = self._抑制权
    self.去抑伙伴 = np.where(rng.random(N) < self.参数["带去抑伙伴比例"],
                            rng.integers(0, N, size=N), -1).astype(np.int32)
    self.去抑权重 = np.zeros(N, dtype=np.float32)
    self._备抑制索引()


def 修前(self):
    rng = self.rng; P = self.参数; N = self.总数; R = P["抑制半径"]
    条 = P["抑制条数"]; 就近条 = int(round(条 * P["抑制就近比例"])); 随机条 = 条 - 就近条
    源 = np.repeat(np.arange(N, dtype=np.int64), 条)
    偏 = np.concatenate([rng.integers(-R, R + 1, size=N * 就近条).astype(np.int64),
                        np.zeros(N * 随机条, dtype=np.int64)])
    目 = 源 + 偏
    目[:N * 随机条] = rng.integers(0, N, size=N * 随机条, dtype=np.int64)
    _装表(self, 源, 目, rng)


def 全新随机(self):
    """每个下标前半段细胞发 条 条全脑随机（等价于把修前那套推广到每个细胞）。"""
    rng = self.rng; P = self.参数; N = self.总数
    条 = P["抑制条数"]
    前半 = N // 2
    源 = np.repeat(np.arange(前半, dtype=np.int64), 条)
    目 = rng.integers(0, N, size=前半 * 条, dtype=np.int64)
    _装表(self, 源, 目, rng)


def 全随机(self):
    """每个细胞都发，全部撒全脑。"""
    rng = self.rng; P = self.参数; N = self.总数; 条 = P["抑制条数"]
    源 = np.repeat(np.arange(N, dtype=np.int64), 条)
    目 = rng.integers(0, N, size=N * 条, dtype=np.int64)
    _装表(self, 源, 目, rng)


def 修后(self):
    rng = self.rng; P = self.参数; N = self.总数; R = P["抑制半径"]
    条 = P["抑制条数"]; 就近条 = int(round(条 * P["抑制就近比例"])); 随机条 = 条 - 就近条
    源 = np.repeat(np.arange(N, dtype=np.int64), 条)
    目 = np.empty(N * 条, dtype=np.int64)
    块 = 目.reshape(N, 条); 本站 = np.arange(N, dtype=np.int64).reshape(N, 1)
    块[:, :就近条] = 本站 + rng.integers(-R, R + 1, size=(N, 就近条))
    块[:, 就近条:] = rng.integers(0, N, size=(N, 随机条))
    _装表(self, 源, 目, rng)


原始 = C.皮层连接网._生抑制
种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]
import 实验_闭环前提_多种子 as E

for 标签, 方法 in (("A 修前", 修前), ("B 修前那套但铺满全部细胞", 全新随机),
                   ("C 每个细胞都发·全撒全脑", 全随机), ("D 每个细胞都发·一半就近（现在的修复版）", 修后)):
    C.皮层连接网._生抑制 = 方法
    站 = []; 走 = []; 末 = []; 边 = 0
    for s in 种子们:
        r = E.跑一趟(s)
        站.append(r["站起来末高"]); 走.append(r["朝红走了"]); 末.append(r["瘫掉末高"])
    站 = np.array(站); 走 = np.array(走); 末 = np.array(末)
    print(f"{标签} | stand {int((站 > 0.20).sum())}/5 med {np.median(站):.3f} min {站.min():.3f}"
          f" | walk {int((走 > 0.30).sum())}/5 med {np.median(走):+.2f} min {走.min():+.2f}"
          f" | limp {int((末 < 0.20).sum())}/5", flush=True)
C.皮层连接网._生抑制 = 原始