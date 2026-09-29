# -*- coding: utf-8 -*-
"""诊断_抑制接线修好后会不会摔.py

临时把「每个细胞都发 50 就近 + 50 全脑随机」的抑制接线装进去（只在内存里替换
皮层连接网._生抑制 这一个方法，不改仓库里任何源文件），再跑 R1 那套协议：
趴着能不能自己站起来 / 站着朝红能不能走 / 什么都不给会不会瘫掉。

用途：给「抑制连接只发了一半」这个 bug 的修复定成本 —— 修完本能还成不成立。

用法：cd model; python 诊断_抑制接线修好后会不会摔.py [种子,种子,...]
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


C.皮层连接网._生抑制 = _生抑制修正

import 实验_闭环前提_多种子 as E

print("【注意】本脚本用的是「修正后的抑制接线」（每个细胞 50 就近 + 50 全脑随机），"
      "不是仓库现在的接线。", flush=True)
E.main()