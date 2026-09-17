# -*- coding: utf-8 -*-
"""验证_感觉网分块.py —— 证明「按行分块抽」和原来的大表抽出来的权重**一模一样**。

做法：把 神经网络.__init__ 整个照抄一遍（包括先抽阈值那一步，它也吃随机数），
老写法 / 新写法各算一次，逐字节比；再和真·视觉网、真·听觉网比。
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from 感觉网_多层 import 出生种子, 出生门槛下限

def 算网(每层数, 层数, 连接半径, 权重范围, 门槛中心, 门槛散布, 输出感受野, 采样数, 老写法):
    出生随机 = np.random.default_rng(出生种子)
    阈值 = [np.clip(门槛中心 + 门槛散布 * 出生随机.uniform(-1.0, 1.0, 每层数), 出生门槛下限, None)
            for _ in range(层数 - 1)]
    来源们, 权重们 = [], []
    for k in range(层数 - 1):
        用半径 = 输出感受野 if k == 层数 - 2 else 连接半径
        连接数 = 2 * 用半径 + 1
        目标 = np.arange(每层数)
        抽着 = 采样数 and 采样数 < 连接数
        if 抽着:
            抽 = 出生随机.integers(0, 连接数, size=(每层数, 采样数))
            来源 = (目标[:, None] - 用半径 + 抽) % 每层数
            幅 = 出生随机.uniform(0.0, 1.0, size=(每层数, 采样数))
        else:
            偏移 = np.arange(-用半径, 用半径 + 1)
            来源 = (目标[:, None] + 偏移[None, :]) % 每层数
            对偏移 = 来源 // 2 - (目标 // 2)[:, None]
            列 = 对偏移 - 对偏移.min()
            if 老写法:
                强度 = 出生随机.uniform(0.0, 1.0, size=(每层数, int(列.max()) + 1))
                幅 = 强度[np.arange(每层数)[:, None], 列]
            else:
                _宽 = int(列.max()) + 1
                _块行 = max(1, min(每层数, 4_000_000 // max(_宽, 1)))
                _幅 = np.empty(列.shape, dtype=np.float64)
                for _起 in range(0, 每层数, _块行):
                    _止 = min(每层数, _起 + _块行)
                    _强 = 出生随机.uniform(0.0, 1.0, size=(_止 - _起, _宽))
                    _行 = np.arange(_止 - _起)[:, None]
                    _幅[_起:_止] = _强[_行, 列[_起:_止]]
                幅 = _幅
        幅 = 幅 / np.sqrt((幅 ** 2).sum(axis=1, keepdims=True))
        符号 = np.where(来源 % 2 == 0, 1.0, -1.0)
        来源们.append(来源.astype(np.int32))
        权重们.append((幅 * 符号).astype(np.float32))
    return 来源们, 权重们, 阈值

def 比(名, 源A, 权A, 源B, 权B):
    同 = all(np.array_equal(a, b) for a, b in zip(权A, 权B)) and all(np.array_equal(a, b) for a, b in zip(源A, 源B))
    print("  %-26s 完全一样：%s" % (名, 同))
    return 同

print("=" * 78)
print("小规模（形状同听觉/视觉，每层数压到 3000）—— 老写法 vs 新写法：")
形听觉 = (3000, 3, 10, 0.1, 0.4, 0.6, 400, 64)
形视觉 = (3000, 4, 10, 0.1, 0.6, 1.0, 600, 8)
形不抽 = (2000, 3, 10, 0.1, 0.4, 0.6, 200, 0)
for 形, 名 in ((形听觉, "听觉参数"), (形视觉, "视觉参数"), (形不抽, "视野比采样数小的那条路")):
    比(名, *算网(*形, 老写法=True)[:2], *算网(*形, 老写法=False)[:2])

print()
print("拿真参数、真宽度 和 真·视觉网/真·听觉网 比（老写法在 30000 要 7.2 GB，只比新的）：")
import 听觉前处理_auditory_preprocess as A
源A, 权A, 阈A = 算网(30000, 3, 10, 0.1, 0.4, 0.6, 400, 64, 老写法=False)
比("真·听觉网", 源A, 权A, A.网.来源, A.网.权重)
print("     阈值也一样：%s" % all(np.array_equal(a, b) for a, b in zip(阈A, A.网.阈值)))
import 视觉前处理_visual_preprocess as V
源V, 权V, 阈V = 算网(int(V.网.每层数), 4, 10, 0.1, 0.6, 1.0, 600, 8, 老写法=False)
比("真·视觉网", 源V, 权V, V.网.来源, V.网.权重)
print("     阈值也一样：%s" % all(np.array_equal(a, b) for a, b in zip(阈V, V.网.阈值)))