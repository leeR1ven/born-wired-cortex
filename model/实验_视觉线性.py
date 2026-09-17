# -*- coding: utf-8 -*-
"""实验_视觉线性.py —— 「外界画面线性地变，视觉皮层是不是也线性地跟着变？」

★ 用户 2026-09-15：
    "出了问题应该直接改，而不是打补丁，不然只会问题越来越多。"
    "'认得出红到什么程度'的测试说明外界画面线性改变，而皮层激活的神经元没有线性改变，
     这有问题，按理说输出应该和输入一样是线性改变的。"

★ 量什么

   画面正中间那片红一点点调暗（亮度档 10 → 0，共 11 档），看视觉皮层上
   "中有红"那一组细胞**被点亮的比例**跟不跟着降下来。
   理想：视比例 ≈ 红亮度（一条直线）。本能的力气就是按这个比例算的
   （本能表里 0.9 的力度摊到源块每根连接上），所以视比例线性 = "认得出红到什么程度"线性。

   ★ 注意：红色区域特意按大像素格对齐（大像素列 5~10），
     这样"调暗"就是纯粹的亮度变化，不会掺进"半个格子"的边角效应。

★ 查出来的病根

   最后一层视野 600 —— 一个细胞把 ±600 个输入**加起来**。权重又按**总和**归一化成 1，
   于是这个和 = 一大片输入取**平均**。平均的数字全往中间挤（细胞之间几乎没差别），
   任何门槛都是"要么几乎全亮、要么几乎全灭"。当初加"取前几名"就是为了硬撑这个。

   真正的根子：**按"总和"归一 = 取平均 = 视野一大就什么都感觉不到。**
   改成按**平方和**归一 = 这个细胞在问"像我认的这种模式吗"（相关），
   视野多大都照样敏感，门槛才有统一的意思，亮几个才会随输入连续变。

★ 命令
    python 实验_视觉线性.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V

每层数 = int(V.池.总数)
图高, 图宽 = 1080, 1920
红左列, 红右列 = 5 * (图宽 // 16), 11 * (图宽 // 16)      # 大像素列 5~10，对齐格子
档数 = 11


def _窗(R):
    目标 = np.arange(每层数)
    偏移 = np.arange(-R, R + 1)
    return ((目标[:, None] + 偏移[None, :]) % 每层数).astype(np.int32)


class 试验网:
    def __init__(self, 层数=4, 半径=10, 输出感受野=600, 阈值初值=0.2,
                 归一="平方和", 采样数=0, 种子=20260914):
        rng = np.random.default_rng(种子)
        self.层数 = 层数
        self.阈值 = [np.full(每层数, 阈值初值) for _ in range(层数 - 1)]
        self.普通层激活 = []
        self.来源, self.权重, self.半径们, self.稀疏层 = [], [], [], []
        for k in range(层数 - 1):
            R = 输出感受野 if k == 层数 - 2 else 半径
            self.半径们.append(R)
            if 采样数 and 采样数 < 2 * R + 1:
                抽 = rng.integers(0, 2 * R + 1, size=(每层数, 采样数))
                绝对 = ((np.arange(每层数)[:, None] - R + 抽) % 每层数).astype(np.int32)
                幅 = rng.uniform(0.0, 1.0, size=(每层数, 采样数))
                self.稀疏层.append(True)
            else:
                绝对 = _窗(R)
                幅 = rng.uniform(0.0, 1.0, size=绝对.shape)
                self.稀疏层.append(False)
            if 归一 == "总和":
                幅 = 幅 / 幅.sum(axis=1, keepdims=True)
            else:
                幅 = 幅 / np.sqrt((幅 ** 2).sum(axis=1, keepdims=True))
            符号 = np.where(绝对 % 2 == 0, 1.0, -1.0)      # 正=兴奋、反=抑制（"对"的两面）
            self.来源.append(绝对)
            self.权重.append((幅 * 符号).astype(np.float32))

    def 前向传播(self, 信号):
        V.池.信号[:] = 信号
        V.池.更新所有对()
        x = V.池.激活.astype(np.float32)
        self.普通层激活 = []
        for k in range(self.层数 - 1):
            if self.稀疏层[k]:
                和 = np.einsum("ij,ij->i", self.权重[k], x[self.来源[k]])
            else:
                R = self.半径们[k]
                xt = np.concatenate([x[每层数 - R:], x, x[:R]])
                X = sliding_window_view(xt, 2 * R + 1)
                和 = np.einsum("ij,ij->i", self.权重[k], X)
            x = 和 >= self.阈值[k]
            self.普通层激活.append(x)
            x = x.astype(np.float32)
        return x


def 图(亮度, 起=None, 止=None, 绿=0, 蓝=0):
    a = np.zeros((图高, 图宽, 3), dtype=np.uint8)
    if 起 is None:
        a[:, :, 0] = 亮度; a[:, :, 1] = 绿; a[:, :, 2] = 蓝
    else:
        a[:, 起:止, 0] = 亮度; a[:, 起:止, 1] = 绿; a[:, 起:止, 2] = 蓝
    return a


探针 = {
    "黑":   图(0),
    "全红": 图(255),
    "全绿": 图(0, 绿=255),
    "全蓝": 图(0, 蓝=255),
    "白":   图(255, 绿=255, 蓝=255),
    "中有红": 图(255, 红左列, 红右列),
    "左有红": 图(255, 0, 图宽 // 2),
    "右有红": 图(255, 图宽 // 2, 图宽),
    "中蓝":  图(0, 红左列, 红右列, 蓝=255),
}


def 中红亮度(档):
    """正中间那片红，红通道 = 第几档（0~10）——直接按档来，不掺图片量化的边角"""
    return 图(int(round(255 * 档 / 10)), 红左列, 红右列)


def 评估(网, 计时=False):
    t0 = time.perf_counter()
    出 = {名: np.asarray(网.前向传播(V.图像转信号(g)), dtype=bool) for 名, g in 探针.items()}
    秒 = (time.perf_counter() - t0) / len(探针)
    基准 = 出["中有红"]
    n = int(基准.sum())
    渐 = []
    for 档 in range(10, -1, -1):
        o = np.asarray(网.前向传播(V.图像转信号(中红亮度(档))), dtype=bool)
        渐.append(float((o & 基准).sum()) / max(n, 1))
    return 出, 渐, n, 秒


def 打分(渐):
    arr = np.asarray(渐)
    s = np.linspace(1.0, 0.0, 档数)
    单调 = float(np.mean(np.diff(arr) <= 0.02))
    贴合 = float(1.0 - np.mean((arr - s) ** 2) / np.var(s))
    跨度 = float(arr[0] - arr[-1])
    return 单调, 贴合, 跨度


def 报(标题, 网):
    出, 渐, n, 秒 = 评估(网)
    单, 贴, 跨 = 打分(渐)
    print(f"[{标题}] 中有红占 {n}/{每层数}（{100 * n / 每层数:.0f}%），一帧 {1000 * 秒:.1f}ms")
    print(f"   亮几个：黑{100 * 出['黑'].sum() / 每层数:.0f}% 全红{100 * 出['全红'].sum() / 每层数:.0f}%"
          f" 中红{100 * n / 每层数:.0f}% 左红{100 * 出['左有红'].sum() / 每层数:.0f}%"
          f" 右红{100 * 出['右有红'].sum() / 每层数:.0f}% 中蓝{100 * 出['中蓝'].sum() / 每层数:.0f}%"
          f" 白{100 * 出['白'].sum() / 每层数:.0f}%")
    print(f"   红渐淡 视比例(%): {' '.join('%3d' % round(100 * v) for v in 渐)}")
    print(f"   -> 单调 {单:.0%}，像一条直线 {贴:.2f}（1=完全线性），跨度 {跨:.0%}")
    return 单, 贴, 跨, 渐, n, 秒


def main():
    print(f"视觉输入：{V.大像素行数}x{V.大像素列数} x RGB x {V.每档对数}对 = {每层数} 个神经元/层")
    print("\n=== 现在生产用的那张网 ===")
    报(f"现网 门槛{V.门槛中心}±{V.门槛散布} 抽{V.采样数}根", V.网)

    print("\n=== 平方和归一：采样数 x 门槛 ===")
    结果 = []
    for 采样数 in (0, 64, 128):
        for 阈值 in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            网 = 试验网(归一="平方和", 采样数=采样数, 阈值初值=阈值)
            单, 贴, 跨, 渐, n, 秒 = 报(f"采样{采样数 or '全'} 阈值{阈值}", 网)
            结果.append((单, 贴, 跨, 采样数, 阈值, 渐, n, 秒))
    print("\n=== 排名（单调 100% 的，按像不像直线）===")
    for 单, 贴, 跨, 采样数, 阈值, 渐, n, 秒 in sorted([r for r in 结果 if r[0] >= 0.99], key=lambda r: -r[1])[:8]:
        print(f"   采样{str(采样数 or '全'):<3} 阈值{阈值:<4} 直线{贴:.2f} 跨度{跨:.0%} "
              f"中有红占{100 * n / 每层数:.0f}% {1000 * 秒:.1f}ms  {' '.join('%3d' % round(100 * v) for v in 渐)}")


if __name__ == "__main__":
    main()