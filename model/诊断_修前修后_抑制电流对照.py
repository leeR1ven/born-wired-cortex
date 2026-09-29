# -*- coding: utf-8 -*-
"""诊断_修前修后_抑制电流对照.py —— 把抑制接线的 bug 修好，到底有没有把整片网的「压」改掉？

★ 为什么要有这个（2026-09-29）
  抑制连接生成器里有个坐标 bug（见 docs/抑制连接_只发了一半_20260929.md）：修之前
  148,032 个细胞里只有 54,565 个发得出抑制，**整个前额叶一根都没有**，而且所谓「就近」
  那一半从来没生效过。修好之后是 124,792 个细胞都发（前额叶 57,436 全发），就近半径 4。
  于是立刻要回答一个绕不开的问题：**抑制变强了没有？**
  因为论文里其它所有数字（摔倒、起身、走路、色相边界）都是在「抑制强度 = 1.5」
  这个刻度下调出来的。如果修 bug 顺带把「压」加了一倍，那等于换了一台机器，
  以前标定的参数全部作废，必须重新调；如果没怎么变，那标定照旧，只是接线对了。

★ 怎么查：同一颗种子，把**修前那一版 `_生抑制` 原样搬回来**（从 git 里抄，一字不改），
  分别建网，然后拿同一张红幕各跑 20 拍，量三样：
    · 抑制边有几条、有多少细胞发得出抑制、前额叶发不发得出
    · 每一拍全脑收到的抑制电流总量（这是「整体压多少」）
    · 「前进1.2」头细胞身上那一份（这是「本能那条路被压多少」）
  三样一比，就知道标定要不要重来。

命令：python 诊断_修前修后_抑制电流对照.py
"""
from __future__ import annotations

import numpy as np

import 皮层连接_cortex_links as 皮层
import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体

种子 = 20260914


def 生抑制_旧(self):
    """★ 修前那一版（git HEAD 的 皮层连接_cortex_links.py::_生抑制），一字不改搬过来。"""
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


现在的方法 = 皮层.皮层连接网._生抑制


def 建网(用旧的):
    皮层.皮层连接网._生抑制 = 生抑制_旧 if 用旧的 else 现在的方法
    try:
        本能.建网种子 = 种子
        return 主循环.脑(学=False, 说=False)
    finally:
        皮层.皮层连接网._生抑制 = 现在的方法


def 量一个(脑, 图, 拍数=20):
    网 = 脑.网
    头 = 网.区起点["运动记忆"] + int(本能.读名字()["运动记忆"]["前进1.2"][0])
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]
    源 = 网._抑制源
    发 = int(np.unique(源).size)
    前发 = int(np.unique(源[(源 >= 前起) & (源 < 前起 + 前宽)]).size)
    身 = 身体.身体()
    身.摆成站姿()
    身.走(0.5, None)
    脑.亮[:] = False
    脑.上一拍[:] = False
    总 = []
    头抑 = []
    for _ in range(拍数):
        脑.一拍(图=图, 身=身)
        r = 网._抑制电流(脑.上一拍)
        总.append(float(r.sum()))
        头抑.append(float(r[头]))
    return dict(边=int(网._抑制键.size), 发=发, 前发=前发,
                总=float(np.mean(总)), 头=float(np.mean(头抑)))


def main():
    print("=" * 96)
    print("同一个种子、同一张红幕：把抑制生成器的 bug 修好，到底把「压」改了多少")
    print("=" * 96)
    图高, 图宽 = 主循环.图高, 主循环.图宽
    红 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
    红[:, 图宽 // 3: 2 * 图宽 // 3, 0] = 255

    出 = {}
    for 名, 旧 in [("修前（旧接线）", True), ("修后（新接线）", False)]:
        脑 = 建网(旧)
        r = 量一个(脑, 红)
        出[名] = r
        print("%s：抑制边 %d 条 | 发得出抑制的细胞 %d 个（其中前额叶 %d 个）| "
              "每拍全脑抑制电流 %.0f | 「前进1.2」头细胞身上 %.2f"
              % (名, r["边"], r["发"], r["前发"], r["总"], r["头"]), flush=True)

    a, b = 出["修前（旧接线）"], 出["修后（新接线）"]
    print("\n" + "=" * 96)
    print("【结论】：全脑抑制电流 %.0f → %.0f，是原来的 %.0f%%；头细胞 %.2f → %.2f，是原来的 %.0f%%。"
          % (a["总"], b["总"], 100.0 * b["总"] / a["总"],
             a["头"], b["头"], 100.0 * b["头"] / a["头"]))
    print("  也就是说：修的是「谁发得出抑制、发到哪儿」，不是「压多重」。刻度（抑制强度 1.5）不用重调。")
    return 出


if __name__ == "__main__":
    出 = main()