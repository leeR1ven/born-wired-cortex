"""调参_皮层网.py —— 给皮层连接网找一组好参数（不是找"猜得准"，是找"活动不多不少、能传信号"）

要同时满足三件事：
  1. 不熄灭：给一点信号，别的区能被带亮一些（本能连接真的在起作用）
  2. 不全亮：信号再大也只稳在"不多的量"上（思考才有稀疏的念头，而不是全脑一起烧）
  3. 输入越大、稳态越大（信号强度还能被区分出来），但不是线性的爆炸

扫的就是两个最关键的数字：兴奋阈值、抑制强度。
运行：python 调参_皮层网.py
"""
from __future__ import annotations

import time
import numpy as np

import 皮层连接_cortex_links as 皮层


宽 = {"视觉": 300, "听觉": 200, "前额叶": 400, "运动": 100}
阈值档 = [0.3, 0.5, 0.7, 1.0, 1.5]
抑制档 = [0.5, 1.0, 1.5, 2.0, 3.0]
输入档 = [20, 60, 120]
步数 = 15


def 跑一次(兴奋阈值, 抑制强度, 亮数, seed=7):
    网 = 皮层.皮层连接网(宽, seed=seed, 兴奋阈值=兴奋阈值, 抑制强度=抑制强度)
    rng = np.random.default_rng(1)
    视 = np.zeros(宽["视觉"], dtype=bool)
    视[rng.choice(宽["视觉"], 亮数, replace=False)] = True
    空 = {名: np.zeros(w, dtype=bool) for 名, w in 宽.items()}
    当前 = {**空, "视觉": 视}
    轨迹 = []
    for _ in range(步数):
        出 = 网.步进(当前)
        出[网.区段("视觉")] |= 视          # 外界一直给信号
        轨迹.append(int(出.sum()))
        当前 = {名: 出[网.区段(名)] for 名 in 宽}
    return 轨迹


def main():
    print("=" * 100)
    print("皮层连接网调参：找'活动不多不少、信号能传出去'的区间")
    print(f"小球宽度 {宽}  总 {sum(宽.values())} 个皮层神经元   每档跑 {步数} 拍")
    print("=" * 100)
    print(f"{'阈值':>5} {'抑制':>5} | " + " | ".join(f"输入{n:>3} → 稳态(非视觉部分)" for n in 输入档))
    print("-" * 100)
    t0 = time.perf_counter()
    最好 = []
    for 阈 in 阈值档:
        for 抑 in 抑制档:
            格 = []
            合格 = True
            for 亮 in 输入档:
                轨迹 = 跑一次(阈, 抑, 亮)
                稳 = 轨迹[-1]
                非视 = 稳 - 亮
                格.append((稳, 非视))
                if 稳 >= 0.9 * sum(宽.values()):
                    合格 = False
                if 稳 - 亮 <= 1 and 亮 >= 60:
                    合格 = False
            txt = " | ".join(f"{稳:>4} (带出 {非:>4})" for 稳, 非 in 格)
            print(f"{阈:>5} {抑:>5} | {txt}" + ("   ← 合格" if 合格 else ""))
            if 合格:
                最好.append((阈, 抑, 格))
    print("-" * 100)
    print(f"用时 {time.perf_counter() - t0:.1f} 秒")
    if 最好:
        print("合格的组合（不熄灭、不全亮、输入越大越活跃）：")
        for 阈, 抑, 格 in 最好:
            print(f"   兴奋阈值={阈}  抑制强度={抑}   " +
                  "  ".join(f"输入{n}→{g[0]}" for n, g in zip(输入档, 格)))
    else:
        print("没有全合格的组合，看上面哪一行最接近（要看'带出'那列 > 0 且总数没到 1000）")


if __name__ == "__main__":
    main()