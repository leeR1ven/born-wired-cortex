# -*- coding: utf-8 -*-
"""实时看身体到底在干什么。

三个画面：
  左：身体侧视。绿圈 = 正踩在地面上的点，红叉 = 重心。
  中：每个关节"想要的角度"（灰）vs"实际角度"（蓝）。差得远 = 关节被卡住了。
  右：它眼睛看到的画面（24x16 马赛克放大）。

用法：
  python 工具_看身体.py                 # 站着
  python 工具_看身体.py 躺下            # 躺下看它稳不稳
  python 工具_看身体.py 站起来          # 跑第一版站起来动作
  python 工具_看身体.py 站起来 录       # 不开窗口，录成 gif
加 "录" 就会保存成 工具_看身体_录像.gif（窗口还是会开）。
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

根 = pathlib.Path(__file__).resolve().parent
if str(根) not in sys.path:
    sys.path.insert(0, str(根))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as 动画
from matplotlib.patches import Circle

import 世界_world as W

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

实时比例 = 30          # 屏幕上每秒画多少帧
物理每帧 = 8           # 每画一帧，物理走几步（240/8 = 30 帧/秒）


def 站起来关键帧():
    """第一版站起来动作（还没调好，先用来复现问题）。"""
    站立 = dict(zip(W.关节名, W.节站立角))
    return {
        "躺下": (0.0, {}),
        "蜷起来": (1.2, {"腰": 95, "左髋": 85, "右髋": 85, "左膝": 80, "右膝": 80,
                          "左踝": 95, "右踝": 95}),
        "前倾": (0.8, {"腰": 75}),
        "伸直": (1.5, {"腰": 178, "左髋": 165, "右髋": 165, "左膝": 153, "右膝": 153,
                        "左踝": 75, "右踝": 75}),
        "接着看": (2.0, {}),
    }


class 看身体:
    def __init__(self, 场景="站着"):
        self.场景 = 场景
        self.w = W.世界()
        self.段 = []
        self.段开始 = 0.0
        self.段号 = 0
        if 场景 == "站起来":
            self.w.摆成躺着(脸朝上=True)
            self.关键帧 = list(站起来关键帧().items())
            self.每段秒 = [x[1][0] for x in self.关键帧]
            self.每段角 = [x[1][1] for x in self.关键帧]
            self.段起角 = self.w.关节.copy()
        elif 场景 == "躺下":
            self.w.摆成躺着(脸朝上=True)

        self.图 = plt.figure(figsize=(15, 5.6))
        网 = self.图.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.28)
        self.轴身 = self.图.add_subplot(网[0, 0])
        self.轴角 = self.图.add_subplot(网[0, 1])
        self.轴眼 = self.图.add_subplot(网[0, 2])
        self.帧 = []

    def _推进一步(self):
        """动作是「关键帧 + 平滑插值」：每段给出目标角度和走的秒数，中间平滑过渡。

        以前这里写错过：整个插值块被塞进 if i > 0 里，结果第 0 段永远不推进、
        段号永远是 0，表现就是「躺在那里一动不动」。别再写回去了。
        """
        w = self.w
        if self.场景 == "站起来":
            if self.段号 == 0:                      # 第 0 段 = 当前躺着这个姿势，直接跳过
                self.段号 = 1
                self.段起角 = w.关节.copy()
                self.段开始 = w.时间
            i = min(self.段号, len(self.关键帧) - 1)
            时长 = self.每段秒[i]
            u = min(1.0, (w.时间 - self.段开始) / max(时长, 1e-6))
            u = u * u * (3 - 2 * u)                 # smoothstep：起落都平滑，别"一顿一顿"
            起 = self.段起角
            终 = 起.copy()
            for n, v in self.每段角[i].items():
                终[W.关节名.index(n)] = v
            w.写关节(起 + (终 - 起) * u)
            if u >= 1.0:
                self.段号 += 1
                self.段起角 = w.关节.copy()
                self.段开始 = w.时间
        for _ in range(物理每帧):
            w.步进()

    def _画身(self):
        ax = self.轴身
        ax.clear()
        w = self.w
        p = w.位置
        贴地 = p[:, 2] <= W.最小离地 + 1e-3
        ax.axhspan(-0.3, W.最小离地, color="0.75", zorder=0)
        ax.axhline(W.最小离地, color="0.45", lw=1, zorder=1)
        for a, b in W.骨架:
            i, j = W.序号[a], W.序号[b]
            ax.plot([p[i, 0], p[j, 0]], [p[i, 2], p[j, 2]], "-", color="tab:blue", lw=4.0, zorder=3)
        for _, a, b in [(x[0], x[1], x[2]) for x in W.被动表]:
            i, j = W.序号[a], W.序号[b]
            ax.plot([p[i, 0], p[j, 0]], [p[i, 2], p[j, 2]], "--", color="0.6", lw=1.0, zorder=2)
        ax.scatter(p[:, 0], p[:, 2], s=(W.质量 * 3.2 + 30), color="tab:orange", zorder=4)
        ax.scatter(p[贴地, 0], p[贴地, 2], s=170, facecolors="none", edgecolors="tab:green", lw=2.2, zorder=5)
        ax.scatter(p[W.序号["头"], 0], p[W.序号["头"], 2], s=70, color="tab:red", zorder=6)
        重 = (p[:, 0] * W.质量).sum() / W.质量.sum()
        重z = (p[:, 2] * W.质量).sum() / W.质量.sum()
        ax.scatter([重], [重z], marker="x", s=150, color="k", lw=2.5, zorder=6)
        最快 = int(np.argmax(np.linalg.norm(w.速度, axis=1)))
        ax.set_title("侧视  t=%.2fs  站直%.2f  重心高%.2f  最快：%s %.1f 米/秒"
                     % (w.时间, w.站直程度(), 重z, W.点名[最快], np.linalg.norm(w.速度, axis=1).max()),
                     fontsize=10)
        ax.set_aspect("equal")
        中 = 0.5 * (p[:, 0].min() + p[:, 0].max())
        ax.set_xlim(中 - 1.5, 中 + 1.5)
        ax.set_ylim(-0.25, 2.0)
        ax.grid(alpha=0.2)

    def _画角(self):
        ax = self.轴角
        ax.clear()
        想 = self.w.关节
        实 = self.w.量关节()
        低 = W.节最小
        高 = W.节最大
        名 = W.关节名[::-1]
        想 = 想[::-1]; 实 = 实[::-1]; 低 = 低[::-1]; 高 = 高[::-1]
        y = np.arange(len(名))
        ax.barh(y, (低 - 低) + (高 - 低), left=低, height=0.75, color="0.9", zorder=1)
        ax.scatter(想, y, marker="|", s=420, color="0.35", lw=3, zorder=3, label="想要")
        ax.scatter(实, y, s=55, color="tab:blue", zorder=4, label="实际")
        ax.set_yticks(y)
        ax.set_yticklabels(名, fontsize=10)
        ax.set_xlim(0, 190)
        ax.set_xlabel("角度（度）", fontsize=10)
        ax.set_title("关节：灰竖线=想要的角度，蓝点=实际角度", fontsize=10)
        ax.grid(alpha=0.2, axis="x")
        ax.legend(fontsize=7, loc="lower left")

    def _画眼(self):
        ax = self.轴眼
        ax.clear()
        图 = self.w.看()
        放大 = np.kron(图, np.ones((10, 10, 1), dtype=np.uint8))
        ax.imshow(放大, interpolation="nearest")
        ax.set_title("它眼睛看到的（%dx%d 马赛克放大）" % (W.世界().画面列, 图.shape[0]), fontsize=10)
        ax.axis("off")

    def 画一帧(self, k):
        self._推进一步()
        self._画身()
        self._画角()
        self._画眼()
        return []


def main():
    参数 = [a for a in sys.argv[1:] if a != "录"]
    场景 = 参数[0] if 参数 else "站着"
    录像 = "录" in sys.argv
    看 = 看身体(场景)

    if 录像:
        帧数 = int(8 * 实时比例)
        print("录 %d 帧 ..." % 帧数)
        出 = []
        for k in range(帧数):
            看.画一帧(k)
            看.图.canvas.draw()
            出.append(np.asarray(看.图.canvas.buffer_rgba())[:, :, :3].copy())
            if (k + 1) % 60 == 0:
                print("  %d/%d  t=%.2fs" % (k + 1, 帧数, 看.w.时间))
        from PIL import Image
        路径 = 根 / ("工具_看身体_%s.gif" % 场景)
        Image.fromarray(np.asarray(出[0])).save(
            路径, save_all=True, append_images=[Image.fromarray(f) for f in 出[1:]],
            duration=int(1000 / 实时比例), loop=0)
        print("录像存到", 路径)
        print("结束状态：", 看.w.摘要())

    print("打开窗口（关掉窗口就退出）")
    动画.FuncAnimation(看.图, 看.画一帧, interval=int(1000 / 实时比例), blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
