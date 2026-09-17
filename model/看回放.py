# -*- coding: utf-8 -*-
"""看回放.py —— 把一段动作录成「双击就能看」的文件，不依赖弹窗。

为什么要这个：这台机器上 matplotlib 的弹窗显示是空白的（Tk 那边的问题），
但图本身是好的。所以干脆不出窗口，直接把每一帧存下来，做成：
    回放_<名字>.gif    ← 用「照片」查看器打开就行
    回放_<名字>.html   ← 用浏览器打开，能播放 / 暂停 / 拖进度条

用法：
    python 看回放.py            # 录当前这套动作
    python 看回放.py 站着 3     # 站着不动录 3 秒
动作写在下面的 动作表 里（就是"关节想要多少度、保持多久"）。
"""
from __future__ import annotations

import base64
import io
import pathlib
import sys

import numpy as np

根 = pathlib.Path(__file__).resolve().parent
if str(根) not in sys.path:
    sys.path.insert(0, str(根))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import 世界_world as W

帧率 = 30                 # 每秒画几帧（录出来的节奏）
物理每帧 = 8              # 每画一帧，物理走几步（240/8 = 30）
画布尺寸 = (12.0, 4.6)    # 英寸
分辨率 = 125


def 取序列(名):
    """返回 [(说明, 目标角度 dict 或 None, 走多少秒), ...]"""
    名 = 名 or "起立"
    if 名 == "站着":
        return [("站着", dict(zip(W.关节名, W.节站立角)), 3.0)]
    if 名 == "躺下":
        w = W.世界()
        return [("躺下", dict(zip(W.关节名, w.量关节())), 3.0)]
    # 默认：趴着起身（目前搜出来的那一套）
    A = dict(zip(W.关节名, [129, 161, 86, 118, 83, 83, 126, 170, 91]))
    B = dict(zip(W.关节名, [79, 143, 180, 176, 65, 65, 160, 156, 122]))
    return [
        ("0 趴着", None, 0.4),
        ("1 折小腿", {"左膝": 60, "右膝": 60}, 0.8),
        ("2 大腿收到肚子下（四肢着地）", {"左髋": 70, "右髋": 70}, 1.2),
        ("3 坐回脚跟", {"左膝": 45, "右膝": 45, "左髋": 60, "右髋": 60, "左踝": 130, "右踝": 130}, 1.0),
        ("4 挺直上身（跪直）", {"左膝": 45, "右膝": 45, "左髋": 175, "右髋": 175, "左踝": 130, "右踝": 130}, 1.0),
        ("5 起身（上）", A, 0.7),
        ("6 起身（下）", B, 1.3),
        ("7 站住", None, 2.0),
    ]


def 一帧(轴, w, 标题):
    p = w.位置
    轴[0].clear(); 轴[1].clear(); 轴[2].clear()
    a = 轴[0]
    a.axhspan(-0.15, W.最小离地, color="0.78", zorder=0)
    a.axhline(W.最小离地, color="0.45", lw=1, zorder=1)
    for x, y in W.骨架:
        i, j = W.序号[x], W.序号[y]
        a.plot([p[i, 0], p[j, 0]], [p[i, 2], p[j, 2]], "-", color="tab:blue", lw=3.4, zorder=3)
    a.scatter(p[:, 0], p[:, 2], s=(W.质量 * 3 + 26), color="tab:orange", zorder=4)
    贴 = p[:, 2] <= W.最小离地 + 1e-3
    a.scatter(p[贴, 0], p[贴, 2], s=150, facecolors="none", edgecolors="tab:green", lw=2, zorder=5)
    心 = (p[:, 0] * W.质量).sum() / W.质量.sum()
    a.scatter([心], [(p[:, 2] * W.质量).sum() / W.质量.sum()], marker="x", s=130, color="k", lw=2.4, zorder=6)
    a.set_aspect("equal")
    中 = 0.5 * (p[:, 0].min() + p[:, 0].max())
    a.set_xlim(中 - 1.25, 中 + 1.25); a.set_ylim(-0.2, 1.85)
    a.set_title("侧视  t=%.1fs  站直%.2f  头高%.2f\n%s" % (w.时间, w.站直程度(), p[0, 2], 标题), fontsize=13)
    a.grid(alpha=0.2)
    a = 轴[1]
    想 = w.关节; 实 = w.量关节()
    y = np.arange(len(W.关节名))
    a.barh(y, W.节最大 - W.节最小, left=W.节最小, height=0.7, color="0.9", zorder=1)
    a.scatter(想, y, marker="|", s=340, color="0.35", lw=3, zorder=3)
    a.scatter(实, y, s=42, color="tab:blue", zorder=4)
    a.set_yticks(y); a.set_yticklabels(W.关节名, fontsize=8)
    a.set_xlim(0, 190); a.set_xlabel("角度（度）", fontsize=8)
    a.set_title("灰竖线=想要的，蓝点=实际的", fontsize=9)
    a.grid(alpha=0.2, axis="x")
    a = 轴[2]
    图 = w.看()
    a.imshow(np.kron(图, np.ones((12, 12, 1), dtype=np.uint8)), interpolation="nearest")
    a.set_title("他眼睛看到的", fontsize=9); a.axis("off")


def 录(名="起立"):
    序列 = 取序列(名)
    w = W.世界()
    if 名 != "站着":
        w.摆成躺着(脸朝上=False)
    图, 轴 = plt.subplots(1, 3, figsize=画布尺寸, gridspec_kw={"width_ratios": [2.3, 1.0, 1.0], "wspace": 0.22})
    出, 标题们 = [], []
    for 说明, 角, 秒 in 序列:
        if 角 is not None:
            w.写关节(角)
        for _ in range(max(1, int(秒 * 帧率))):
            for _ in range(物理每帧):
                w.步进()
            一帧(轴, w, 说明)
            图.canvas.draw()
            缓冲 = io.BytesIO()
            图.savefig(缓冲, format="png", dpi=分辨率, bbox_inches="tight")
            出.append(缓冲.getvalue())
            标题们.append(说明)
    plt.close(图)

    gif路 = 根 / ("回放_%s.gif" % 名)
    from PIL import Image
    Image.open(io.BytesIO(出[0])).convert("RGB").save(
        gif路, save_all=True,
        append_images=[Image.open(io.BytesIO(b)).convert("RGB") for b in 出[1:]],
        duration=int(1000 / 帧率), loop=0, optimize=True)

    html路 = 根 / ("回放_%s.html" % 名)
    b64 = [base64.b64encode(b).decode() for b in 出]
    html路.write_text("""<!doctype html><meta charset="utf-8">
<title>回放 %s</title>
<style>
 body{margin:0;background:#111;color:#eee;font:14px/1.5 "Microsoft YaHei",sans-serif;text-align:center}
 #框{margin:8px auto;background:#fff;display:inline-block;line-height:0}
 img{max-width:96vw}
 .条{margin:10px}
 input[type=range]{width:60vw;vertical-align:middle}
 button{font-size:16px;padding:6px 16px;margin:0 6px;border-radius:6px;border:0;cursor:pointer}
</style>
<div class="条"><button id="播">▶ 播放</button><span id="说明"></span>　<span id="计"></span></div>
<div id="框"><img id="图"></div>
<div class="条"><input type="range" id="滑" min="0" max="%d" value="0"></div>
<script>
const 帧=%s, 题=%s, 总=帧.length; let i=0, 开=false, 计时=null;
const 图=document.getElementById("图"), 滑=document.getElementById("滑"),
      按=document.getElementById("播"), 说=document.getElementById("说明"), 计=document.getElementById("计");
function 画(){图.src="data:image/png;base64,+帧[i];滑.value=i;说.textContent=" "+题[i];计.textContent=(i+1)+" / "+总;}
滑.oninput=()=>{i=+滑.value;画();};
按.onclick=()=>{开=!开;按.textContent=开?"⏸ 暂停":"▶ 播放";
  if(开){计时=setInterval(()=>{i=(i+1)%%总;画();},%d);}else{clearInterval(计时);}};
画();
</script>""" % (名, len(出) - 1, "[" + ",".join('"%s"' % t for t in 标题们) + "]",
                  "[" + ",".join('"%s"' % b for b in b64) + "]", 1000 // 帧率),
        encoding="utf-8", newline="\n")
    print("录了 %d 帧（%.1f 秒）" % (len(出), len(出) / 帧率))
    print("  双击看：", gif路)
    print("  浏览器看：", html路)


if __name__ == "__main__":
    录(sys.argv[1] if len(sys.argv) > 1 else "起立")