# -*- coding: utf-8 -*-
"""看_大脑开车.py —— 让**皮层模型自己**开一遍车，把每一拍录成能看的文件。

和 看回放.py 的区别（重要）：那边是**人**直接摆关节角，这边是：
  外界信号 -> 感觉皮层 -> 整片皮层（143,796 个神经元 + 308,809 根本能连接）-> 12 块肌肉
代码里**没有任何一行**直接摆姿势或写动作。身体动多少，完全看皮层发出什么。

四幕（每一幕只换"外界信号"这一件事）：
  A 趴着，不给画面        -> 靠身体感觉知道自己趴着 -> 自己站起来
  B 站着，正前方一片红    -> 朝红走过去
  C 站着，黑屏            -> 一步不走
  D 什么都不给            -> 肌肉松掉，瘫下去

用法：
    python 看_大脑开车.py            # 录全部四幕
    python 看_大脑开车.py B          # 只录 B
"""
from __future__ import annotations

import base64
import io
import os
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

根 = pathlib.Path(__file__).resolve().parent
if str(根) not in sys.path:
    sys.path.insert(0, str(根))
os.environ.setdefault("前额叶不保留", "1")      # 论文 R1-R8 那一档

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

帧秒 = 主循环.帧秒
图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")

字号 = pathlib.Path(r"C:\Windows\Fonts\msyh.ttc")
字号粗 = pathlib.Path(r"C:\Windows\Fonts\msyhbd.ttc")
字体 = lambda n, b=False: ImageFont.truetype(str(字号粗 if b else 字号), n)

版式 = dict(宽=1010, 高=470, 头=34, 左宽=580)
渲染宽, 渲染高 = 580, 390

幕们 = [
    dict(键="A", 名="A  趴着，不给画面", 起点="趴", 图=None, 秒=6.0,
         注="只靠身体感觉：它感觉到自己趴着 → 出生就有的本能让它站起来；站定之后一步没走"),
    dict(键="B", 名="B  站着，正前方一片红", 起点="站", 图="红", 秒=4.0,
         注="眼睛看到红 → 视觉那一半 + 前额叶那一半，两股电流加起来过阈值 → 往前走"),
    dict(键="C", 名="C  只看一眼红，半秒后就把红撤掉", 起点="站", 图="看一眼红", 秒=5.0,
         注="红只亮了半秒，眼前全黑了它还在走 —— 前额叶里那点“想”没有跟着刺激一起消失（论文 R8）"),
    dict(键="D", 名="D  什么都不给", 起点="站", 图=None, 感觉=False, 秒=1.5,
         注="连身体感觉都不给 → 肌肉松掉，瘫下去。说明动作是皮层下的令，不是物理自己动"),
]


def 取图(是哪个, 秒=0.0):
    """给名字或给一个函数（按时间变）都行。"""
    if callable(是哪个):
        return 是哪个(秒)
    if 是哪个 == "红":
        return 红图
    if 是哪个 == "黑":
        return 黑图
    if 是哪个 == "看一眼红":
        return 红图 if 秒 < 0.5 else 黑图
    return None


def 数各区的亮(出, 网):
    行 = []
    for 区, 起 in 网.区起点.items():
        行.append((区, int(np.count_nonzero(出[起:起 + 网.区宽[区]])), 网.区宽[区]))
    return 行


def 画一帧(身, 出, 网, 名, 注, 秒, 起xy, 站起来块, 走路块, 前一块, 眼图):
    W, H, 头, 左宽 = 版式["宽"], 版式["高"], 版式["头"], 版式["左宽"]
    画 = Image.new("RGB", (W, H), (24, 24, 28))
    笔 = ImageDraw.Draw(画)

    笔.rectangle([0, 0, W, 头], fill=(38, 38, 46))
    笔.text((10, 8), 名, font=字体(17, True), fill=(255, 255, 255))
    前进 = float(np.hypot(身.数据.qpos[0] - 起xy[0], 身.数据.qpos[1] - 起xy[1]))
    笔.text((W - 360, 6), "t = %5.2f 秒" % 秒, font=字体(15), fill=(200, 200, 210))
    笔.text((W - 360, 20), "机身 %.3f 米    歪 %2.0f 度    挪了 %.2f 米"
             % (身.机身高度(), 身.机身歪了(), 前进), font=字体(13), fill=(230, 200, 140))

    # ① 三维画面：相机跟着角色走
    看向 = (float(身.数据.qpos[0]), float(身.数据.qpos[1]), 0.30)
    图3 = Image.fromarray(身.画(相机=100.0, 俯角=-12.0, 距离=1.35, 看向=看向))
    画.paste(图3.resize((左宽, 渲染高), Image.LANCZOS), (0, 头))
    笔.text((8, 头 + 渲染高 + 5), 注, font=字体(13), fill=(180, 220, 180))

    # ② 它眼睛看到的
    x0, y0 = 左宽 + 12, 头 + 4
    笔.text((x0, y0), "它眼睛看到的：", font=字体(14, True), fill=(255, 255, 255))
    if 眼图 is None:
        笔.rectangle([x0, y0 + 24, x0 + 372, y0 + 24 + 128], fill=(8, 8, 8), outline=(90, 90, 100))
        笔.text((x0 + 120, y0 + 76), "（没有给画面）", font=字体(14), fill=(140, 140, 150))
    else:
        小 = Image.fromarray(眼图).resize((372, 128), Image.BILINEAR)
        画.paste(小, (x0, y0 + 24))
        笔.rectangle([x0, y0 + 24, x0 + 372, y0 + 24 + 128], outline=(90, 90, 100))

    # ③ 皮层活动条
    y = y0 + 184
    笔.text((x0, y), "皮层亮着的细胞：", font=字体(14, True), fill=(255, 255, 255))
    y += 22
    区行 = 数各区的亮(出, 网)
    要显 = [("视觉", None), ("听觉", None), ("运动", None), ("运动记忆", None), ("前额叶", None), ("多巴胺", None)]
    显 = {a: b for a, b, c in 区行}
    for 区, _ in 要显:
        if 区 not in 显:
            continue
        总 = dict((a, c) for a, b, c in 区行)[区]
        v = 显[区]
        w = int(248 * min(1.0, v / max(1, 总)))
        笔.text((x0, y - 1), "%-6s" % 区, font=字体(12), fill=(200, 200, 210))
        笔.rectangle([x0 + 62, y + 2, x0 + 62 + 248, y + 14], fill=(55, 55, 65))
        笔.rectangle([x0 + 62, y + 2, x0 + 62 + w, y + 14], fill=(90, 170, 255))
        笔.text((x0 + 318, y - 1), "%6d/%d" % (v, 总), font=字体(11), fill=(190, 190, 200))
        y += 18

    # ④ 两条动作链 —— 运动记忆是"一拍走一格"，所以要报"正放到第几拍"
    y += 6
    前 = int(np.count_nonzero(出[前一块]))

    def 放到第几拍(块):
        ix = np.nonzero(出[块])[0]
        return (int(ix[0]) + 1, len(块)) if ix.size else (0, len(块))

    for 名2, (第拍, 总2) in (("站起来链", 放到第几拍(站起来块)),
                            ("走路链", 放到第几拍(走路块))):
        亮不 = 第拍 > 0
        笔.rectangle([x0, y, x0 + 178, y + 20], fill=(70, 120, 70) if 亮不 else (55, 55, 62))
        笔.text((x0 + 8, y + 3), "%s  第 %d/%d 拍" % (名2, 第拍, 总2), font=字体(12, True),
                fill=(255, 255, 255) if 亮不 else (150, 150, 155))
        y += 24
    笔.text((x0, y + 2), "前额叶亮 %d 个" % 前, font=字体(12), fill=(200, 200, 210))
    return 画


def 录一幕(键=None):
    # ★ 每一幕都从**刚出生的同一颗脑子**开始（同一个种子、同一套出生连线），
    #   这样画面里的数就是论文里的数，不会带着上一幕学出来的东西。
    本能.建网种子 = 20260914
    出表 = []
    for 幕 in 幕们:
        if 键 and 幕["键"] != 键:
            continue
        print("=== %s ===" % 幕["名"], flush=True)
        脑 = 主循环.脑(学=True, 说=False)
        网 = 脑.网
        身 = 身体.身体()
        运记起 = 网.区起点["运动记忆"]
        站起来块 = 运记起 + np.asarray(运动记忆.时刻们("站起来"), dtype=np.int64)
        走路块 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
        前一块 = 脑.段["前额叶"]
        身.摆成趴姿() if 幕["起点"] == "趴" else 身.摆成站姿()
        身.走(0.5, None)
        脑.亮[:] = False
        脑.上一拍[:] = False
        起xy = 身.数据.qpos[:2].copy()
        用感觉 = 幕.get("感觉", True)
        帧们, 标们 = [], []
        t0 = time.time()
        总拍 = int(幕["秒"] / 帧秒)
        for k in range(总拍):
            图 = 取图(幕["图"], k * 帧秒)
            出 = 脑.一拍(图=图, 声=None, 身=(身 if 用感觉 else None))
            身.步进(脑.肌肉发力())
            if k % 2 == 0:
                画 = 画一帧(身, 出, 网, 幕["名"], 幕["注"], k * 帧秒, 起xy,
                            站起来块, 走路块, 前一块, 图)
                帧们.append(画)
                标们.append("%s  t=%.2fs" % (幕["名"], k * 帧秒))
        前进 = float(np.hypot(身.数据.qpos[0] - 起xy[0], 身.数据.qpos[1] - 起xy[1]))
        print("   %s：末高 %.3f 米、挪了 %.2f 米、歪 %.0f 度（%.0f 秒录完）"
              % (幕["名"], 身.机身高度(), 前进, 身.机身歪了(), time.time() - t0), flush=True)
        出表.append(dict(幕=幕, 帧们=帧们, 标们=标们, 末高=身.机身高度(), 前进=前进,
                         歪=身.机身歪了(), 用时=time.time() - t0))
    return 出表


def 写文件(出表):
    帧率 = 25
    for r in 出表:
        名 = r["幕"]["键"]
        gif路 = 根.parent / "playback" / ("回放_大脑_%s.gif" % 名)
        P = [f.convert("P", palette=Image.ADAPTIVE, colors=160) for f in r["帧们"]]
        P[0].save(gif路, save_all=True, append_images=P[1:],
                  duration=int(1000 / 帧率), loop=0, optimize=True)
        b64 = [base64.b64encode(_png(f).getvalue()).decode() for f in r["帧们"]]
        html = """<!doctype html><meta charset="utf-8"><title>%s</title>
<style>body{margin:0;background:#111;color:#eee;font:15px/1.6 "Microsoft YaHei",sans-serif;text-align:center}
img{max-width:97vw}#框{margin:10px auto}button{font-size:16px;padding:6px 18px;margin:0 6px;border:0;border-radius:6px;cursor:pointer}
input[type=range]{width:70vw;vertical-align:middle}</style>
<h2>%s</h2><p>%s</p>
<div><button id="播">▶ 播放</button> <span id="计"></span></div>
<div id="框"><img id="图"></div>
<div><input type="range" id="滑" min="0" max="%d" value="0"></div>
<script>const 帧=%s,总=帧.length;let i=0,开=false,计=null;
const 图=document.getElementById("图"),滑=document.getElementById("滑"),按=document.getElementById("播"),计数=document.getElementById("计");
function 画(){图.src="data:image/png;base64," + 帧[i];滑.value=i;计数.textContent=(i+1)+" / "+总;}
滑.oninput=()=>{i=+滑.value;画();};
按.onclick=()=>{开=!开;按.textContent=开?"⏸ 暂停":"▶ 播放";
 if(开){计=setInterval(()=>{i=(i+1)%%总;画();},%d);}else{clearInterval(计);}};
画();</script>""" % (r["幕"]["名"], r["幕"]["名"], r["幕"]["注"], len(r["帧们"]) - 1,
                     "[" + ",".join('"%s"' % b for b in b64) + "]", 1000 // 帧率)
        (根.parent / "playback" / ("回放_大脑_%s.html" % 名)).write_text(html, encoding="utf-8", newline="\n")
        print("  写好 %s（%.1f MB）和 回放_大脑_%s.html" % (gif路.name, gif路.stat().st_size / 1e6, 名))


def _png(im):
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b


if __name__ == "__main__":
    键 = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None
    出表 = 录一幕(键)
    写文件(出表)