# -*- coding: utf-8 -*-
"""出一张图_动作库.py —— 把"想一下就能做出来的动作"画成图，人看。

跑完会出两张：
  动作库_全景.png   一个动作一格，看得见狗真的摆出来了
  动作库_站起来.gif 完整过程（双击就能看）
"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能

帧秒 = 1.0 / 50.0
宽, 高 = 480, 340


def 准备():
    名字 = 本能.读名字()
    网, _ = 本能.装(说=False)
    return 网, 名字, 网.区起点


def 摆(网, 名字, 身, 想什么=None, 秒=1.0, 方式="一直", 存=None, 每几拍=3):
    """跑一段；存 是列表就每 每几拍 拍存一张（给 GIF 用）。"""
    前 = 网.区起点["运动记忆"]
    运起 = 网.区起点["运动"]
    亮 = np.zeros(网.总数, dtype=bool)
    if 想什么:
        亮[前 + np.array(名字["运动记忆"][想什么], dtype=int)] = True
    for k in range(int(秒 / 帧秒)):
        出 = 网.步进(亮)
        发力 = 运动.解码发力(出[运起:运起 + 运动.运动区宽度])
        身.步进(发力)
        if 方式 == "一直" and 想什么:
            出[前 + np.array(名字["运动记忆"][想什么], dtype=int)] = True
        亮 = 出
        if 存 is not None and k % 每几拍 == 0:
            存.append(身.画(相机=100.0, 俯角=-14.0, 距离=1.15,
                            看向=(身.数据.qpos[0], 身.数据.qpos[1], 0.26)))
    return 身


def main():
    import pathlib as _pl
    from PIL import ImageDraw, ImageFont
    字体 = None
    for 路径 in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\simsun.ttc"):
        if _pl.Path(路径).exists():
            字体 = ImageFont.truetype(路径, 15)
            break
    网, 名字, 起点 = 准备()
    身 = 身体.身体(渲染宽=宽, 渲染高=高)
    身.摆成站姿()

    print("正在跑各个动作 ...")
    格 = []
    def 快照(标题):
        图 = Image.fromarray(身.画(相机=100.0, 俯角=-14.0, 距离=1.15,
                                   看向=(身.数据.qpos[0], 身.数据.qpos[1], 0.26)))
        d = ImageDraw.Draw(图)
        d.rectangle([0, 0, 480, 26], fill=(0, 0, 0))
        d.text((6, 6), 标题 + "   机身 %.2f 米  歪 %.0f 度"
               % (身.机身高度(), 身.机身歪了()), fill=(255, 255, 255), font=字体)
        格.append(图)

    # 一、先想"趴姿"
    摆(网, 名字, 身, "趴姿", 1.0)
    快照("1 一直想「趴姿」→ 趴下")

    # 二、想一下"站起来"（只点一拍）
    gif = []
    摆(网, 名字, 身, "站起来", 2.4, 方式="一下", 存=gif, 每几拍=2)
    快照("2 想一下「站起来」→ 自己站起来并保持")

    # 三、几个姿势
    for 名, 秒, 标 in (("坐", 0.8, "3 一直想「坐」"), ("蹲", 0.8, "4 一直想「蹲」"),
                      ("抬左前腿", 0.7, "5 一直想「抬左前腿」"),
                      ("站高", 0.8, "6 一直想「站高」")):
        摆(网, 名字, 身, 名, 秒)
        快照(标)

    # 四、想一下"中走"
    摆(网, 名字, 身, "站起来", 1.8, 方式="一下")
    摆(网, 名字, 身, "中走", 1.4, 方式="一下", 存=gif, 每几拍=2)
    快照("7 想一下「中走」→ 步态自己在皮层里转")

    拼 = Image.new("RGB", (宽 * 4, 高 * 2), (255, 255, 255))
    for i, 图 in enumerate(格[:8]):
        拼.paste(图, ((i % 4) * 宽, (i // 4) * 高))
    拼.save(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "动作库_全景.png"))
    print("存了 动作库_全景.png")

    if gif:
        小 = [Image.fromarray(f).resize((320, 227)) for f in gif[::2]]
        小[0].save(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "动作库_站起来.gif"), save_all=True,
                   append_images=小[1:], duration=100, loop=0)
        print("存了 动作库_站起来.gif（%d 帧）" % len(小))


if __name__ == "__main__":
    main()