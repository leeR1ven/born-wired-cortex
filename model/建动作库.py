# -*- coding: utf-8 -*-
"""建动作库.py —— 把三处动作合并成一份「动作库.json」。

  · 手编的静态姿势（10 个）和慢动作序列（4 个）  ← 动作库_go2.py
  · 从训练好的模型抄下来的步态（13 个）          ← 动作库_训练好的.json

动作库.json 的每一条：
    名字 -> {"拍": [[12 块肌肉的发力], ...], "循环": 1 或 0, "来源": "..."}

  「拍」= 大脑一拍（1/30 秒）的动作。1 拍 = 一个固定姿势；
        多拍 = 一串动作（循环=1 会自己转圈接着放，循环=0 放完就停）。
"""
from __future__ import annotations

import json
import pathlib

根 = pathlib.Path(__file__).resolve().parent


def 建(说=True):
    import 动作库_go2 as 手编
    动作 = {}
    for 名, 发 in 手编.静态.items():
        # 姿势 = 一个在原地转圈的 2 拍动作（两拍一模一样）。
        # 为什么要 2 拍：皮层里"保持一个姿势"必须是一小圈自己点自己的神经元，
        # 只点一下是保持不住的（神经元不会一直亮着）。这也正是真脑里"持续放电"的样子。
        p = [round(float(v), 3) for v in 发]
        动作[名] = {"拍": [p, p], "循环": 1, "来源": "手编", "说明": "固定姿势"}
    收尾 = {"站起来": "站姿", "趴下去": "趴姿", "坐下": "坐"}
    for 名, 造 in 手编.序列.items():
        拍 = [[round(float(v), 3) for v in 发] for 发 in 造()]
        动作[名] = {"拍": 拍, "循环": 0, "来源": "手编",
                    "说明": "慢动作，做完接着保持「%s」" % 收尾.get(名, "")}
        if 名 in 收尾:
            动作[名]["接"] = 收尾[名]
    训练文件 = 根 / "动作库_训练好的.json"
    训练 = json.loads(训练文件.read_text(encoding="utf-8")) if 训练文件.exists() else {}
    丢掉 = 0
    for 名, v in 训练.items():
        if v.get("最低机身", 1.0) <= 0.20:
            丢掉 += 1                    # 老师自己在这个指令下就站不住，抄来的不能要
            continue
        动作[名] = {"拍": v["发力"], "循环": 1, "来源": "抄训练好的模型",
                    "说明": "指令 %s，步态周期 %.2f 秒" % (v["指令"], v["周期秒"])}
    if 说 and 丢掉:
        print("（抄下来的里面有 %d 个是老师自己也没站住的，没进库）" % 丢掉)
    (根 / "动作库.json").write_text(json.dumps(动作, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    if 说:
        拍总数 = sum(len(v["拍"]) for v in 动作.values())
        print("动作库.json 建好：%d 个动作，一共 %d 拍" % (len(动作), 拍总数))
        for 名, v in 动作.items():
            print("   %-10s %2d 拍  %s  %s" % (名, len(v["拍"]),
                  "会转圈" if v["循环"] else "放完停", v["说明"]))
    return 动作


if __name__ == "__main__":
    建()
