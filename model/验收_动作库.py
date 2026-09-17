# -*- coding: utf-8 -*-
"""验收_动作库.py —— 逐个动作验一遍：只点亮头一拍，看身体到底做出来没有。

判定标准（都很直白，没有"打分"）：
  · 撑住 4 秒没倒        = 机身没掉到 0.18 米以下、也没歪过 60 度
  · 走了多远 / 转了多少度 = 位置和朝向真的变了没有
"""
from __future__ import annotations
import pathlib

import json
import numpy as np

import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能

帧秒 = 1.0 / 50.0


起手 = {'站起来': '趴姿'}


def 试一个(网, 名字, 动作名, 秒=4.0):
    前 = 网.区起点["运动记忆"]
    运起 = 网.区起点["运动"]
    身 = 身体.身体()
    if 起手.get(动作名, '站姿') == '趴姿':
        身.摆成趴姿()
        身.走(0.6, None)
    else:
        身.摆成站姿()
        身.走(0.5, None)
    亮 = np.zeros(网.总数, dtype=bool)
    亮[前 + np.array(名字["运动记忆"][动作名], dtype=int)] = True   # 只点一拍
    起 = 身.数据.qpos[:2].copy()
    起向 = float(np.arctan2(2 * (身.数据.qpos[3] * 身.数据.qpos[6]
                                + 身.数据.qpos[4] * 身.数据.qpos[5]),
                            1 - 2 * (身.数据.qpos[4] ** 2 + 身.数据.qpos[5] ** 2)))
    撑 = 0
    最高 = 0.0
    for k in range(int(秒 / 帧秒)):
        出 = 网.步进(亮)
        发力 = 运动.解码发力(出[运起:运起 + 运动.运动区宽度])
        身.步进(发力)
        亮 = 出
        最高 = max(最高, 身.机身高度())
        if 身.机身歪了() < 75:          # 摔 = 歪过 75 度（躺倒/翻过去），不是矮
            撑 = k
        else:
            break
    p = 身.数据.qpos
    w, x, y, z = p[3:7]
    末向 = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    转 = np.degrees(末向 - 起向)
    转 = (转 + 180) % 360 - 180
    return dict(撑住秒=撑 * 帧秒, 走了米=float(np.hypot(p[0] - 起[0], p[1] - 起[1])),
                转角度=float(转), 最高机身=float(最高), 末高=身.机身高度())


def main():
    名字 = 本能.读名字()
    网, _ = 本能.装(说=False)
    动作库 = 本能.读动作库()
    print("%-10s %6s %8s %8s %8s  %s" % ("动作", "拍数", "撑住", "走了", "转了", "结论"))
    print("-" * 66)
    行 = {}
    for 名, v in 动作库.items():
        r = 试一个(网, 名字, 名)
        行[名] = r
        if 名 in ("慢走", "中走", "快走", "疾跑", "后退", "左横移", "右横移",
                      "左转", "右转", "原地踏步", "斜着走", "边走边左转", "边走边右转"):
            结果 = "走/转出来了" if r["撑住秒"] >= 3.9 else "走一半摔了"
        elif 名 in ("站起来", "趴下去", "坐下"):
            结果 = "做完并接上了" if r["撑住秒"] >= 3.9 else "做一半摔了"
        else:
            结果 = "摆出来了" if r["撑住秒"] >= 3.9 else "没稳住"
        print("%-10s %6d %7.1f秒 %7.2f米 %7.1f度  %s"
              % (名, len(v["拍"]), r["撑住秒"], r["走了米"], r["转角度"], 结果))
    json.dump(行, open(str(pathlib.Path(__file__).resolve().parent / "验收_动作库.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()