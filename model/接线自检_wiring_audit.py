"""接线自检_wiring_audit.py —— 跑一遍模型，报告哪些部件根本没被用到

用途：防止"功能写了但没接上"。程序跑起来一切正常，可那段代码从没执行过，
出了问题也发现不了。本工具用运行时计数代替人工核对。

计数分两个阶段，都算数：
  导入阶段：模块加载时执行的（例如神经网络出生时的输出校准）
  运行阶段：闭环主流程执行时调用的

实现要点：用 builtins.__build_class__ 钩子在"类被创建的那一刻"就把计数器装上，
所以模块顶层直接 new 出来的单例、以及导入时调用的方法都能被统计到。
（只在导入之后才装计数器会漏掉这一类，这是上一版的错误。）

用法：python 接线自检_wiring_audit.py [帧数]      默认 8 帧
"""
from __future__ import annotations
import pathlib

import builtins
import collections
import importlib
import os
import sys
import time
import types

import numpy as np

帧数 = int(sys.argv[1]) if len(sys.argv) > 1 else 8

底层模块名 = [
    "海马体时间区_hippocampal_time",
    "视觉前处理_visual_preprocess",
    "听觉前处理_auditory_preprocess",
    "运动输出区_motor_output",
    "前额叶区_prefrontal",
    "视觉记忆区_visual_memory",
    "听觉记忆区_auditory_memory",
    "运动记忆区_motor_memory",
    "权重连接管理_weight_manager",
    "皮层连接_cortex_links",
    "本能工具_instincts",
    "身体_go2",
]
主流程模块名 = "主循环_完整的一拍"      # 现在这条主线：时钟 + 三感觉 + 视/听 + 前额叶 + 返回线
                                      # + 运动记忆 + 皮层步进 + 赫布学习（2026-09-15 起）

计数 = collections.Counter()      # (模块, 部件, 名字) -> 次数
阶段次数 = collections.Counter()   # 同上，但只统计导入阶段
种类 = {}                         # (模块, 部件) -> "类"/"模块函数"
成员种类 = {}                     # (模块, 部件, 方法) -> "方法"
目标模块 = set(底层模块名)


def 包一层(键, 函数):
    def 包装(*a, **kw):
        计数[键] += 1
        return 函数(*a, **kw)
    包装.__name__ = getattr(函数, "__name__", 键[-1])
    包装.__doc__ = getattr(函数, "__doc__", None)
    包装.__wrapped__ = 函数
    return 包装


def 给类装计数器(类):
    模块名 = 类.__module__
    if 模块名 not in 目标模块:
        return
    部件 = 类.__name__
    种类[(模块名, 部件)] = "类"
    for 名字, 成员 in list(vars(类).items()):
        if 名字.startswith("__") and 名字.endswith("__"):
            continue
        if isinstance(成员, (staticmethod, classmethod)):
            内层 = 成员.__func__
            if isinstance(内层, types.FunctionType):
                成员种类[(模块名, 部件, 名字)] = "方法"
                setattr(类, 名字, type(成员)(包一层((模块名, 部件, 名字), 内层)))
        elif isinstance(成员, types.FunctionType):
            成员种类[(模块名, 部件, 名字)] = "方法"
            setattr(类, 名字, 包一层((模块名, 部件, 名字), 成员))
    if isinstance(vars(类).get("__init__"), types.FunctionType):
        setattr(类, "__init__", 包一层((模块名, 部件, "（实例化）"), 类.__init__))


_原build_class = builtins.__build_class__


def _挂钩build_class(func, name, *bases, **kw):
    类 = _原build_class(func, name, *bases, **kw)
    try:
        给类装计数器(类)
    except Exception as 错:
        print(f"  [警告] 给类 {name} 装计数器失败: {错}")
    return 类


def 装模块级函数计数器(模块名, 模块):
    for 名字, 成员 in list(vars(模块).items()):
        if 名字.startswith("_") or isinstance(成员, type):
            continue
        if isinstance(成员, types.FunctionType) and 成员.__module__ == 模块名:
            种类[(模块名, 名字)] = "模块函数"
            setattr(模块, 名字, 包一层((模块名, 名字, "（调用）"), 成员))


def 主流程():
    print(f"接线自检：给 {len(底层模块名)} 个底层模块装计数器，再运行闭环主流程（{帧数} 帧）")
    print("-" * 100)
    builtins.__build_class__ = _挂钩build_class
    模块们 = {}
    for 名 in 底层模块名:
        模块们[名] = importlib.import_module(名)
    builtins.__build_class__ = _原build_class
    for 名, 模块 in 模块们.items():
        装模块级函数计数器(名, 模块)
    阶段次数.update(计数)          # 到此为止 = 导入阶段

    单例们 = []
    for 名, 模块 in 模块们.items():
        for 候选 in ("网", "池", "时钟", "前额网", "返回线"):
            值 = getattr(模块, 候选, None)
            if 值 is not None and not isinstance(值, (int, float, str, bool)):
                单例们.append((名, 候选, type(值).__name__))

    sys.argv = [主流程模块名, "不要学"]
    t0 = time.perf_counter()
    主流程 = importlib.import_module(主流程模块名)
    if hasattr(主流程, "主流程"):                 # 老式闭环脚本
        主流程.主流程()
    else:                                        # 主循环_完整的一拍：手动走几拍，好控制帧数
        大脑 = 主流程.脑(学=True, 说=False)
        身 = 主流程.身体.身体()
        身.摆成站姿()
        身.走(0.5, None)
        图 = 主流程.红图("中")
        声 = np.zeros(4096)
        声[100:600] = 1.0                       # 一段低频声音，把听觉那一路也走一遍
        for _ in range(帧数):
            大脑.一拍(图=图, 声=声, 身=身)
            身.步进(大脑.肌肉发力())
    用时 = time.perf_counter() - t0

    按部件 = collections.defaultdict(collections.Counter)
    for (模块名, 部件, 名字), 次 in 计数.items():
        按部件[(模块名, 部件)][名字] += 次

    print()
    print("=" * 100)
    print(f"接线自检报告（主流程实跑 {用时:.1f} 秒）")
    print("=" * 100)
    print("\n【一】每个类/模块函数被调用的次数（导入阶段 / 运行阶段分开看）")
    for 模块名 in 底层模块名:
        条目 = [(k, v) for k, v in 按部件.items() if k[0] == 模块名]
        总 = sum(sum(v.values()) for _, v in 条目)
        print(f"\n  ── {模块名}  （合计 {总:,} 次）")
        if not 条目:
            print("       （没有进入这个模块）")
            continue
        for (_, 部件), 次 in sorted(条目, key=lambda x: -sum(x[1].values())):
            导入次 = sum(v for (m, p, n), v in 阶段次数.items() if m == 模块名 and p == 部件)
            运行次 = sum(次.values()) - 导入次
            明细 = ", ".join(f"{n}×{c}" for n, c in sorted(次.items(), key=lambda x: -x[1])[:5])
            print(f"     {部件:22s} 导入 {导入次:6,d} | 运行 {运行次:8,d}   {明细}")

    print("\n【二】加载时就存在的单例")
    for 名, 属性, 类型 in 单例们:
        print(f"     {名}.{属性}  ->  {类型}")

    从没碰过的 = []
    for 键 in sorted(种类):
        模块名, 部件 = 键
        if 键 not in 按部件 or sum(按部件[键].values()) == 0:
            从没碰过的.append(f"{模块名}.{部件}")
    从没跑过的方法 = []
    for (模块名, 部件, 方法) in sorted(成员种类):
        if 计数.get((模块名, 部件, 方法), 0) == 0:
            从没跑过的方法.append(f"{模块名}.{部件}.{方法}")

    print("\n【三】整个部件（类 / 模块函数）从头到尾没被碰到")
    print("     无。" if not 从没碰过的 else "")
    for 名 in 从没碰过的:
        print(f"     ✗ {名}")
    print("\n【四】部件被用到了，但它里面这些方法一次都没执行")
    print("     无。" if not 从没跑过的方法 else "")
    for 名 in 从没跑过的方法:
        print(f"     ✗ {名}")
    print()
    return 从没碰过的, 从没跑过的方法


if __name__ == "__main__":
    os.chdir(str(pathlib.Path(__file__).resolve().parent))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    主流程()