# -*- coding: utf-8 -*-
"""规模试验_测量.py —— 只量：建网多久、一拍多久、占多少内存。不跑行为，不看画面。

用法：
    python 规模试验_测量.py                 # 当前大小
    AGI列数=32 python ...                   # 放大（Windows 上用 $env:AGI列数=32）
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import time

import numpy as np

import 本能工具_instincts as 本能
import 皮层连接_cortex_links as 皮层


class 内存计数(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


# ★ 2026-09-16 修的坑：不声明 argtypes 时，ctypes 会把 GetCurrentProcess() 返回的
#   64 位句柄按 32 位传进去 -> 截断 -> 调用失败 -> 读出来永远是 0 MB（实测）。
#   那次"建网后内存 0 MB"就是这么来的，不是真的不占内存。
_k = ctypes.windll.kernel32
_p = ctypes.windll.psapi
_取内存 = getattr(_k, "K32GetProcessMemoryInfo", None) or _p.GetProcessMemoryInfo
_取内存.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(内存计数), ctypes.wintypes.DWORD]
_取内存.restype = ctypes.wintypes.BOOL


def 内存():
    c = 内存计数()
    c.cb = ctypes.sizeof(c)
    好 = _取内存(_k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not 好:
        raise OSError("读进程内存失败")
    return c.WorkingSetSize / 1048576.0, c.PeakWorkingSetSize / 1048576.0


def 系统内存():
    """整机物理内存 (已用 GB, 总共 GB)"""
    class 状态(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.wintypes.DWORD), ("dwMemoryLoad", ctypes.wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = 状态()
    st.dwLength = ctypes.sizeof(st)
    if not _k.GlobalMemoryStatusEx(ctypes.byref(st)):
        return 0.0, 0.0
    return (st.ullTotalPhys - st.ullAvailPhys) / 2 ** 30, st.ullTotalPhys / 2 ** 30


行 = 本能.视觉.大像素行数
列 = 本能.视觉.大像素列数
宽 = 本能.各区宽度
总 = sum(宽.values())
已用GB, 整机GB = 系统内存()
print("=" * 70)
print("整机物理内存：%.1f GB / %.1f GB（这个进程还没开始吃）" % (已用GB, 整机GB))
print("规模：画面 %d x %d 格；视觉 %d、视觉细节 %d、听觉 %d、前额叶 %d"
      % (行, 列, 宽["视觉"], 宽["视觉细节"], 宽["听觉"], 宽["前额叶"]))
print("皮层神经元总数 %d" % 总)

t0 = time.time()
网 = 本能.建网()
建网秒 = time.time() - t0
兴奋边 = int(网._兴奋权.size)
抑制边 = int(网._抑制权.size)
print("建网 %.1f 秒；兴奋连接 %d 根、抑制连接 %d 根" % (建网秒, 兴奋边, 抑制边))
现, 峰 = 内存()
print("建网后内存 %.0f MB（峰值 %.0f MB）" % (现, 峰))

rng = np.random.default_rng(1)
亮 = np.zeros(总, dtype=bool)
亮[rng.choice(总, max(1, 总 // 50), replace=False)] = True     # 2% 亮着，接近真实

拍数 = 30
t0 = time.time()
for _ in range(拍数):
    亮 = np.asarray(网.步进(亮), dtype=bool)
步进秒 = time.time() - t0
现, 峰 = 内存()
print("步进 %d 拍 %.1f 秒 = 一拍 %.3f 秒（%.1f 拍/秒）" % (拍数, 步进秒, 步进秒 / 拍数, 拍数 / 步进秒))
print("跑完内存 %.0f MB（峰值 %.0f MB）" % (现, 峰))

t0 = time.time()
上 = 亮.copy()
for _ in range(10):
    这 = np.asarray(网.步进(上), dtype=bool)
    网.学习(上, 这)
    上 = 这
学秒 = time.time() - t0
现, 峰 = 内存()
print("带学习 %d 拍 %.1f 秒 = 一拍 %.3f 秒" % (10, 学秒, 学秒 / 10))
print("最终内存 %.0f MB（峰值 %.0f MB）" % (现, 峰))