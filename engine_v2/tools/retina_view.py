# -*- coding: utf-8 -*-
"""把视网膜那张表还原成均匀画面 —— 只为了给人看。

表是「画面中段格子密、边缘格子疏」的非均匀采样：每个格子是它自己那一小块画面的平均。
直接当成均匀网格画出来，中段被放大、边缘被压扭，看着像画面畸变；按每个格子实际盖住的
画面位置还原，才是这只眼真正看到的画面 —— 中段清楚、边缘糊，和几何无关。
"""
import numpy as np


def as_seen(eyes, sheet):
    """视网膜表 -> 均匀画面（画出「这只眼看到的」，形状 (drawn高, drawn宽, 3)）。"""
    low_rows, high_rows = (np.asarray(part) for part in eyes._rows)
    low_columns, high_columns = (np.asarray(part) for part in eyes._columns)
    drawn_h, drawn_w = int(high_rows[-1]), int(high_columns[-1])
    which_rows = np.clip(np.searchsorted(low_rows, np.arange(drawn_h), side="right") - 1,
                         0, eyes.height - 1)
    which_columns = np.clip(np.searchsorted(low_columns, np.arange(drawn_w), side="right") - 1,
                            0, eyes.width - 1)
    return np.asarray(sheet)[np.ix_(which_rows, which_columns)]


def picture(eyes, sheet, scale):
    """还原并放大，外加一圈黄框，好让人一眼看出哪块是眼睛画面。"""
    seen = as_seen(eyes, sheet)
    tile = np.repeat(np.repeat(seen, scale, 0), scale, 1).copy()
    tile[:2, :] = tile[-2:, :] = (255, 255, 0)
    tile[:, :2] = tile[:, -2:] = (255, 255, 0)
    return tile