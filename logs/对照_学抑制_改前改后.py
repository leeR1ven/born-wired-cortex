import pathlib
# -*- coding: utf-8 -*-
"""把旧版文件临时装回去跑一遍、再装回新版跑一遍，比较"真实闭环里的一拍"。"""
import os, shutil, subprocess, sys
py = sys.executable
根 = str(pathlib.Path(__file__).resolve().parent)
文件 = os.path.join(根, "皮层连接_cortex_links.py")
备份 = 文件 + ".bak_学抑制优化前"
暂存 = 文件 + ".新版暂存"
shutil.copyfile(文件, 暂存)
try:
    shutil.copyfile(备份, 文件)          # 装旧版
    r1 = subprocess.run([py, "-X", "utf8", "诊断_整拍耗时.py"], cwd=根, capture_output=True)
    print("===== 改之前（旧写法）=====")
    sys.stdout.write(r1.stdout.decode("utf-8", "replace"))
    sys.stdout.flush()
finally:
    shutil.copyfile(暂存, 文件)          # 一定装回新版
    os.remove(暂存)
r2 = subprocess.run([py, "-X", "utf8", "诊断_整拍耗时.py"], cwd=根, capture_output=True)
print("===== 改之后（新写法）=====")
sys.stdout.write(r2.stdout.decode("utf-8", "replace"))
print("确认现在是新版（含 _备抑制索引）：",
      "_备抑制索引" in open(文件, encoding="utf-8").read())