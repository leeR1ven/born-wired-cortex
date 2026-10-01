#!/bin/bash
# 让这个 Python 认得 CUDA。
#
# 2026-10-01 的实情：这台机器是 RTX 4090，渲染一直在显卡上跑（MUJOCO_GL=egl）。
# 但 CUDA 计算用不了 —— 内核模块是 550.144.03，而 10-01 01:57 装上的用户态库是
# 550.163.01，两边版本对不上，nvidia-smi 直接报 driver/library version mismatch，
# torch.cuda.is_available() 就是 False。于是 born_wired 的 torch_execution.resolve()
# 返回 None，整个大脑的张量和眼睛的收缩求和全退回 CPU 的 numpy 去算。
#
# 这里把用户态那几个库软链指回内核那一版（/hy-tmp/bwc/nvfix 里就是这些软链）。
# 只在原样认不出显卡时才挂，驱动修好了这条就自动失效 —— 这是绕，不是治本；
# 治本是让恒源云把驱动两半对齐。
if ! nvidia-smi -L > /dev/null 2>&1; then
  if [ -d /hy-tmp/bwc/nvfix ]; then
    export LD_LIBRARY_PATH=/hy-tmp/bwc/nvfix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  fi
fi