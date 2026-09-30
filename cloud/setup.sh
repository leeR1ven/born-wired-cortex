#!/usr/bin/env bash
# 云服务器第一次用之前跑这一遍：装系统库 + python 依赖 + 验一眼眼睛能渲染。
#
#   bash cloud/setup.sh
#
# 显卡机上 MuJoCo 要无头渲染（云上没有显示器），所以需要 EGL 那几套系统库，
# 并把 MUJOCO_GL 设成 egl。少这一句，眼睛渲染会直接报错，训练跑不起来。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"

echo "仓库      $ROOT"
echo "处理器    $(nproc) 核"
echo "内存      $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}')"
"$PY" -c 'import sys, torch; print("python   ", sys.version.split()[0]); print("torch    ", torch.__version__, "| cuda 可用:", torch.cuda.is_available())' 2>/dev/null || echo "torch    还没装"

if command -v apt-get >/dev/null 2>&1; then
    echo "装系统库（无头渲染要的 EGL/GL）..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        libegl1 libgl1 libosmesa6 libglib2.0-0 libsm6 libxext6 libxrender1
fi

echo "装 python 依赖..."
"$PY" -m pip install -q -U pip
if "$PY" -c 'import torch' 2>/dev/null; then
    echo "镜像里已经有 torch，跳过它，只补其余的"
    grep -v -i '^torch' "$ROOT/engine_v2/requirements.txt" > /tmp/req_no_torch.txt
    "$PY" -m pip install -q -r /tmp/req_no_torch.txt
else
    "$PY" -m pip install -q -r "$ROOT/engine_v2/requirements.txt"
fi

export MUJOCO_GL=egl
if ! grep -q 'MUJOCO_GL' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export MUJOCO_GL=egl' >> "$HOME/.bashrc"
    echo "已把 MUJOCO_GL=egl 写进 ~/.bashrc"
fi

cd "$ROOT/engine_v2"
echo "冒烟：生 4 只孩子，确认建得出来、跑得动..."
PYTHONIOENCODING=utf-8 "$PY" -X utf8 -u tools/evolve.py --candidates 4 --generations 1 \
    --workers 4 --quiet --out-prefix artifacts/_上云冒烟 --seed 1

echo
echo "装好了。下一步：bash cloud/bench.sh 量一下这台机器开几个进程最划算。"