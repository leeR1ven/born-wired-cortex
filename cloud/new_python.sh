#!/usr/bin/env bash
# 镜像只有 python 3.8 时用这个：在机器上自己造一个 python 3.11，装在仓库里的 .venv，
# 不碰系统 python。造完之后 setup/bench/run 三个脚本会自己认这个 .venv。
#
#   bash cloud/new_python.sh
#   bash cloud/setup.sh
#
# 为什么不能将就用 3.8：我们钉的 mujoco 3.13 要求 python >= 3.10、numpy 2.4 要求 >= 3.11，
# 3.8 装不上。硬降级到 3.8 能装的旧版（mujoco 3.2 + numpy 1.24）会换掉物理引擎，
# 之前量出来的那些成绩就不能比了 —— 所以宁可造个新 python，也不动库的版本。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
export PATH="$HOME/.local/bin:$PATH"

echo "仓库 $ROOT"

# 一、机器上要是已经有 3.10 以上的 python，直接拿它建 venv 就够了
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        echo "这台机器已经有 $candidate（$("$candidate" -V 2>&1)），用它建环境"
        if "$candidate" -m venv "$VENV" && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
            echo
            echo "好了：$VENV/bin/python"
            exit 0
        fi
        echo "$candidate 建环境不成（可能缺 python3-venv），换别的办法"
        rm -rf "$VENV"
    fi
done

# 二、有 conda 就用 conda 现成的 python 3.11
if command -v conda >/dev/null 2>&1; then
    echo "用 conda 造一个 python 3.11（放在 $VENV）..."
    conda create -y -p "$VENV" python=3.11 pip
    echo
    echo "好了：$VENV/bin/python"
    exit 0
fi

# 三、都没有就装 uv —— 它能把一份独立的 CPython 直接下载下来，不用 root、不用编译
echo "装 uv，用它拉一份独立的 python 3.11..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
mkdir -p "$(dirname "$VENV")"
uv venv --python 3.11 "$VENV"
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
"$VENV/bin/python" -m pip install -q -U pip 2>/dev/null || \
    echo "（这个环境没有 pip，等一下 setup.sh 会用 uv pip 装）"
echo
echo "好了：$VENV/bin/python"