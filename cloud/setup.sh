#!/usr/bin/env bash
# 云服务器第一次用之前跑这一遍：装系统库 + python 依赖 + 验一眼眼睛能渲染。
#
#   bash cloud/setup.sh
#
# 显卡机上 MuJoCo 要无头渲染（云上没有显示器），所以需要 EGL 那几套系统库，
# 并把 MUJOCO_GL 设成 egl。少这一句，眼睛渲染会直接报错，训练跑不起来。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 解释器：优先用 cloud/new_python.sh 造在仓库里的 .venv（镜像只有 python 3.8 时用它），
# 其次用 PY=... 指定的，最后才用系统的 python3。
if [ -n "${PY:-}" ]; then :
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY=python3
fi

echo "仓库      $ROOT"
echo "处理器    $(nproc) 核"
echo "内存      $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}')"
"$PY" -c 'import sys, torch; print("python   ", sys.version.split()[0]); print("torch    ", torch.__version__, "| cuda 可用:", torch.cuda.is_available())' 2>/dev/null || echo "torch    还没装"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || echo "注意：这台机器的 python 比 3.10 老，有几个工具用了 X | Y 这种写法，会跑不了"

if command -v apt-get >/dev/null 2>&1; then
    echo "装系统库（无头渲染要的 EGL/GL）..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        libegl1 libgl1 libosmesa6 libglib2.0-0 libsm6 libxext6 libxrender1
fi

echo "装 python 依赖..."
"$PY" -m pip install -q -U pip

# torch 交给镜像（镜像里那份跟驱动是对口的），其余照 requirements.txt 装。
# 一处版本关系要照顾：torch 比 2.3 老的话跟 numpy 2.x 不兼容，那就把 numpy 压回 1.x。
NUMPY_SPEC=$("$PY" - <<'PYEOF'
import importlib.metadata as meta
try:
    version = meta.version("torch").split("+")[0].split(".")
    older = (int(version[0]), int(version[1])) < (2, 3)
except Exception:
    older = False
print("numpy<2" if older else "numpy==2.4.6")
PYEOF
)
echo "numpy 装这一档：$NUMPY_SPEC"
sed -e 's/^torch.*/torch  # 交给镜像/' -e "s/^numpy==.*/$NUMPY_SPEC/" \
    "$ROOT/engine_v2/requirements.txt" | grep -v '交给镜像' > /tmp/req_cloud.txt
"$PY" -m pip install -q -i "${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    -r /tmp/req_cloud.txt \
    || "$PY" -m pip install -q -i "${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
        "$NUMPY_SPEC" mujoco scipy numba pillow

export MUJOCO_GL=egl
if ! grep -q 'MUJOCO_GL' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export MUJOCO_GL=egl' >> "$HOME/.bashrc"
    echo "已把 MUJOCO_GL=egl 写进 ~/.bashrc"
fi

cd "$ROOT/engine_v2"
# 冒烟用的爹：就用随包带来的那份 keep 台账。别用 evolve.py 的默认值 —— 默认那份是 6 MB 的
# 比赛台账，上云的精简包里没有。
SMOKE_PARENTS="${SMOKE_PARENTS:-artifacts/题1f_演化_g01_keep.jsonl}"
if [ ! -f "$SMOKE_PARENTS" ]; then
    echo "冒烟要用的台账不见了：$SMOKE_PARENTS"
    echo "（精简包只带这一份 keep 台账；缺了就用 SMOKE_PARENTS=... 指一份）"
    exit 1
fi
echo "冒烟：生 4 只孩子，确认建得出来、跑得动（爹用 $SMOKE_PARENTS）..."
PYTHONIOENCODING=utf-8 "$PY" -X utf8 -u tools/evolve.py --parents "$SMOKE_PARENTS" \
    --candidates 4 --generations 1 --workers 4 --quiet --out-prefix artifacts/_上云冒烟 --seed 1

echo
echo "装好了。下一步：bash cloud/bench.sh 量一下这台机器开几个进程最划算。"