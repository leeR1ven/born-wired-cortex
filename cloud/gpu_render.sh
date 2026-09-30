#!/usr/bin/env bash
# 把「渲染用的那半驱动」装齐，让 EGL 真的落在 N 卡上，而不是落到 Mesa 的软件光栅。
#
#   bash cloud/gpu_render.sh
#
# 为什么需要这一步：云镜像经常只装了驱动里跑 CUDA 的那半（libcuda、libnvidia-compute），
# 没装渲染的那半（libEGL_nvidia、libGLX_nvidia）。少了它，MuJoCo 用 MUJOCO_GL=egl
# 还能跑起来（它不会报错，会安安静静地用 Mesa），但代价是灾难性的：在 4090 那台机器上
# 量到「取一次眼睛画面」要 464.6 毫秒，装完这个包是同一条代码 2.6 毫秒 —— 差 177 倍。
# 表现出来就是：nvidia-smi 上 GPU 几乎闲的（软件光栅在 CPU 上干活），一整只孩子要 55 秒，
# 一台 4090 的机器比本机还慢。
set -euo pipefail

if ldconfig -p | grep -q libEGL_nvidia; then
    echo "渲染库已经在：$(ldconfig -p | grep libEGL_nvidia | head -1 | sed "s/^[[:space:]]*//")"
    exit 0
fi

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d "[:space:]")
[ -n "$DRIVER" ] || { echo "nvidia-smi 读不到驱动版本 —— 先确认这台机器的显卡驱动是好的"; exit 1; }
MAJOR=${DRIVER%%.*}
PKG="libnvidia-gl-${MAJOR}"
echo "宿主驱动 $DRIVER，装 $PKG=${DRIVER}-0ubuntu1 ..."

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PKG}=${DRIVER}-0ubuntu1" \
    || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$PKG"

# 坑：装这个包会顺手把 libnvidia-compute 升到比宿主内核模块更新的版本，符号链接跟着
# 指过去，CUDA 立刻变成 “Driver/library version mismatch”、torch.cuda.is_available() 返回
# False；而且这些 .so 是宿主挂进来的（bind mount），dpkg 换不动它们，想降级也降不下去
# （会报 unable to make backup link ... Invalid cross-device link）。所以这里把符号链接
# 手动拨回挂进来的、跟宿主驱动对口的那个版本。
cd /usr/lib/x86_64-linux-gnu
for link in $(ls -1 | grep -E "^lib(nvidia|cuda)[^/]*\.so(\.[0-9]+)*$"); do
    target=$(readlink "$link" 2>/dev/null) || continue
    case "$target" in
        *".$DRIVER") continue ;;                      # 已经是对口的
        *.so.*) ;;
        *) continue ;;                               # 没有版本号的那种，不碰
    esac
    stem=${target%%.so.*}                            # libcuda
    pinned="${stem}.so.${DRIVER}"
    if [ -e "$pinned" ]; then
        ln -sfn "$pinned" "$link"
        echo "  链接拨回 $link -> $pinned"
    fi
done

echo
echo "现在这台机器的显卡：$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tr "\n" " ")"
echo "验一眼渲染真的上了卡（这条命令每秒几毫秒才对，几百毫秒就是又掉软件光栅了）："
echo "  cd engine_v2 && MUJOCO_GL=egl python -c \"import mujoco;...\""
