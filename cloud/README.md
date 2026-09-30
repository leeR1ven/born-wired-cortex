# 上云跑训练：五步

## 0. 租什么机器

实测（`logs/测速_云选型.log`，本机 i5-12600KF + 3070 Ti，同一批 64 只孩子）：

| 跑法 | 64 只花多久 | 折合每只 | 每小时 |
| --- | --- | --- | --- |
| 8 进程，用显卡 | 2.9 分钟 | 21.8 秒 | 1320 只 |
| 8 进程，强制纯 CPU | 5.3 分钟 | 39.8 秒 | 723 只 |

所以：**租带显卡的机器**，纯 CPU 机型不要。

- 推荐：1× RTX 4090（24 GB）+ 16~24 核 + 64~128 GB 内存。
- 一个进程吃 1.1 GB 内存、占一个大核；进程数别超过「核数」和「内存 GB ÷ 2」里小的那个。
- 显存不是瓶颈（现在整台机器只用 4.6/8 GB），**内存和核数是**。
- 单独跑那个几十万细胞的大模型（`live_config.json` 里那档）时才需要 L40S 48 GB / A100 80 GB。
- 系统盘往往很小，仓库和产物尽量放数据盘。

## 0.5 镜像选什么

**带 PyTorch 的镜像，Python 3.10 或 3.11，CUDA 11.8 或 12.x**（恒源云镜像列表里 PyTorch 那几档都行）。

- 代码要求 Python ≥ 3.10：有几个工具用了 `X | Y` 这种写法。
- 镜像里的 torch 越新越好（≥ 2.3）。比 2.3 老的话它跟 numpy 2.x 不兼容，`setup.sh` 会自动把
  numpy 压回 1.x —— 能跑，但没必要，能挑新的就挑新的。
- 别选 Python 3.8 / 3.9 的镜像，也别选 torch 1.x 的。
- 镜像里**不用**带 mujoco：`setup.sh` 自己装（mujoco 3.13 + EGL 那几套系统库）。
- 镜像里连 torch 都没有也能跑：引擎会退回 numpy 那份 host 引擎（慢很多，但不至于跑不动）。
- 系统盘给大一点（仓库 336 MB + torch 几个 GB），产物尽量写数据盘。

## 0.8 镜像只有 python 3.8 怎么办

先造一个新 python，再照常走：

```bash
bash cloud/new_python.sh     # 优先用机器上已有的 3.10+，没有就用 conda，再没有就装 uv 拉一份独立的 3.11
bash cloud/setup.sh          # 三个脚本会自动认仓库里这个 .venv
```

为什么不能将就用 3.8：钉住的 mujoco 3.13 要求 python ≥ 3.10、numpy 2.4 要求 ≥ 3.11，3.8 装不上。
硬降级到 3.8 能装的旧版（mujoco 3.2 + numpy 1.24）会换掉物理引擎，之前量出来的成绩就不能比了 ——
所以宁可造个新 python，也不动库的版本。

## 1. 拉代码

```bash
git clone https://github.com/leeR1ven/born-wired-cortex.git
cd born-wired-cortex
```

## 2. 装环境（一条命令）

```bash
bash cloud/setup.sh
```

它做四件事：装 EGL/GL 系统库（云上无头渲染要的）、装 `engine_v2/requirements.txt`（镜像里
已有 torch 就跳过 torch）、把 `MUJOCO_GL=egl` 写进 `~/.bashrc`（**少这一句，眼睛渲染直接
报错**）、生 4 只孩子冒烟。看到冒烟那行的「没摔/倒了」就是装好了。

## 3. 量这台机器开几个进程最划算

```bash
bash cloud/bench.sh              # 默认试 8 / 16 / 24 个进程
WORKERS="8 12 16" KIDS=64 bash cloud/bench.sh
```

它按「每只多少秒 / 每小时多少只」打印。挑每小时最多的那一档。

## 4. 挂长跑

```bash
WORKERS=20 GENERATIONS=20 bash cloud/run.sh
tail -f logs/上云演化_*.log      # 看进度
grep 代收工 logs/上云演化_*.log  # 看每一代的成绩单
```

后台跑，关掉终端不停。一代 300 只，跑完立刻挑爹生下一代：6 只没摔的 + 4 只最快的
（快的哪怕会摔也留着，不然速度基因会断种）。

**断了怎么接**（比如机器被回收）：看日志最后那行说的 keep 文件和代数，这样接：

```bash
PARENTS=artifacts/题1f_演化_云_g07_keep.jsonl START_GEN=8 bash cloud/run.sh
```

## 5. 把成绩拿回来

台账在 `engine_v2/artifacts/题1f_演化_云_gNN.jsonl` 和 `_keep.jsonl`（几十 KB 到几百 KB），
日志在 `logs/`。云上要 `git push` 得先在机器上配 GitHub 令牌；最省事的办法是把这几个文件
下载回本地，由本地统一入库。