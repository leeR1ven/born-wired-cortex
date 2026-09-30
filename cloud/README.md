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