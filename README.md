# Born wired —— 天生的皮层连接 + 局部可塑性，就够了

> **网页版（想快速了解就看这个）**：https://leer1ven.github.io/born-wired-cortex/  
> 一页读完摘要、关键数字和“什么没声明”；往下是机器人自己站起来、自己走向红块的四幕动画。  
> 本地看：双击 `index.html`（首页）或 `playback/index.html`（四幕动画）。

**一句话**：143,796 个皮层神经元、308,809 根**出生时就写好的兴奋连接**、只有一条局部规则
（上一拍亮的细胞 → 这一拍亮的细胞，连接强一点）。没有奖励函数、没有反向传播、没有目标函数、
没有"输入 token / 输出 token"。结果是：身体自己从趴着站起来、看见红块自己走过去、
眼睛自己盯住移动的小球。

本仓库是论文 **《Born wired: innate cortical connectivity plus local plasticity is enough
to stand, walk and look》** 的代码、日志、图和数据。

|  |  |
|---|---|
| 作者 | 李秩文（独立研究者 / Independent Researcher） |
| 单位 | No. 67 Yuanren Street, Huangjing Town, Taicang, Suzhou, Jiangsu, China |
| ORCID | 0009-0005-8289-6393 |
| 邮箱 | rivenlee94@gmail.com |
| 论文正文 | `paper/Born_wired_manuscript.md` |
| 论文 PDF（全长版，10 图和补充说明都在） | `paper/Born_wired_manuscript.pdf` |
| 论文正文（Nature Machine Intelligence 短版：摘要 146 词 / 正文 3,476 词 / 6 图） | `paper/Born_wired_manuscript_NatureMI.md` |
| 论文 PDF（Nature Machine Intelligence 短版） | `paper/Born_wired_manuscript_NatureMI.pdf` |
| 短版用的 10 张图（按 Nature 顺序重新编号） | `paper/figures_NatureMI/` |
| **实际投出去的稿子**（2026-09-17 投 Nature Machine Intelligence，原件存档并附 SHA-256） | `paper/submitted_20260917/` |
| **投稿记录**（稿件号 NATMACHINTELL-A26095390、状态时间线、预印本 rs-11070398） | `docs/投稿记录_NatureMachineIntelligence_20260917.md` |

## 现在能看到什么

- **论文全文**：`paper/Born_wired_manuscript.md`（全长版）／`paper/Born_wired_manuscript_NatureMI.md`（20 页短版）
- **论文 PDF**：`paper/Born_wired_manuscript.pdf`（全长 37 页）／`paper/Born_wired_manuscript_NatureMI.pdf`（20 页）／`paper/submitted_20260917/`（实际投出去的那一版，带行号）
- **模型自己走路的回放**：双击 `playback/index.html`，不用装任何东西
- **预印本**：Research Square `rs-11070398`（平台审核中，上线后此处补 DOI）
- **引用方式**：见 `CITATION.cff`，GitHub 页面右上会出现 "Cite this repository"


---

## 目录说明

| 目录 | 里面是什么 |
|---|---|
| `paper/` | 论文正文（Markdown 与 PDF）、十张图（`paper/figures/`）、实际投稿的 PDF 原件（`paper/submitted_20260917/`） |
| `model/` | 皮层模拟器的全部代码（156 个脚本）+ 数据表（`本能表.txt`、`动作库*.json`、`皮层名字*.json`…） |
| `logs/` | 论文引用的全部运行日志（292 份）。论文里每一个数字都能在这里找到出处 |
| `snapshots/` | 过程留影（站起来、走路、眼睛跟随的截图与小动画） |
| `playback/` | 四幕回放页。双击 `playback/index.html`，不用装任何东西就能看模型自己的行为 |
| `experiments/` | 各阶段子实验（运动库蒸馏、色标导航、闭环仿真、认知实验、文字网络实验…） |
| `docs/` | 架构说明、规则总表、接线审计、复现说明、进展记录 |

**这个仓库只装这篇论文相关的东西。** 与论文无关的内容（3.9 GB 的"宠物之家"仿真、
56 GB 的文字语料、旧架构归档、对外联络材料）都留在旧的开发仓库里，不在这里。

---

## 先看什么

1. **想快速看懂**：`paper/Born_wired_manuscript.md`（正文），或 `docs/架构导读_中文.md`
2. **想亲眼看到行为**：双击 `playback/index.html`
3. **想验证数字**：见下面"怎么自己验一遍"

## 怎么自己验一遍

论文里的每一个数字，背后都有一个脚本 + 一份日志。脚本可以重跑，日志可以当场对照。

```powershell
cd model
$py = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"   # 换成你自己的 python
& $py -X utf8 核对_论文数字.py        # 几秒钟：程序自己去翻 logs/，把数字抠出来和论文比，应打印"对上 52 条，对不上 0 条"
& $py -X utf8 核对_论文数字.py 重跑    # 约 6 分钟：当场重跑 R1，看真跑出来的数是不是论文里那个
& $py -X utf8 看_大脑开车.py          # 让模型自己开一遍机器人，生成 playback/ 里的四幕回放页
```

论文 5.8 节列了**每个结果对应哪个脚本**（R1–R11，一共四十多个）；论文里每个数字后面
也写着"跑哪个脚本、看哪份日志"。`docs/复现说明.md` 是一张完整的对照表。

**环境**：核心只需要 Python 3 + NumPy；身体用 MuJoCo（Go2 模型缺了会自动从
MuJoCo Menagerie 下载）；画图用 matplotlib + Pillow；只有第三方对照实验
（`对照_RL_训练.py`、`对照_现成RL_驱动.py`）需要 PyTorch 和一个外部 checkpoint。
全部跑在 CPU 上。

**两个开关**（环境变量，不改代码）：`前额叶不保留=1` 关掉前额叶回响（R1–R8 用这档，
论文 Methods 5.5）；`AGI列数`、`AGI数据后缀` 换皮层大小（R11 用）。

---

## 诚实声明：哪里还会摔

### 已修好：出生就会的那一走（2026-09-18）

以前走到第 66 拍（1.3 秒）就摔。原因不是学习（开/关学习逐拍数字一模一样），
也不是走路链乱了（始终只有 1 个时刻亮），而是**前额叶在累加**：
它一亮就靠自己的环一直亮，2608 → 2818 还在涨，经天生互惠返回线把运动区灌到 96/160 个细胞，
身体从第一拍就开始歪。把前额叶压回稀疏（动态整体抑制，现已改为默认开）后，
同一颗脑走 **600 拍、6.62 米、一次没歪过 75 度**。完整证据和对照表：
`docs/修复_摔倒_20260918.md`。修完之后论文那两条结论仍然成立：前额叶按灭 → 走路动作 **0/200 拍**；
全黑不给画面 → 走路动作 **0/200 拍**；趴着不给信号 → **自己站起来**（末高 0.260、歪 0）。

### 还没修好：“教它听见声音就走”那一段（论文 R9）

教学时“红 → 走”和天生的“响 → 原地踏步”两个动作同时被点着，运动记忆区同时亮 2~3 个时刻，
两串肌肉力叠在一起，第 107 拍摔。这是**另一个问题**：两个动作抢一个身体。
论文 4.6 节把它单列成一条缺口，没有藏；排查过程在 `docs/论文进展说明_20260916.md`。
已经排除的：不是权重在长（进运动区 265,436 根边，考试 200 拍里变化 **0 根**）；
不是返回线的错；也不是整体降权能修的（三档实测反而更早摔，论文 4.6 引了这个负结果）。
两条候选修法写在 `docs/修复_摔倒_20260918.md` 末尾。

---

## 许可

MIT，见 `LICENSE`。© 2026 李秩文。可用、可学、可在此基础上构建，只请保留署名。

## 关于 AI 协作

按论文 5.10 节的声明：代码、实验协议和论文正文，是在作者主导下，与大语言模型
交互完成的（编程与实验执行由 AI 承担，架构、判断和方向由作者给出）。
