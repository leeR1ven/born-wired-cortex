# 附录 B　每个数字配哪个脚本 + 哪份日志

> **本附录的用途**：写下这本书里**每一个数字的出处**。
> **规矩**：**追不出来的数字，不该写进书里。**
>
> ⚠️ **一句话说明**：这个仓库有**两套**东西——
> **`model/`** 是**最初那一套**（文件名是中文，例如 `实验_切断回响.py`、`核对_论文数字.py`），
> 论文里的数字基本来自这里；
> **`engine_v2/`** 是**后来重写的引擎**（文件名是英文，例如 `tools/probe_model_scale.py`），
> 规模、眼睛、题库那条线基本来自这里。
> **两边都留着**，因为**能对上原始日志**比"统一目录"更重要。

---

## B.1 四层证据（这套书里的数怎么才能"被追到"）

| 层 | 是什么 | 在哪 |
|---|---|---|
| **①** | **论文自带的索引** | `paper/Born_wired_manuscript.md` |
| **②** | **论文数字核对脚本** | `model/核对_论文数字.py`（**52 条全过**）+ `docs/论文数字复核_20260918.md` |
| **③** | **重跑一遍** | `model/重跑日志_20260918/`、`model/_重跑_修抑制_总.log` |
| **④** | **能看的回放** | 双击 `playback/回放_大脑_A.html`（另有 B / C / D 三幕、`回放_大脑_四幕.html`） |

> **要验一个数字，按 ①→②→③→④ 的顺序走**。四层里任何一层对不上，都要停下来查清。

---

## B.2 论文 R1–R21 → 脚本与日志

| 编号 | 内容 | 脚本（`model/` 下） | 日志 |
|---|---|---|---|
| **R1** | 天生接线就能闭环（站起 / 走红 / 无感觉 / 清空本能表） | `实验_闭环前提_多种子.py`、`诊断_白脑`系列、`实验_本能分阶段.py` | `日志_闭环多种子_新.log`、`日志_白脑_新.log`、`日志_分阶段_新.log` |
| **R2** | 行为由特定细胞群算出来（按灭 0/200） | `实验_前额叶_多种子.py`、`诊断_前额叶电流去哪.py` | `日志_七路_R2多种子.txt`、`日志_前额叶多种子_新A.log` |
| **R3** | 加结构不重训（221→581；"被音叫停" 0/3） | `实验_本能分阶段.py`、`实验_声音叫停.py` | `日志_分阶段_新.log` |
| **R4** | 认得出的边界（色相 96° / 78°） | `实验_颜色渐变.py`、`实验_渐变_正式协议.py`、`实验_放大_色相红块.py` | `日志_渐变正式_新.log`、`日志_渐变边界_无环.log` |
| **R5** | 层级压缩的代价（第 1 层准 / 压缩层散） | `实验_前几层看得见小东西吗.py`、`实验_追踪靠的是哪一层.py`、`实验_小东西的名字.py` | `日志_前几层.log` |
| **R6** | 注视是长出来的（穿 5.6° / 突现 4.8° / 空场 0.0°） | `实验_眼睛跟随.py`、`实验_眼睛底噪对照.py` | — |
| **R7** | 天生地图偏移（−3~+3 列；存活曲线） | `实验_本能表整体偏移.py`、`实验_模糊化_存活曲线.py` | `日志_存活曲线_新.log` |
| **R8** | 刺激发令、皮层运行（撤红仍走 495/495；掐回响 0/3） | `实验_刺激撤掉还走吗.py`、`实验_切断回响.py` | `日志_切断回响_有…`（见 `model/` 目录） |
| **R9** | 奖励信号学到一条新连接（4 阶段 1 个脑） | `实验_发育_仿生奖励_三臂.py`、`实验_发育_声音叫走路.py`、`诊断_R9_逐拍.py` | `日志_R9*`、`诊断_R9权重.py` |
| **R10** | 奖励由感觉传入（触觉；吃了 48 万根还是 0/200） | `触觉区_touch.py`、`实验_发育_*` | `日志_触觉版_天生站起来.log` |
| **R11** | 更宽皮层的代价（148,032→925,236） | `量规模_三档.py`、`量规模_七档.py`、`规模试验_测量.py` | `engine_v2/artifacts/scale_all_regions.log` |
| **R13** | 重建引擎的代价 | `engine_v2/tools/probe_step_cost.py`、`probe_sparse_cost.py`、`probe_execution_limits.py` | `engine_v2/logs/` |
| **R14** | 信号不只从最后一层来（中间层能分开末层分不开的） | `实验_前几层看得见小东西吗.py`、`实验_视觉线性*.py` | `日志_前几层.log`、`日志_分层.log` |
| **R15** | 网要多大才够（"模型太小"可量） | `诊断_力气分档.py`、`实验_想还在吗.py`、`诊断_参数一致.py` | — |
| **R16** | 会聚 / 聚焦 / 看变化处（**已撤回**的那条） | `engine_v2/tools/compare_honest_and_warped_retina.py`、`measure_retina_angles.py` | `engine_v2/artifacts/bank_honest_vs_warped.log` |
| **R17** | 耳朵分前后 | `engine_v2/born_wired/binaural_senses.py`、`engine_v2/tests/test_binaural_senses.py` | — |
| **R18** | 两个脑合成一个（17 细胞、5+5 条边、+1.000、5/5） | `engine_v2/tools/merge_models.py`、`fuse_parents.py`、`check_merge_identity.py` | `engine_v2/_merge_test.log`、`_merge10*.log` |
| **R19** | 每种感觉都是成对的（黑屏 0.96985 等） | `engine_v2/born_wired/stereo_senses.py`、`binaural_senses.py` | — |
| **R20** | 用得多就守得住、用得少也不一直掉 | `engine_v2/tools/measure_weight_drift.py`、`measure_forgetting.py` | — |
| **R21** | 什么挡住"一个念头点亮整张网" | `model/实验_专注_无关就互压.py`、`实验_专注_无关信号扩不出去.py`、`诊断_互斥到底压没压.py` | `日志_专注_无关就互压.log`、`日志_专注_无关信号扩不出去.log` |

---

## B.3 数字 → 脚本（按主题）

### 细胞与连接 / 抑制

| 数字 | 脚本 | 日志 |
|---|---|---|
| 抑制边 5,102,714 → 7,239,774 | `model/自检_抑制神经元.py`、`接线自检_wiring_audit.py` | `诊断_抑制修前修后_当前架构对照.log` |
| 总抑制电流 18,933 → 21,485（113%） | `model/诊断_修前修后_抑制电流对照.py` | `诊断_修前修后_抑制电流对照.log` |
| 就近半径 24 vs 4（站起 5/12 vs 12/12） | `model/诊断_抑制半径_多种子.py`、`诊断_抑制就近比例扫一扫.py` | `诊断_抑制半径_多种子.log` |
| 抑制强度扫描 | `model/诊断_抑制强度扫一扫.py` | `诊断_抑制强度扫一扫.log` |
| 出生抑制权重扫描 | `model/诊断_出生抑制权重扫一扫.py` | `诊断_出生抑制权重扫一扫.log` |
| 冻结真的免疫 | `engine_v2/tools/probe_freeze.py` | — |

### 学习规则 / 权重

| 数字 | 脚本 | 日志 |
|---|---|---|
| 稀疏执行逐位等价 | `engine_v2/tools/check_sparse_engine.py` | — |
| 逐位一致（改前后） | `engine_v2/tools/check_model_fingerprint.py` | — |
| 重建引擎 vs 赫布引擎 | `engine_v2/tools/compare_engines.py` | — |
| 权重漂移（R20 那三个读数） | `engine_v2/tools/measure_weight_drift.py` | — |
| 托儿所四臂对照（R12） | `engine_v2/tools/nursery.py` | `engine_v2/logs/`（nursery） |

### 规模与耗时

| 数字 | 脚本 | 日志 |
|---|---|---|
| 每档细胞 / 连接 / 一步耗时 | `engine_v2/tools/probe_model_scale.py` | `engine_v2/artifacts/scale_all_regions.log` |
| 一步分项（学习 10.4 ms 等） | `engine_v2/tools/profile_dog_step.py`、`profile_live_step.py` | — |
| 实时窗口 | `engine_v2/tools/probe_live_config.py` | — |
| 渲染耗时 | `engine_v2/tools/probe_render_cost.py`、`probe_render_speed.py` | — |
| 稀疏代价 | `engine_v2/tools/probe_sparse_scaling.py` | — |

### 眼睛 / 视网膜

| 数字 | 脚本 | 日志 |
|---|---|---|
| 25/25 命中、中位 0.045、最差 0.088 | `engine_v2/tools/wire_red_gaze.py` + `measure_eye_gaze_routes.py` | `engine_v2/logs/眼大_2.5_0.35_0.1_0.2.log` |
| 抖动 0.503 → 0.056 | 同上（扫 `--gain`） | `engine_v2/logs/眼扫5_0.01_3.5_1.00_0.40.log` |
| 绿球对照 2/25 | 同上 | `engine_v2/logs/对照_绿球.log` |
| 不分两档 25/25 | 同上 | `engine_v2/logs/对照_不分档.log` |
| 红驱动 0.041~0.138 → 0.17~0.40 | `engine_v2/tools/probe_red_drive.py` | `engine_v2/logs/红驱动探针.log`、`红驱动探针2.log` |
| 位置野 ±0.36 → ±0.50（936 位置） | `engine_v2/tools/sweep_red_ball_features.py` | `engine_v2/logs/红球位置野_红_大.log` |
| 诚实 vs 变形采样 | `engine_v2/tools/compare_honest_and_warped_retina.py`、`compare_retina_modes.py` | `engine_v2/artifacts/bank_honest_vs_warped.log` |
| +yaw / +pitch 的正负号 | `engine_v2/tools/probe_eye_axis.py` | — |
| 眼肌格子 / 单位 | `engine_v2/tools/probe_eye_motor_units.py`、`probe_eye_loop.py` | — |
| 眼睛单测 | `engine_v2/tests/test_eye_muscles.py`、`test_embodied.py`（**20 个全过**） | `engine_v2/logs/单测_眼睛.log` |

### 题库 / 演化 / 融合

| 数字 | 脚本 | 日志 |
|---|---|---|
| 130 道题与合格线 | `engine_v2/tools/taskbank.py`、`test_taskbank.py` | — |
| 题库自审（废题 / 错读法 / 死文本） | `engine_v2/tools/audit_taskbank.py`、`task_health.py`、`compare_audits.py` | `engine_v2/docs/题库与训练方式自审_20260929.md` |
| 出生动物跑全库 1,609 秒 | `engine_v2/tools/race.py`、`screen_candidates.py` | `engine_v2/logs/` |
| 追球演化 | `engine_v2/tools/evolve_chase.py`、`hunt_chase.py`、`verify_chase.py`、`confirm_kept.py` | `engine_v2/_追球_批0*.log`、`_复核追球*.log` |
| **留种综合评分的十格权重**（追到球 **.27** / 跑得快 **.22** / 站得稳 **.15** / 球在正前方 .10 / 落后 .08 / 静止球 .05 / 追近 .04 / 最近 .03 / 两边 .03 / 走直 .03） | `engine_v2/tools/evolve_chase.py` → `SCORE_WEIGHTS`、`overall()`、`lowest_up()` | `engine_v2/logs/追球_云_七批_*.log`（每代收工那几行） |
| **「跑得快」为什么除以 6 米**（六批 193 只没摔的：中位 3.85 m、最远 9.29 m；÷5 → **26%** 顶满分，÷6 → **8%**）与**「站得稳」的 0.3 / 0.9 两条线**（同一批：最低直立分量 0.394 ~ 1.000） | 同上 | `engine_v2/artifacts/云端/六批/追球_云_六批_g00.jsonl` |
| 云上批处理 | `engine_v2/tools/_pull_cloud.ps1`（带批次参数）、`_batch_report.py`、`_云七批.sh`、`_watchdog7.sh`、`_gpu_env.sh` | `engine_v2/logs/追球_云_*批_*.log`、`watchdog.log` |
| 融合身份核对 | `engine_v2/tools/check_merge_identity.py` | `engine_v2/_merge_test.log` |
| 眼睛融合（云端线） | `engine_v2/tools/merge_cloud_eye.py` | `engine_v2/logs/` |

### 本能表 / 标定

| 数字 | 脚本 | 文件 |
|---|---|---|
| 585 条规则 | `model/生成本能表.py`、`本能工具_instincts.py` | `model/本能表.txt`（686 行）、`model/本能表_列32.txt` |
| 反射表 4 条 | `model/算反射.py` | `model/反射表.txt` |
| 动作库 221 条 | `model/建动作库.py`、`取动作_从训练好的模型.py`、`验收_动作库.py` | `model/动作库.json`、`动作库_训练好的.json` |
| 皮层名字（标定结果） | `model/皮层连接_cortex_links.py`、`感觉区_共同.py` | `model/皮层名字.json`、`皮层名字_第1~4层.json` |

---

## B.4 复现入口（照这四步走）

| 步 | 做什么 | 文件 |
|---|---|---|
| 1 | 看总说明 | `docs/复现说明.md`（第三节是**论文数字对照表**） |
| 2 | 跑数字核对 | `model/核对_论文数字.py`（应 **52 条全过**） |
| 3 | 看复核记录 | `docs/论文数字复核_20260918.md` |
| 4 | 看回放 | `playback/回放_大脑_A.html`（B / C / D、`回放_大脑_四幕.html`） |

---

## B.5 一条使用纪律

> **任何写进书里、写进论文、写进汇报的数字，都必须能在这张表里查到脚本和日志。**
>
> 查不到的，只有两种下场：
> **要么补一张日志，要么把这个数字撤下来。**

---

## B.6 自检三问

1. 为什么这个仓库要**保留两套**（`model/` 中文名 + `engine_v2/` 英文名），而不是统一成一个？
2. 如果你拿到一个数字（比如"站起 12/12"），**你要走哪几步**才能确认它？
3. 为什么规矩是"**追不出来的数字不许写进书里**"？如果放宽这条规矩，会发生什么？