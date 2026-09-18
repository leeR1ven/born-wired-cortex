# 投稿记录：Nature Machine Intelligence（2026-09-17 提交，2026-09-18 建档）

这份记录只记一件事：**投出去的是哪一版稿子、投到哪、当时勾了什么、后来状态怎么变**。
方便以后回头核对，也避免和别的项目混在一起。

## 一、稿件

| 项目 | 内容 |
|---|---|
| 期刊 | Nature Machine Intelligence |
| 稿件类型 | Article（研究论文） |
| 标题 | Born wired: innate cortical connectivity plus local plasticity is enough to stand, walk and look |
| 稿件号 | **NATMACHINTELL-A26095390** |
| 提交日期（投稿系统记录） | 2026-09-17 |
| 投稿系统 | https://mts-natmachintell.nature.com |
| 作者（唯一作者，兼通讯作者） | 李秩文 / Zhiwen Li，Independent Researcher |
| 邮箱 | rivenlee94@gmail.com |
| ORCID | 0009-0005-8289-6393 |
| 通信地址 | No. 67 Yuanren Street, Huangjing Town, Taicang, Suzhou, Jiangsu, China（邮编 215426） |
| 投稿时仓库版本 | commit `fd91967`（提交那一刻仓库 HEAD） |

## 二、实际投出去的文件（原件已存档）

| 系统里的文件类型 | 系统里的文件名 | 本地原件 | 字节数 | SHA-256 |
|---|---|---|---|---|
| 文章文件 | Born_wired_manuscript.pdf | `paper/submitted_20260917/Born_wired_NatureMI_line_numbers.pdf` | 3,137,255 | `10713f48e8af634be764798765454f1754cc5a9b95b9f34a99a475c017d2d969` |
| 作者求职信 | 求职信.pdf | `paper/submitted_20260917/Cover_letter.pdf` | 37,542 | `157d58411a2422914c7597bb2fc5abd0111e1d1bf4d8ea2f4b26df65bd181014` |

几点说明：

* 文章文件是**全文带行号**的 PDF（20 页，6 张主图 + 4 张补充图）。Nature 要求 PDF 稿件逐行编号。
* 上传时文件名里的"行号"两个字被投稿系统转成了 `XXXXXX`，投稿过程中已替换成 `Born_wired_manuscript.pdf`，内容未变。
* `paper/Born_wired_manuscript_NatureMI.pdf` 是同一版内容的**不带行号**版本，留作对照。
* 求职信正文（Markdown 源）见 `docs/` 同一批投稿材料；本地原件在 `投稿_20260917/上传用_upload/Cover_letter.pdf`。
* 系统在 2026-09-17 把求职信和文章都做过一次"批准 PDF"确认，两次都点过"验证 PDF"再勾"批准"。

## 三、投稿时在表单里填的选项

| 项目 | 当时填的 |
|---|---|
| 学科词（5 条） | 生物科学/神经科学/计算神经科学；生物科学/神经科学/突触可塑性；神经科学/感觉运动处理；生物科学/神经科学/运动控制；物理科学/数学与计算/计算机科学 |
| 利益冲突 | 不存在利益冲突 |
| 双重出版 | 否（结果/数据/图表未在他处发表，也未被他刊考虑） |
| 预印本 / In Review | 投稿时选"不希望从《审稿中》获益"→ 2026-09-18 改为参加（见第五节） |
| 同行评审方式 | 单匿名（向审稿人显示作者信息） |
| Research Square 作者面板 | 同意与该平台共享稿件与个人数据 |
| 研究数据存储（Figshare） | 否 |
| 代码可用性 | 是；不使用 Code Ocean；URL：https://github.com/leeR1ven/born-wired-cortex |
| 先前互动 | 以上皆非（投稿前未与任何编辑联系过） |

## 四、状态时间线

| 时间（UTC） | 事件 | 来源 |
|---|---|---|
| 2026-09-17 07:17 | 投稿提交成功（submitted） | Research Square 时间线 |
| 2026-09-17 12:22 | **编辑已指派**（editor assigned） | Research Square 时间线 |
| 2026-09-18 | 投稿系统"当前阶段：稿件已收到" | 投稿系统稿件页 |
| 2026-09-18 | 邮件回执（仅确认收稿，不代表会送外审） | 编辑部邮件 |

> 投稿当晚出现过一次断网，正好发生在点"批准提交"的瞬间。事后核对系统：稿件已收到、文件齐全，未受影响。

## 五、Research Square 预印本

| 项目 | 内容 |
|---|---|
| RSID | **rs-11070398** |
| 私有仪表盘 | https://www.researchsquare.com/article/rs-11070398/private/timeline |
| 2026-09-18 | 选择加入 In Review（"审核中"预印本服务），系统标记 `hasOptedInToPreprint = true` |
| 当前状态 | 预印本尚未发布（`isPublished = false`，草稿） |
| 本次选择 | **等待自动发布**：最多 3 周内，预印本连同"正在 Nature Machine Intelligence 审议中"的期刊信息一起发布，并分配 DOI |
| 另一条路（未选） | 页面上的"发布我的预印本"可立即发布，但暂不带期刊信息；期刊信息后续自动补上 |

## 六、后续待办

1. 等 Research Square 邮件（预印本上线通知，含 DOI）→ 把 DOI 补进第五节。
2. 等编辑部决定（送外审 or 退稿）。收到任何编辑部来信，原文转给 AI 处理。
3. 若被退稿：全长版（`paper/Born_wired_manuscript.pdf`，含 Author Summary、218 词摘要）已备好，可直接转投 PLOS Computational Biology。

## 七、怎么自己核对

* 投稿系统记录：登录 https://mts-natmachintell.nature.com → 作者首页 → 活体手稿 → 查看稿件编号。
* 预印本记录：登录 https://www.researchsquare.com → 仪表板 → 我的文章。
* 提交的到底是哪一版：比对本节第二节两张 PDF 的 SHA-256。
