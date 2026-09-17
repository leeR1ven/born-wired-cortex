"""Build a factual report and inspect preselected helpful/harmful connections."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from hebb_text import CharacterHebb

HERE = Path(__file__).resolve().parent
OUT = HERE / 'short_context_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    summary = read(OUT / 'summary.json')
    assert summary['completed']
    name = summary['selected']
    selected = read(OUT / name / 'test_and_context.json')
    old = read(OUT / 'decay_0p8' / 'test_and_context.json')
    b = CharacterHebb.load(OUT / name / 'model.npz')
    saved = (b.weight_hash(), b.activity.copy(), b.clock)
    full = selected['context_interventions']['full']['rows']
    last = selected['context_interventions']['last_only']['rows']
    examples = []
    # First chronological case in each category; not chosen for eloquent prose.
    for kind in ('helped', 'harmed'):
        eligible = [(a, c) for a, c in zip(full, last)
                    if (a['rank'] == 1 and c['rank'] != 1 if kind == 'helped'
                        else a['rank'] != 1 and c['rank'] == 1)]
        a, c = eligible[0]
        b.reset_activity_for_probe()
        for ch in a['prefix']:
            b.observe(ch, learn=False)
        destinations = list(dict.fromkeys([a['target'], a['predicted'], c['predicted']]))
        connections = []
        for i, activation in b.activity.items():
            for destination in destinations:
                j = b.ids[destination]
                connections.append(dict(source=b.vocabulary[i], activation=activation,
                    destination=destination, weight=float(b.weights[i, j]),
                    current=float(np.float32(activation) * b.weights[i, j])))
        currents = b.currents()
        examples.append(dict(kind=kind, position=a['position'], prefix=a['prefix'],
            target=a['target'], full_prediction=a['predicted'], last_only_prediction=c['predicted'],
            activity=[dict(character=b.vocabulary[i], amplitude=v) for i, v in b.activity.items()],
            connections=connections,
            total_currents={ch: float(currents[b.ids[ch]]) for ch in destinations}))
    b.activity, b.clock = saved[1], saved[2]
    assert b.weight_hash() == saved[0]
    (OUT / 'connection_examples.json').write_text(json.dumps(dict(
        selection='first chronological helped case and first chronological harmed case',
        model=name, weight_sha256=saved[0], examples=examples), ensure_ascii=False, indent=2), encoding='utf-8')

    p = summary['protocol']
    v = summary['candidates'][name]
    sm = selected['test']['metrics']
    om = old['test']['metrics']
    effect = selected['context_effect']
    lines = [
        '# 短上下文字符赫布实验', '',
        '日期：2026-09-13。完成目录：`short_context_v2/`。', '',
        f'沿用去标点语料，单独比较约 2、3、6、10、18 字范围。按验证集概率损失选出的约 {v["span"]} 字版本，在同一批新测试位置上答对 '
        f'{round(sm["top1"] * sm["n"])} / {sm["n"]}（{sm["top1"]:.2%}）；原约 18 字版本答对 '
        f'{round(om["top1"] * om["n"])} / {om["n"]}（{om["top1"]:.2%}）。缩短范围有效，但自由续写仍出现循环。', '',
        '## 本次改了什么', '',
        '- 只调整活动衰减参数；字符赫布学习、权重增长函数、训练顺序、语料和读出规则保持一致。没有添加答案、词组表或任务专用连接。',
        '- 每个字符代表一个单元，使用同一张有向连接矩阵；本次是用户提出的简化字符实验，不是完整宠物模型或大前额叶。',
        f'- 训练 {p["train_characters"]:,} 个字符，{p["vocabulary_characters"]:,} 种字符。仅删除 Unicode 标点，空白保留。矩阵占用 {v["stats"]["matrix_bytes"]:,} 字节（约 74.2 MiB），在本地 CPU 完成。',
        '- 学习规则为 `Δw = 0.25 × 源单元活动 / (1 + 原权重)`。本次选中活动衰减 0.25、删除阈值 0.02：最近三个位置的活动为 1、0.25、0.0625；更早位置降至阈值以下。重复字符仍共用一个单元，保留最近一次活动。',
        '- 输出是各活跃字符沿已学习连接传递的电流之和。它利用近期多字，但没有额外的“整个三字组合”单元。', '',
        '## 验证集选参', '',
        '五个版本分别从零学习同一条完整训练流。预先规定用验证集的平均负对数概率损失选参；先保存 selection.json，再计算测试分数。BPC 与该损失只差固定换算，越低越好。Top-1 指首选字符正确，Top-5 指前五个候选包含正确字符。', '',
        '| 有效范围 | 活动衰减 | 验证 Top-1 | 验证 Top-5 | 验证 BPC ↓ |',
        '|---|---:|---:|---:|---:|',
    ]
    for candidate_name, candidate in summary['candidates'].items():
        m = candidate['validation']
        label = f'约 {candidate["span"]} 字' + ('（选中）' if candidate_name == name else '')
        lines.append(f'| {label} | {candidate["config"]["trace_decay"]} | {m["top1"]:.2%} | {m["top5"]:.2%} | {m["bpc_known"]:.3f} |')
    lines += ['', '约 2 字版本的验证首选准确率较高，约 3 字版本的概率损失略低。两者差距很小；这里遵循事先选择的概率损失指标，没有根据测试结果更换标准，也不能据此认定 3 字是普遍最优范围。', '',
        '## 同一批测试位置的比较', '',
        '测试选取同一留出文本中此前未计分的 1,024 个汉字位置，排除前两轮用过的两批位置。它们仍来自相同书籍的留出文本，部分文字曾作为其他预测的上下文出现，不是全新书籍上的独立外部测试。', '',
        '| 条件 | 答对 / 1,024 | Top-1 | Top-5 | BPC ↓ |',
        '|---|---:|---:|---:|---:|',
    ]
    comparisons = [('原约 18 字版本', om),
        ('选中约 3 字版本', sm),
        ('同一个 3 字模型，测试时只输入最后一个字', selected['context_interventions']['last_only']['metrics']),
        ('独立的一字相邻连接对照', summary['one_character_control']),
        ('3 字模型再沿同一图传播两轮', selected['two_recurrent_steps']['metrics'])]
    for label, m in comparisons:
        lines.append(f'| {label} | {round(m["top1"] * m["n"])} | {m["top1"]:.2%} | {m["top5"]:.2%} | {m["bpc_known"]:.3f} |')
    lines += ['',
        '这里的“正确”严格指与原文实际下一字一致，不是判断语言是否合理。例如原文“我想回家”，输入“我想回”，输出“家”算命中，输出“来”算未命中；但“我想回来”同样可以是合理表达。原文只是一个参考续写，并不穷尽合理答案。', '',
        '计分时每个位置都使用真实原文的前文，不把上一次预测接回去；因此这个分数衡量真实上下文下的下一字预测，不能等同于自由对话或连续生成能力。自由续写才会把模型自己的输出接回输入，二者分别检查。', '',
        '验证位置中有 1 个未登录字符，测试位置中有 3 个未登录字符；准确率将它们算作失败，BPC 仅统计模型词表中存在的目标。测试不更新权重，每次检查后恢复训练活动和时间状态。', '',
        '## 前面的字有没有用上', '',
        f'用上了。保持同一模型的权重不变，只移除最后一个字之前的输入，{effect["prediction_changed_with_last_only"]} / {effect["n"]} 个位置的首选字符发生变化。较早上下文使 {effect["earlier_context_helped"]} 个位置从错误变正确，也使 {effect["earlier_context_harmed"]} 个位置从正确变错误，净少答对 9 个。', '',
        '不过，保留多字上下文时 BPC 从 8.223 降到 7.942，正确字符的整体概率质量有所改善。因此不能简单说上下文完全没用：它改善了概率分配，但还没有稳定提高首选字符准确率。独立的一字模型是另一组训练权重，不应与“同一个模型只输入最后一字”的干预混为一谈。', '',
        '补充干预保持最后一字不变，打乱它之前的 63 字：', '',
        '| 3 字模型输入 | Top-1 | Top-5 | BPC ↓ |',
        '|---|---:|---:|---:|',
    ]
    for key, label in [('full', '原连续上下文'), ('last_only', '只保留最后一字'), ('shuffled_earlier', '打乱较早 63 字')]:
        m = selected['context_interventions'][key]['metrics']
        lines.append(f'| {label} | {m["top1"]:.2%} | {m["top5"]:.2%} | {m["bpc_known"]:.3f} |')
    lines += ['',
        '打乱干预也改变了进入有效短窗口的字符内容，因此不是纯词序消融。原上下文没有表现出优于这个打乱条件的明确优势；当前概率改善可能包含一般的概率平滑效应，尚不能据此声称学会了语义关系。', '',
        '## 直接查看连接', '',
        '以下取时间顺序上的第一个“上下文帮对”与第一个“上下文干扰”案例，实际连接、电流和活动保存在 `short_context_v2/connection_examples.json`。', '',
    ]
    for example in examples:
        lines += [f'**{"帮对" if example["kind"] == "helped" else "干扰"}，测试位置 {example["position"]}**：输入末尾“{example["prefix"][-20:]}”，真实下一字“{example["target"]}”；完整短上下文输出“{example["full_prediction"]}”，只留最后一字输出“{example["last_only_prediction"]}”。', '',
            '| 来源字 | 活动 | 输出候选 | 连接权重 | 电流贡献 |',
            '|---|---:|---|---:|---:|']
        for item in example['connections']:
            lines.append(f'| {item["source"]} | {item["activation"]:.4f} | {item["destination"]} | {item["weight"]:.4f} | {item["current"]:.4f} |')
        lines += ['', '候选总电流：' + '；'.join(f'“{ch}” {current:.4f}' for ch, current in example['total_currents'].items()) + '。', '']
    lines += ['## 自由续写', '',
        '提示固定取冻结测试位置序列中每隔 256 个位置的前 32 字，共四条；每条贪心续写 48 字。以下原样展示第一条，全部输出在 test_and_context.json。没有用语言模型润色或补写。', '',
        f'> 输入：{selected["continuations"][0]["prompt"]}', '',
        f'> 输出：{selected["continuations"][0]["greedy"]}', '',
        '四条续写都进入类似“的人道你們……”的循环。额外传播两轮使首选正确数从 169 增至 171，仍未消除续写循环；这里的传播只是同一字符图的两轮激活，不能当作复杂前额叶规划的证据。', '',
        '## 复核与文件', '',
        f'- 本轮五版本训练与评估总耗时 {summary["elapsed_seconds"]:.2f} 秒；最终证据目录为 `short_context_v2/`。',
        '- 第一轮 short_context_v1 在比较 float32 累加结果时使用了过严的逐位概率相等检查：仅一个位置有约 1.86×10⁻¹² 的概率差，排名和首选均相同。原始文件和失败诊断已保留。第二轮只改核查的浮点容差并重跑，训练核心、参数、选参规则与测试位置均未改。',
        '- 独立复核脚本：`verify_short_context.py`；复核结论：`VERIFIED_SHORT_CONTEXT.md`。',
        '- 模型：`short_context_v2/decay_0p25/model.npz`；完整分数：`short_context_v2/summary.json`；逐位置记录和续写：`short_context_v2/decay_0p25/test_and_context.json`。',
        '- 运行代码：`run_short_context_v2.py`。为保护已冻结证据，它拒绝覆盖现有输出目录。', '',
        '本轮支持的结论是：短活动窗口可以显著减轻当前字符赫布图的高频输出干扰；近期多字确实影响了结果，但目前仍主要表现为字符关联与概率平滑，尚未形成稳定的连续表达。本次没有增加组合单元或改变完整架构，结果只适用于这个简化实现。', '',
    ]
    (HERE / '短上下文实验结果.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote 短上下文实验结果.md and short_context_v2/connection_examples.json')


if __name__ == '__main__':
    main()
