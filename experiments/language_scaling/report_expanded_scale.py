"""Summarize scaling, rare-character coverage and raw unchanged continuations."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'expanded_scale_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label(ch):
    return {' ':'空格','\n':'换行'}.get(ch,ch)


def main():
    summary=read(OUT/'summary.json')
    rare=read(HERE/'rare_character_probe_v1/summary.json')
    assert summary['completed'] and rare['completed']
    first=summary['stages'][0]
    final=summary['stages'][-1]
    start=first['stage']
    end=final['stage']
    first_result=read(OUT/f'{start}_persistent8_weak.json')
    last_result=read(OUT/f'{end}_persistent8_weak.json')
    coverage=read(OUT/f'{end}_coverage.json')
    han=[r for r in coverage['per_character'] if '\u3400'<=r['character']<='\u9fff']
    seldom=[r for r in han if 1<=r['count']<=9]
    common=[r for r in han if r['count']>=1000]
    common_share=sum(r['count'] for r in common)/sum(r['count'] for r in han)
    lines=['# 扩大语料、词频覆盖与多候选实验', '',
        f'训练从 {start:,} 字符连续增加到 {end:,} 字符（{end/start:.2f} 倍）。新增作品带来了更多连接和部分预测改善；词频覆盖仍高度不均，保存的自由续写仍重复。本轮没有得出“数据量已经足够”或“数据量是唯一原因”的结论。', '',
        '## 先看真实续写', '',
        '以下使用固定的前8名候选、总活动预算0.25。各阶段使用相同输入、相同规则、同一张持续训练的图，输出没有润色。第一条是冻结列表的第一条旧提示，第二条是第一条新增提示。', '']
    for index in (0,4):
        lines += [f'**例{1 if index==0 else 2}：**', '', '> 输入：'+first_result['trajectories'][index]['prompt'], '']
        for stage in summary['stages']:
            result=read(OUT/f'{stage["stage"]}_persistent8_weak.json')
            lines += [f'训练 {stage["stage"]:,} 字符后：', '', '> '+result['trajectories'][index]['emitted'], '']
    lines += ['这些输出说明学习规模会改变联想，例如新增了“西→岐”“西→門外面”等局部衔接，但本轮没有持续接着原场景叙述。局部词串变化不等于理解了原场景。全部8条提示、三种固定回忆模式的输出都保留在各阶段JSON中。', '',
        '## 每个字究竟学了多少次', '',
        f'本轮固定词表有 {summary["protocol"]["vocabulary_characters"]:,} 个字符，其中按 U+3400–U+9FFF 统计的汉字为 {len(han):,} 个。四阶段预先分配同一词表，尚未出现的字对应的连接仍为零。', '',
        '| 汉字在训练流中出现的次数 | 初始阶段字种数 | 全量阶段字种数 |',
        '|---|---:|---:|']
    for bucket in ('0','1-9','10-99','100-999','1000+'):
        lines.append(f'| {bucket} | {first["coverage"]["han_character_counts"].get(bucket,0):,} | {final["coverage"]["han_character_counts"].get(bucket,0):,} |')
    lines += ['',
        f'全量阶段仍有 {len(seldom):,} 个汉字只见过1–9次，占已见汉字种类的 {len(seldom)/len(han):.1%}，它们平均只有 {sum(r["outgoing_connections"] for r in seldom)/len(seldom):.2f} 条非零出边。出现至少1,000次的 {len(common):,} 个汉字贡献了 {common_share:.1%} 的汉字输入，平均出边 {sum(r["outgoing_connections"] for r in common)/len(common):.2f} 条。', '',
        f'所有字符的非零连接总数从 {first["stats"]["connections"]:,} 增至 {final["stats"]["connections"]:,}。新增连接不等于形成了能区分句意的组合；同时，当前结果也不支持把“仍循环”解释成完全没有学习。', '',
        '## 专门检查少见字', '',
        '自然抽取的新作品256个目标在全量阶段没有1–9次这一组，容易漏掉稀字问题。因此另外冻结了事后诊断：按全量训练频次分为五组，每组最多128个新测试位置，排除主实验位置；每组在初始和全量模型上使用相同目标。分组依据始终是全量频次，不随阶段变化。', '',
        '下表是前8名、弱反馈模型。BPC描述对原文实际下一字分配的概率，越低越好；首选命中也单独报告。', '',
        '| 全量训练频次 | 目标数 | 初始命中 | 全量命中 | 初始BPC ↓ | 全量BPC ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    r0=rare['results'][f'{start}_persistent8_weak']
    r1=rare['results'][f'{end}_persistent8_weak']
    for bucket in ('0','1-9','10-99','100-999','1000+'):
        if bucket not in r1:
            continue
        a,c=r0[bucket],r1[bucket]
        a_bpc='—' if a['bpc_known'] is None else f'{a["bpc_known"]:.3f}'
        c_bpc='—' if c['bpc_known'] is None else f'{c["bpc_known"]:.3f}'
        lines.append(f'| {bucket} | {c["n"]} | {round(a["top1"]*a["n"])} | {round(c["top1"]*c["n"])} | {a_bpc} | {c_bpc} |')
    lines += ['',
        '稀字组的目标概率有所改善，首选命中仍很低。这一分层样本故意增加了低频目标比例，不能将各组简单混合成自然文本总体准确率。频次0组的120个目标在全量训练词表外，当前字符身份编码无法输出它们，不能用这一点单独评判推理能力。其他组在初始阶段虽然有预分配的身份，也可能尚未在训练中出现过。', '',
        '## 是否只是背出原文', '',
        '在主实验的旧256个和新256个目标中，前16字、前64字，以及它们连同真实下一字的连续字符串，在全量训练流中完全出现的数量均为0。前3字则分别有119个和95个在训练中出现过。', '',
        '这些只读查找不进入学习或预测。长前文没出现过，只能排除这些具体长字符串被完整照搬的解释，不能排除已学短片段、常见搭配或较短激活环；也不能直接证明逻辑推理。模型实际依赖的近期活动范围和候选反馈都远小于整段文字的所有信息。', '',
        '## 同一批自然文本位置的比较', '',
        '| 训练字符数 | 旧作品命中/256 | 新作品命中/256 | 旧BPC ↓ | 新BPC ↓ |',
        '|---|---:|---:|---:|---:|']
    for stage in summary['stages']:
        c=stage['conditions']['persistent8_weak']
        a,b=c['old_test'],c['new_test']
        lines.append(f'| {stage["stage"]:,} | {round(a["top1"]*256)} | {round(b["top1"]*256)} | {a["bpc_known"]:.3f} | {b["bpc_known"]:.3f} |')
    lines += ['',
        '新增作品上的预测改善，旧作品上的首选命中略降。本轮先学习旧语料，再连续追加新作品，既增加经历量，也改变了经历内容和分布；不能把这些变化全部归为“数量”一个因素。这些是同作品留出内容，不是跨独立书籍或广泛领域的能力测试。', '',
        '## 同时活跃的字群', '',
        '以下是全量模型在第一条输入之后停止发声、只做内部传播的真实记录。每行中的字同时活跃，并非一次只激活一个；列出的是当前前8名活动强度。', '',
        '| 内部推进次数 | 活跃字及强度 |',
        '|---|---|']
    for step in (0,4,16,63):
        state=last_result['silent_trajectories'][0]['rows'][step]['persistent']['top'][:8]
        lines.append(f'| {step} | '+'；'.join(f'{label(r["character"])} {r["amplitude"]:.3f}' for r in state)+' |')
    lines += ['',
        '当前记录仍主要集中在少量常见字，不能仅凭活跃字群变化判定它在持续讨论一个主题。模型可以维持已激活单元的活动：下一步信号并未排除当前活跃者，本轮没有加入不应期、疲劳或重复禁止规则。', '',
        '## 语料与复现', '',
        '新增六部作品来自Project Gutenberg官方文本；官方条目标注Public domain in the USA，原始许可随下载文件保留：', '']
    for book in read(HERE/'corpus_expansion_v1/manifest.json')['books']:
        lines.append(f'- [{book["title"]}]({book["landing_page"]})')
    lines += ['',
        '完整gzip响应、CRC/ISIZE、严格UTF-8、原文/正文哈希和PG边界均已校验。统一清洗显式移除PG制作/尾注行、孤立排版残留和Unicode P类标点，保留原繁简字形，不做自动翻译。按完整段落及可识别章节组成连续块，新作品按固定种子约80/10/10分配块；源文本有很长段落，因此各块长度不完全相等。跨全部旧/新六分区30字及以上、去空白的完全相同段落交集均为0，未声称排除近重复或共同情节。', '',
        '四阶段词表和回忆配置均在计分前固定；学习规则一直为 `0.25×源活动/(1+w)`，输入痕迹衰减0.25。保留候选只改变回忆活动，猜测不进入权重学习；训练时间流没有重置。起始阶段在旧字符上的完整权重投影逐位等于原模型。', '',
        f'本轮运行耗时 {summary["elapsed_seconds"]:.2f} 秒，矩阵 {final["stats"]["matrix_bytes"]:,} 字节（约 {final["stats"]["matrix_bytes"]/1024**2:.1f} MiB），在本地CPU完成，未使用租用GPU。', '',
        '独立核验：VERIFIED_EXPANDED_SCALE.md；验证器：verify_expanded_scale.py。稀字检查见 rare_character_probe_v1/。', '',
        f'最终模型：expanded_scale_v1/model_{end}.npz。完整逐位置结果、所有固定提示的逐步活动、每字频次与出边数均在 expanded_scale_v1/。', '',
        '可以用 query_candidates.py 的 `--model` 指定这个检查点，以 `--mode silent` 查看整组活动，以 `--mode speak` 查看续写。例如：', '',
        '```powershell',
        f'& "F:\\一种AGI架构\\.venv\\Scripts\\python.exe" -X utf8 "F:\\一种AGI架构\\语言规模实验_language_scaling_20260913\\query_candidates.py" "見他進來" --model "F:\\一种AGI架构\\语言规模实验_language_scaling_20260913\\expanded_scale_v1\\model_{end}.npz" --mode silent --steps 12',
        '```', '',
        '本轮证据支持继续关注稀字与上下文覆盖，尚不能判定再增加多少类似文本就会形成连贯思考。后续若改变采样分布、增长函数或激活后的响应规则，应继续保留独立对照，避免把多个改变的效果混在一起。', '',
    ]
    (HERE/'扩大语料实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote 扩大语料实验结果.md')


if __name__=='__main__':
    main()
