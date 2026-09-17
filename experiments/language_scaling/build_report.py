"""Report every condition; model selection uses validation loss only."""
import json
import platform
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results_v1'
NAMES={'adjacent_linear':'相邻字／线性增强','adjacent_diminishing':'相邻字／递减增强',
       'context_diminishing':'短上下文／递减增强','context_saturating':'短上下文／饱和增强',
       'long_context_diminishing':'长上下文／递减增强'}


def main():
    result=json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))
    assert result['completed']
    protocol=result['protocol'];conditions=result['conditions']
    selected=min(conditions,key=lambda k:conditions[k]['stages'][-1]['validation']['mean_nll_known'])
    lines=['# 文字直接建立赫布连接：真实文本实验结果','',
           '2026-09-13。按用户最新要求，本轮省略完整宠物模型，让每个字符直接代表一个活动单元，学习有向字间连接。所有版本、阶段和原始续写都保留。','',
           f'本轮实际训练 **{protocol["train_characters"]:,} 个字符／每版本**，词表 **{protocol["characters"]:,} 个字符单元**。运行了 {len(conditions)} 个版本；总运行时间 {result["elapsed_seconds"]:.2f} 秒（包含评估、保存和恢复检查）。没有租用或使用GPU。','',
           '## 材料与有效范围','',
           '材料来自Project Gutenberg公版《紅樓夢》《吶喊》，保留原始繁体及标点。完整段落组成连续块再划分训练、验证和测试，长重复段落先去重。不同分区仍来自同两部作品，可能共享人物、写法和情节；这是同来源未见段落预测，不是跨领域知识或AGI测试。来源与文件哈希见data_sources.md和data/manifest.json。','',
           f'每阶段使用相同的 {protocol["eval_count"]} 个非空白测试位置，预测在揭示该位置的字符之前完成。测试文本不更新连接。全部阶段预先知道训练词表的字符身份，但早期阶段没有未来连接。','',
           '## 学习量曲线','',
           'Top-1/Top-5分母包含测试中的未知字符；BPC（每字符负对数预测概率，越低越好）只计算已知字符。不是阅读理解正确率。','',
           '|版本|已训练字符|Top-1|Top-5|BPC（已知字）|未知字位置|连接数|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,condition in conditions.items():
        for stage in condition['stages']:
            m=stage['test']
            lines.append(f'|{NAMES[name]}|{stage["stage"]:,}|{m["top1"]:.2%}|{m["top5"]:.2%}|{m["bpc_known"]:.3f}|{m["oov_n"]}/{m["n"]}|{stage["stats"]["connections"]:,}|')
    first=next(iter(conditions.values()))['stages'][-1]['unigram']
    lines+=['',f'同一训练材料的单字频率基线：Top-1={first["top1"]:.2%}，Top-5={first["top5"]:.2%}，BPC={first["bpc_known"]:.3f}。这个基线只用于外部比较，不参与模型输出。','',
            f'按照最终阶段的验证集损失选择的版本是 **{NAMES[selected]}**。选择没有使用测试分数。相邻字线性与递减版本的权重都是转移次数的单调函数，故Top-1相同，不把它们算作两个独立结构性成功。','']
    inspection_path=RESULTS/'failure_inspection.json'
    if inspection_path.exists():
        inspection=json.loads(inspection_path.read_text(encoding='utf-8'))
        lines+=['## 标点与汉字分开看','',
                '整体准确率容易掩盖输出偏向标点。下表对最终1024个位置进一步分组：汉字定义为U+3400至U+9FFF，标点按Unicode P类；另有1个其他字符。','',
                '|版本|汉字命中|标点命中|实际预测出的不同字符数|', '|---|---:|---:|---:|']
        for name,item in inspection['conditions'].items():
            h=item['by_target_type']['han'];p=item['by_target_type']['punctuation']
            lines.append(f'|{NAMES[name]}|{h["correct"]}/{h["n"]}|{p["correct"]}/{p["n"]}|{item["predicted_types"]}|')
        ex=inspection['example']
        if ex:
            win=ex['explanation']['actual_model_winner'];target=ex['explanation']['heldout_target']
            lines+=['',f'实际连接审计的一个失败：输入末尾为 `{ex["prefix"][-32:].replace(chr(10),"↵")}`，原文下一字为 `{ex["target"]}`。短上下文模型给 `{win["character"]}` 的总电流为 {win["total_current"]:.4f}，给目标字的电流为 {target["total_current"]:.4f}。逐条来源、活动和连接权重保存在results_v1/failure_inspection.json。这个例子按明确的失败条件选取，用于定位，不用作总体准确率估计。','']
    lines+=[
            '## 上下文与反复传播对照','',
            '以下干预使用同一批128个位置，重新呈现前64个字符、只呈现最后一个字，或打乱前面的字符但保留最后一个字。长上下文模型的这一64字窗口干预不等于整篇连续评估。','',
            '|版本|64字上下文 BPC|仅最后一字 BPC|打乱较早字符 BPC|直接读出 BPC|两次同图传播 BPC|',
            '|---|---:|---:|---:|---:|---:|']
    for name,c in conditions.items():
        iv=json.loads((RESULTS/name/'context_interventions.json').read_text(encoding='utf-8'))
        a,b,d=[iv[k]['metrics']['bpc_known'] for k in ('full','last_only','shuffled_earlier')]
        recur=c['recurrence'];e=recur['no_recurrence']['bpc_known'];f=recur['two_steps']['bpc_known']
        lines.append(f'|{NAMES[name]}|{a:.3f}|{b:.3f}|{d:.3f}|{e:.3f}|{f:.3f}|')
    lines+=['','同图传播使用原来学到的连接，中间最多保留32个活动字符，混合比例0.25；没有添加词组、句子、角色或解题节点。多传播几步本身不算思考成功。','',
            '## 未筛选的贪心续写','',
            '下面按固定位置抽取每个版本的第一个例子，其他例子和两步传播版本均在各目录continuations.json中。不做人工修文、不删除重复。']
    for name in conditions:
        examples=json.loads((RESULTS/name/'continuations.json').read_text(encoding='utf-8'))
        ex=examples[0]
        lines+=['',f'### {NAMES[name]}','', '输入：`'+ex['prompt'].replace('\n','↵').replace('`','ˋ')+'`','',
                '输出：`'+ex['greedy'].replace('\n','↵').replace('`','ˋ')+'`']
    lines+=['','## 能说明与尚未说明的事','',
            '- 预测改善只支持这个字符关联机制从自然文本中学到可复用的统计联系；不能把它直接等同于逻辑思考、对话或AGI。',
            '- 每字一个单元、活动衰减与正权重叠加，会把同一字在不同位置和角色下的活动合在一起。本轮没有实现新概念单元或独立的对象—关系绑定机制。',
            '- 这些限制属于本轮简化版本；本轮没有检验完整前额叶—连续记忆架构的上限，也没有检验无限扩容后的能力。',
            '- 输入为明确的字符身份，不包含真实字形识别、声学特征、肌肉发声或虚拟环境动作。',
            '- 本轮参数在正式运行前冻结，没有根据测试答案逐题加连接；若后续调整，必须作为新实验保留旧结果。','',
            '## 复现与验证','',
            '运行方法见README.md；冻结协议在results_v1/frozen_protocol.json。每个最终model.npz保存所有连接、活动与时钟，query.py可直接读取续写和实际连接。独立验证器及VERIFIED.md另记是否完成重放核验。','',
            f'Python {platform.python_version()}；NumPy {np.__version__}；平台 {platform.platform()}。不同运行库的浮点求和和同分竞争可能影响逐位复现。']
    (HERE/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (RESULTS/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
        selected_by_final_validation=selected),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected':selected,'report':str(HERE/'实验结果.md')},ensure_ascii=False))


if __name__=='__main__':main()
