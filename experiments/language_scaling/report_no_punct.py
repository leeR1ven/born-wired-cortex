"""Report paired retained-character targets, not incomparable overall scores."""
import json
from pathlib import Path
import sys
from collections import Counter

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_no_punct_v1'
NAMES={'adjacent_linear':'相邻字／线性','adjacent_diminishing':'相邻字／递减',
       'context_diminishing':'短上下文／递减','context_saturating':'短上下文／饱和',
       'long_context_diminishing':'长上下文／递减'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def display(s):return s.replace('\n','↵').replace('\r','').replace('\t','⇥').replace(' ','␠').replace('`','ˋ')


def main():
    summary=read(OUT/'summary.json');matched=read(OUT/'matched_comparison.json')
    manifest=read(HERE/'data_no_punct_v1/manifest.json')
    tr=manifest['splits']['train']
    lines=['# 删除标点后重新训练：配对比较','',
           '2026-09-13。按照用户要求，仅删除Unicode P类标点，保留空格、换行、字符顺序和原训练／验证／测试分区。5个版本的模型代码、参数、增权规则与上一轮相同，全部从零训练。','',
           f'原训练文本{tr["original_characters"]:,}字符，删除{tr["removed_count"]:,}个标点后剩{tr["characters"]:,}字符。仍包含{tr["whitespace_count"]:,}个空白字符。每个版本都学习整个过滤后的训练分区。运行共{summary["elapsed_seconds"]:.2f}秒（包含训练、常规评估及存档校验）。','',
           '## 同一批汉字上的结果','',
           f'沿原测试索引逐字映射，比较同一批 **{len(matched["targets"])} 个汉字目标**。没有把原来含标点的22.1%与新测试集的总体分数直接比较。删除输入标点也改变了相邻关系及以字符计的上下文跨度；因此这是整个过滤操作的效果。','',
           '输出屏蔽对照使用旧模型和原输入，只把标点候选的输出概率置零再归一化，不重训。它用于区分候选变化与删标点重训的效果。','',
           '|版本|原模型|只屏蔽标点输出|删标点后重训|改善位置／退步位置|',
           '|---|---:|---:|---:|---:|']
    for name,c in matched['conditions'].items():
        p=c['paired_summary']
        vals=[c[k]['metrics']['top1'] for k in ('original','output_mask_only','no_punct')]
        lines.append(f'|{NAMES[name]}|{vals[0]:.2%}|{vals[1]:.2%}|{vals[2]:.2%}|{p["helped"]}/{p["harmed"]}|')
    adjacent=matched['conditions']['adjacent_linear']['paired_summary']
    long=matched['conditions']['long_context_diminishing']['paired_summary']
    lines+=['',f'相邻字版本在这批目标上从{adjacent["original_correct"]}/{adjacent["n"]}提升到{adjacent["no_punct_correct"]}/{adjacent["n"]}。长上下文版本虽然命中{long["no_punct_correct"]}个目标，但实际只输出一种字符；应结合下面的输出分布判断，不能把高频字命中当作上下文区分成功。','']
    lines+=['','## 输出是否转向其他高频字符','',
            '|版本|配对测试中预测过的字符种类|最常预测的字符与次数|','|---|---:|---|']
    for name,c in matched['conditions'].items():
        p=c['paired_summary']
        top='、'.join(f'`{display(ch)}` {count}次' for ch,count in p['no_punct_most_predicted'][:4])
        lines.append(f'|{NAMES[name]}|{p["no_punct_predicted_types"]}|{top}|')
    lines+=['','␠表示空格，↵表示换行。本轮保留这些空白字符。','',
            '## 学习量变化','',
            '下表来自过滤后新抽取的固定1024个非空白位置，可用于本轮内部学习曲线；跨轮比较以上面的859个配对汉字为准。','',
            '|版本|训练字符|Top-1|Top-5|已知字BPC|','|---|---:|---:|---:|---:|']
    for name,c in summary['conditions'].items():
        for stage in c['stages']:
            m=stage['test']
            lines.append(f'|{NAMES[name]}|{stage["stage"]:,}|{m["top1"]:.2%}|{m["top5"]:.2%}|{m["bpc_known"]:.3f}|')
    lines+=['','## 原始续写','',
            '每个版本取保存文件中的第一个固定样例，不筛选、不修文。其余续写保存在各模型目录continuations.json。']
    for name in matched['conditions']:
        ex=read(OUT/name/'continuations.json')[0]
        lines+=['',f'### {NAMES[name]}','',f'输入：`{display(ex["prompt"])}`','',f'输出：`{display(ex["greedy"])}`']
    lines+=['','## 文件与验证','',
            '- data_no_punct_v1/manifest.json记录删除规则、每分区数量和来源哈希；old_to_new_positions.npz保留位置映射。',
            '- results_no_punct_v1/matched_comparison.json保存859个目标的逐项原预测、新预测、输出屏蔽对照、改善和退步数。',
            '- 核验结果另见VERIFIED_NO_PUNCT.md；原模型与旧结果保留。',
            '- 本实验检验去标点是否缓解当前字符关联模型的输出偏置，不把下一字预测等同于对话、逻辑或AGI。','',
            '复跑训练：','', '```powershell',
            "& '.\\.venv\\Scripts\\python.exe' -X utf8 '语言规模实验_language_scaling_20260913/run_experiment.py' --data '语言规模实验_language_scaling_20260913/data_no_punct_v1' --out '语言规模实验_language_scaling_20260913/results_no_punct_rerun' --max-chars 800000",'```']
    path=HERE/'去标点实验结果.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(path))


if __name__=='__main__':main()
