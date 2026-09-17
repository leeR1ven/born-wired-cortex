"""Render raw examples, simultaneous activity, and the actual reactivation edges."""
import json
from pathlib import Path
import numpy as np
from hebb_text import CharacterHebb

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label(ch):
    return {' ': '空格', '\n': '换行', '\t': '制表'}.get(ch, ch)


def main():
    first = HERE/'persistent_candidates_v1'
    second = HERE/'candidate_budget_v1'
    s1 = read(first/'summary.json')
    s2 = read(second/'summary.json')
    assert s1['completed'] and s2['completed']
    a = read(first/'diminishing_658462_k4_test.json')
    b = read(second/(s2['selected']+'_test.json'))
    strong8 = read(second/'k8_b1p0_test.json')
    baseline = read(second/'baseline_test.json')
    graph = CharacterHebb.load(first/'diminishing_658462.npz')
    before = graph.weight_hash()
    row = a['silent_trajectories'][0]['rows'][4]
    activity = {graph.ids[x['character']]:x['amplitude'] for x in row['persistent']['all']}
    ids = np.array(list(activity), dtype=np.int64)
    values = np.array(list(activity.values()), dtype=np.float32)
    raw = (graph.weights[ids]*values[:,None]).sum(axis=0)
    target = graph.ids['個']
    contributions = [dict(source=graph.vocabulary[i], activity=value,
        weight=float(graph.weights[i,target]), current=float(np.float32(value)*graph.weights[i,target]))
        for i,value in activity.items()]
    without_self = raw.copy()
    for i,value in activity.items():
        without_self[i] -= np.float32(value)*graph.weights[i,i]
    def ranking(values):
        return [dict(character=graph.vocabulary[int(i)], current=float(values[i]))
                for i in np.argsort(-values, kind='stable')[:8]]
    case = dict(source='persistent_candidates_v1/diminishing_658462_k4_test.json',
        trajectory='silent_trajectories[0]', step=4, activity=row['persistent']['all'],
        injected_candidates=row['injected_candidates'], next_activity=row['next_persistent']['all'],
        target='個', contributions=contributions, actual_top8=ranking(raw),
        without_self_edges_top8=ranking(without_self),
        counterfactual='Subtract every active self-edge contribution for THIS STEP only; no retraining or rollout claim',
        model_weight_sha256=before)
    assert graph.weight_hash() == before
    (HERE/'candidate_reactivation_audit.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 多候选持续激活实验', '',
        '日期：2026-09-13。实现了跨步保留多个候选，以及不强制输出字符的静默联想模式。多个字确实同时活跃；当前结果仍会集中于常见字，尚未形成围绕输入场景持续展开的表达。', '',
        '## 实际续写', '',
        '以下均取冻结提示列表的第一条，没有挑选最好看的输出。前三种条件使用同一组训练权重，只有回忆时的候选保留不同。', '',
        '> 输入：'+a['trajectories'][0]['prompt'], '',
        '**不保留预测候选（仅近期输入活动）：**', '',
        '> '+baseline['trajectories'][0]['emitted'], '',
        '**保留前4名候选，总活动预算1：**', '',
        '> '+a['trajectories'][0]['emitted'], '',
        '**保留前8名候选，总活动预算1：**', '',
        '> '+strong8['trajectories'][0]['emitted'], '',
        '输出出现了较长的局部词串，但随后重复同一片段；局部看起来更像一句话，不代表模型已经按前文的意思作答。较弱反馈通过验证集选出前8名、总预算0.25，其第一条输出与原五字循环相同。', '',
        '## 多个字是同时激活的', '',
        '静默模式在读完提示后不再把赢家字符回灌，也不强迫模型组成一句话，只让整组活动沿同一连接图继续传播。下面是前4名、预算1模式下同一条提示的真实活动；每行是同一时刻同时活跃的一组字符，顺序按强度排列。', '',
        '| 内部推进次数 | 同时活跃的前8个字符及强度 |',
        '|---|---|']
    for step in (0,1,4,8,16,32,63):
        state = a['silent_trajectories'][0]['rows'][step]['persistent']['top'][:8]
        lines.append(f'| {step} | '+ '；'.join(f'{label(x["character"])} {x["amplitude"]:.3f}' for x in state)+' |')
    lines += ['',
        '“轮换”指活动强度和排名变化，不是一次只激活一个字。前文相关的“進、來、去”等活动较快消退，随后常集中于“一、個、人、不、是、了”等字。这些记录能显示关联活动，却不能仅凭字群就确定模型在思考某个具体意思。不同提示最后的排名不同，也不等于形成了不同的语义主题。', '',
        '## 为什么已激活的字还会被激活', '',
        '当前实现只记录连续活动强度。每步先将旧活动乘0.25，再把前K候选的归一化激励加回；候选可以包含已经活跃的字。每个字的活动上限为1，低于0.02删除。同一组字可以通过组内连接持续维持彼此，不需要每一步都走向从未活跃的字。', '',
        '例如上表第4步，“一、個、說、是、不、的”同时活跃。共同传递电流后的前4名是“人、個、是、一”：既有之前未活跃的“人”，也有已经活跃的三个字。', '',
        '已活跃的“個”得到以下实际电流：', '',
        '| 来源 | 源活动 | 到“個”的权重 | 电流贡献 |',
        '|---|---:|---:|---:|']
    for c in contributions:
        lines.append(f'| {c["source"]} | {c["activity"]:.3f} | {c["weight"]:.3f} | {c["current"]:.3f} |')
    lines += ['',
        '其中最大的贡献来自“一→個”，约8.458；“個→個”自身的贡献约1.381。只在这一步扣除所有活跃字的自连接贡献，前四名仍是“人、是、個、好”，“個”仍会得到再次激励。这不是仅靠自连接造成的。这个反事实只计算一个步骤，没有声称删除自连接后整条轨迹完全相同。', '',
        '目前还没有“点火后暂时不能再点火”的不应期、随重复活动上升的阈值或疲劳状态，也没有强制排除已经活跃的候选。若要规定当前活跃者只能驱动新的激活事件，需要明确增加相应状态规则；扩大候选数量本身不会自动实现这种约束。', '',
        '## 三个因素的对照', '',
        '第一轮冻结12个验证条件：训练量100,000或658,462字符 × 递减增长或上限增长 × 不保留、保留4名、保留8名候选。先在256个验证位置选出完整数据的多候选版本，再在512个新测试位置比较对应交叉条件。第二轮固定已选中的权重，另用验证位置比较候选总预算0.25、0.5、1；选择后才评分另512个新测试位置。', '',
        '- **数据量：**100,000字版本的多候选输出集中于换行；训练到658,462字后可产生上面的词串，概率损失也改善，但仍循环。这说明更多经历有影响，不能据此断定扩大到什么规模就会形成逻辑。训练使用顺序前缀，增加数量也改变了接触的内容，不能把变化完全归为数量本身。',
        '- **增长函数：**原规则已按 `0.25×活动/(1+w)` 减速，但没有上限。另从零连续训练 `0.25×活动×max(0,1-w/4)` 版本，让弱边相对更易增长、强边接近4时停止增长。它没有稳定改善本轮表现；前4名强反馈下反而变成三字循环。这个结果只针对上限4等本轮参数。',
        '- **多候选：**候选已经跨步保存，未被输出的字也实际参与后续。反馈过强会增加常见字之间的相互维持，减弱后可以缓解对真实输入的干扰，但本次仍未提高原文命中率或消除输出循环。', '',
        '第二轮同一批512个测试位置：', '',
        '| 模式 | 原文下一字命中 | BPC ↓ | 平均持久活跃数 |',
        '|---|---:|---:|---:|']
    for key, title in [('baseline','不保留候选'), (s2['selected'],'前8名，预算0.25'), ('k8_b1p0','前8名，预算1')]:
        result = s2['test_results'][key]
        metrics = result['metrics']
        lines.append(f'| {title} | {round(metrics["top1"]*metrics["n"])}/{metrics["n"]} | {metrics["bpc_known"]:.3f} | {result["mean_active_count"]:.2f} |')
    lines += ['',
        '评分标准为匹配原文实际下一字，不是判定所有其他续写都不合理。第二轮测试有1个词表外目标，命中率将其计为未命中，BPC仅统计另外511个目标。两个实验使用不同测试位置，百分比不能跨轮直接比较。新位置仍来自此前两本书的同一留出文本，不是独立新书。', '',
        '## 规则和检查', '',
        '- 所有训练仍只输入真实语料，使用同一张字符连接矩阵、一个连续训练时间流；没有按句子创建记忆或添加答案连接。本轮没有让猜出的候选参与权重学习。',
        '- 回忆只用一个活动字典和一个时钟。每一步先从旧活动计算电流、选候选，再加入实际输入；没有使用尚未呈现的目标调整当前预测。',
        '- 改变K时保持候选总激励预算相同，加入活动后上限1、最多64个单元。因为阈值和重合字合并，保留后的实际单元数不必等于K。',
        '- 测试各取真实前64字来启动瞬态活动，评分时不更新权重。自由生成会回灌输出字；静默联想不回灌赢家。三种过程分开记录。',
        '- 没有采样、禁止重复、隐藏语言模型或根据原文答案选择输出。静默活动不被包装为可读句子。',
        '- 活动总量受限。这里说的正反馈是组内偏向和维持，不是数值无限增长。输出周期按最后32个字符检查1至16步重复；没有精确状态周期不等于产生了语言能力。',
        '- 原核心、旧实验和宠物模型没有修改。新增实验模块 persistent_candidates.py；核心等价、因果性、候选跨步保留、活动预算与静默模式的小测试已通过。',
        f'- 两轮实验运行耗时分别为{s1["elapsed_seconds"]:.2f}秒、{s2["elapsed_seconds"]:.2f}秒，本地CPU完成。独立核验记录：VERIFIED_PERSISTENT_CANDIDATES.md。', '',
        '## 查看自己的输入会激活哪些字', '',
        'query_candidates.py 默认使用第二轮选中的前8名、预算0.25，以静默模式显示每一步的活跃字和强度。例：', '',
        '```powershell',
        '& "F:\\一种AGI架构\\.venv\\Scripts\\python.exe" -X utf8 "F:\\一种AGI架构\\语言规模实验_language_scaling_20260913\\query_candidates.py" "見他進來" --mode silent --steps 12',
        '```', '',
        '用 `--mode speak` 可查看逐字续写，`--k 4 --budget 1` 可切换到上面的强反馈对照；这些选项只改变当前回忆活动，不改存档权重。', '',
        '完整数据：persistent_candidates_v1/、candidate_budget_v1/。已激活字再次得到电流的逐边记录：candidate_reactivation_audit.json。旧模型候选覆盖：候选覆盖核查.md。', '',
    ]
    (HERE/'多候选持续激活实验结果.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote 多候选持续激活实验结果.md and candidate_reactivation_audit.json')


if __name__ == '__main__':
    main()
