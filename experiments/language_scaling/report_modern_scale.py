"""Report actual completed evidence, never intended corpus size as learned size."""
from pathlib import Path
from run_modern_scale import read,OUT,DATA,HERE

def main():
    s=read(OUT/'summary.json');assert s['completed']
    p=read(OUT/'frozen_protocol.json');m=read(DATA/'manifest.json')
    final=s['stages'][str(s['training_characters'])];main_name=p['main_condition']
    common=p['probes']['common_cues'];cue_map={x['prompt']:x for x in common['fixed_cues']}
    outputs=final['outputs'][main_name]
    cyclic=sum(x['tail_period'] is not None for x in outputs)
    lines=['# 亿字级现代中文与日常对话实验','',
        f"本轮实际学习 **{s['training_characters']:,} 个清洗后字符**，只遍历一遍训练流。用同一套参数记录各规模阶段；规模数字不包含留出集、下载但未使用的文本，也不是把少量文本重复计数。",'',
        f"**实际结果：**主要配置的 {cyclic}/{len(outputs)} 个固定输入，在连续读出64字的测试中都检出1—16字的短周期。熟悉问法也有重复；扩大这批现代语料后，当前简化回路仍未摆脱循环。下面保留出现次数、原始输出及连接核查，不把下字命中率等同于逻辑能力。",'',
        '本轮是“文字本身代表神经元”的简化赫布图实验；它不等同于完整前额叶和连续情景记忆模型的训练。','',
        '## 实际数据与来源','',
        '| 来源 | 训练整段数 | 训练字符 | 其中汉字 |','|---|---:|---:|---:|']
    for name,st in m['source_stats'].items():lines.append(f"| {name} | {st.get('train_documents',0):,} | {st.get('train_characters',0):,} | {st.get('train_han_characters',0):,} |")
    lines += ['',
        'LCCC-base 是中文社交媒体对话，保留原有多轮顺序；来源：[官方介绍](https://github.com/thu-coai/CDial-GPT)、[官方数据卡](https://huggingface.co/datasets/thu-coai/lccc)。官方 README 对数据声明仅限科研，本地此次按科研用途使用；网页上的 MIT 标识不被解释为无条件商业授权。',
        '现代百科文章来自 [Wikimedia 中文维基百科 20231101](https://huggingface.co/datasets/wikimedia/wikipedia)，数据卡标注 CC-BY-SA-3.0/GFDL。使用 shard 00001 的实际正文，来源版本、原文件哈希和许可已保留。','',
        f"清洗后训练词表有 {m['vocabulary_size']:,} 个不同字符，每个字符一个独立单元，没有把生僻字合并为常见字。矩阵占 {m['dense_matrix_bytes']/1024**2:,.1f} MiB。",'',
        '删除 Unicode P 标点、空白和 Cc/Cf 控制/格式字符；LCCC 的分词空格只是数据格式，故去除。保留简繁、大小写和其他字符身份。**这比旧小说实验多删除了空白/控制字符，因此旧小说与本轮不是只有数据规模这一项不同；本轮各阶段的处理则完全一致。**','',
        '先保留 LCCC 官方验证/测试对话，再按完整清洗文本 SHA256 去重；其余整篇/整段按哈希留出各约 0.5%。没有做近重复或短语去重；常见问句可以在不同对话中重复，这正是本轮要检测的熟悉输入。','',
        '## 规则与连续性','',
        '每个训练字符按原先顺序更新其前面活动单元指向当前单元的连接；增量为 ηa/(1+w)，η=0.25，活动每步乘0.25，低于0.02消退。弱连接更容易增长。编译加速逐次执行同样的 float32 运算，没有用字符频数近似替代更新。',
        '对话按约10万字符块输入，文章块按两类训练文本的总字数比例调整，交替输入，使各学习阶段的题材比例接近。同一个活动状态和时钟连续走完。不同对话/文章拼接处也会形成连接；没有虚构分隔符、说话人或额外的按任务记忆。',
        '主要读出沿用此前冻结的8候选、总反馈预算0.25；另测无候选反馈及4候选/预算1。每次输出仍展示最强字符；同时活动的多个候选会继续参与下一步。没有重复禁令、随机改字、检索回答或外部语言模型。','',
        '## 同一批问题上的规模变化','',
        '| 学习字符数 | 对话内部下一字 | 百科内部下一字 | 回复首字 |',
        '|---:|---:|---:|---:|']
    for stage in s['stages'].values():
        metrics=stage['metrics'][main_name]
        fields=[f"{metrics[k]['top1']}/{metrics[k]['n']}" for k in ('lccc_base','wikipedia','reply_first')]
        lines.append(f"| {stage['characters']:,} | {' | '.join(fields)} |")
    lines += ['',
        '每组最多128个整段抽样样本，记录在冻结协议中。前两组每篇/段随机一个非首字符；第三组预测已有上文后的回复首字。输入真实前文，打分之后也不学习测试答案。原文中的下一个字只是一种参考续写，命中率不是“逻辑正确率”。','',
        '**测试边界：**自由读出固定进行64步，当前字符图没有话轮结束/停止发声的单独表示，且本轮删除了标点与空白。它是在测试给定输入后的联想与续写，不是一个已经具备轮流交谈协议的完整聊天系统。循环应结合最初几字、参考回复和活动状态一起看，不能仅凭“没有自行停下”判定整个架构的能力。','',
        '## 常见问法：全部原始输出','',
        '“出现次数”来自实际训练对话与文章的清洗文本：子串出现和对话整句相等分开计数。协议中另存文章/对话各自出现次数。输出按预先固定问法列表展示，未挑选看起来最好的回答。','',
        '| 输入 | 训练子串次数 | 训练整句次数 | 最终模型原始续写 |','|---|---:|---:|---|']
    final_outputs=final['outputs'][main_name]
    for row in final_outputs:
        cue=cue_map.get(row['prompt']);a=str(cue['training_substring_occurrences']) if cue else '见协议';b=str(cue['training_exact_utterances']) if cue else '见协议'
        lines.append('| '+row['prompt'].replace('|','\\|')+' | '+a+' | '+b+' | '+row['emitted'].replace('|','\\|')+' |')
    coverage_path=OUT/'phrase_coverage_audit.json'
    if coverage_path.exists():
        coverage=read(coverage_path)
        lines += ['', '另外全量扫描了训练对话，单独统计“这句话之后，同一段对话里确实还有下一轮回复”的次数：','',
                  '| 输入 | 有真实下一轮回复的训练次数 |','|---|---:|']
        for phrase in ('你好','谢谢','为什么','什么意思','你在干什么'):
            lines.append(f"| {phrase} | {coverage['phrases'][phrase]['with_following_reply']:,} |")
        lines += ['',
            '这项计数不包括作为对话末句的出现，也不把下一篇/下一段的开头当作回复。审计另记录了真实回复首字的分布，用于查看训练提供了哪些关联；这些统计没有输入模型。','']
    lines += ['',
        '## 陌生说法与澄清','',
        '最后三个输入是预先写定的新奇/虚构说法，**并非已证明模型主观“不懂”**。这里只检查是否自行生成有意义的询问、表达不确定或请求补充。评估器会标注疑问关键词，但关键词命中不能自动算作理解、澄清成功或逻辑能力。没有把“我不知道”写成兜底回答。','']
    for row in final_outputs:
        if row['prompt'] in p['probes']['novel_prompts']:lines += [f"- 输入：{row['prompt']}；原始续写：{row['emitted']}"]
    if coverage_path.exists():
        assert all(sum(v.values())==0 for v in coverage['novel_occurrences'].values())
        lines += ['', '后续覆盖核查确认：这三个完整输入在保留的训练文章/对话内出现次数均为0；其中的单字和常见片段可能已经见过。它们的本轮输出没有生成有效的澄清请求。']
    lines += ['',
        '## 多个候选同时激活','',
        '每阶段 evaluation 文件保留“你好”“为什么”“你在干什么”和一个虚构说法的64步无发声轨迹：没有把最强字符回灌为听觉输入，只有候选群继续传播。可以查看活动幅度及新注入候选，而不是把同时激活的候选误当成一句已说出的话。','',
        f"最终主条件 {len(final_outputs)} 个固定输入中，尾部32字符被1—16字符周期精确覆盖的有 {sum(r['tail_period'] is not None for r in final_outputs)} 个。这个指标只描述已观察到的输出循环；未检出短周期不等于语言有意义。",'',
        '## 已观察到的连接原因','']
    audit_path=OUT/'cycle_audit.json'
    if audit_path.exists():
        audit=read(audit_path)
        assert audit['model_weight_sha256']==final['weight_sha256']
        lines += ['对4个预定输入另做160步只读核查。固定权重下，**决定下一步输出的完整活动状态**会精确重现，不仅是展示的一个字符重复。单调时钟不属于这里比较的状态，因为它不参与当前读出计算。','',
                  '| 输入 | 首次状态重现 | 周期 | 最后一步收到正电流的候选数 |',
                  '|---|---|---:|---:|']
        for a in audit['results']:
            repeat=a['first_exact_output_state_repeat']
            lines.append(f"| {a['prompt']} | 第{repeat['first_step']}步→第{repeat['repeated_at']}步 | {repeat['period']} | {a['rows'][-1]['positive_candidates']:,} |")
        lines += ['',
            '例如“你好”轨迹最后一帧，有11,543个字符收到正电流；“不→是”的连接提供554.92电流，“是”自身连接提供51.60。只在该帧去掉自连接后，“是”仍是最强候选。因此至少这个回路存在大量其他可能性，同时仍被既有的相互激活顺序锁住；不能把它解释为网络里已没有别的连接。',
            '上述去自连接只是单步反事实计算，没有改权重，也没有声称已经验证整段删除自连接后的效果。完整候选、活动幅度及贡献值见 modern_scale_v1/cycle_audit.json。','']
    lines += [
        '## 核对与复现','',
        'fast_hebb_verification.json：9组增长/衰减配置，共90,000字符，分块前后所有连接与状态逐项等于原实现。modern_pipeline_verification.json：合成小样本完整流程核对，确保多个检查点、尾块、状态连续和只读评估正确；这些完整性测试不计入语言成绩。',
        '所有原始来源、清洗计数、训练文件哈希、冻结协议、每阶段权重、逐样本得分和未经润色的输出均在本目录。独立复查结论另见 VERIFIED_MODERN_SCALE.md（若已生成）。','',
        f"本轮运行耗时（不含下载与清洗）：{s['elapsed_seconds']:.1f} 秒。最终模型：`modern_scale_v1/{s['final_model']}`。",'',
        '交互读取示例（在本实验目录运行）：','',
        '```powershell',
        "& 'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python313\\python.exe' -X utf8 query_modern.py '你在干什么'",
        "& 'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python313\\python.exe' -X utf8 query_modern.py '你好' --silent --steps 12",'```','',
        '本轮能直接回答的是：在这套固定的字符表示、更新与读出规则下，学习这些现代语料后，常见输入和陌生输入实际激活了什么。对完整架构的能力结论，需要完整架构自身的实验证据。','']
    (HERE/'亿字级现代中文实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT WRITTEN',flush=True)

if __name__=='__main__':main()
