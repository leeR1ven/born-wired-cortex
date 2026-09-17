"""Read-only independent audit of candidate budget, pruning and winner feedback.

Only creates modern_candidate_mechanism_audit_v1; does not train or change the
frozen model, controller, experiment or old reports.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from verify_expanded_scale import Recall, require, same

HERE = Path(__file__).resolve().parent
OUT = HERE / "modern_candidate_mechanism_audit_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rank(raw):
    return np.lexsort((np.arange(len(raw)), -raw))


def describe_frame(recall, label):
    raw = recall.current()
    source_ids = np.array(list(recall.a), np.int64)
    amplitudes = np.array(list(recall.a.values()), np.float32)
    contributions = recall.w[source_ids] * amplitudes[:, None]
    require(np.array_equal(contributions.sum(axis=0), raw), "Complete directed current sum")
    top = rank(raw)[:16]
    targets = []
    for target in top:
        parts = [dict(source=recall.v[int(source)], amplitude=float(amplitudes[j]),
                      weight=float(recall.w[source, target]), current=float(contributions[j, target]),
                      diagonal=bool(source == target)) for j, source in enumerate(source_ids)]
        targets.append(dict(character=recall.v[int(target)], current=float(raw[target]),
                            contributions=parts))
    no_self = contributions.copy()
    for j, source in enumerate(source_ids):
        no_self[j, source] = 0
    altered = no_self.sum(axis=0)
    return dict(label=label, activity=recall.activity(), positive_destinations=int(np.count_nonzero(raw)),
                source_rows=[dict(character=recall.v[int(source)], amplitude=float(amplitudes[j]),
                                  nonzero_out_degree=int(np.count_nonzero(recall.w[source])),
                                  all_destination_current=float(contributions[j].sum(dtype=np.float64)),
                                  self_weight=float(recall.w[source, source]),
                                  self_current=float(contributions[j, source]))
                             for j, source in enumerate(source_ids)],
                top16=targets, no_self_top8=[dict(character=recall.v[int(i)], current=float(altered[i]))
                                           for i in rank(altered)[:8]])


def diagnostic_step(recall, observed):
    raw = recall.current()
    count = recall.cfg["candidate_count"]
    candidates = [int(i) for i in rank(raw)[:count] if raw[i] > 0]
    total = math.fsum(float(raw[i]) for i in candidates)
    previous = recall.a.copy()
    decayed = {i: a*.25 for i, a in previous.items() if a*.25 >= .02}
    allocations = []
    for i in candidates:
        amount = recall.cfg["candidate_budget"] * float(raw[i]) / total
        allocations.append(dict(character=recall.v[i], current=float(raw[i]), injection=amount,
                                old_amplitude=previous.get(i, 0.), survived_pre_prune=i in decayed,
                                fresh_injection_reaches_floor=amount >= .02,
                                before_observed_clamp=min(1., decayed.get(i, 0.) + amount)))
    recall.step(observed, raw)
    for entry in allocations:
        i = recall.ids[entry["character"]]
        entry["final_amplitude"] = recall.a.get(i, 0.)
    return dict(observed=observed, previous_active_count=len(previous),
                pre_prune_survivors=len(decayed), injected_count=len(candidates),
                fresh_survivors=sum(x["fresh_injection_reaches_floor"] for x in allocations),
                injection_sum=math.fsum(x["injection"] for x in allocations),
                allocations=allocations, next_activity=recall.activity())


def main():
    model = HERE / "modern_scale_v1/model_344787049"
    meta = read(model / "metadata.json")
    v = meta["vocabulary"]
    w = np.load(model / "weights.npy", mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256(v.encode("utf8"))
    for start in range(0, len(v), 128):
        digest.update(w[start:start+128].tobytes())
    require(digest.hexdigest() == meta["weight_sha256"], "Full frozen model digest")
    cfg = dict(candidate_count=8, candidate_budget=.25, decay=.25, floor=.02, active_limit=64)
    original = Recall(w, v, cfg)
    original.warm("你好")
    direct = describe_frame(original, "Immediately after two observed characters; nine active units")
    winner = v[int(np.argmax(original.current()))]
    branches = {}
    for branch, observed in (("silent", None), ("winner_feedback", winner)):
        oracle = Recall(w, v, cfg)
        oracle.a = original.a.copy()
        branches[branch] = diagnostic_step(oracle, observed)
        branches[branch]["following_frame"] = describe_frame(oracle, branch)
    change = {x["character"]: x["amplitude"] for x in branches["winner_feedback"]["next_activity"]["all"]}
    for x in branches["silent"]["next_activity"]["all"]:
        change[x["character"]] -= x["amplitude"]
    change = {c: a for c, a in change.items() if a}
    require(set(change) == {winner}, "Only winner clamping differs in controlled pair")

    # Separate K changes on the same state from changes that already happened
    # while hearing the prompt. Both are useful and explicitly labelled.
    common_state_sweep, end_to_end = {}, {}
    for k in (8, 16, 32):
        oracle = Recall(w, v, dict(cfg, candidate_count=k))
        oracle.a = original.a.copy()
        common_state_sweep[str(k)] = diagnostic_step(oracle, None)
        oracle = Recall(w, v, dict(cfg, candidate_count=k))
        oracle.warm("你好")
        warm = oracle.activity()
        trajectory = []
        first_empty = None
        for step in range(8):
            trace = diagnostic_step(oracle, None)
            trace["silent_steps_completed"] = step+1
            trajectory.append(trace)
            if not oracle.a and first_empty is None:
                first_empty = step+1
        end_to_end[str(k)] = dict(after_prompt=warm, silent_trajectory=trajectory,
                                 first_empty_after_silent_steps=first_empty)

    oracle = Recall(w, v, cfg)
    oracle.warm("你好")
    for _ in range(4):
        oracle.step(None)
    require(len(oracle.a) == 8, "Exactly eight active units in specified silent frame")
    eight = describe_frame(oracle, "After 你好 followed by four silent advances; before silent advance 5")
    eight["advance5"] = diagnostic_step(oracle, None)
    eight["following_frame"] = describe_frame(oracle, "After silent advance 5")

    diagonal = np.diag(w)
    report = dict(completed=True, model=str(model.relative_to(HERE)), weight_sha256=digest.hexdigest(),
                  learning_disabled=True, config=cfg,
                  source_sha256={name: sha(HERE/name) for name in
                                 ("hebb_text.py", "persistent_candidates.py", "verify_expanded_scale.py", Path(__file__).name)},
                  diagonal_stats=dict(nonzero=int(np.count_nonzero(diagonal)), maximum=float(diagonal.max())),
                  immediately_after_prompt=direct, controlled_feedback_pair=branches,
                  winner_clamp_increment=change, same_old_state_k_sweep=common_state_sweep,
                  each_k_own_prompt_state_sweep=end_to_end, exactly_eight_source_frame=eight)
    OUT.mkdir(exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    brief = dict(weight_sha256=digest.hexdigest(), direct_active_count=direct["activity"]["count"],
                 direct_effective_count=direct["activity"]["effective_count"],
                 winner=winner, winner_clamp_increment=change,
                 branch_next_top={key:value["following_frame"]["top16"][0]["character"] for key,value in branches.items()},
                 branch_next_totals={key:value["next_activity"]["total"] for key,value in branches.items()},
                 same_state_fresh_survivors={k:r["fresh_survivors"] for k,r in common_state_sweep.items()},
                 same_state_injection_range={k:[min(x["injection"] for x in r["allocations"]), max(x["injection"] for x in r["allocations"])]
                                            for k,r in common_state_sweep.items()},
                 own_warm_counts={k:r["after_prompt"]["count"] for k,r in end_to_end.items()},
                 first_empty={k:r["first_empty_after_silent_steps"] for k,r in end_to_end.items()},
                 eight_sources=eight["activity"]["all"],
                 eight_next=eight["advance5"]["next_activity"]["all"],
                 eight_winner=eight["top16"][0])
    (OUT / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf8")
    lines = [
        "# 多候选活动与自反馈机制核查", "",
        "检查现代最终模型 344,787,049 字检查点；只读，无新训练。证据由独立 Recall 实现重算，模型完整权重 SHA256 与冻结记录一致。", "",
        "当前实现确实把所有活动字符的出边电流相加，然后保留前 K 个候选继续传播。展示最强字符是另一步；发声时还会把这个字符的活动强制置为 1。自连接 w[i,i] 和这种赢家回灌是两种不同机制。", "",
        "代码依据：persistent_candidates.py 38–43 行汇总全部活动出边；63–64 行先衰减并删除弱活动，67–74 行分配总预算，75–80 行将观察/自输出字符置为 1，81 行再次删除弱活动。hebb_text.py 57–66 行学习活动源→新字符，没有排除源和目标相同，故自连接也能由真实训练中短距离重现同字形成。", "",
        "## 固定预算和活动阈值", "",
        "在同一个 K8 的“你好”末状态，固定预算 0.25 和 floor 0.02，仅改变下一步 K：", "",
        "| K | 平均注入预算 | 最小–最大实际注入 | 单独达到阈值的候选 |",
        "|---:|---:|---:|---:|",
    ]
    for k, r in common_state_sweep.items():
        injections = [x["injection"] for x in r["allocations"]]
        lines.append(f"| {k} | {.25/int(k):.8f} | {min(injections):.5f}–{max(injections):.5f} | {r['fresh_survivors']} |")
    lines += [
        "", "这不是 K32 无连接：电流已经算出，只是分配后每个新候选都低于阈值。仅靠总预算 0.25，至多 12 个新候选能各拿到至少 0.02。已有强活动可能另行存活，因此这不是全部活动数的绝对上限。",
        "旧候选若低于 0.08，乘 0.25 后会在新注入之前被删除；例如 0.03 的旧活动不会先保留 0.0075 再与本步注入相加。这是当前两次删弱顺序的明确效果。",
        "各 K 分别从“你好”开始时，活动数为 9/5/2；K32 在第 3 次静默推进后全空，此后无外部输入便保持空状态。该核查没有把全灭解释为联想成功。", "",
        "## 发声赢家回灌的独立影响", "",
        f"“你好”刚输入完实际有 {direct['activity']['count']} 个活动单元，有效活动数为 {direct['activity']['effective_count']:.3f}。最强候选是“{winner}”。相同旧状态下，静默推进给“的”注入 0.044071；发声则把它置为 1，额外增加 0.955929。其余活动逐项相同。",
        "结果：下一状态总活动由 0.5625 增至 1.518429，随后最强字由静默条件的“的”变为发声条件的“我”。不能只用屏蔽对角边代替这个回灌对照。", "",
        "## 明确的一帧：8 个活动单元共同驱动下一批", "",
        "“你好”后进行 4 次静默推进，此刻恰好 8 个活动源。下表给出它们对下一最强候选“是”的全部贡献。电流乘法按 float32，活动持久保存按 float64。", "",
        "| 活动源 | 活动量 | 指向“是”的权重 | 本帧电流 |",
        "|---|---:|---:|---:|",
    ]
    for entry in eight["top16"][0]["contributions"]:
        lines.append(f"| {entry['source']} | {entry['amplitude']:.6f} | {entry['weight']:.3f} | {entry['current']:.3f} |")
    lines += [
        "", "合计电流为 72.520；其中“是→是”自连接只贡献 6.160。“不→是”贡献 18.527，比自连接更大。在这一帧去掉所有对角边后，“是”仍以 66.361 位居第一。",
        f"这一帧共有 {eight['positive_destinations']:,} 个目标收到正电流。保留下批顺序是“是、我、不、的、一、你、好、了”，活动范围 0.02346–0.03601，总量 0.25；八个字符确实都参加下一次出边汇总。它们的集合与上一帧相同，但幅度和排序变化，不能据此说整个精确状态已重现，也不能说语义联想已变丰富。", "",
        "## 新对照的审查建议", "",
        "把对角边开关、赢家回灌、K 和短暂不应期分别列清楚。相同权重/输入下先做单步相同状态的受控比较，再看完整轨迹；如果从 warm 阶段就应用不同规则，要明确初始活动也随规则而变。",
        "增加 K 并保持平均候选活动 0.03125，K16/K32 总预算会变为 0.5/1；这同时改变总反馈强度。若总预算固定 0.25，则需另处理 floor 才可能容纳更多候选；这些应作为明确的不同条件。",
        "短暂不应期要说明约束的是输出选择、候选注入还是源电流；外部真实输入能否覆盖它也要固定。只屏蔽展示的赢家仍让该单元通过候选传播，不能当作抑制全部回路。",
        "应记录实际候选存活数、删除前后活动总量、有效活动数、空状态，以及完整幅度/排序和输出周期。单帧去自连接结果只支持该帧的因果分解，不能代替新规则的完整轨迹。", "",
        "原始证据：evidence.json；简表：brief.json；复现脚本：../audit_candidate_mechanism.py。既有核心、模型及报告均未修改。", "",
    ]
    (OUT / "机制核查.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(brief, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
