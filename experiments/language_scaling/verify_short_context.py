"""Independent replay, selection, and intervention audit for short_context_v2.

Uses the existing independent Reference implementation as a read-only library.
Only this script and VERIFIED_SHORT_CONTEXT.md belong to this verification run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import verify_results as vr

HERE = Path(__file__).resolve().parent
DECAYS = (.1, .25, .5, .65, .8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    protected = [HERE / name for name in
                 ("hebb_text.py", "run_experiment.py", "check_core.py", "verify_results.py",
                  "verify_no_punct.py", "VERIFIED.md", "VERIFIED_NO_PUNCT.md",
                  "run_short_context.py")]
    for directory in ("results_v1", "results_no_punct_v1", "short_context_v1"):
        protected.extend(sorted(p for p in (HERE / directory).rglob("*") if p.is_file()))
    return {str(p.resolve()): sha(p) for p in protected}


def sample_positions(text, seed, excluded=()):
    blocked = set(excluded)
    eligible = [i for i in range(64, len(text))
                if "\u3400" <= text[i] <= "\u9fff" and i not in blocked]
    return sorted(map(int, np.random.default_rng(seed).choice(eligible, min(1024, len(eligible)), replace=False)))


def span(decay, floor):
    amplitude, length = 1.0, 0
    while amplitude >= floor:
        amplitude *= decay
        length += 1
    return length


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        vr.require(set(archive.files) == {"weights", "metadata"}, "Checkpoint fields")
        metadata = json.loads(str(archive["metadata"]))
        reference = vr.Reference(metadata["vocabulary"], metadata["config"])
        reference.w = archive["weights"].copy()
    vr.require(reference.w.dtype == np.float32 and reference.w.shape == (len(reference.vocabulary),) * 2,
               "Checkpoint type and shape")
    vr.require(np.isfinite(reference.w).all() and np.all(reference.w >= 0), "Finite nonnegative weights")
    reference.activity = {int(i): float(a) for i, a in metadata["activity"]}
    reference.clock, reference.learned = metadata["clock"], metadata["learned_characters"]
    reference.updates = metadata["update_events"]
    return reference


def verify_context_effect(actual, interventions):
    full, last, shuffled = [interventions[key]["rows"] for key in ("full", "last_only", "shuffled_earlier")]
    expected = dict(n=len(full), full_correct=sum(r["rank"] == 1 for r in full),
                    last_only_correct=sum(r["rank"] == 1 for r in last),
                    earlier_context_helped=sum(a["rank"] == 1 and b["rank"] != 1 for a, b in zip(full, last)),
                    earlier_context_harmed=sum(a["rank"] != 1 and b["rank"] == 1 for a, b in zip(full, last)),
                    prediction_changed_with_last_only=sum(a["predicted"] != b["predicted"] for a, b in zip(full, last)),
                    prediction_changed_with_shuffled_earlier=sum(a["predicted"] != b["predicted"] for a, b in zip(full, shuffled)),
                    full_most_predicted=[list(item) for item in Counter(r["predicted"] for r in full).most_common(8)])
    vr.require(actual == expected, "Context effect summary")
    vr.require(expected["full_correct"]-expected["last_only_correct"] ==
               expected["earlier_context_helped"]-expected["earlier_context_harmed"], "Context paired accounting")
    return expected


def verify_connection_examples(out, selected, reference, interventions):
    saved = vr.read(out / "connection_examples.json")
    vr.require(saved["model"] == selected and saved["weight_sha256"] ==
               vr.weight_hash(reference.vocabulary, reference.w), "Connection example checkpoint identity")
    vr.require(saved["selection"] == "first chronological helped case and first chronological harmed case",
               "Connection example selection declaration")
    vr.require([example["kind"] for example in saved["examples"]] == ["helped", "harmed"], "Example categories")
    full, last = interventions["full"]["rows"], interventions["last_only"]["rows"]
    for example in saved["examples"]:
        eligible = [(a, b) for a, b in zip(full, last) if
                    ((a["rank"] == 1 and b["rank"] != 1) if example["kind"] == "helped"
                     else (a["rank"] != 1 and b["rank"] == 1))]
        a, b = eligible[0]
        vr.require((example["position"], example["prefix"], example["target"],
                    example["full_prediction"], example["last_only_prediction"]) ==
                   (a["position"], a["prefix"], a["target"], a["predicted"], b["predicted"]),
                   "First chronological example and target")
        previous = (reference.activity, reference.clock)
        reference.activity = {}
        for character in a["prefix"]:
            reference.advance(character)
        expected_activity = [dict(character=reference.vocabulary[i], amplitude=amplitude)
                             for i, amplitude in reference.activity.items()]
        vr.require(example["activity"] == expected_activity, "Independent example activity")
        destinations = list(dict.fromkeys((a["target"], a["predicted"], b["predicted"])))
        connections = []
        for i, activation in reference.activity.items():
            for destination in destinations:
                j = reference.ids[destination]
                connections.append(dict(source=reference.vocabulary[i], activation=activation,
                                        destination=destination, weight=float(reference.w[i, j]),
                                        current=float(np.float32(activation) * reference.w[i, j])))
        vr.require(example["connections"] == connections, "Every actual connection and individual current")
        indices = np.asarray(list(reference.activity), dtype=np.int64)
        amplitudes = np.asarray(list(reference.activity.values()), dtype=np.float32)
        currents = (reference.w[indices] * amplitudes[:, None]).sum(axis=0)
        vr.require(example["total_currents"] ==
                   {ch: float(currents[reference.ids[ch]]) for ch in destinations}, "Exact summed example currents")
        reference.activity, reference.clock = previous
        vr.CHECKS["connection_examples"] += 1
        vr.CHECKS["individual_connections"] += len(connections)


def verify(out):
    protocol = vr.read(out / "frozen_protocol.json")
    selection = vr.read(out / "selection.json")
    summary = vr.read(out / "summary.json")
    vr.require(summary["completed"] is True, "Incomplete short-context experiment")
    vr.require(summary["protocol"] == protocol, "Summary protocol")
    vr.require(protocol["version"] == 2 and protocol["corpus"] == "data_no_punct_v1", "Protocol version/corpus")
    first_protocol = vr.read(HERE / "short_context_v1/frozen_protocol.json")
    vr.require({k: v for k, v in protocol.items() if k not in ("version", "source_sha256")} ==
               {k: v for k, v in first_protocol.items() if k not in ("version", "source_sha256")},
               "Rerun preserves every scientific protocol field")
    first_selection = vr.read(HERE / "short_context_v1/selection.json")
    vr.require(not (HERE / "short_context_v1/summary.json").exists(), "Original failed attempt remains incomplete")
    for name, expected in protocol["source_sha256"].items():
        vr.require(sha(HERE / name) == expected, "Frozen source hash: " + name)
        vr.CHECKS["source_hashes"] += 1
    old_protocol = vr.read(HERE / "results_no_punct_v1/frozen_protocol.json")
    old_summary = vr.read(HERE / "results_no_punct_v1/summary.json")
    for name in ("hebb_text.py", "run_experiment.py"):
        vr.require(protocol["source_sha256"][name] == old_protocol["sources"][name], "Unchanged core: " + name)
    data = HERE / "data_no_punct_v1"
    texts = {split: (data / (split + ".txt")).read_text(encoding="utf-8") for split in ("train", "validation", "test")}
    for split in texts:
        name = str((data / (split + ".txt")).relative_to(HERE))
        vr.require(protocol["source_sha256"][name] == old_protocol["sources"][name], "Unchanged input split: " + split)
    train, validation, test = [texts[k] for k in ("train", "validation", "test")]
    vocabulary = "".join(sorted(set(train)))
    vr.require(protocol["train_characters"] == len(train) == 658462, "Full fixed training amount")
    vr.require(protocol["vocabulary_characters"] == len(vocabulary), "Fixed training vocabulary")
    matched = vr.read(data / "matched_positions.json")
    excluded = sorted(set(old_protocol["test_positions"]) | set(matched["new_positions"]))
    vr.require(protocol["excluded_test_positions"] == excluded, "Union of previously scored test positions")
    val_positions = sample_positions(validation, 19351)
    test_positions = sample_positions(test, 19352, excluded)
    vr.require(protocol["validation_positions"] == val_positions and protocol["test_positions"] == test_positions,
               "Seeded validation/new-test positions")
    vr.require(not set(test_positions).intersection(excluded), "Fresh test target overlap")
    vr.require(protocol["selection"] == "minimum validation mean_nll_known; ties follow candidate order", "Selection criterion")
    expected_configs = {
        f"decay_{str(decay).replace('.', 'p')}": dict(trace_decay=decay, trace_floor=.02,
                                                     learning_rate=.25, growth="diminishing", ceiling=4.0)
        for decay in DECAYS}
    vr.require(list(protocol["candidates"]) == list(expected_configs), "Frozen candidate order")
    vr.require(list(selection["candidates"]) == list(expected_configs), "Selection candidate set/order")
    vr.require(selection["candidates"] == summary["candidates"] and selection["test_scored_yet"] is False,
               "Saved pre-test selection record")
    vr.require(summary["selection_file_sha256"] == sha(out / "selection.json"), "Unchanged selected-parameter file")
    training_seconds = 0.0
    for name, cfg in expected_configs.items():
        candidate = selection["candidates"][name]
        for key in ("config", "span", "validation", "weight_sha256", "stats"):
            vr.require(candidate[key] == first_selection["candidates"][name][key], name + ": identical first-attempt candidate " + key)
        effective_span = span(cfg["trace_decay"], cfg["trace_floor"])
        vr.require(protocol["candidates"][name] == cfg | {"effective_character_span": effective_span}, name + ": protocol config")
        vr.require(candidate["config"] == cfg and candidate["span"] == effective_span >= 2, name + ": multi-character candidate")
        reference = vr.Reference(vocabulary, cfg)
        for character in train:
            reference.advance(character, training=True)
        directory = out / name
        vr.require(candidate["stats"] == reference.stats(), name + ": complete replay statistics")
        vr.require(candidate["weight_sha256"] == vr.weight_hash(vocabulary, reference.w), name + ": complete replay weight hash")
        validation_record = vr.read(directory / "validation.json")
        actual_metrics = vr.verify_rows(reference, validation_record, validation, val_positions, name + "/validation")
        vr.verify_metrics(candidate["validation"], actual_metrics, name + ": validation selection metrics")
        checkpoint = load_checkpoint(directory / "model.npz")
        vr.require(checkpoint.vocabulary == vocabulary and checkpoint.config == cfg, name + ": checkpoint vocabulary/config")
        vr.require(np.array_equal(checkpoint.w, reference.w), name + ": every checkpoint weight")
        vr.require(checkpoint.activity == reference.activity and checkpoint.clock == reference.clock and
                   checkpoint.learned == reference.learned and checkpoint.updates == reference.updates,
                   name + ": checkpoint complete state")
        if name == "decay_0p8":
            vr.require(candidate["weight_sha256"] == old_summary["conditions"]["context_diminishing"]["final_checkpoint_weight_sha256"],
                       "18-character candidate exactly reproduces prior full training")
        from hebb_text import CharacterHebb
        loaded = CharacterHebb.load(directory / "model.npz")
        loaded.train(train[:128])
        for ch in train[:128]:
            reference.advance(ch, training=True)
        vr.require(np.array_equal(loaded.weights, reference.w) and loaded.activity == reference.activity and
                   loaded.clock == reference.clock and loaded.learned_characters == reference.learned and
                   loaded.update_events == reference.updates, name + ": independent exact 128-character resumed learning")
        vr.require(np.isfinite(candidate["training_seconds"]) and candidate["training_seconds"] >= 0, name + ": timing")
        training_seconds += candidate["training_seconds"]
        vr.CHECKS["fully_replayed_candidates"] += 1
        print(json.dumps({"verified_candidate": name, "span": effective_span}), flush=True)
        del loaded, checkpoint, reference
    selected = min(expected_configs, key=lambda name: selection["candidates"][name]["validation"]["mean_nll_known"])
    vr.require(selection["selected"] == summary["selected"] == first_selection["selected"] == selected,
               "Validation-only selection unchanged by numerical-check repair")
    tested = list(dict.fromkeys((selected, "decay_0p8")))
    vr.require(list(summary["test_results"]) == tested, "Only selected and prespecified 18-character controls scored")
    analyses = {}
    for name in expected_configs:
        vr.require((out / name / "test_and_context.json").exists() == (name in tested), name + ": test artifact eligibility")
    for name in tested:
        reference = load_checkpoint(out / name / "model.npz")
        record = vr.read(out / name / "test_and_context.json")
        compact = summary["test_results"][name]
        vr.require(record["probe_state_restored"] is True, name + ": frozen test probe declaration")
        measured = vr.verify_rows(reference, record["test"], test, test_positions, name + "/new_test")
        vr.verify_metrics(compact["metrics"], measured, name + ": summary test")
        interventions = record["context_interventions"]
        vr.require(set(interventions) == {"full", "last_only", "shuffled_earlier"}, name + ": intervention set")
        for condition, section in interventions.items():
            vr.verify_rows(reference, section, test, test_positions, name + "/" + condition, intervention=condition)
        if name == selected:
            verify_connection_examples(out, selected, reference, interventions)
        deltas, changed = [], []
        for a, b in zip(record["test"]["rows"], interventions["full"]["rows"]):
            vr.require((a["position"], a["known"], a["rank"], a["predicted"]) ==
                       (b["position"], b["known"], b["rank"], b["predicted"]), name + ": continuous/prefix decisions")
            if a["known"]:
                delta = abs(a["probability"]-b["probability"])
                vr.require(math.isclose(a["probability"], b["probability"], rel_tol=1e-7, abs_tol=1e-12),
                           name + ": continuous/prefix numeric tolerance")
                deltas.append(delta)
                if delta:
                    changed.append(dict(position=a["position"], rank_a=a["rank"], rank_c=b["rank"],
                                        probability_a=a["probability"], probability_c=b["probability"],
                                        absolute_difference=delta, predicted_a=a["predicted"], predicted_c=b["predicted"]))
        expected_check = dict(rank_and_prediction_exact=True, probability_relative_tolerance=1e-7,
                              probability_absolute_tolerance=1e-12, max_absolute_probability_difference=max(deltas, default=0.))
        vr.require(record["continuous_vs_prefix_check"] == expected_check, name + ": recorded numeric-path audit")
        if name == "decay_0p8":
            diagnostic = vr.read(HERE / "short_context_v1/failure_diagnostic.json")
            vr.require(diagnostic["different_probability_rows"] == len(changed) and
                       diagnostic["different_ranks"] == diagnostic["different_predictions"] == 0,
                       "First-attempt failure diagnosis counts")
            vr.require(diagnostic["max_absolute_probability_difference"] == max(deltas, default=0.) and
                       diagnostic["differences"] == changed, "Independently reproduced first-attempt floating-point difference")
            vr.close(diagnostic["max_relative_probability_difference"],
                     max((row["absolute_difference"]/abs(row["probability_a"]) for row in changed), default=0.),
                     "First-attempt relative numerical difference")
        first_test = HERE / "short_context_v1" / name / "test_and_context.json"
        if first_test.exists():
            original_record = vr.read(first_test)
            vr.require(original_record == {key: record[key] for key in original_record},
                       name + ": first-attempt completed test record exactly reproduced")
        effect = verify_context_effect(record["context_effect"], interventions)
        vr.require(compact["context_effect"] == effect, name + ": summary context effects")
        recurrent = vr.verify_rows(reference, record["two_recurrent_steps"], test, test_positions, name + "/two_steps", steps=2)
        vr.verify_metrics(compact["two_recurrent_steps"], recurrent, name + ": summary recurrence")
        expected_prompts = [test[max(0, pos-32):pos] for pos in test_positions[::256]][:4]
        vr.require([example["prompt"] for example in record["continuations"]] == expected_prompts, name + ": unscreened fixed prompts")
        for example in record["continuations"]:
            for key, steps in (("greedy", 0), ("greedy_two_steps", 2)):
                vr.require(len(example[key]) == 48 and example[key] == reference.generate(example["prompt"], 48, steps),
                           name + ": independent continuation")
                vr.CHECKS["continuations"] += 1
        analyses[name] = dict(test=measured, recurrence=recurrent, context=effect,
                              last_only=interventions["last_only"]["metrics"],
                              shuffled_earlier=interventions["shuffled_earlier"]["metrics"])
        del reference
    baseline = load_checkpoint(HERE / "results_no_punct_v1/adjacent_diminishing/model.npz")
    vr.require(vr.weight_hash(baseline.vocabulary, baseline.w) ==
               old_summary["conditions"]["adjacent_diminishing"]["final_checkpoint_weight_sha256"], "Complete one-character control hash")
    vr.require(baseline.config["trace_decay"] == 0.0 and baseline.clock == len(train) == baseline.learned, "One-character control state")
    baseline_metrics = vr.verify_rows(baseline, vr.read(out / "one_character_control.json"), test, test_positions, "one_character_control")
    vr.verify_metrics(summary["one_character_control"], baseline_metrics, "Summary one-character control")
    vr.require(np.isfinite(summary["elapsed_seconds"]) and summary["elapsed_seconds"] >= training_seconds, "Wall-time accounting")
    return selected, analyses, baseline_metrics, selection, len(excluded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "--out", dest="results", type=Path, default=HERE / "short_context_v2")
    args = parser.parse_args()
    report = HERE / "VERIFIED_SHORT_CONTEXT.md"
    started = time.perf_counter()
    protected = snapshot()
    vr.CHECKS.clear()
    try:
        selected, analyses, baseline, selection, excluded = verify(args.results.resolve())
        vr.require(snapshot() == protected, "Old core, verification files and results changed")
    except Exception as error:
        report.write_text("# 短上下文实验独立核验\n\nFAIL: " + type(error).__name__ + ": " + str(error) + "\n", encoding="utf-8")
        raise
    elapsed = time.perf_counter() - started
    lines = ["# 短上下文实验独立核验", "", "结论：PASS。候选完整重放、验证集选型、新测试目标、上下文干预与逐项评分均通过独立核验。", "",
             f"- 结果目录：{args.results.resolve()}", f"- 选择结果：{selected}",
             f"- 核验耗时：{elapsed:.2f} 秒", f"- 环境：Python {platform.python_version()} / NumPy {np.__version__}",
             f"- 核验器SHA-256：{sha(Path(__file__))}", f"- 只读参考实现SHA-256：{sha(HERE / 'verify_results.py')}",
             f"- 核验计数：{json.dumps(dict(vr.CHECKS), ensure_ascii=False)}", "",
             "五个候选分别从出生连续重放全部658462个去标点字符，完整矩阵、活动、时钟、训练量和更新计数逐项相同；恢复后续训128字也与独立实现完全相同。18字候选权重精确复现上一轮同配置结果。新实验没有改学习核心、旧结果或旧核验报告。", "",
             "首轮short_context_v1因概率逐位等号过严中止，已保留全部原文件。重跑v2的候选配置、验证/测试位置、验证指标、模型权重及选择结果均与首轮相同。独立复现18字模型仅1行约1.8572e-12的概率差，名次和预测无变化；v2使用明确容差核对该数值路径，没有改学习机制。", "",
             "连接示例也逐项核验：按时间顺序取第一个帮对和第一个干扰位置；活动、实际边权、12条电流贡献和候选总电流与独立重算完全相同。案例没有用于取代整体统计。", "",
             f"独立重建固定随机抽样，1024个新测试汉字目标避开两轮共{excluded}个曾评分位置。候选顺序和验证集最小NLL规则均匹配；NLL与BPC只差正常数，因此选型等价。selection.json中的决定及SHA-256保持一致。一字模型只作为对照，未进入多字候选选择。", "",
             "## 验证集候选", "", "| 候选 | 有效字符跨度 | 验证BPC | 验证Top1 |", "|---|---:|---:|---:|"]
    for name, candidate in selection["candidates"].items():
        lines.append(f"| {name} | {candidate['span']} | {candidate['validation']['bpc_known']:.6f} | {candidate['validation']['top1']:.2%} |")
    lines += ["", "## 冻结新位置上的测试", "", "| 模型 | Top1 | BPC | 同模型仅末字Top1 | 更早上下文帮助/损害 |", "|---|---:|---:|---:|---:|"]
    for name, record in analyses.items():
        lines.append(f"| {name} | {record['test']['top1']:.2%} | {record['test']['bpc_known']:.6f} | {record['last_only']['top1']:.2%} | {record['context']['earlier_context_helped']}/{record['context']['earlier_context_harmed']} |")
    lines.append(f"| 一字对照 | {baseline['top1']:.2%} | {baseline['bpc_known']:.6f} | — | — |")
    lines += ["", "## 解释边界", "",
              "- 新的是测试目标位置，文本仍来自此前两部作品及同一测试分区；不代表独立的新语料。旧预测保存的前缀可能覆盖新目标，哈希也不能证明外部意义上的预注册时间。",
              "- full使用64字前缀，足以覆盖所有候选的活动跨度。shuffled_earlier打乱64字前缀中末字之前的字符，保留末字；它同时改变近期内容和顺序，不能单独证明顺序理解。两条评估路径分别独立重算，避免把浮点累加顺序差异当成学习差异。",
              "- 同模型last_only消融反映当前读出中更早字符的作用；该模型权重仍通过多字活动训练，不能直接等同于独立训练的一字模型。",
              "- OOV计入准确率失败，NLL/BPC仅对已知字符计算；重复字符共用一个单元，模型没有词序角色状态。两步传播和全部固定续写分别核验，不用生成个例替代总体指标。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status="PASS", selected=selected, counts=dict(vr.CHECKS),
                          elapsed_seconds=round(elapsed, 3), analyses=analyses, one_character_control=baseline),
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
