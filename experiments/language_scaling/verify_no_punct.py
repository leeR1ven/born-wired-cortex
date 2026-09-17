"""Verify the derived experiment without changing the original verifier/report.

Imports the independent Reference and verification functions as a library.
The old CLI is never called. Its manifest hook is replaced only in memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import unicodedata

import numpy as np
import verify_results as vr

HERE = Path(__file__).resolve().parent
CORE = ("hebb_text.py", "run_experiment.py", "check_core.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot(original_results):
    files = [HERE / name for name in (*CORE, "verify_results.py", "VERIFIED.md")]
    files += sorted(path for path in original_results.rglob("*") if path.is_file())
    return {str(path.resolve()): sha(path) for path in files}


def is_han(character):
    return "\u3400" <= character <= "\u9fff"


def verify_derivative(data, original_data, original_results):
    protocol = vr.read(original_results / "frozen_protocol.json")
    records, texts, mappings = {}, {}, {}
    for split in ("train", "validation", "test"):
        source_path, new_path = original_data / (split + ".txt"), data / (split + ".txt")
        source = source_path.read_text(encoding="utf-8", errors="strict")
        actual = new_path.read_text(encoding="utf-8", errors="strict")
        expected, categories = [], Counter()
        mapping = np.full(len(source), -1, dtype=np.int64)
        for old_position, character in enumerate(source):
            category = unicodedata.category(character)
            if category.startswith("P"):
                categories[category] += 1
            else:
                mapping[old_position] = len(expected)
                expected.append(character)
        vr.require(actual == "".join(expected), split + ": exact Unicode-P-only deletion")
        vr.require(not any(unicodedata.category(c).startswith("P") for c in actual), split + ": punctuation remains")
        vr.require("".join(c for c in source if c.isspace()) == "".join(c for c in actual if c.isspace()),
                   split + ": whitespace changed")
        source_key = str(source_path.relative_to(HERE))
        vr.require(sha(source_path) == protocol["sources"][source_key], split + ": original split changed")
        records[split] = dict(original_characters=len(source), new_characters=len(actual),
                             removed_characters=len(source)-len(actual), removed_categories=dict(categories),
                             original_sha256=sha(source_path), new_sha256=sha(new_path))
        texts[split], mappings[split] = (source, actual), mapping
        vr.CHECKS["derivative_splits"] += 1
    old_positions = [p for p in protocol["test_positions"] if is_han(texts["test"][0][p])]
    new_positions = [int(mappings["test"][p]) for p in old_positions]
    vr.require(len(old_positions) == 859, "Expected original 859 Han targets")
    vr.require(len(new_positions) == len(set(new_positions)) and new_positions == sorted(new_positions),
               "Unique monotone paired positions")
    for old_pos, new_pos in zip(old_positions, new_positions):
        vr.require(texts["test"][0][old_pos] == texts["test"][1][new_pos], "Paired target changed")
    return records, texts, mappings, old_positions, new_positions


class MaskedReference(vr.Reference):
    def distribution(self, steps=0):
        probabilities = super().distribution(steps)
        permitted = np.array([not unicodedata.category(c).startswith("P") for c in self.vocabulary])
        probabilities[~permitted] = 0
        probabilities /= probabilities.sum()
        return probabilities


def load_reference(path, masked=False):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        cls = MaskedReference if masked else vr.Reference
        reference = cls(metadata["vocabulary"], metadata["config"])
        reference.w = archive["weights"].copy()
    reference.activity = {int(i): float(a) for i, a in metadata["activity"]}
    reference.clock = metadata["clock"]
    reference.learned = metadata["learned_characters"]
    reference.updates = metadata["update_events"]
    return reference


def verify_derived_records(data, derivative, mappings, old_positions, new_positions, texts):
    manifest = vr.read(data / "manifest.json")
    vr.require(manifest["version"] == 1, "Derivative manifest version")
    vr.require(manifest["transformation"] == "delete characters whose Unicode category starts with P" and
               manifest["whitespace"] == "preserved exactly" and manifest["original_split_assignment"] == "unchanged",
               "Derivative transform declarations")
    vr.require(manifest["unicode_version"] == unicodedata.unidata_version, "Unicode version")
    vr.require(manifest["builder_sha256"] == sha(HERE / "prepare_no_punct.py"), "Derivative builder hash")
    vr.require(set(manifest["splits"]) == set(derivative), "Derivative split set")
    with np.load(data / "old_to_new_positions.npz", allow_pickle=False) as archive:
        vr.require(set(archive.files) == set(mappings), "Saved mapping fields")
        for split, expected_map in mappings.items():
            vr.require(archive[split].dtype == np.int64 and np.array_equal(archive[split], expected_map),
                       split + ": every saved original-to-new index")
            vr.CHECKS["mapped_input_positions"] += len(expected_map)
    for split, record in derivative.items():
        actual = manifest["splits"][split]
        vr.require(actual["source_sha256"] == record["original_sha256"] and actual["sha256"] == record["new_sha256"],
                   split + ": manifest hashes")
        vr.require(actual["original_characters"] == record["original_characters"] and
                   actual["characters"] == record["new_characters"] and actual["removed_count"] == record["removed_characters"],
                   split + ": manifest character counts")
        removed = Counter(c for c in texts[split][0] if unicodedata.category(c).startswith("P"))
        vr.require(actual["removed_characters"] == [list(item) for item in removed.most_common()],
                   split + ": removed character inventory")
        vr.require(actual["whitespace_count"] == sum(c.isspace() for c in texts[split][1]), split + ": whitespace count")
        vr.require(actual["output"] == split + ".txt" and sha(HERE / actual["source"]) == record["original_sha256"],
                   split + ": manifest source/output paths")
    paired = vr.read(data / "matched_positions.json")
    vr.require(paired["original_positions"] == old_positions and paired["new_positions"] == new_positions,
               "Saved target mapping")
    vr.require(paired["targets"] == [texts["test"][0][p] for p in old_positions], "Saved mapped targets")
    vr.require(paired["target_definition"] == "U+3400..U+9FFF", "Han target definition")


def verify_matched(path, original_results, results, texts, old_positions, new_positions):
    saved = vr.read(path)
    vr.require(saved["original_positions"] == old_positions, "Original paired positions")
    vr.require(saved["new_positions"] == new_positions, "New paired positions")
    targets = [texts["test"][0][pos] for pos in old_positions]
    vr.require(list(saved["targets"]) == targets, "Saved paired target identities")
    protocol = vr.read(results / "frozen_protocol.json")
    original_summary = vr.read(original_results / "summary.json")
    new_summary = vr.read(results / "summary.json")
    vr.require(saved["training_characters"] == dict(original=original_summary["protocol"]["train_characters"],
                                                   no_punct=protocol["train_characters"]), "Paired training amounts")
    for name, expected in saved["source_hashes"].items():
        vr.require(sha(HERE / name) == expected, "Paired frozen helper hash: " + name)
        vr.CHECKS["paired_source_hashes"] += 1
    vr.require(set(saved["conditions"]) == set(protocol["conditions"]), "Paired condition set")
    results_summary = {}
    for name, sections in saved["conditions"].items():
        original_stage = original_summary["protocol"]["train_characters"]
        all_old = vr.read(original_results / name / f"stage_{original_stage}.json")["test"]["rows"]
        selected_old = set(old_positions)
        vr.require(sections["original"]["rows"] == [row for row in all_old if row["position"] in selected_old],
                   name + ": exact original final-stage row subset")
        for key, directory, text, selected, masked in (
            ("original", original_results, texts["test"][0], old_positions, False),
            ("no_punct", results, texts["test"][1], new_positions, False),
            ("output_mask_only", original_results, texts["test"][0], old_positions, True),
        ):
            reference = load_reference(directory / name / "model.npz", masked=masked)
            checkpoint_summary = original_summary if directory == original_results else new_summary
            vr.require(vr.weight_hash(reference.vocabulary, reference.w) ==
                       checkpoint_summary["conditions"][name]["final_checkpoint_weight_sha256"],
                       name + "/" + key + ": complete checkpoint hash")
            vr.require(reference.clock == checkpoint_summary["protocol"]["train_characters"] and
                       reference.learned == reference.clock, name + "/" + key + ": checkpoint counters")
            vr.verify_rows(reference, sections[key], text, selected, name + "/paired/" + key)
            del reference
        vr.require(sections["output_mask_only"]["weights_unchanged"] is True, name + ": output-mask declaration")
        before = [row["rank"] == 1 for row in sections["original"]["rows"]]
        after = [row["rank"] == 1 for row in sections["no_punct"]["rows"]]
        expected = dict(n=len(before), original_correct=sum(before), no_punct_correct=sum(after),
                        helped=sum((not a) and b for a, b in zip(before, after)),
                        harmed=sum(a and (not b) for a, b in zip(before, after)),
                        no_punct_predicted_types=len({row["predicted"] for row in sections["no_punct"]["rows"]}),
                        no_punct_most_predicted=[list(item) for item in Counter(row["predicted"] for row in sections["no_punct"]["rows"]).most_common(8)],
                        masked_most_predicted=[list(item) for item in Counter(row["predicted"] for row in sections["output_mask_only"]["rows"]).most_common(8)])
        vr.require(sections["paired_summary"] == expected, name + ": paired change and prediction-concentration summary")
        vr.require(expected["no_punct_correct"] - expected["original_correct"] == expected["helped"] - expected["harmed"],
                   name + ": paired accounting identity")
        results_summary[name] = {key: sections[key]["metrics"] for key in ("original", "no_punct", "output_mask_only")}
        results_summary[name]["paired_summary"] = expected
    return results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "--out", dest="results", type=Path, default=HERE / "results_no_punct_v1")
    parser.add_argument("--data", type=Path, default=HERE / "data_no_punct_v1")
    parser.add_argument("--original-results", type=Path, default=HERE / "results_v1")
    parser.add_argument("--original-data", type=Path, default=HERE / "data")
    parser.add_argument("--matched", type=Path)
    args = parser.parse_args()
    results, data = args.results.resolve(), args.data.resolve()
    original_results, original_data = args.original_results.resolve(), args.original_data.resolve()
    report = HERE / "VERIFIED_NO_PUNCT.md"
    started = time.perf_counter()
    protected = protected_snapshot(original_results)
    original_hook = vr.verify_manifest
    vr.CHECKS.clear()
    try:
        derivative, texts, mappings, old_positions, new_positions = verify_derivative(data, original_data, original_results)
        verify_derived_records(data, derivative, mappings, old_positions, new_positions, texts)
        old_protocol = vr.read(original_results / "frozen_protocol.json")
        new_protocol = vr.read(results / "frozen_protocol.json")
        vr.require(new_protocol["conditions"] == old_protocol["conditions"], "All five unchanged configurations")
        vr.require(new_protocol["train_characters"] == len(texts["train"][1]), "Full derived training split")
        for name in CORE:
            vr.require(new_protocol["sources"][name] == old_protocol["sources"][name] == sha(HERE / name),
                       "Unchanged frozen core: " + name)
        # The derivative has been reconstructed independently, character by character.
        # The old raw-source/block manifest format does not describe this data.
        vr.verify_manifest = lambda supplied: vr.require(supplied.resolve() == data, "Derivative manifest hook path")
        conditions, _ = vr.verify_experiment(results, data)
        matched_path = args.matched.resolve() if args.matched else results / "matched_comparison.json"
        paired = verify_matched(matched_path, original_results, results, texts, old_positions, new_positions)
        vr.require(protected_snapshot(original_results) == protected, "Original core/results/verifier/report modified")
    except Exception as error:
        report.write_text("# 无标点实验独立核验\n\nFAIL: " + type(error).__name__ + ": " + str(error) + "\n", encoding="utf-8")
        raise
    finally:
        vr.verify_manifest = original_hook
    elapsed = time.perf_counter() - started
    lines = ["# 无标点实验独立核验", "", "结论：PASS。删标点文本、完整训练与逐项预测、859个同目标配对比较均通过独立核验。", "",
             f"- 新结果目录：{results}", f"- 耗时：{elapsed:.2f} 秒",
             f"- 核验环境：Python {platform.python_version()} / NumPy {np.__version__} / Unicode {unicodedata.unidata_version}",
             f"- 新核验器SHA-256：{sha(Path(__file__))}",
             f"- 复用只读参考实现SHA-256：{sha(HERE / 'verify_results.py')}",
             f"- 核验计数：{json.dumps(dict(vr.CHECKS), ensure_ascii=False)}", "",
             "三个分区分别逐字符重建，仅删除Unicode类别P*字符，空白的内容和顺序完全保留。另固定原859个汉字目标，核对删除操作前后的唯一索引映射。五种配置复用完全未改的核心，从出生连续重放全部训练阶段并核对哈希、活动、计数器和所有评分。终态权重逐元素相等、有限非负；载入后续训128字仍逐项一致。", "",
             "配对比较逐项重算原模型、新模型及仅屏蔽标点输出三组概率/rank。屏蔽对照保留旧模型与旧输入，只将标点输出概率置零并归一化。词表变化同时影响概率归一化和底噪，概率损失差异包含这一影响。", "",
             "旧VERIFIED.md、旧verify_results.py、三个核心文件及原results_v1全部文件在核验前后SHA-256完全一致。旧实验与其结论完整保留。", "",
             "## 分区变换", "", "| 分区 | 原字符数 | 新字符数 | 删除P*数 |", "|---|---:|---:|---:|"]
    for split, record in derivative.items():
        lines.append(f"| {split} | {record['original_characters']:,} | {record['new_characters']:,} | {record['removed_characters']:,} |")
    lines += ["", "## 解释边界", "",
              "新固定随机测试位置与原位置不同，不能把原含标点1024位置准确率直接作为新集配对基线。配对准确率用同一859个汉字目标计算。删除标点同时改变相邻关系及按字符计量的有效上下文，应解释为输入表示干预。", "",
              "数值核验支持记录与给定实现一致，不证明语义理解、推理、AGI或跨作品泛化。数据仍来自同两部作品；不排除短片段和近重复。NLL/BPC仅对已知字符计算；同分采用固定词表顺序。", "",
              "## 配对核验结果", "", json.dumps(paired, ensure_ascii=False, indent=2), ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status="PASS", counts=dict(vr.CHECKS), elapsed_seconds=round(elapsed, 3),
                          derivative=derivative, paired=paired, conditions=conditions), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
