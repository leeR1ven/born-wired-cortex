"""Independently replay the frozen experiment and verify every saved prediction.

This file does not import the experiment's evaluation or training routines.
Only the final checkpoint continuation cross-check imports CharacterHebb.
Outputs: a concise JSON result on stdout and VERIFIED.md beside this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKS = Counter()


def require(ok, message):
    if not ok:
        raise AssertionError(message)
    CHECKS["assertions"] += 1


def close(actual, expected, message):
    if expected is None:
        require(actual is None, message)
    else:
        require(actual is not None and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-12),
                f"{message}: {actual!r} != {expected!r}")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def weight_hash(vocabulary, matrix):
    h = hashlib.sha256(vocabulary.encode("utf-8"))
    h.update(matrix.tobytes())
    return h.hexdigest()


class Reference:
    """One identity per character, causal directed float32 Hebbian updates."""

    def __init__(self, vocabulary, config):
        self.vocabulary = vocabulary
        self.ids = dict(zip(vocabulary, range(len(vocabulary))))
        self.config = config
        self.w = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.float32)
        self.activity = {}
        self.clock = self.learned = self.updates = 0

    def advance(self, character, training=False):
        identity = self.ids.get(character)
        if training:
            require(identity is not None, "Training OOV")
            if self.activity:
                indices = np.asarray(list(self.activity), dtype=np.int64)
                amplitudes = np.asarray(list(self.activity.values()), dtype=np.float32)
                values = self.w[indices, identity]
                increments = self.config["learning_rate"] * amplitudes
                if self.config["growth"] == "diminishing":
                    increments = increments / (1.0 + values)
                elif self.config["growth"] == "saturating":
                    increments = increments * np.maximum(0.0, 1.0 - values / self.config["ceiling"])
                else:
                    require(self.config["growth"] == "linear", "Unknown growth rule")
                self.w[indices, identity] = values + increments
                self.updates += len(indices)
            self.learned += 1
        decay, floor = self.config["trace_decay"], self.config["trace_floor"]
        self.activity = {i: value * decay for i, value in self.activity.items()
                         if value * decay >= floor}
        if identity is not None:
            self.activity[identity] = 1.0
        self.clock += 1

    def distribution(self, steps=0):
        size = len(self.vocabulary)
        signal = np.zeros(size, dtype=np.float32)
        if self.activity:
            indices = np.asarray(list(self.activity), dtype=np.int64)
            amplitudes = np.asarray(list(self.activity.values()), dtype=np.float32)
            signal = (self.w[indices] * amplitudes[:, None]).sum(axis=0)
        original = signal.copy()
        for _ in range(steps):
            if not np.any(signal > 0):
                break
            selected = np.argpartition(signal, -min(32, size))[-min(32, size):]
            selected = selected[signal[selected] > 0]
            drives = signal[selected] / signal[selected].sum()
            incoming = (self.w[selected] * drives[:, None]).sum(axis=0)
            if incoming.sum() > 0:
                incoming *= max(float(original.sum()), 1e-12) / incoming.sum()
            signal = 0.75 * original + 0.25 * incoming
        signal = signal.astype(np.float64)
        total = float(signal.sum())
        result = np.full(size, 1.0 / size) if total <= 0 else 0.999 * signal / total + 0.001 / size
        require(np.isfinite(result).all() and np.all(result > 0), "Probability finite/positive")
        close(float(result.sum()), 1.0, "Probability normalization")
        return result

    def probe(self, prefix, steps=0):
        saved = (self.activity, self.clock)
        self.activity = {}
        for character in prefix:
            self.advance(character)
        result = self.distribution(steps)
        self.activity, self.clock = saved
        return result

    def generate(self, prompt, length, steps):
        saved = (self.activity, self.clock)
        self.activity = {}
        for ch in prompt:
            self.advance(ch)
        result = []
        for _ in range(length):
            ch = self.vocabulary[int(np.argmax(self.distribution(steps)))]
            result.append(ch)
            self.advance(ch)
        self.activity, self.clock = saved
        return "".join(result)

    def stats(self):
        positive = self.w[self.w > 0]
        return dict(characters=len(self.vocabulary), learned_characters=self.learned,
                    clock=self.clock, update_events=self.updates,
                    connections=len(positive), matrix_bytes=self.w.nbytes,
                    weight_max=float(positive.max()) if len(positive) else 0.0,
                    weight_mean=float(positive.mean()) if len(positive) else 0.0,
                    active_characters=len(self.activity))


def positions(text, count, seed):
    eligible = np.array([i for i in range(64, len(text)) if not text[i].isspace()])
    return sorted(map(int, np.random.default_rng(seed).choice(eligible, min(count, len(eligible)), replace=False)))


def metrics(rows):
    probabilities = [r["probability"] for r in rows if r["known"]]
    n = len(rows)
    return dict(n=n, known_n=len(probabilities), oov_n=n-len(probabilities),
                top1=sum(r["rank"] == 1 for r in rows)/n,
                top5=sum(1 <= r["rank"] <= 5 for r in rows)/n,
                mean_nll_known=sum(-math.log(p) for p in probabilities)/len(probabilities) if probabilities else None,
                bpc_known=sum(-math.log2(p) for p in probabilities)/len(probabilities) if probabilities else None)


def verify_metrics(actual, expected, label):
    require(set(actual) == set(expected), label + " metric keys")
    for key, value in expected.items():
        close(actual[key], value, label + "/" + key)


def verify_manifest(data):
    path = data / "manifest.json"
    if not path.exists():
        return
    manifest = read(path)
    CHECKS["manifest_files"] += 1
    builder = HERE / "build_corpus.py"
    require(hashlib.sha256(builder.read_bytes()).hexdigest() == manifest["script_sha256"], "Corpus builder hash")
    for source in manifest["sources"]:
        raw = (data / source["raw_file"]).read_bytes()
        require(len(raw) == source["raw_bytes"] and hashlib.sha256(raw).hexdigest() == source["raw_sha256"], "Raw source hash/length")
        raw.decode("utf-8", errors="strict")
        CHECKS["raw_source_hashes"] += 1
    paragraphs = {}
    identifiers = [(b["source"], b["block"]) for b in manifest["blocks"]]
    require(len(identifiers) == len(set(identifiers)), "Unique source block assignments")
    for name, spec in manifest["splits"].items():
        raw = (data / spec["file"]).read_bytes()
        content = raw.decode("utf-8", errors="strict").replace("\r\n", "\n")
        require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "Manifest split hash")
        require(len(content) == spec["characters"] and len(set(content)) == spec["unique_characters"], "Manifest split counts")
        blocks = sorted((b for b in manifest["blocks"] if b["split"] == name), key=lambda b: (b["block"], b["source"]))
        require(len(blocks) == spec["blocks"], "Manifest block count")
        cursor = 0
        for index, block in enumerate(blocks):
            piece = content[cursor:cursor+block["characters"]]
            require(hashlib.sha256(piece.encode("utf-8")).hexdigest() == block["sha256"], "Manifest block content")
            cursor += block["characters"]
            boundary = "\n" if index == len(blocks)-1 else "\n\n"
            require(content[cursor:cursor+len(boundary)] == boundary, "Manifest block boundary")
            cursor += len(boundary)
            CHECKS["corpus_blocks"] += 1
        require(cursor == len(content), "Manifest complete block coverage")
        paragraphs[name] = {p for p in content.rstrip("\n").split("\n\n") if len(p) >= 30}
    overlaps = {a+"_"+b: len(paragraphs[a] & paragraphs[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    require(overlaps == manifest["cross_split_exact_paragraph_overlap_ge30"] and not any(overlaps.values()), "Independent paragraph overlap audit")


def verify_rows(reference, section, text, selected, label, *, steps=0, intervention=None, baseline=None):
    rows = section["rows"]
    require([r["position"] for r in rows] == list(selected), label + " fixed positions")
    saved_state = (reference.activity, reference.clock)
    continuous = intervention is None and baseline is None
    if continuous:
        reference.activity = {}
    cursor = 0
    for row in rows:
        pos = row["position"]
        original = text[max(0, pos-64):pos]
        require(row["prefix"] == original and row["target"] == text[pos], label + " prefix/target")
        prefix = original
        if intervention == "last_only":
            prefix = prefix[-1:]
        elif intervention == "shuffled_earlier":
            earlier = list(prefix[:-1])
            np.random.default_rng(pos+91507).shuffle(earlier)
            prefix = "".join(earlier) + prefix[-1:]
        if intervention is not None:
            require(row["presented_prefix"] == prefix, label + " presented prefix")
        if continuous:
            # Replay the whole prefix, preserving float32 sum order as well as
            # activity. Repeated identities can retain old dictionary positions
            # even after their initial appearance is older than the trace horizon.
            for character in text[cursor:pos]:
                reference.advance(character)
            cursor = pos
            p = reference.distribution(steps)
        else:
            p = baseline if baseline is not None else reference.probe(prefix, steps)
        identity = reference.ids.get(text[pos])
        known = identity is not None
        require(row["known"] == known, label + " OOV flag")
        # Independent exact rank with stable identity-index tie breaking.
        rank = 0 if not known else 1 + int(np.sum(p > p[identity])) + int(np.sum(p[:identity] == p[identity]))
        require(row["rank"] == rank, label + " target rank")
        close(row["probability"], float(p[identity]) if known else None, label + " target probability")
        ordered = np.argsort(-p, kind="stable")
        require(row["predicted"] == reference.vocabulary[int(ordered[0])], label + " predicted")
        require(row["top5"] == "".join(reference.vocabulary[int(i)] for i in ordered[:5]), label + " top5")
        CHECKS["prediction_rows"] += 1
    reference.activity, reference.clock = saved_state
    calculated = metrics(rows)
    verify_metrics(section["metrics"], calculated, label)
    CHECKS["metric_sections"] += 1
    if "weight_sha256" in section:
        require(section["weight_sha256"] == weight_hash(reference.vocabulary, reference.w), label + " frozen weights")
        require(section["weights_unchanged"] is True and section["training_state_restored"] is True,
                label + " evaluation state declarations")
    return calculated


def verify_experiment(out, data):
    protocol = read(out / "frozen_protocol.json")
    summary = read(out / "summary.json")
    require(summary["completed"] is True, "Experiment incomplete")
    require(summary["protocol"] == protocol, "Summary protocol differs")
    require(set(summary["conditions"]) == set(protocol["conditions"]), "Condition set")
    require(protocol["version"] == 1 and protocol["architecture"] == "character-directed-Hebb-only", "Protocol version")
    for name, expected in protocol["sources"].items():
        path = HERE / name
        require(path.is_file(), "Missing frozen source: " + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Frozen hash changed: " + name)
        CHECKS["source_hashes"] += 1
    verify_manifest(data)
    train = (data / "train.txt").read_text(encoding="utf-8")[:protocol["train_characters"]]
    validation = (data / "validation.txt").read_text(encoding="utf-8")
    test = (data / "test.txt").read_text(encoding="utf-8")
    require(len(train) == protocol["train_characters"], "Train length")
    require(len(validation) == protocol["validation_characters"] and len(test) == protocol["test_characters"], "Evaluation lengths")
    vocabulary = "".join(sorted(set(train)))
    require(len(vocabulary) == protocol["characters"] and protocol["vocabulary_preallocated_from_training_only"] is True,
            "Training-only vocabulary declaration")
    val_positions = positions(validation, protocol["eval_count"], 19301)
    test_positions = positions(test, protocol["eval_count"], 19302)
    require(val_positions == protocol["validation_positions"] and test_positions == protocol["test_positions"], "Seeded positions")
    require(protocol["stages"] == sorted(set(min(k, len(train)) for k in (1000, 10000, 100000, len(train)))), "Training stages")
    for cfg in protocol["conditions"].values():
        require(0 <= cfg["trace_decay"] < 1 and 0 < cfg["trace_floor"] <= 1, "Finite activity horizon")
    results = {}
    for name, cfg in protocol["conditions"].items():
        reference = Reference(vocabulary, cfg)
        directory = out / name
        condition = summary["conditions"][name]
        require(len(condition["stages"]) == len(protocol["stages"]), name + " stage count")
        verify_rows(reference, read(directory / "birth_test.json"), test, test_positions, name + "/birth")
        previous = 0
        for end, compact in zip(protocol["stages"], condition["stages"]):
            for character in train[previous:end]:
                reference.advance(character, training=True)
            stage = read(directory / f"stage_{end}.json")
            label = f"{name}/{end}"
            require(stage["stage"] == end and stage["increment"] == end-previous, label + " training increment")
            require(math.isfinite(stage["training_seconds"]) and stage["training_seconds"] >= 0, label + " elapsed time")
            require(stage["stats"] == reference.stats(), label + " independently replayed training statistics")
            require(stage["stats"]["clock"] == end and stage["stats"]["learned_characters"] == end, label + " clock/amount")
            verify_rows(reference, stage["validation"], validation, val_positions, label + "/validation")
            verify_rows(reference, stage["test"], test, test_positions, label + "/test")
            counts = Counter(train[:end])
            baseline = np.asarray([counts[c]+1 for c in vocabulary], dtype=float)
            baseline /= baseline.sum()
            verify_rows(reference, stage["unigram"], test, test_positions, label + "/unigram", baseline=baseline)
            for key in ("stage", "increment", "training_seconds", "stats"):
                require(compact[key] == stage[key], label + " summary " + key)
            for key in ("validation", "test", "unigram"):
                verify_metrics(compact[key], stage[key]["metrics"], label + "/summary/" + key)
            previous = end
            CHECKS["replayed_stages"] += 1
            print(json.dumps({"verified_condition": name, "stage": end}, ensure_ascii=False), flush=True)
        selected = test_positions[:min(128, len(test_positions))]
        interventions = read(directory / "context_interventions.json")
        require(set(interventions) == {"full", "last_only", "shuffled_earlier"}, name + " interventions")
        for intervention, section in interventions.items():
            verify_rows(reference, section, test, selected, name + "/intervention/" + intervention, intervention=intervention)
        recurrence = read(directory / "recurrence.json")
        require(recurrence["active_limit"] == 32 and recurrence["recurrent_mix"] == .25, name + " recurrence parameters")
        for key, steps in (("no_recurrence", 0), ("two_steps", 2)):
            recalculated = verify_rows(reference, recurrence[key], test, selected, name + "/" + key, steps=steps)
            verify_metrics(condition["recurrence"][key], recalculated, name + "/summary/" + key)
        examples = read(directory / "continuations.json")
        prompts = [test[max(0, pos-32):pos] for pos in test_positions[::max(1, len(test_positions)//4)][:4]]
        require([example["prompt"] for example in examples] == prompts, name + " fixed prompts")
        for example in examples:
            for key, steps in (("greedy", 0), ("greedy_two_steps", 2)):
                require(len(example[key]) == 48, name + " continuation length")
                require(example[key] == reference.generate(example["prompt"], 48, steps), name + " generated sequence")
                CHECKS["continuations"] += 1
        with np.load(directory / "model.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"weights", "metadata"}, name + " NPZ fields")
            weights = archive["weights"]
            metadata = json.loads(str(archive["metadata"]))
            require(weights.shape == reference.w.shape and weights.dtype == np.dtype("float32"), name + " matrix shape/type")
            require(np.isfinite(weights).all() and np.all(weights >= 0), name + " finite nonnegative weights")
            require(np.array_equal(weights, reference.w), name + " full independent matrix reconstruction")
            require(metadata["vocabulary"] == vocabulary and metadata["config"] == cfg, name + " saved config")
            require(metadata["clock"] == reference.clock and metadata["learned_characters"] == reference.learned and
                    metadata["update_events"] == reference.updates, name + " saved counters")
            saved_activity = {int(i): float(a) for i, a in metadata["activity"]}
            require(saved_activity == reference.activity, name + " saved activity")
            digest = weight_hash(vocabulary, weights)
            require(digest == condition["final_checkpoint_weight_sha256"], name + " final checkpoint hash")
            asymmetry = int(np.count_nonzero(weights != weights.T))
        from hebb_text import CharacterHebb
        loaded = CharacterHebb.load(directory / "model.npz")
        loaded.train(train[:128])
        for ch in train[:128]:
            reference.advance(ch, training=True)
        require(np.array_equal(loaded.weights, reference.w), name + " continued learning matrix")
        require(loaded.activity == reference.activity and loaded.clock == reference.clock and
                loaded.learned_characters == reference.learned and loaded.update_events == reference.updates,
                name + " continued learning state/counters")
        require(condition["resume_128_exact"] is True, name + " resume declaration")
        training_seconds = sum(s["training_seconds"] for s in condition["stages"])
        require(condition["total_seconds"] >= training_seconds and math.isfinite(condition["total_seconds"]), name + " condition timing")
        results[name] = dict(stages_replayed=len(protocol["stages"]),
                             checkpoint_sha256=digest, asymmetric_matrix_entries=asymmetry,
                             resumed_characters=min(128, len(train)))
        del loaded, reference, weights
    require(summary["elapsed_seconds"] >= sum(c["total_seconds"] for c in summary["conditions"].values()), "Total timing")
    return results, protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results_v1")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        results, protocol = verify_experiment(args.out.resolve(), args.data.resolve())
    except Exception as error:
        message = f"FAIL: {type(error).__name__}: {error}"
        (HERE / "VERIFIED.md").write_text("# 独立核验失败\n\n" + message + "\n", encoding="utf-8")
        print(message, file=sys.stderr)
        raise
    elapsed = time.perf_counter() - start
    verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = ["# 独立核验记录", "", "结论：PASS。冻结实验所有条件、所有阶段及保存的逐项预测均通过独立重算。", "",
             f"- 结果目录：`{args.out.resolve()}`", f"- 训练字符数：{protocol['train_characters']:,}",
             f"- 核验耗时：{elapsed:.2f} 秒", f"- 参考实现：`verify_results.py`，SHA-256 `{verifier_sha}`",
             f"- 核验统计：`{json.dumps(dict(CHECKS), ensure_ascii=False)}`", "",
             "独立参考实现从零开始连续重放完整训练流，每阶段检查权重哈希、连接数、活动数、时钟、训练量及更新次数；包括出生与1000字阶段。终态NPZ的全部float32权重逐元素相等、有限且非负，并核对词表、配置、活动、计数器。载入原实现后续训128字，与独立实现逐元素及全部状态对照。", "",
             "核对冻结源码/数据SHA-256、固定随机抽样位置；重新计算每个预测的目标概率、稳定破同分rank、top1、top5、OOV、NLL/BPC及JSON汇总；重算出生预测、unigram、上下文干预、循环读出与生成样例。时间记录只验证有限性及加总一致性，不能事后证明墙钟实测值。", "",
             ("语料清单另通过独立核对：原始文件SHA-256与UTF-8严格解码、构建脚本哈希、三个分区哈希/长度/词表大小、119个来源块的唯一分区与逐块内容哈希、完整分区覆盖；三个分区两两交叉的至少30字完全相同段落数均为0。"
              if CHECKS.get("manifest_files") else "本次目录没有语料清单，未执行原始来源与段落划分核验。"), "",
             "## 模型与评分的解释边界", "",
             "- 未发现目标先输入后评分：预测使用目标之前的字符。权重独立重放一致，支持验证/测试没有进入训练；哈希只能证明当前文件与协议匹配，不能证明预注册时间或外部数据来源真实性。",
             "- 全部阶段预先知道完整训练词表，1000字阶段因此已知后续训练字符身份，但边权只来自当阶段已观察字符；出生同样知道词表。",
             "- rank同分按照排序词表的稳定索引决定，出生top1不是随机猜测的经验估计。top1/top5把OOV作为错误；NLL/BPC仅对已知字符计算。评估抽样只包含第64位以后的非空白字符，不能当作整篇文本困惑度。",
             "- 上下文活动为字符身份的衰减记忆，重复字符覆盖该身份活动，没有独立的词、句、角色或顺序隐状态；较长上下文效果不能证明组合推理。段落连续还会学到段落边界字符关联。",
             "- context_interventions的full仅指64字窗口，长痕迹条件的整篇连续评估最多能保留约228字历史，因此这两处full不是同一上下文条件；核验已分别按实际历史长度重算。",
             "- 循环读出固定使用同一权重图，额外包含32个活跃单元竞争、0.25混合与0.001均匀读出底噪；这些是明确的读出规则，不是学习出来的语言能力。",
             "- 核验保证实现和已记录结果一致；不证明语义理解、AGI能力、跨文体泛化或语料规模与性能的普遍因果关系。训练、验证和测试来自同两部作品；零长段落精确重合不等于不存在短片段或近重复，也不等于跨作品泛化。", "",
             "## 终态检查点", ""]
    for name, record in results.items():
        lines.append(f"- `{name}`：{record['stages_replayed']} 个阶段完整重放；方向不对称矩阵项 {record['asymmetric_matrix_entries']:,}；SHA-256 `{record['checkpoint_sha256']}`。")
    (HERE / "VERIFIED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", counts=dict(CHECKS), elapsed_seconds=round(elapsed, 3), conditions=results), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
