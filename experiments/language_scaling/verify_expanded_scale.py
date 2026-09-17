"""Independent expanded-corpus reconstruction, continuation, scoring and dynamics.

Does not import experiment builders, model, evaluation or candidate controllers.
Only writes VERIFIED_EXPANDED_SCALE.md. Run after summary.completed becomes true.
"""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import time
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "expanded_scale_v1"
COUNTS = Counter()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def require(ok, label):
    if not ok:
        raise AssertionError(label)


def same(a, b, path=""):
    if isinstance(b, dict):
        require(isinstance(a, dict) and set(a) == set(b), path + " keys")
        for k in b:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(b, (list, tuple)):
        require(isinstance(a, (list, tuple)) and len(a) == len(b), path + " length")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    elif isinstance(b, (float, np.floating)):
        require(a is not None and math.isfinite(a) and
                math.isclose(a, float(b), rel_tol=1e-8, abs_tol=1e-12),
                f"{path}: {a} != {b}")
    else:
        require(a == b, f"{path}: {a!r} != {b!r}")


def check_hashes(mapping):
    for filename, digest in mapping.items():
        require(sha((HERE / filename).read_bytes()) == digest, "Source hash: " + filename)
        COUNTS["source_hashes"] += 1


def reconstruct_data():
    raw_root, data = HERE / "corpus_expansion_v1", HERE / "data_expanded_v1"
    m = read(data / "manifest.json")
    original_manifest = read(raw_root / "manifest.json")
    require(sha((raw_root / "manifest.json").read_bytes()) == m["source_manifest_sha256"],
            "Original source manifest")
    require(sha((HERE / "prepare_expanded_corpus.py").read_bytes()) == m["builder_sha256"],
            "Builder hash")
    check_hashes(m["script_inputs"])
    old = {s: (HERE / "data_no_punct_v1" / (s + ".txt")).read_text(encoding="utf8")
           for s in ("train", "validation", "test")}
    remove_space = lambda s: "".join(s.split())
    seen = {remove_space(p) for t in old.values() for p in t.split("\n\n") if len(p) >= 30}
    blocks, books = [], []
    chunks = {name: [] for name in old}
    total_duplicates = 0
    for b in original_manifest["books"]:
        raw = (raw_root / b["raw_file"]).read_bytes()
        require(sha(raw) == b["raw_sha256"], "Raw source bytes")
        t = raw.decode("utf-8-sig")
        start = re.search(r"\*\*\* START OF .*?\*\*\*", t)
        end = re.search(r"\*\*\* END OF .*?\*\*\*", t)
        require(start and end and start.end() < end.start(), "Source boundaries")
        body = t[start.end():end.start()].replace("\r\n", "\n").replace("\r", "\n").strip()
        require(sha((raw_root / b["body_file"]).read_bytes()) == b["body_sha256"], "Saved body hash")
        require((raw_root / b["body_file"]).read_text(encoding="utf8").strip() == body,
                "Saved source body")
        lines, removed_lines, headings = [], [], 0
        for line in body.splitlines():
            q = line.strip()
            if q.startswith(("Produced by ", "End of Project Gutenberg")) or q in {"/p", "p/", "<p>", "</p>"}:
                removed_lines.append(q)
            else:
                if re.match(r"^第[一二三四五六七八九十百千零〇0-9]+[回卷章]", q):
                    lines.append("")
                    headings += 1
                lines.append(line)
        paragraphs, punctuation_removed, duplicate_removed = [], 0, 0
        for chunk in re.split(r"\n\s*\n", "\n".join(lines)):
            joined = "".join(x.strip() for x in chunk.splitlines()).strip()
            paragraph = "".join(x for x in joined if unicodedata.category(x)[0] != "P")
            punctuation_removed += len(joined) - len(paragraph)
            if not paragraph:
                continue
            if len(paragraph) >= 30:
                key = remove_space(paragraph)
                if key in seen:
                    duplicate_removed += 1
                    continue
                seen.add(key)
            paragraphs.append(paragraph)
        grouped, pending, size = [], [], 0
        for p in paragraphs:
            if pending and size >= 5000:
                grouped.append("\n\n".join(pending))
                pending, size = [], 0
            pending.append(p)
            size += len(p)
        if pending:
            grouped.append("\n\n".join(pending))
        shuffled = list(range(len(grouped)))
        random.Random(19401 + int(b["id"])).shuffle(shuffled)
        n = max(1, round(.1 * len(grouped)))
        split_ids = {"validation": set(shuffled[:n]), "test": set(shuffled[n:2*n])}
        for index, chunk in enumerate(grouped):
            split = next((name for name in split_ids if index in split_ids[name]), "train")
            chunks[split].append((int(b["id"]), index, chunk))
            blocks.append(dict(source=int(b["id"]), block=index, split=split,
                               characters=len(chunk), sha256=sha(chunk.encode("utf8"))))
        books.append(dict(id=b["id"], title=b["title"], paragraphs=len(paragraphs), blocks=len(grouped),
                          punctuation_removed=punctuation_removed, duplicate_paragraphs_removed=duplicate_removed,
                          metadata_lines_removed=removed_lines, explicit_chapter_boundaries=headings,
                          characters=sum(map(len, paragraphs))))
        total_duplicates += duplicate_removed
    same(m["blocks"], blocks, "Data blocks")
    same(m["books"], books, "Book preprocessing")
    require(m["duplicate_paragraphs_removed"] == total_duplicates, "Duplicate count")
    texts = {s: "\n\n".join(x[2] for x in sorted(parts, key=lambda x: (x[1], x[0]))) + "\n"
             for s, parts in chunks.items()}
    training = old["train"] + "\n\n" + texts["train"]
    expected_files = {"train.txt": training, "new_validation.txt": texts["validation"], "new_test.txt": texts["test"]}
    for filename, text in expected_files.items():
        require((data / filename).read_text(encoding="utf8") == text, "Complete reconstructed file " + filename)
        require(not any(unicodedata.category(c)[0] == "P" for c in text), "No Unicode punctuation")
        spec = dict(characters=len(text), han_characters=sum("\u3400" <= c <= "\u9fff" for c in text),
                    unique_characters=len(set(text)), sha256=sha((data / filename).read_bytes()))
        same(m["files"][filename], spec, filename)
    sets = {f"old_{s}": {remove_space(p) for p in text.split("\n\n") if len(p) >= 30} for s, text in old.items()}
    sets.update({f"new_{s}": {remove_space(p) for p in text.split("\n\n") if len(p) >= 30} for s, text in texts.items()})
    overlaps = {a + "__" + b: len(sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2)}
    same(m["exact_paragraph_overlap_ge30"], overlaps, "All 15 split intersections")
    require(not any(overlaps.values()), "No long exact paragraph overlap")
    require(len(old["train"]) == m["old_training_prefix_characters"], "Prefix size")
    require(sha(old["train"].encode("utf8")) == m["old_training_prefix_sha256"], "Prefix logical hash")
    COUNTS["corpus_blocks"] = len(blocks)
    COUNTS["source_books"] = len(books)
    return training, old, texts, m


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"weights", "metadata"}, "Checkpoint fields")
        w = archive["weights"].copy()
        m = json.loads(str(archive["metadata"]))
    require(w.dtype == np.float32 and w.shape == (len(m["vocabulary"]),) * 2, "Matrix shape/dtype")
    require(np.isfinite(w).all() and np.min(w) >= 0, "Finite nonnegative weights")
    return w, m


def digest(vocab, w):
    return sha(vocab.encode("utf8") + w.tobytes())


def frequency_bin(n):
    return next((name for ceiling, name in [(1, "0"), (10, "1-9"), (100, "10-99"), (1000, "100-999")]
                 if n < ceiling), "1000+")


def sample_positions(text, seed, excluded=()):
    blocked = set(excluded)
    pool = [p for p in range(64, len(text)) if p not in blocked and "\u3400" <= text[p] <= "\u9fff"]
    return sorted(int(p) for p in np.random.default_rng(seed).choice(pool, 256, replace=False))


def metrics(rows):
    known = [r for r in rows if r["known"]]
    return dict(n=len(rows), known_n=len(known), oov_n=len(rows)-len(known),
                top1=sum(r["rank"] == 1 for r in rows)/len(rows),
                top5=sum(1 <= r["rank"] <= 5 for r in rows)/len(rows),
                mean_nll_known=math.fsum(-math.log(r["probability"]) for r in known)/len(known) if known else None,
                bpc_known=math.fsum(-math.log2(r["probability"]) for r in known)/len(known) if known else None)


class Recall:
    def __init__(self, w, vocabulary, cfg):
        self.w, self.v, self.cfg = w, vocabulary, cfg
        self.ids = {c: i for i, c in enumerate(vocabulary)}
        self.a = {}

    def current(self):
        if not self.a:
            return np.zeros(len(self.v), np.float32)
        index = np.asarray(list(self.a), np.int64)
        amplitude = np.asarray(list(self.a.values()), np.float32)
        return np.sum(self.w[index] * amplitude[:, None], axis=0)

    def step(self, character, signal=None):
        c = self.cfg
        if signal is None and c["candidate_count"]:
            signal = self.current()
        updated = {i: a*c["decay"] for i, a in self.a.items() if a*c["decay"] >= c["floor"]}
        injections = []
        if c["candidate_count"] and np.any(signal > 0):
            # Independent stable ordering, explicit positive candidates.
            ranked = np.lexsort((np.arange(len(signal)), -signal))[:c["candidate_count"]]
            selected = [int(i) for i in ranked if signal[i] > 0]
            total = math.fsum(float(signal[i]) for i in selected)
            for i in selected:
                x = c["candidate_budget"] * float(signal[i]) / total
                updated[i] = min(1., updated.get(i, 0.) + x)
                injections.append(dict(character=self.v[i], amount=x))
        if character in self.ids:
            updated[self.ids[character]] = 1.
        updated = {i: a for i, a in updated.items() if a >= c["floor"]}
        if len(updated) > c["active_limit"]:
            keep = set(sorted(updated, key=lambda i: (-updated[i], i))[:c["active_limit"]])
            updated = {i: a for i, a in updated.items() if i in keep}
        self.a = updated
        return injections

    def warm(self, prefix):
        for c in prefix:
            self.step(c)

    def activity(self):
        rank = sorted(self.a, key=lambda i: (-self.a[i], i))
        values = np.array(list(self.a.values()), np.float64)
        total = float(values.sum())
        square = float((values*values).sum())
        return dict(count=len(values), total=total,
                    maximum_share=float(values.max()/total) if total else 0.,
                    effective_count=total*total/square if square else 0.,
                    top=[dict(character=self.v[i], amplitude=self.a[i]) for i in rank[:16]],
                    all=[dict(character=self.v[i], amplitude=a) for i, a in self.a.items()])


def diagnose(train, counts, text, positions):
    result = {}
    for p in positions:
        prefix, target = text[p-64:p], text[p]
        d = dict(training_frequency=counts[target], frequency_bin=frequency_bin(counts[target]))
        for n in (3, 16, 64):
            d[f"seen_prefix{n}"] = prefix[-n:] in train
            if n != 3:
                d[f"seen_prefix{n}_plus_target"] = prefix[-n:]+target in train
        result[p] = d
    return result


def verify_score(saved, w, vocab, cfg, text, positions, diagnostics):
    require([r["position"] for r in saved["rows"]] == positions, "Fixed score positions")
    computed = []
    for row, pos in zip(saved["rows"], positions):
        probe = Recall(w, vocab, cfg)
        prefix = text[pos-64:pos]
        probe.warm(prefix)
        raw = probe.current().astype(np.float64)
        total = float(raw.sum())
        probabilities = .999*raw/total + .001/len(vocab) if total else np.full(len(vocab), 1/len(vocab))
        identity = probe.ids.get(text[pos])
        order = np.lexsort((np.arange(len(vocab)), -probabilities))
        rank = (1 + int(np.count_nonzero(probabilities > probabilities[identity])) +
                int(np.count_nonzero(probabilities[:identity] == probabilities[identity]))) if identity is not None else 0
        expected = dict(position=pos, prefix=prefix, target=text[pos], known=identity is not None, rank=rank,
                        probability=float(probabilities[identity]) if identity is not None else None,
                        predicted=vocab[int(order[0])], top5="".join(vocab[int(i)] for i in order[:5]),
                        activity=probe.activity(), **diagnostics[pos])
        same(row, expected, f"Prediction/{pos}")
        computed.append(expected)
    summary = metrics(computed)
    bins = {name: metrics(rows) for name in ("0", "1-9", "10-99", "100-999", "1000+")
            if (rows := [r for r in computed if r["frequency_bin"] == name])}
    exposures = {key: sum(r[key] for r in computed) for key in diagnostics[positions[0]] if key.startswith("seen_")}
    same(saved["metrics"], summary, "Score metrics")
    same(saved["by_training_frequency"], bins, "Score frequency groups")
    same(saved["prefix_exposure_counts"], exposures, "Score exposure counts")
    same(saved["mean_active_count"], np.mean([r["activity"]["count"] for r in computed]), "Mean count")
    same(saved["mean_effective_count"], np.mean([r["activity"]["effective_count"] for r in computed]), "Mean effective")
    COUNTS["prediction_rows"] += len(computed)
    COUNTS["frequency_metric_groups"] += len(bins)
    return summary, bins, exposures


def period(s):
    tail = s[-32:]
    return next((p for p in range(1, 17) if len(tail) == 32 and
                 all(tail[j] == tail[j % p] for j in range(len(tail)))), None)


def verify_trajectory(saved, w, vocab, cfg, prompt, silent):
    require(saved["prompt"] == prompt and saved["silent"] == silent and len(saved["rows"]) == 64, "Trajectory settings")
    probe = Recall(w, vocab, cfg)
    probe.warm(prompt)
    outputs = []
    for index, row in enumerate(saved["rows"]):
        raw = probe.current()
        chosen = vocab[int(np.argmax(raw))] if np.any(raw > 0) else None
        expected = dict(step=index, persistent=probe.activity(), strongest_candidate=chosen)
        expected["injected_candidates"] = probe.step(None if silent else chosen, raw)
        expected["next_persistent"] = probe.activity()
        same(row, expected, f"Trajectory/{index}")
        if not silent and chosen is not None:
            outputs.append(chosen)
    emitted = "".join(outputs)
    same(saved["emitted"], None if silent else emitted, "Complete continuation")
    same(saved["tail_period"], None if silent else period(emitted), "Tail period")
    COUNTS["trajectories"] += 1
    COUNTS["trajectory_steps"] += 64


def main():
    started = time.perf_counter()
    summary = read(OUT / "summary.json")
    require(summary["completed"] is True, "Experiment must be complete")
    protocol = read(OUT / "frozen_protocol.json")
    same(summary["protocol"], protocol, "Frozen protocol")
    check_hashes(protocol["source_sha256"])
    train, old_texts, new_texts, manifest = reconstruct_data()
    vocabulary = "".join(sorted(set(train)))
    require(len(vocabulary) == protocol["vocabulary_characters"] == 7652, "Full training vocabulary")
    require(len(train) == protocol["train_characters"] == 2841685, "Full training count")
    require(protocol["stages"] == [658462, 1000000, 2000000, len(train)], "Frozen stages")
    prior = read(HERE / "candidate_budget_v1/summary.json")
    excluded = sorted(set(prior["protocol"]["excluded_test_positions"]) | set(prior["protocol"]["test_positions"]))
    same(protocol["old_excluded_positions"], excluded, "Old position exclusions")
    old_positions = sample_positions(old_texts["test"], 19431, excluded)
    new_positions = sample_positions(new_texts["test"], 19432)
    same(protocol["old_test_positions"], old_positions, "Old positions seed")
    same(protocol["new_test_positions"], new_positions, "New positions seed")
    prompts = prior["protocol"]["generation_prompts"][:4] + [new_texts["test"][p-32:p] for p in new_positions[::64]]
    same(protocol["generation_prompts"], prompts, "Fixed eight prompts")
    require(len(prompts) == 8, "Prompt count")
    graph_cfg = dict(trace_decay=.25, trace_floor=.02, learning_rate=.25, growth="diminishing", ceiling=4.)
    same(protocol["graph_config"], graph_cfg, "Graph parameters")
    common = dict(decay=.25, floor=.02, active_limit=64)
    recall_cfgs = dict(recent_only=dict(candidate_count=0, candidate_budget=0., **common),
                       persistent8_weak=dict(candidate_count=8, candidate_budget=.25, **common),
                       persistent4_strong=dict(candidate_count=4, candidate_budget=1., **common))
    same(protocol["recall_configs"], recall_cfgs, "Recall parameters")
    old_w, old_meta = load_model(HERE / "short_context_v2/decay_0p25/model.npz")
    ids = {c: i for i, c in enumerate(vocabulary)}
    old_indices = np.array([ids[c] for c in old_meta["vocabulary"]], np.int64)
    weights = np.zeros((len(vocabulary),)*2, np.float32)
    weights[np.ix_(old_indices, old_indices)] = old_w
    del old_w
    activity = {ids[old_meta["vocabulary"][int(i)]]: a for i, a in old_meta["activity"]}
    clock, learned, updates = old_meta["clock"], old_meta["learned_characters"], old_meta["update_events"]
    require(clock == learned == 658462, "Starting checkpoint training count")
    same(old_meta["config"], graph_cfg, "Starting checkpoint graph parameters")
    consumed = 658462
    counts = Counter(train[:consumed])
    final_summary = []
    for record, stage in zip(summary["stages"], protocol["stages"]):
        require(record["stage"] == stage, "Stage order")
        expected_increment = stage if stage == 658462 else stage - consumed
        require(record["increment"] == expected_increment, "Stage increment")
        for ch in train[consumed:stage]:
            target = ids[ch]
            if activity:
                sources = np.array(list(activity), np.int64)
                amplitudes = np.array(list(activity.values()), np.float32)
                previous = weights[sources, target]
                weights[sources, target] = previous + (.25 * amplitudes) / (1. + previous)
                updates += len(sources)
            activity = {i: a*.25 for i, a in activity.items() if a*.25 >= .02}
            activity[target] = 1.
        clock += stage-consumed
        learned += stage-consumed
        counts.update(train[consumed:stage])
        saved_weights, metadata = load_model(OUT / f"model_{stage}.npz")
        require(np.array_equal(weights, saved_weights), f"Exact continued matrix at {stage}")
        del saved_weights
        same(metadata, dict(vocabulary=vocabulary, config=graph_cfg, clock=clock,
                           learned_characters=learned, update_events=updates, activity=list(activity.items())),
             f"Checkpoint metadata/{stage}")
        require(clock == learned == stage, "Continuous training clock")
        require(digest(vocabulary, weights) == record["weight_sha256"], "Stage weight hash")
        require(record["old_weight_projection_exact"] == (True if stage == 658462 else None), "Old projection flag")
        positive = weights[weights > 0]
        expected_stats = dict(characters=len(vocabulary), learned_characters=stage, clock=stage, update_events=updates,
                              connections=len(positive), matrix_bytes=weights.nbytes,
                              weight_max=float(positive.max()), weight_mean=float(positive.mean()),
                              active_characters=len(activity))
        same(record["stats"], expected_stats, "Weight statistics")
        del positive
        degrees = np.count_nonzero(weights, axis=1)
        per_character = [dict(character=c, count=counts[c], frequency_bin=frequency_bin(counts[c]),
                              outgoing_connections=int(degrees[i])) for i, c in enumerate(vocabulary)]
        coverage = dict(character_counts=dict(Counter(r["frequency_bin"] for r in per_character)),
                        han_character_counts=dict(Counter(r["frequency_bin"] for r in per_character if "\u3400" <= r["character"] <= "\u9fff")),
                        token_counts={name: sum(r["count"] for r in per_character if r["frequency_bin"] == name)
                                      for name in ("0", "1-9", "10-99", "100-999", "1000+")},
                        most_common=counts.most_common(20), per_character=per_character)
        same(read(OUT / f"{stage}_coverage.json"), coverage, "Character coverage")
        same(record["coverage"], {k: v for k, v in coverage.items() if k != "per_character"}, "Coverage summary")
        require(sum(counts.values()) == stage, "Total observed character frequency")
        diagnostic_old = diagnose(train[:stage], counts, old_texts["test"], old_positions)
        diagnostic_new = diagnose(train[:stage], counts, new_texts["test"], new_positions)
        small = {"stage": stage, "conditions": {}}
        for name, config in recall_cfgs.items():
            evidence = read(OUT / f"{stage}_{name}.json")
            old_metrics, old_bins, old_exposure = verify_score(evidence["old_test"], weights, vocabulary, config,
                                                             old_texts["test"], old_positions, diagnostic_old)
            new_metrics, new_bins, new_exposure = verify_score(evidence["new_test"], weights, vocabulary, config,
                                                             new_texts["test"], new_positions, diagnostic_new)
            require(len(evidence["trajectories"]) == 8, "All eight spoken trajectories")
            require(len(evidence["silent_trajectories"]) == (8 if config["candidate_count"] else 0), "Silent trajectory count")
            # All stages' trajectory output and full activity are replayed; no older experiments are rerun.
            for prompt, trajectory in zip(prompts, evidence["trajectories"]):
                verify_trajectory(trajectory, weights, vocabulary, config, prompt, False)
            for prompt, trajectory in zip(prompts, evidence["silent_trajectories"]):
                verify_trajectory(trajectory, weights, vocabulary, config, prompt, True)
            expected = dict(old_test=old_metrics, new_test=new_metrics, old_by_frequency=old_bins, new_by_frequency=new_bins,
                            old_prefix_exposure_counts=old_exposure, new_prefix_exposure_counts=new_exposure,
                            suffix_periods=[t["tail_period"] for t in evidence["trajectories"]],
                            old_mean_active_count=evidence["old_test"]["mean_active_count"],
                            new_mean_active_count=evidence["new_test"]["mean_active_count"])
            same(record["conditions"][name], expected, "Stage condition summary")
            small["conditions"][name] = dict(old_hits=round(256*old_metrics["top1"]), new_hits=round(256*new_metrics["top1"]),
                                             old_bpc=old_metrics["bpc_known"], new_bpc=new_metrics["bpc_known"])
        final_summary.append(small)
        COUNTS["stages"] += 1
        COUNTS["character_coverage_rows"] += len(per_character)
        print(json.dumps({"verified_stage": stage, "predictions": COUNTS["prediction_rows"]}), flush=True)
        consumed = stage
    require(len(summary["stages"]) == 4, "Exactly four stage records")
    check_hashes(protocol["source_sha256"])
    elapsed = time.perf_counter() - started
    proof = dict(status="PASS", elapsed_seconds=elapsed, counts=dict(COUNTS), stages=final_summary,
                 independently_replayed_new_training_characters=len(train)-658462,
                 starting_checkpoint="Previously verified old 658462-character model, projected into the fixed full-training vocabulary",
                 full_weight_arrays_exact=True, protocol_sha256=sha((OUT / "frozen_protocol.json").read_bytes()),
                 manifest_sha256=sha((HERE / "data_expanded_v1/manifest.json").read_bytes()),
                 verifier_sha256=sha(Path(__file__).read_bytes()))
    lines = [
        "# 扩大语料实验独立核验", "", "状态：PASS。", "",
        f"独立重建六本清洗、整段分块、固定种子分区和追加训练流；核对 {COUNTS['corpus_blocks']} 个来源块、全部文件哈希和 15 组新旧分区长段落交集。",
        "训练前 658,462 字符与原训练流精确一致。四阶段固定使用 7,652 字符的全量训练词表；从此前已核验的旧权重投影后，独立续训剩余 2,183,223 字符，四阶段完整矩阵、活动、clock、learned_characters 和边更新次数均逐位一致。",
        f"独立复现 {COUNTS['prediction_rows']:,} 行预测及 {COUNTS['trajectories']} 条完整轨迹共 {COUNTS['trajectory_steps']:,} 步，核对预测概率、名次、前五、每一步候选注入及全部活动、尾部周期。复算 {COUNTS['character_coverage_rows']:,} 行频次与非零出边覆盖、频次分组分数和前文出现诊断。",
        "未导入训练核心、候选控制器或原评分函数；未修改数据或旧模型。", "",
        "| 训练字符数 | 回忆条件 | 旧测试命中/256 | 新测试命中/256 | 旧 BPC | 新 BPC |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for s in final_summary:
        for name, r in s["conditions"].items():
            lines.append(f"| {s['stage']:,} | {name} | {r['old_hits']} | {r['new_hits']} | {r['old_bpc']:.4f} | {r['new_bpc']:.4f} |")
    lines += [
        "", "解释边界：",
        "",
        "- known 表示字符在预分配词表中；起始阶段可包含训练频次为 0 的字符。频次分组以当时实际读入字符为准。",
        "- 这些是同书未见块及固定真实 64 字前文的下一字预测；没有验证现代知识、跨书泛化或自由续写的语义正确性。",
        "- 完全长段落交集为 0、16/64 字原样前文未出现，都不能排除近重复、短片段背诵或共同人物情节；本轮没有近重复去重。",
        "- 追加数据同时改变书目组成和字符频次，不能把全部变化归因为数据量单一因素。四阶段共用测试位置，协议没有据测试选择参数。",
        "- 活动范围和尾部周期是动态指标；更宽活动或更长循环不足以证明语言能力改善。",
        "", f"独立核验耗时 {elapsed:.2f} 秒。", "", "原始核验摘要：", "", "~~~json",
        json.dumps(proof, ensure_ascii=False, indent=2), "~~~", "",
    ]
    (HERE / "VERIFIED_EXPANDED_SCALE.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(proof, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
