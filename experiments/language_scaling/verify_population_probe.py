"""Independent read-only oracle for population_probe_v1.

Imports our earlier independent weighted oracle only; never imports either new
propagation module or the experiment runner. No semantic scoring or retraining.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from verify_expanded_scale import Recall, same, require

HERE = Path(__file__).resolve().parent
OUT = HERE / "population_probe_v1"
CHECKS = Counter()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_digest(weights, vocabulary):
    result = hashlib.sha256(vocabulary.encode("utf8"))
    for start in range(0, len(vocabulary), 128):
        result.update(weights[start:start+128].tobytes())
    return result.hexdigest()


def current(weights, activity, no_self):
    if not activity:
        return np.zeros(len(weights), np.float32)
    ids = np.array(list(activity), np.int64)
    amplitudes = np.array(list(activity.values()), np.float32)
    pieces = weights[ids] * amplitudes[:, None]
    if no_self:
        for row, identity in enumerate(ids):
            pieces[row, identity] = 0
    return np.sum(pieces, axis=0)


class WeightedOracle(Recall):
    def __init__(self, weights, vocabulary, spec):
        cfg = dict(candidate_count=spec["k"], candidate_budget=spec["budget"],
                   decay=.25, floor=.02, active_limit=64)
        super().__init__(weights, vocabulary, cfg)
        self.no_self = spec["no_self"]

    def current(self):
        return current(self.w, self.a, self.no_self)


class BinaryOracle:
    def __init__(self, weights, vocabulary, spec):
        self.w, self.v, self.spec = weights, vocabulary, spec
        self.ids = {c: i for i, c in enumerate(vocabulary)}
        self.a, self.rest = {}, {}

    def warm(self, prompt):
        for character in prompt:
            self.a = {i: a*.25 for i, a in self.a.items() if a*.25 >= .02}
            if character in self.ids:
                self.a[self.ids[character]] = 1.
        self.a = dict(sorted(self.a.items()))
        if self.spec["refractory"]:
            self.rest = {i: self.spec["refractory"] for i in self.a}

    def current(self):
        # Incoming rest gates never modify the outgoing source sum.
        return current(self.w, self.a, self.spec["no_self"])

    def step(self, raw):
        allowed = np.array([i for i in np.flatnonzero(raw > 0) if i not in self.rest], np.int64)
        order = np.lexsort((allowed, -raw[allowed]))
        chosen = allowed[order[:self.spec["k"]]]
        require(len(chosen) == min(self.spec["k"], len(allowed)), "Exactly K eligible positive candidates")
        require(all(int(i) not in self.rest for i in chosen), "Resting units cannot receive")
        self.a = {int(i): 1. for i in sorted(chosen)}
        self.rest = {i: d-1 for i, d in self.rest.items() if d > 1}
        if self.spec["refractory"]:
            for i in chosen:
                self.rest[int(i)] = self.spec["refractory"]
        CHECKS["binary_simultaneous_updates"] += 1
        if self.spec["refractory"]:
            CHECKS["incoming_rest_gate_updates"] += 1
        return chosen


def state_key(engine):
    active = tuple((int(i), float(a).hex()) for i, a in engine.a.items())
    rest = tuple(sorted(engine.rest.items())) if isinstance(engine, BinaryOracle) else ()
    return active, rest


def mean(values):
    return math.fsum(values) / len(values)


def replay(weights, vocabulary, spec, result, steps):
    binary = spec["kind"] == "binary"
    oracle = BinaryOracle(weights, vocabulary, spec) if binary else WeightedOracle(weights, vocabulary, spec)
    oracle.warm(result["prompt"])
    union = set(oracle.a)
    seen, repeated, sets, records, emitted = {}, None, [], [], []
    for tick, row in enumerate(result["rows"]):
        key = state_key(oracle)
        if key in seen and repeated is None:
            repeated = dict(first=seen[key], again=tick, period=tick-seen[key])
        seen.setdefault(key, tick)
        old = set(oracle.a)
        sets.append(frozenset(old))
        raw = oracle.current()
        order = np.lexsort((np.arange(len(vocabulary)), -raw))[:8]
        winner = int(order[0]) if raw[order[0]] > 0 else None
        values = np.array(list(oracle.a.values()), np.float64)
        total = float(values.sum())
        expected = dict(step=tick, active_count=len(old),
                        effective_count=total*total/float(np.square(values).sum()) if total else 0.,
                        top_current=None if winner is None else vocabulary[winner],
                        positive_candidates=int(np.count_nonzero(raw > 0)))
        same(row["activity"], [[vocabulary[i], a] for i, a in oracle.a.items()], "Every ordered activity amplitude")
        same(row["top8_currents"], [[vocabulary[int(i)], float(raw[i])] for i in order if raw[i] > 0],
             "Every raw top-eight current")
        if binary:
            same(row["cooldown"], [[vocabulary[i], d] for i, d in sorted(oracle.rest.items())], "Every local cooldown")
            if tick:
                require(all(a == 1. for a in oracle.a.values()), "Every post-seed activity is one")
            if oracle.rest and oracle.a:
                require(all(i in oracle.rest for i in oracle.a), "Already-fired sources remain active during rest")
                CHECKS["resting_outgoing_frames"] += 1
        contributors_expected = winner is not None and (tick < 6 or tick >= steps-4)
        require(("contributors" in row) == contributors_expected, "Exactly declared connection frames")
        if contributors_expected:
            contributions = [dict(source=vocabulary[i], activation=a, weight=float(weights[i, winner]),
                                  used_current=0. if spec["no_self"] and i == winner else float(np.float32(a)*weights[i, winner]),
                                  self_edge=i == winner) for i, a in oracle.a.items()]
            same(row["contributors"], contributions, "All contributing edge values and applied current")
            CHECKS["connection_decomposition_frames"] += 1
        if binary:
            oracle.step(raw)
        else:
            observed = vocabulary[winner] if spec["spoken"] and winner is not None else None
            oracle.step(observed, raw)
            if observed is not None:
                emitted.append(observed)
        new = set(oracle.a)
        expected.update(new_vs_previous=len(new-old), new_ever=len(new-union), next_active_count=len(new))
        union.update(new)
        same({name: row[name] for name in expected}, expected, "Every per-step dynamics measurement")
        records.append(expected)
        CHECKS["independent_trajectory_steps"] += 1
    expected = dict(prompt=result["prompt"], emitted="".join(emitted) if spec.get("spoken") else None,
                    first_exact_repeat=repeated, distinct_units=len(union), final_active_count=len(oracle.a),
                    last32_distinct_member_sets=len(set(sets[-32:])),
                    mean_late_turnover=mean([r["new_vs_previous"] for r in records[-32:]]),
                    late_first_time_units=sum(r["new_ever"] for r in records[-32:]),
                    late_effective_count=mean([r["effective_count"] for r in records[-32:]]),
                    final_members=[vocabulary[i] for i in sorted(oracle.a)])
    same({name: result[name] for name in expected}, expected, "Complete independent trajectory aggregate")
    CHECKS["independent_trajectories"] += 1


def summarize(results):
    return dict(prompts=len(results), exact_state_repeats=sum(r["first_exact_repeat"] is not None for r in results),
                repeat_periods=dict(Counter(str(r["first_exact_repeat"]["period"]) if r["first_exact_repeat"] else "none" for r in results)),
                final_extinctions=sum(r["final_active_count"] == 0 for r in results),
                mean_final_active=mean([r["final_active_count"] for r in results]),
                mean_distinct_lifetime_units=mean([r["distinct_units"] for r in results]),
                mean_last32_member_sets=mean([r["last32_distinct_member_sets"] for r in results]),
                mean_late_turnover=mean([r["mean_late_turnover"] for r in results]),
                mean_late_effective_count=mean([r["late_effective_count"] for r in results]),
                mean_late_first_time_units=mean([r["late_first_time_units"] for r in results]))


def main():
    started = time.perf_counter()
    summary = read(OUT / "summary.json")
    require(summary["completed"], "Wait for experiment completion")
    protocol = read(OUT / "frozen_protocol.json")
    require(sha(OUT / "frozen_protocol.json") == summary["protocol_sha256"], "Frozen protocol SHA")
    for name, digest in protocol["source_sha256"].items():
        require(sha(HERE / name) == digest, "Frozen source " + name)
    require(protocol["steps"] == 160 and len(protocol["prompts"]) == 46 and len(protocol["specs"]) == 12,
            "Frozen condition/prompt/step counts")
    same(protocol["trace_prompts"], ["你好", "为什么", "地球", "请解释一下咕噜帕索是什么意思"], "Four predeclared detailed prompts")
    modern_protocol = read(HERE / "modern_scale_v1/frozen_protocol.json")
    same(protocol["prompts"], modern_protocol["probes"]["prompts"], "All original fixed prompts")
    meta = read(HERE / protocol["model"] / "metadata.json")
    weights = np.load(HERE / protocol["model"] / "weights.npy", mmap_mode="r", allow_pickle=False)
    vocabulary = meta["vocabulary"]
    require(weight_digest(weights, vocabulary) == meta["weight_sha256"] == protocol["weight_sha256"] == summary["weight_sha256"],
            "Entire frozen model identity")
    original = read(HERE / "modern_scale_v1/evaluation_344787049.json")["persistent8_weak"]["spoken"]
    baseline = {r["prompt"]: r["emitted"] for r in original}
    collected = {}
    result_hashes = {}
    for name, spec in protocol["specs"].items():
        path = OUT / (name + ".json")
        result_hashes[path.name] = sha(path)
        evidence = read(path)
        same(evidence["spec"], spec, "Saved condition matches protocol")
        results = evidence["results"]
        same([r["prompt"] for r in results], protocol["prompts"], "All prompts kept in original order")
        for result in results:
            rows = result["rows"]
            require(len(rows) == 160, "All propagation steps retained")
            require([r["step"] for r in rows] == list(range(160)), "No missing/reordered tick")
            for previous, following in zip(rows, rows[1:]):
                require(previous["next_active_count"] == following["active_count"], "Activity count continuity")
            require(rows[-1]["next_active_count"] == result["final_active_count"] == len(result["final_members"]), "Final activity counts")
            require(rows[0]["active_count"] + sum(r["new_ever"] for r in rows) == result["distinct_units"], "Lifetime distinct count")
            same(result["mean_late_turnover"], mean([r["new_vs_previous"] for r in rows[-32:]]), "Late turnover summary")
            same(result["late_effective_count"], mean([r["effective_count"] for r in rows[-32:]]), "Effective count summary")
            require(result["late_first_time_units"] == sum(r["new_ever"] for r in rows[-32:]), "First-time late units")
            if name == "weighted8_spoken":
                require(result["emitted"][:64] == baseline[result["prompt"]], "All original baseline outputs")
            else:
                require(result["emitted"] is None, "No population rank list decoded as speech")
            if result["prompt"] in protocol["trace_prompts"]:
                replay(weights, vocabulary, spec, result, protocol["steps"])
            else:
                require(all("activity" not in r for r in rows), "Declared compact evidence scope")
            CHECKS["aggregate_trajectories"] += 1
            CHECKS["aggregate_step_records"] += len(rows)
        computed = summarize(results)
        same(summary["conditions"][name], computed, "All condition summary metrics")
        collected[name] = computed
        print(json.dumps(dict(verified=name, detailed_trajectories=4)), flush=True)
    require(weight_digest(weights, vocabulary) == protocol["weight_sha256"], "Model unchanged at end of audit")
    for name, digest in protocol["source_sha256"].items():
        require(sha(HERE / name) == digest, "Frozen source unchanged " + name)
    proof = dict(status="PASS", seconds=time.perf_counter()-started, checks=dict(CHECKS),
                 verifier_sha256=sha(Path(__file__)), independent_weighted_helper_sha256=sha(HERE / "verify_expanded_scale.py"),
                 protocol_sha256=sha(OUT / "frozen_protocol.json"), result_sha256=result_hashes, conditions=collected)
    lines = ["# 同步多字符传播独立核验", "", "状态：PASS。", "",
             "原最终模型只读，完整权重哈希及本轮冻结源哈希一致。独立实现没有导入 population_propagation.py 或 run_population_probe.py。", "",
             "完整重放四个事前指定详细提示 × 12 个条件 × 160 步，共 7,680 步；逐步核对活动身份/幅度、原始电流前八、休息计数、成员更新、首次精确状态重现，以及首尾指定帧的全部赢家连接贡献。全部 46 × 12 条轨迹的 88,320 步保存记录用于计数连续性和指标汇总重算。其余 42 个提示未独立重算全部电流或精确状态周期。", "",
             "二值条件在初始感觉痕迹之后，将每个正信号且允许接收的前 K 个单元同步置为 1；足够合格候选时恰好 K 个。候选排名不作为句子输出。对角边屏蔽只影响当前电流计算，原矩阵未改。休息计数只限制新接收，已活动单元仍传出，并已独立核对。", "",
             "解释细节：top_current/top8_currents 记录的是休息门之前的原始电流，可能指向暂时禁止接收的单元；实际新群体见下一帧 activity。初始感觉种子幅度仍有 .25/.0625；首次推进后才都是 1。带休息条件在播种末统一将所有感觉种子计数设为 D，而非按提示内的历史发放时间计算。", "",
             "加权条件保留原先衰减后先删弱、注入后再删弱的顺序。K32/预算1 同时改变候选数与总反馈强度；二值规则也改变幅度与残留机制，因此不能将所有差异单独归因于 K。活动轮换、存活或周期变化均不是语义正确性或完整架构能力的评分。", "",
             "| 条件 | 检出精确周期的提示 | 最终全灭 | 平均最终活动数 | 尾部平均不同成员集合数 |", "|---|---:|---:|---:|---:|"]
    for name, r in collected.items():
        lines.append(f"| {name} | {r['exact_state_repeats']}/46 | {r['final_extinctions']} | {r['mean_final_active']:.3f} | {r['mean_last32_member_sets']:.3f} |")
    lines += ["", "完整核验记录：", "", "~~~json", json.dumps(proof, ensure_ascii=False, indent=2), "~~~", ""]
    (HERE / "VERIFIED_POPULATION_PROBE.md").write_text("\n".join(lines), encoding="utf8")
    (HERE / "population_probe_verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(dict(status="PASS", seconds=proof["seconds"], checks=dict(CHECKS)), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
