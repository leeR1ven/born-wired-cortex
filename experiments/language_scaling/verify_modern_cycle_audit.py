"""Independent short-trajectory audit; no experiment implementation imports.

Run after verify_modern_scale.py. Only writes our verification JSON and adds
an appendix to VERIFIED_MODERN_SCALE.md.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from verify_expanded_scale import Recall, same, require

HERE = Path(__file__).resolve().parent
OUT = HERE / "modern_scale_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    summary = read(OUT / "summary.json")
    protocol = read(OUT / "frozen_protocol.json")
    evidence = read(OUT / "cycle_audit.json")
    path = OUT / summary["final_model"]
    metadata = read(path / "metadata.json")
    vocabulary = metadata["vocabulary"]
    weights = np.load(path / "weights.npy", mmap_mode="r", allow_pickle=False)
    require(evidence["completed"] and evidence["learning_disabled"], "Complete read-only audit")
    require(evidence["model_weight_sha256"] == metadata["weight_sha256"], "Final model identity")
    actual = read(OUT / f"evaluation_{protocol['stages'][-1]}.json")["persistent8_weak"]["spoken"]
    cfg = protocol["recall_configs"]["persistent8_weak"]
    prompts = ["你好", "为什么", "地球", "请解释一下咕噜帕索是什么意思"]
    same([r["prompt"] for r in evidence["results"]], prompts, "Four fixed probes")
    verified = []
    for result in evidence["results"]:
        recall = Recall(weights, vocabulary, cfg)
        recall.warm(result["prompt"])
        seen, repeated, output = {}, None, []
        recorded = {r["step"]: r for r in result["rows"]}
        require(set(recorded) == set(range(8)) | set(range(152, 160)), "All first/last frames")
        for step in range(160):
            # The clock is intentionally excluded: neither current nor step
            # reads it. Ordered exact binary64 amplitudes determine the future.
            state = tuple((int(i), a.hex()) for i, a in recall.a.items())
            if state in seen and repeated is None:
                repeated = dict(first_step=seen[state], repeated_at=step, period=step-seen[state])
            seen.setdefault(state, step)
            current = recall.current()
            order = np.lexsort((np.arange(len(vocabulary)), -current))
            winner = int(order[0])
            output.append(vocabulary[winner])
            if step in recorded:
                row = recorded[step]
                require(row["winner"] == vocabulary[winner], "Winner")
                same(row["active"], recall.activity(), "Complete activity frame")
                require(row["positive_candidates"] == int(np.count_nonzero(current > 0)), "Positive candidate count")
                share = float(current[winner] / current.sum()) if current.sum() > 0 else None
                same(row["winner_current_share"], share, "Winner current share")
                same(row["strongest"], [dict(character=vocabulary[int(i)], current=float(current[i]))
                     for i in order[:8]], "Eight strongest currents")
                sources = np.array(list(recall.a), np.int64)
                amplitudes = np.array(list(recall.a.values()), np.float32)
                edges = weights[sources, winner]
                drive = edges * amplitudes
                indices = sorted(range(len(sources)), key=lambda j: (-float(drive[j]), j))
                same(row["winner_contributors"], [dict(character=vocabulary[int(sources[j])],
                     activity=float(amplitudes[j]), weight=float(edges[j]), current=float(drive[j]))
                     for j in indices], "Every winner contribution")
                alternative = weights[sources].copy()
                for j, identity in enumerate(sources):
                    alternative[j, identity] = 0
                alternative = np.sum(alternative * amplitudes[:, None], axis=0)
                indices = np.lexsort((np.arange(len(vocabulary)), -alternative))[:8]
                same(row["one_step_self_edges_removed"], [dict(character=vocabulary[int(i)],
                     current=float(alternative[i])) for i in indices], "One-step self-edge counterfactual")
            recall.step(vocabulary[winner], current)
        require(result["emitted"] == "".join(output), "All 160 emitted characters")
        require(next(r["emitted"] for r in actual if r["prompt"] == result["prompt"]) == "".join(output[:64]),
                "Original frozen 64-character output")
        same(result["first_exact_output_state_repeat"], repeated, "First exact ordered state recurrence")
        verified.append(dict(prompt=result["prompt"], first_repeat=repeated,
                             final_positive_candidates=recorded[159]["positive_candidates"]))
    proof = dict(status="PASS", independently_replayed_steps=640, complete_activity_frames=64,
                 winner_decompositions=64, single_step_self_edge_counterfactuals=64,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 cycle_audit_sha256=hashlib.sha256((OUT / "cycle_audit.json").read_bytes()).hexdigest(),
                 results=verified)
    (HERE / "modern_scale_verification_cycles.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf8")
    report = HERE / "VERIFIED_MODERN_SCALE.md"
    if report.exists():
        marker = "\n## 循环活动核验补充\n"
        text = report.read_text(encoding="utf8").split(marker)[0]
        text += marker + "\n独立复算 4 个固定输入的各 160 步输出，及每条首尾共 16 帧全部活动、正电流候选数、赢家贡献和单步去自连接电流。首次精确状态重现与原 64 字输出均一致。这里的状态不含未参与读出的单调时钟；单步去自连接并未改写实际轨迹。\n\n~~~json\n"
        text += json.dumps(proof, ensure_ascii=False, indent=2) + "\n~~~\n"
        report.write_text(text, encoding="utf8")
    print(json.dumps(proof, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
