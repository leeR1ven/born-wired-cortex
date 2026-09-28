#!/usr/bin/env python3
"""Run deterministic bounded-synapse stress checks and write a JSON report."""

from __future__ import annotations

import json
import platform
import sys
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.synapses import BoundedSynapses
from born_wired.topology import make_edges


N_NEURONS = 32
LOCAL_DEGREE = 3
RANDOM_DEGREE = 2
SEEDS = tuple(range(5))
STEPS_PER_STAGE = 2000
DT = 0.02
BUDGET_TOLERANCE = 1e-10
TIME_LIMIT_SECONDS = 120.0
MAX_BUDGET_RATIO_SCOPE = (
    "maximum over 100-step and stage-end checks; not a per-step exact maximum"
)
PHASES = (
    "repeated_stimulation",
    "random_sparse_stimulation",
    "strong_positive_modulation",
    "strong_negative_modulation",
    "no_input",
)


def _phase_activities(phase, step, rng, repeated_pre, repeated_post):
    if phase == "repeated_stimulation":
        return repeated_pre, repeated_post, 1.0
    if phase == "random_sparse_stimulation":
        sparse_pre = (rng.random(N_NEURONS) < 0.10) * rng.random(N_NEURONS) * 0.8
        sparse_post = (rng.random(N_NEURONS) < 0.10) * rng.random(N_NEURONS) * 0.8
        return sparse_pre, sparse_post, 1.0
    if phase == "strong_positive_modulation":
        return np.ones(N_NEURONS), np.ones(N_NEURONS), 1.0
    if phase == "strong_negative_modulation":
        return np.ones(N_NEURONS), np.ones(N_NEURONS), -1.0
    if phase == "no_input":
        return np.zeros(N_NEURONS), np.zeros(N_NEURONS), 1.0
    raise ValueError(f"unknown phase: {phase}")


def _max_budget_ratio(totals, budgets):
    ratios = np.zeros(N_NEURONS, dtype=np.float64)
    positive = budgets > 0
    ratios[positive] = totals[positive] / budgets[positive]
    ratios[~positive] = np.where(totals[~positive] > BUDGET_TOLERANCE, np.inf, 0.0)
    return float(np.max(ratios))


def _check_state(syn, label):
    weights = syn.weights
    totals = syn.budget_totals()
    if not np.all(np.isfinite(weights)):
        raise AssertionError(f"{label}: non-finite weights")
    if not np.all(np.isfinite(syn.w_max)):
        raise AssertionError(f"{label}: non-finite w_max")
    if not np.all(np.isfinite(syn.budgets)):
        raise AssertionError(f"{label}: non-finite budgets")
    if np.any(weights < -BUDGET_TOLERANCE):
        raise AssertionError(f"{label}: negative weight")
    if np.any(weights > syn.w_max + BUDGET_TOLERANCE):
        raise AssertionError(f"{label}: weight exceeds cap")
    ratios = []
    for group_name, total in zip(("excitatory", "inhibitory"), totals):
        if not np.all(np.isfinite(total)):
            raise AssertionError(f"{label}: non-finite {group_name} budget total")
        if np.any(total < -BUDGET_TOLERANCE):
            raise AssertionError(f"{label}: negative {group_name} budget total")
        if np.any(total > syn.budgets + BUDGET_TOLERANCE):
            raise AssertionError(f"{label}: {group_name} budget exceeded")
        ratios.append(_max_budget_ratio(total, syn.budgets))
    return max(ratios), totals


def _run_seed(seed):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    positions = np.column_stack(
        (np.arange(N_NEURONS, dtype=np.float64), (np.arange(N_NEURONS) % 7) ** 2)
    )
    src, dst = make_edges(positions, LOCAL_DEGREE, RANDOM_DEGREE, seed=seed)
    signs = np.where(np.arange(N_NEURONS) % 2 == 0, 1.0, -1.0)
    weights = np.full(src.size, 0.05, dtype=np.float64)
    syn = BoundedSynapses(
        src,
        dst,
        weights,
        signs,
        N_NEURONS,
        w_max=1.0,
        budgets=2.0,
        learning_rate=0.01,
        decay=0.001,
        plasticity=1.0,
    )
    repeated_bits = (np.arange(N_NEURONS) % 4 == 0).astype(np.float64)
    repeated_pre = repeated_bits * 0.8
    repeated_post = np.roll(repeated_bits, 1) * 0.6
    start_weights = syn.weights.copy()
    max_weight = float(np.max(start_weights)) if start_weights.size else 0.0
    max_budget_ratio = 0.0
    checks = 0
    phase_end_weight_sums = []

    initial_ratio, _ = _check_state(syn, f"seed={seed} initial")
    max_budget_ratio = max(max_budget_ratio, initial_ratio)
    checks += 1

    step_count = 0
    for phase in PHASES:
        for phase_step in range(STEPS_PER_STAGE):
            pre, post, modulator = _phase_activities(
                phase, phase_step, rng, repeated_pre, repeated_post
            )
            syn.update(pre, post, modulator=modulator, dt=DT)
            step_count += 1
            max_weight = max(max_weight, float(np.max(syn.weights)))
            if step_count % 100 == 0 or phase_step == STEPS_PER_STAGE - 1:
                ratio, _ = _check_state(
                    syn, f"seed={seed} phase={phase} step={step_count}"
                )
                max_budget_ratio = max(max_budget_ratio, ratio)
                checks += 1
        phase_end_weight_sums.append(float(np.sum(syn.weights)))

    end_weights = syn.weights.copy()
    difference = end_weights - start_weights
    return {
        "seed": seed,
        "passed": True,
        "total_steps": step_count,
        "checks": checks,
        "max_weight": max_weight,
        "max_budget_ratio": max_budget_ratio,
        "max_budget_ratio_scope": MAX_BUDGET_RATIO_SCOPE,
        "start_weight_sum": float(np.sum(start_weights)),
        "end_weight_sum": float(np.sum(end_weights)),
        "weight_change_abs_sum": float(np.sum(np.abs(difference))),
        "weight_change_max_abs": float(np.max(np.abs(difference))),
        "phase_end_weight_sums": phase_end_weight_sums,
        "elapsed_seconds": time.perf_counter() - started,
        "error": None,
    }


def _run_independent_cases():
    result = {}
    try:
        correlated = BoundedSynapses(
            [0, 1],
            [2, 2],
            [0.1, 0.1],
            [1, 1, 1],
            3,
            learning_rate=0.1,
            decay=0,
            budgets=2,
        )
        for _ in range(100):
            correlated.update([1, 0, 0], [0, 0, 1])
        correlated_gain = float(correlated.weights[0] - 0.1)
        uncorrelated_gain = float(correlated.weights[1] - 0.1)
        result["correlated_growth"] = {
            "correlated_edge_gain": correlated_gain,
            "uncorrelated_edge_gain": uncorrelated_gain,
            "passed": correlated_gain > uncorrelated_gain + 0.15 > 0,
        }
    except Exception as exc:
        result["correlated_growth"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        weakened = BoundedSynapses(
            [0],
            [1],
            [0.5],
            [1, 1],
            2,
            learning_rate=0.1,
            decay=0,
            budgets=2,
        )
        initial = float(weakened.weights[0])
        for _ in range(10):
            weakened.update([1, 0], [0, 1], modulator=-1)
        final = float(weakened.weights[0])
        result["negative_modulation"] = {
            "initial_weight": initial,
            "final_weight": final,
            "passed": final < initial,
        }
    except Exception as exc:
        result["negative_modulation"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    result["passed"] = all(case["passed"] for key, case in result.items() if key != "passed")
    return result


def _run_unittests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TestResult()
    suite.run(result)
    failures = [
        {"test": str(test), "message": traceback_text.strip().splitlines()[-1]}
        for test, traceback_text in result.failures
    ]
    errors = [
        {"test": str(test), "message": traceback_text.strip().splitlines()[-1]}
        for test, traceback_text in result.errors
    ]
    return {
        "tests_run": result.testsRun,
        "failures": failures,
        "errors": errors,
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
    }


def main():
    script_started = time.perf_counter()
    test_result = _run_unittests()
    seed_results = []
    for seed in SEEDS:
        started = time.perf_counter()
        try:
            seed_results.append(_run_seed(seed))
        except Exception as exc:
            seed_results.append(
                {
                    "seed": seed,
                    "passed": False,
                    "total_steps": 0,
                    "checks": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    independent = _run_independent_cases()
    stress_elapsed = sum(item["elapsed_seconds"] for item in seed_results)
    elapsed = time.perf_counter() - script_started
    all_seeds_passed = all(item["passed"] for item in seed_results)
    total_steps = sum(item["total_steps"] for item in seed_results)
    within_target = stress_elapsed < TIME_LIMIT_SECONDS
    report = {
        "task": "02_verify_components",
        "status": "pass"
        if (
            test_result["passed"]
            and all_seeds_passed
            and independent["passed"]
            and within_target
        )
        else "fail",
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
        "config": {
            "n_neurons": N_NEURONS,
            "local_degree": LOCAL_DEGREE,
            "random_degree": RANDOM_DEGREE,
            "seeds": list(SEEDS),
            "steps_per_seed": STEPS_PER_STAGE * len(PHASES),
            "steps_per_stage": STEPS_PER_STAGE,
            "phases": list(PHASES),
            "dt": DT,
            "w_max": 1.0,
            "budgets": 2.0,
            "learning_rate": 0.01,
            "decay": 0.001,
            "plasticity": 1.0,
            "budget_tolerance": BUDGET_TOLERANCE,
            "strong_modulation_inputs": "all ones pre/post",
            "max_budget_ratio_scope": MAX_BUDGET_RATIO_SCOPE,
        },
        "unittest": test_result,
        "stress": {
            "passed": all_seeds_passed,
            "total_steps": total_steps,
            "seed_results": seed_results,
            "elapsed_seconds": stress_elapsed,
            "target_seconds": TIME_LIMIT_SECONDS,
            "within_target": within_target,
            "max_budget_ratio_scope": MAX_BUDGET_RATIO_SCOPE,
        },
        "independent_cases": independent,
        "elapsed_seconds": elapsed,
    }
    report["passed"] = report["status"] == "pass"
    report_path = ROOT / "artifacts" / "task02_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
