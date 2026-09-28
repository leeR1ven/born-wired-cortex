#!/usr/bin/env python3
"""Run deterministic dynamics demonstrations and save a JSON report."""

from __future__ import annotations

import json
import hashlib
import platform
import sys
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.dynamics import ConductanceNetwork
from born_wired.synapses import BoundedSynapses
from born_wired.topology import make_edges


N_NEURONS = 64
LOCAL_DEGREE = 3
RANDOM_DEGREE = 2
SEEDS = tuple(range(5))
LONG_RUN_STEPS = 10000
DEMO_WINDOW_STEPS = 250
WITHDRAWAL_STEPS = 150
TIME_LIMIT_SECONDS = 120.0
ACTIVITY_TOLERANCE = 1e-12
BUDGET_TOLERANCE = 1e-10


def _neuron_positions():
    return np.column_stack((np.arange(N_NEURONS) % 8, np.arange(N_NEURONS) // 8))


def _spatial_groups():
    grid = np.arange(N_NEURONS).reshape(8, 8)
    return (
        grid[:4, :4].reshape(-1),
        grid[:4, 4:].reshape(-1),
        grid[4:, :4].reshape(-1),
        grid[4:, 4:].reshape(-1),
    )


def _build_network(seed):
    rng = np.random.default_rng(seed)
    src, dst = make_edges(_neuron_positions(), LOCAL_DEGREE, RANDOM_DEGREE, seed=seed)
    signs = np.where(rng.random(N_NEURONS) < 0.75, 1.0, -1.0)
    synapses = BoundedSynapses(
        src,
        dst,
        np.full(src.size, 0.02, dtype=np.float64),
        signs,
        N_NEURONS,
        w_max=1.0,
        budgets=2.0,
        learning_rate=0.01,
        decay=0.001,
        plasticity=1.0,
    )
    network = ConductanceNetwork(synapses, tau=0.02, leak=1.0, threshold=0.1)
    return rng, synapses, network


def _long_run_seed(seed):
    started = time.perf_counter()
    rng, synapses, network = _build_network(seed)
    groups = _spatial_groups()
    voltage_min = 1.0
    voltage_max = 0.0
    activity_min = 1.0
    activity_max = 0.0
    weight_min = float(np.min(synapses.weights))
    weight_max = float(np.max(synapses.weights))
    positive_source_activity_sum = 0.0
    negative_source_activity_sum = 0.0
    external_input_sum = 0.0
    all_dead_steps = 0
    near_all_bright_steps = 0
    stimulated_steps = 0
    stimulated_dead_steps = 0
    budget_violations = 0
    max_budget_ratio_sampled = 0.0
    budget_saturated_observations = 0
    budget_observations = 0
    positive_sources = synapses.signs > 0
    negative_sources = synapses.signs < 0

    for step in range(LONG_RUN_STEPS):
        phase = (step // 500) % 4
        external = np.zeros(N_NEURONS)
        if phase == 0:
            external[groups[0]] = 0.8
        elif phase == 1:
            external[groups[1]] = 0.8
        elif phase == 2:
            external[rng.random(N_NEURONS) < 0.10] = 0.8
        activity = network.step(external, learn=True)
        voltage = network.voltage
        voltage_min = min(voltage_min, float(np.min(voltage)))
        voltage_max = max(voltage_max, float(np.max(voltage)))
        activity_min = min(activity_min, float(np.min(activity)))
        activity_max = max(activity_max, float(np.max(activity)))
        current_weights = synapses.weights
        weight_min = min(weight_min, float(np.min(current_weights)))
        weight_max = max(weight_max, float(np.max(current_weights)))
        positive_source_activity_sum += float(np.sum(activity[positive_sources]))
        negative_source_activity_sum += float(np.sum(activity[negative_sources]))
        external_input_sum += float(np.sum(external))
        if np.all(activity <= ACTIVITY_TOLERANCE):
            all_dead_steps += 1
        if np.any(external > 0):
            stimulated_steps += 1
            stimulated_dead_steps += int(np.all(activity <= ACTIVITY_TOLERANCE))
        if np.mean(activity >= 0.9) >= 0.9:
            near_all_bright_steps += 1
        if step % 10 == 0:
            exc_total, inh_total = synapses.budget_totals()
            for total in (exc_total, inh_total):
                budget_violations += int(np.count_nonzero(
                    ~np.isfinite(total) | (total > synapses.budgets + BUDGET_TOLERANCE)))
                max_budget_ratio_sampled = max(
                    max_budget_ratio_sampled, float(np.max(total / synapses.budgets)))
            saturated = (exc_total >= synapses.budgets - BUDGET_TOLERANCE) | (
                inh_total >= synapses.budgets - BUDGET_TOLERANCE
            )
            budget_saturated_observations += int(np.count_nonzero(saturated))
            budget_observations += N_NEURONS

    positive_count = int(np.count_nonzero(positive_sources))
    negative_count = int(np.count_nonzero(negative_sources))
    result = {
        "seed": seed,
        "steps": LONG_RUN_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "voltage_range": [voltage_min, voltage_max],
        "activity_range": [activity_min, activity_max],
        "weight_range": [weight_min, weight_max],
        "positive_source_activity_mean": positive_source_activity_sum
        / (LONG_RUN_STEPS * positive_count),
        "negative_source_activity_mean": negative_source_activity_sum
        / (LONG_RUN_STEPS * negative_count),
        "external_input_mean_per_cell_step": external_input_sum
        / (LONG_RUN_STEPS * N_NEURONS),
        "budget_saturation_ratio": budget_saturated_observations / budget_observations,
        "all_dead_step_fraction": all_dead_steps / LONG_RUN_STEPS,
        "near_all_bright_step_fraction": near_all_bright_steps / LONG_RUN_STEPS,
        "stimulated_steps": stimulated_steps,
        "stimulated_dead_step_fraction": stimulated_dead_steps / stimulated_steps if stimulated_steps else None,
        "budget_violations": budget_violations,
        "max_budget_ratio_sampled": max_budget_ratio_sampled,
    }
    result["passed"] = (
        np.isfinite(result["voltage_range"]).all()
        and np.isfinite(result["activity_range"]).all()
        and np.isfinite(result["weight_range"]).all()
        and result["voltage_range"][0] >= -1e-14
        and result["voltage_range"][1] <= 1.0 + 1e-14
        and result["activity_range"][0] >= -1e-14
        and result["activity_range"][1] <= 1.0 + 1e-14
        and result["weight_range"][0] >= -1e-14
        and result["weight_range"][1] <= 1.0 + BUDGET_TOLERANCE
        and budget_violations == 0
        and stimulated_steps > 0
        and stimulated_dead_steps < stimulated_steps
        and near_all_bright_steps < LONG_RUN_STEPS
    )
    return result


def _run_demonstration():
    started = time.perf_counter()
    _, synapses, network = _build_network(seed=100)
    groups = _spatial_groups()
    before = synapses.weights.copy()
    windows = []
    group_partition = []

    for group_index, group in enumerate(groups):
        target = np.zeros(N_NEURONS, dtype=bool)
        target[group] = True
        label = f"region_{group_index}"
        run_name = f"{label}_input"
        activity_sum_target = 0.0
        activity_sum_other = 0.0
        for _ in range(DEMO_WINDOW_STEPS):
            external = np.zeros(N_NEURONS)
            external[group] = 1.0
            activity = network.step(external, learn=True)
            activity_sum_target += float(np.sum(activity[target]))
            activity_sum_other += float(np.sum(activity[~target]))
        windows.append(
            {
                "window": run_name,
                "steps": DEMO_WINDOW_STEPS,
                "target_mean_activity": activity_sum_target
                / (DEMO_WINDOW_STEPS * len(group)),
                "other_mean_activity": activity_sum_other
                / (DEMO_WINDOW_STEPS * (N_NEURONS - len(group))),
            }
        )

        run_name = f"{label}_withdrawal"
        activity_sum_target = 0.0
        activity_sum_other = 0.0
        for _ in range(WITHDRAWAL_STEPS):
            activity = network.step(
                np.zeros(N_NEURONS), learn=True
            )
            activity_sum_target += float(np.sum(activity[target]))
            activity_sum_other += float(np.sum(activity[~target]))
        windows.append(
            {
                "window": run_name,
                "steps": WITHDRAWAL_STEPS,
                "target_mean_activity": activity_sum_target
                / (WITHDRAWAL_STEPS * len(group)),
                "other_mean_activity": activity_sum_other
                / (WITHDRAWAL_STEPS * (N_NEURONS - len(group))),
            }
        )
        group_partition.append(
            {
                "group": label,
                "cells": [int(value) for value in group],
            }
        )

    after = synapses.weights.copy()
    difference = after - before
    weight_checks = {
        "changed_edges": int(np.count_nonzero(np.abs(difference) > 1e-12)),
        "l1_change": float(np.sum(np.abs(difference))),
        "max_abs_change": float(np.max(np.abs(difference))),
        "before_sum": float(np.sum(before)),
        "after_sum": float(np.sum(after)),
    }
    finite = all(
        np.isfinite(value)
        for window in windows
        for value in (
            window["target_mean_activity"],
            window["other_mean_activity"],
        )
    )
    input_windows = [window for window in windows if window["window"].endswith("_input")]
    passed = (
        finite
        and weight_checks["changed_edges"] > 0
        and all(window["target_mean_activity"] > window["other_mean_activity"] + ACTIVITY_TOLERANCE
                for window in input_windows)
    )
    return {
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "group_definition_is_label_only": True,
        "group_partition": group_partition,
        "windows": windows,
        "weight_change": weight_checks,
    }


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
    started = time.perf_counter()
    test_result = _run_unittests()
    long_runs = []
    for seed in SEEDS:
        try:
            long_runs.append(_long_run_seed(seed))
        except Exception as exc:
            long_runs.append({"seed": seed, "steps": None, "passed": False,
                              "error": f"{type(exc).__name__}: {exc}"})
    try:
        demonstration = _run_demonstration()
    except Exception as exc:
        demonstration = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - started
    all_long_runs_passed = all(item["passed"] for item in long_runs)
    all_passed = (
        test_result["passed"]
        and all_long_runs_passed
        and demonstration["passed"]
        and elapsed <= TIME_LIMIT_SECONDS
    )
    report = {
        "task": "03_neural_dynamics",
        "status": "pass" if all_passed else "fail",
        "model": "conductance_rate_network",
        "implementation_assistant": "Codex B / deepseek-flash; reviewed by Codex A",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [*sorted((ROOT / "born_wired").glob("*.py")), Path(__file__)]},
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
            "long_run_steps_per_seed": LONG_RUN_STEPS,
            "demo_input_steps_per_group": DEMO_WINDOW_STEPS,
            "demo_withdrawal_steps_per_group": WITHDRAWAL_STEPS,
            "model_semantics": "rate approximation; no action layer",
        },
        "unittest": test_result,
        "long_runs": {
            "passed": all_long_runs_passed,
            "seeds": long_runs,
            "total_steps": sum(item["steps"] or 0 for item in long_runs),
            "requested_steps": LONG_RUN_STEPS * len(SEEDS),
        },
        "demonstration": demonstration,
        "limitations": [
            "The model is a rate approximation, not a biological neuron.",
            "No action, robot, navigation, or state-machine behavior is implemented.",
            "Labels are used only for reporting stimulus-group statistics.",
            "Bounded numerical behavior does not establish long-term stability.",
        ],
        "elapsed_seconds": elapsed,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
    }
    report["passed"] = report["status"] == "pass"
    report_path = ROOT / "artifacts" / "task03_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
