#!/usr/bin/env python3
"""Draw the validation summary from artifact data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "validation_summary.png"

CONDITION_LABELS = {
    "paired": "配对",
    "unpaired": "错配",
    "frozen": "关学习",
}
CONDITION_COLORS = {
    "paired": "#1f77b4",
    "unpaired": "#d95f02",
    "frozen": "#4d4d4d",
}
CASE_LABELS = {
    "stand": "站立",
    "rise": "起立",
    "flex": "屈曲",
    "perturb": "扰动",
    "rhythm": "节律",
    "silenced": "silenced",
}


def load_json(name: str) -> dict[str, Any] | None:
    path = ARTIFACTS / name
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{name}: JSON 顶层必须是对象")
    return value


def find_result(
    payload: dict[str, Any],
    condition: str | None = None,
    seed: int | None = None,
    hidden_routes: bool | None = None,
    learn: bool | None = None,
) -> dict[str, Any] | None:
    items = payload.get("results")
    if not isinstance(items, list):
        items = payload.get("cases", [])
    for result in items:
        if not isinstance(result, dict):
            continue
        if condition is not None and result.get("condition") != condition:
            continue
        if seed is not None and result.get("seed") != seed:
            continue
        if hidden_routes is not None and result.get("hidden_routes") is not hidden_routes:
            continue
        if learn is not None and result.get("learn") is not learn:
            continue
        return result
    return None


def draw_conditioning(ax: plt.Axes, payload: dict[str, Any]) -> None:
    plotted = []
    for condition in ("paired", "unpaired", "frozen"):
        result = find_result(payload, condition=condition, seed=0)
        if result is None:
            continue
        samples = result.get("samples", [])
        points = [
            (sample.get("time"), sample.get("height"))
            for sample in samples
            if isinstance(sample, dict)
        ]
        points = [(time, height) for time, height in points if time is not None and height is not None]
        if not points:
            continue
        time = np.asarray([point[0] for point in points], dtype=float)
        height = np.asarray([point[1] for point in points], dtype=float)
        ax.plot(
            time,
            height,
            color=CONDITION_COLORS[condition],
            linewidth=1.8,
            label=CONDITION_LABELS[condition],
        )
        plotted.append(condition)

    ax.set_title("条件实验：seed 0 高度随时间", loc="left", fontweight="bold", pad=30)
    ax.set_xlabel("仿真时间（秒）")
    ax.set_ylabel("身体高度（米）")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    if plotted:
        ax.legend(frameon=False, ncols=3)
    ax.text(
        0.0,
        1.01,
        "提示输入为条件线索，不是声音识别",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
        va="bottom",
    )


def draw_hidden_routes(ax: plt.Axes, payload: dict[str, Any]) -> None:
    result = find_result(
        payload,
        seed=0,
        hidden_routes=True,
        learn=True,
    )
    if result is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "hidden_routes.json 缺少 hidden=真、学习开启、seed 0 的真实结果",
            ha="center",
            va="center",
            wrap=True,
        )
        return

    after = result.get("after", [])
    motors = [
        np.asarray(item.get("motor", []), dtype=float)
        for item in after
        if isinstance(item, dict)
    ]
    if len(motors) < 2 or any(motor.size < 2 for motor in motors):
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "hidden_routes.json 缺少训练后两刺激的两个运动输出",
            ha="center",
            va="center",
            wrap=True,
        )
        return

    indices = np.arange(2)
    width = 0.34
    ax.bar(
        indices - width / 2,
        motors[0][:2],
        width,
        color="#1f77b4",
        label="运动输出 1",
    )
    ax.bar(
        indices + width / 2,
        motors[1][:2],
        width,
        color="#e6a020",
        label="运动输出 2",
    )
    ax.set_xticks(indices, ["合成刺激 A", "合成刺激 B"])
    ax.set_ylabel("训练后运动输出")
    ax.set_title("隐藏直连：训练后两刺激输出", loc="left", fontweight="bold", pad=30)
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.text(
        0.0,
        1.01,
        "隐藏实验使用两种合成位置特征；评估时无运动教学输入",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
        va="bottom",
    )


def draw_robot_matrix(ax: plt.Axes, payload: dict[str, Any] | None) -> None:
    if payload is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "robot_matrix_60s.json 未生成\n60 秒情景验证未完成",
            ha="center",
            va="center",
            fontsize=14,
            color="#444444",
        )
        return

    results = [result for result in payload.get("results", []) if isinstance(result, dict)]
    cases = []
    for result in results:
        case = result.get("case")
        if case and case not in cases:
            cases.append(case)

    labels = []
    fractions = []
    annotation = []
    for case in cases:
        case_results = [result for result in results if result.get("case") == case]
        labels.append(CASE_LABELS.get(case, case))
        if case == "silenced":
            fractions.append(np.nan)
            annotation.append(f"消融\n不计成功\nn={len(case_results)}")
            continue
        valid = [
            result
            for result in case_results
            if "error" not in result and isinstance(result.get("metrics"), dict)
        ]
        falls = sum(result["metrics"].get("fallen_any") is True for result in valid)
        fraction = falls / len(case_results) if case_results else np.nan
        fractions.append(fraction)
        suffix = f"；错误 {len(case_results) - len(valid)}" if len(valid) != len(case_results) else ""
        annotation.append(f"{falls}/{len(case_results)}{suffix}")

    values = np.asarray(fractions, dtype=float)
    x = np.arange(len(cases))
    valid_bars = ~np.isnan(values)
    ax.bar(x[valid_bars], values[valid_bars], color="#3b7a57", width=0.68)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("发生跌倒的试验比例")
    ax.set_title("60 秒情景：跌倒数 / 试验数", loc="left", fontweight="bold", pad=30)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    for xpos, value, text in zip(x, values, annotation):
        ypos = value + 0.04 if np.isfinite(value) else 0.08
        ax.text(xpos, ypos, text, ha="center", va="bottom", fontsize=9)
    ax.text(
        0.0,
        1.01,
        "silenced 为运动消融对照，不判定成功",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
        va="bottom",
    )


def extract_stand_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("case") not in (None, "stand"):
                continue
            samples = result.get("samples")
            if isinstance(samples, list) and samples:
                return [sample for sample in samples if isinstance(sample, dict)]
    samples = payload.get("samples")
    if isinstance(samples, list):
        return [sample for sample in samples if isinstance(sample, dict)]
    return []


def draw_stand_hour(ax: plt.Axes, payload: dict[str, Any] | None) -> None:
    if payload is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "stand_1hour.json 未生成\n1 小时站立验证未完成",
            ha="center",
            va="center",
            fontsize=14,
            color="#444444",
        )
        return

    samples = extract_stand_samples(payload)
    points = [
        (sample.get("time"), sample.get("body_height"), sample.get("body_up_z"))
        for sample in samples
    ]
    points = [
        point
        for point in points
        if all(value is not None for value in point)
    ]
    if not points:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "stand_1hour.json 缺少 body_height/body_up_z 曲线",
            ha="center",
            va="center",
            fontsize=13,
            color="#444444",
        )
        return

    time = np.asarray([point[0] for point in points], dtype=float)
    height = np.asarray([point[1] for point in points], dtype=float)
    up_z = np.asarray([point[2] for point in points], dtype=float)

    ax.plot(time, height, color="#1f77b4", linewidth=1.4, label="身体高度")
    ax.set_xlabel("仿真时间（秒）")
    ax.set_ylabel("身体高度（米）", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.8)

    twin = ax.twinx()
    twin.plot(time, up_z, color="#d95f02", linewidth=1.4, label="up_z")
    twin.set_ylabel("身体 up_z", color="#d95f02")
    twin.tick_params(axis="y", labelcolor="#d95f02")

    ax.set_title("1 小时站立：高度与 up_z", loc="left", fontweight="bold", pad=30)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    ax.text(
        0.0,
        1.01,
        "仅代表本次固定种子与配置；数值约束不是永久可靠保证",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
        va="bottom",
    )


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    conditioning = load_json("conditioning.json")
    hidden_routes = load_json("hidden_routes.json")
    robot_matrix = load_json("robot_matrix_60s.json")
    stand_hour = load_json("stand_1hour.json")
    if conditioning is None or hidden_routes is None:
        missing = [
            name
            for name, payload in (
                ("conditioning.json", conditioning),
                ("hidden_routes.json", hidden_routes),
            )
            if payload is None
        ]
        raise FileNotFoundError("缺少必需真实数据: " + ", ".join(missing))

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    draw_conditioning(axes[0, 0], conditioning)
    draw_hidden_routes(axes[0, 1], hidden_routes)
    draw_robot_matrix(axes[1, 0], robot_matrix)
    draw_stand_hour(axes[1, 1], stand_hour)
    figure.suptitle(
        "Born-Wired Cortex 仿真验证摘要",
        x=0.055,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.015,
        "仅绘制 artifacts 中真实存在的数据；不生成或插值未完成实验",
        color="#555555",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.03, 0.04, 0.98, 0.93), h_pad=2.8, w_pad=2.4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=100, facecolor="white")
    plt.close(figure)
    print(f"已生成 {OUTPUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
