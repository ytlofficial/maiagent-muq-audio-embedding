#!/usr/bin/env python3
"""Split a chart into density-aware sections from measure density curves."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_measure_density import (
    chart_measure_densities,
    combined_handtrip_summary,
    read_json,
    resolve_chart,
    same_button_triple_tap_summary,
    slide_movement_summary,
    tap_distance_summary,
)
from scripts.simai_note_counter import (
    button_distribution_from_counts,
    note_mix_from_counts,
    special_ratios_from_counts,
)


MIN_DENSITY_PROFILE_MEASURE_SECONDS = 0.4
DEFAULT_SEGMENT_SCORE_BASELINES = Path("outputs/six_dimension/global_six_dimension_intermediates.json")
DEFAULT_STEADY_DENSITY_STANDARDS = Path("outputs/steady_density_standards/steady_density_by_level.json")


@dataclass(frozen=True)
class SegmentationConfig:
    min_segments: int = 5
    max_segments: int = 5
    fixed_segments: int | None = 5
    edge_exclusion: int = 8
    min_segment_measures: int | None = None
    max_segment_length_factor: float = 1.75
    window: int = 4
    balance_weight: float = 1.0
    extra_segment_penalty: float = 0.35


@dataclass(frozen=True)
class SegmentationPlan:
    segment_count: int
    boundaries: list[int]
    score: float
    boundary_score_average: float
    balance_penalty: float


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def population_stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def level_value(level: Any) -> float | None:
    if level is None:
        return None
    text = str(level).strip()
    if not text:
        return None
    bonus = 0.5 if text.endswith("+") else 0.0
    if bonus:
        text = text[:-1]
    try:
        return float(text) + bonus
    except ValueError:
        return None


def load_steady_density_standards(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    standards = payload.get("recommended_standards", [])
    if not isinstance(standards, list):
        raise ValueError("steady density standards must contain a recommended_standards list")
    return {
        "source_file": str(path),
        "description": payload.get("description"),
        "standards": standards,
        "by_level": {
            str(item.get("level")): item
            for item in standards
            if isinstance(item, dict) and item.get("level") is not None
        },
    }


def steady_density_standard_for_level(
    standards: dict[str, Any] | None,
    level: Any,
) -> dict[str, Any] | None:
    parsed = level_value(level)
    if standards is None:
        return None
    by_level = standards.get("by_level", {})
    if isinstance(by_level, dict):
        exact = by_level.get(str(level).strip()) if level is not None else None
        if isinstance(exact, dict):
            return exact

    if parsed is None:
        return None
    if parsed < 11:
        return {
            "level": str(level),
            "level_value": parsed,
            "low": "all",
            "mid": "none",
            "high": "none",
            "source": "forced_below_11_fallback",
        }
    if parsed >= 15:
        return {
            "level": str(level),
            "level_value": parsed,
            "low": "none",
            "mid": "none",
            "high": "all",
            "source": "forced_15_or_above_fallback",
        }

    candidates = [
        item
        for item in standards.get("standards", [])
        if isinstance(item, dict) and level_value(item.get("level")) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((level_value(item.get("level")) or parsed) - parsed))


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_percentile(value: float, values: list[float]) -> float:
    q05 = percentile(values, 0.05)
    q95 = percentile(values, 0.95)
    if math.isclose(q05, q95):
        return 0.5 if value > 0 else 0.0
    return clip((value - q05) / (q95 - q05))


def normalize_to_q95(value: float, values: list[float]) -> float:
    q95 = percentile(values, 0.95)
    if q95 <= 0:
        return 0.0
    return clip(value / q95)


def saturated_ratio_density_score(
    *,
    ratio_value: float,
    density_value: float,
    ratio_98: float,
    density_98: float,
    ratio_weight: float = 0.65,
    density_weight: float = 0.35,
    epsilon: float = 0.001,
) -> dict[str, float]:
    if ratio_value <= 0 or density_value <= 0:
        normalized_ratio = 0.0
        normalized_density = 0.0
    else:
        normalized_density = clip(density_value / density_98) if density_98 > 0 else 1.0
        normalized_ratio = clip(ratio_value / ratio_98) if ratio_98 > 0 else 1.0
    fused = (
        ((1 + epsilon) * normalized_density * normalized_ratio)
        / (ratio_weight * normalized_ratio + density_weight * normalized_density + epsilon)
        if normalized_density > 0 and normalized_ratio > 0
        else 0.0
    )
    return {
        "score": 200 * clip(fused),
        "density": density_value,
        "ratio": ratio_value,
        "normalized_density": normalized_density,
        "normalized_ratio": normalized_ratio,
        "density_98": density_98,
        "ratio_98": ratio_98,
        "mapping": "ratio_density_p98_harmonic_to_200",
    }


def burst_peak_score(
    burst_density: float,
    burst_density_98: float,
    *,
    floor_score: float = 10.0,
    alpha: float = 1.35,
) -> dict[str, float]:
    if burst_density_98 <= 0:
        normalized = 0.0
    else:
        normalized = clip(burst_density / burst_density_98)
    score = floor_score + (200.0 - floor_score) * (normalized**alpha)
    return {
        "score": clip(score, 0.0, 200.0),
        "density": burst_density,
        "density_98": burst_density_98,
        "normalized_density": normalized,
        "floor_score": floor_score,
        "alpha": alpha,
    }


def cubic_hermite(
    t: float,
    y0: float,
    y1: float,
    slope0: float,
    slope1: float,
) -> float:
    return (
        (2 * t**3 - 3 * t**2 + 1) * y0
        + (t**3 - 2 * t**2 + t) * slope0
        + (-2 * t**3 + 3 * t**2) * y1
        + (t**3 - t**2) * slope1
    )


def tricky_final_curve(
    base_score: float,
    *,
    midpoint_score: float = 84.0,
    left_slope: float = 1.35,
    middle_slope: float = 0.20,
    right_slope: float = 1.75,
) -> float:
    """Remap tricky score so the curve is steep-flat-steep.

    The first Hill score keeps the timing meaning. This final curve compresses
    medium tricky charts and preserves a sharper high-end rise.
    """

    x = clip(base_score / 200.0)
    midpoint = clip(midpoint_score / 200.0)
    if x <= 0.5:
        return 200 * clip(
            cubic_hermite(
                x / 0.5,
                0.0,
                midpoint,
                left_slope * 0.5,
                middle_slope * 0.5,
            )
        )
    return 200 * clip(
        cubic_hermite(
            (x - 0.5) / 0.5,
            midpoint,
            1.0,
            middle_slope * 0.5,
            right_slope * 0.5,
        )
    )


def boundary_change_scores(densities: list[dict[str, Any]], window: int) -> dict[int, float]:
    values = [float(item["density"]) for item in densities]
    n = len(values)
    global_mean = mean(values)
    global_stdev = population_stdev(values)
    low_threshold = global_mean - 0.35 * global_stdev
    long_window = max(window * 3, 10)
    scores: dict[int, float] = {}
    for boundary in range(1, n):
        left = values[max(0, boundary - window) : boundary]
        right = values[boundary : min(n, boundary + window)]
        before = values[max(0, boundary - 2 * window) : max(0, boundary - window)]
        after = values[min(n, boundary + window) : min(n, boundary + 2 * window)]
        long_left = values[max(0, boundary - long_window) : boundary]
        long_right = values[boundary : min(n, boundary + long_window)]
        mean_shift = abs(mean(right) - mean(left))
        slope_shift = abs((mean(right) - mean(left)) - (mean(after) - mean(before)))
        broad_shift = abs(mean(long_right) - mean(long_left))

        low_plateau_bonus = 0.0
        if long_left and right:
            long_left_mean = mean(long_left)
            right_mean = mean(right)
            if long_left_mean <= low_threshold and right_mean > long_left_mean:
                low_depth = max(0.0, global_mean - long_left_mean)
                low_plateau_bonus = 0.9 * (right_mean - long_left_mean) + 0.35 * low_depth

        climb_bonus = 0.0
        if before and left and right and after:
            previous_slope = mean(left) - mean(before)
            next_slope = mean(after) - mean(right)
            current_slope = mean(right) - mean(left)
            if current_slope > 0 and previous_slope >= -0.25 and next_slope >= -0.25:
                climb_bonus = 0.55 * current_slope + 0.25 * max(0.0, broad_shift)

        scores[boundary] = (
            mean_shift
            + 0.35 * slope_shift
            + 0.45 * broad_shift
            + low_plateau_bonus
            + climb_bonus
        )
    return scores


def effective_min_segment_measures(n: int, config: SegmentationConfig) -> int:
    if config.min_segment_measures is not None:
        return max(1, config.min_segment_measures)
    return max(1, min(n // 10, 10))


def candidate_boundaries(n: int, config: SegmentationConfig) -> list[int]:
    min_segment_measures = effective_min_segment_measures(n, config)
    low = max(config.edge_exclusion + 1, min_segment_measures)
    high = min(n - config.edge_exclusion - 1, n - min_segment_measures)
    if high < low:
        low = min_segment_measures
        high = n - min_segment_measures
    return list(range(low, high + 1)) if high >= low else []


def segment_lengths(n: int, boundaries: list[int]) -> list[int]:
    points = [0, *boundaries, n]
    return [points[index + 1] - points[index] for index in range(len(points) - 1)]


def balance_penalty(n: int, boundaries: list[int], segment_count: int) -> float:
    target = n / segment_count
    if target <= 0:
        return 0.0
    return sum(abs(length - target) / target for length in segment_lengths(n, boundaries)) / segment_count


def boundaries_are_valid(n: int, boundaries: list[int], config: SegmentationConfig) -> bool:
    min_segment_measures = effective_min_segment_measures(n, config)
    lengths = segment_lengths(n, boundaries)
    if any(length < min_segment_measures for length in lengths):
        return False
    if any(boundary <= config.edge_exclusion for boundary in boundaries):
        return False
    if any(boundary >= n - config.edge_exclusion for boundary in boundaries):
        return False
    return True


def max_segment_length(n: int, segment_count: int, config: SegmentationConfig) -> int:
    min_segment_measures = effective_min_segment_measures(n, config)
    return max(
        min_segment_measures,
        math.ceil((n / segment_count) * config.max_segment_length_factor),
    )


def aggregate_note_counts(densities: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for density in densities:
        note_counts = density.get("note_counts", {})
        if not isinstance(note_counts, dict):
            continue
        for key, value in note_counts.items():
            if key in {"note_mix", "special_note_ratios", "button_distribution", "notes"}:
                continue
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    totals["total"] = (
        totals.get("tap", 0)
        + totals.get("hold", 0)
        + totals.get("slide", 0)
        + totals.get("touch", 0)
        + totals.get("touch_hold", 0)
    )
    totals["note_mix"] = note_mix_from_counts(totals)
    totals["special_note_ratios"] = special_ratios_from_counts(totals)
    totals["button_distribution"] = button_distribution_from_counts(totals)
    return totals


def burst_note_count(density: dict[str, Any]) -> int:
    note_counts = density.get("note_counts", {})
    if not isinstance(note_counts, dict):
        return 0
    return int(note_counts.get("tap", 0)) + int(note_counts.get("hold", 0))


def four_measure_burst_summary(
    densities: list[dict[str, Any]],
    *,
    window_size: int = 4,
) -> dict[str, Any]:
    if not densities:
        return {
            "density": 0.0,
            "note_count": 0,
            "duration_seconds": 0.0,
            "window_size": window_size,
            "start_measure": None,
            "end_measure": None,
        }

    actual_window = min(window_size, len(densities))
    best: dict[str, Any] | None = None
    for start in range(0, len(densities) - actual_window + 1):
        window = densities[start : start + actual_window]
        note_count = sum(burst_note_count(item) for item in window)
        duration_seconds = sum(float(item.get("duration_seconds") or 0.0) for item in window)
        density = note_count / duration_seconds if duration_seconds > 0 else 0.0
        candidate = {
            "density": density,
            "note_count": note_count,
            "duration_seconds": duration_seconds,
            "window_size": actual_window,
            "start_measure": window[0].get("index"),
            "end_measure": window[-1].get("index"),
        }
        if best is None or density > float(best["density"]):
            best = candidate
    return best if best is not None else {
        "density": 0.0,
        "note_count": 0,
        "duration_seconds": 0.0,
        "window_size": actual_window,
        "start_measure": None,
        "end_measure": None,
    }


def aggregate_moments(
    densities: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    moments: list[dict[str, Any]] = []
    beat_cursor = 0.0
    seconds_cursor = 0.0
    for density in densities:
        raw_moments = density.get(key, [])
        if isinstance(raw_moments, list):
            for moment in raw_moments:
                if not isinstance(moment, dict):
                    continue
                copied = dict(moment)
                copied["beat"] = beat_cursor + float(copied.get("local_beat", copied.get("beat", 0.0)))
                copied["time_seconds"] = seconds_cursor + float(
                    copied.get("local_seconds", copied.get("time_seconds", 0.0))
                )
                moments.append(copied)
        beat_cursor += float(density.get("beats") or 0.0)
        seconds_cursor += float(density.get("duration_seconds") or 0.0)
    return moments


def aggregate_tap_moments(densities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_moments(densities, "tap_moments")


def aggregate_handtrip_tap_moments(densities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_moments(densities, "handtrip_tap_moments")


def aggregate_tap_only_moments(densities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_moments(densities, "tap_only_moments")


def aggregate_slide_paths(densities: list[dict[str, Any]]) -> list[list[int]]:
    paths: list[list[int]] = []
    for density in densities:
        slide_movement = density.get("slide_movement", {})
        raw_paths = slide_movement.get("paths", []) if isinstance(slide_movement, dict) else []
        if not isinstance(raw_paths, list):
            continue
        for item in raw_paths:
            if not isinstance(item, dict):
                continue
            lanes = item.get("lanes", [])
            if isinstance(lanes, list):
                paths.append([int(lane) for lane in lanes])
    return paths


def best_boundaries_for_count(
    n: int,
    segment_count: int,
    candidates: list[int],
    change_scores: dict[int, float],
    config: SegmentationConfig,
) -> SegmentationPlan | None:
    needed_boundaries = segment_count - 1
    if needed_boundaries <= 0:
        return None
    min_segment_measures = effective_min_segment_measures(n, config)
    max_length = max_segment_length(n, segment_count, config)

    dp: dict[tuple[int, int], tuple[float, list[int]]] = {(0, 0): (0.0, [])}
    for candidate in candidates:
        next_dp = dict(dp)
        for (chosen, last_boundary), (score, boundaries) in dp.items():
            if chosen >= needed_boundaries:
                continue
            if candidate - last_boundary < min_segment_measures:
                continue
            if candidate - last_boundary > max_length:
                continue
            remaining = needed_boundaries - chosen - 1
            if n - candidate < remaining * min_segment_measures:
                continue
            if remaining == 0 and n - candidate > max_length:
                continue
            key = (chosen + 1, candidate)
            new_boundaries = [*boundaries, candidate]
            new_score = score + change_scores.get(candidate, 0.0)
            if key not in next_dp or new_score > next_dp[key][0]:
                next_dp[key] = (new_score, new_boundaries)
        dp = next_dp

    best: SegmentationPlan | None = None
    for (chosen, _), (raw_score, boundaries) in dp.items():
        if chosen != needed_boundaries:
            continue
        if not boundaries_are_valid(n, boundaries, config):
            continue
        boundary_average = raw_score / needed_boundaries
        balance = balance_penalty(n, boundaries, segment_count)
        score = (
            raw_score
            - config.balance_weight * balance
            - config.extra_segment_penalty * (segment_count - config.min_segments)
        )
        plan = SegmentationPlan(
            segment_count=segment_count,
            boundaries=boundaries,
            score=score,
            boundary_score_average=boundary_average,
            balance_penalty=balance,
        )
        if best is None or plan.score > best.score:
            best = plan
    return best


def choose_segmentation(
    densities: list[dict[str, Any]],
    config: SegmentationConfig,
) -> SegmentationPlan:
    n = len(densities)
    if n == 0:
        return SegmentationPlan(0, [], 0.0, 0.0, 0.0)

    min_segment_measures = effective_min_segment_measures(n, config)
    max_possible = max(1, n // max(1, min_segment_measures))
    min_segments = config.fixed_segments or config.min_segments
    max_segments = config.fixed_segments or config.max_segments
    min_segments = max(1, min(min_segments, max_possible))
    max_segments = max(min_segments, min(max_segments, max_possible))

    candidates = candidate_boundaries(n, config)
    scores = boundary_change_scores(densities, config.window)
    plans = [
        plan
        for segment_count in range(min_segments, max_segments + 1)
        if (
            plan := best_boundaries_for_count(
                n,
                segment_count,
                candidates,
                scores,
                config,
            )
        )
        is not None
    ]
    if plans:
        return max(plans, key=lambda plan: plan.score)

    fallback_count = min_segments
    step = n / fallback_count
    boundaries = [round(step * index) for index in range(1, fallback_count)]
    boundaries = sorted(set(max(1, min(n - 1, boundary)) for boundary in boundaries))
    return SegmentationPlan(
        segment_count=len(boundaries) + 1,
        boundaries=boundaries,
        score=0.0,
        boundary_score_average=0.0,
        balance_penalty=balance_penalty(n, boundaries, len(boundaries) + 1),
    )


def summarize_segment(
    densities: list[dict[str, Any]],
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    selected = densities[start_index:end_index]
    weighted_sum = sum(float(item["weighted_sum"]) for item in selected)
    duration_seconds = sum(float(item["duration_seconds"]) for item in selected)
    profile_items = [
        item
        for item in selected
        if float(item.get("duration_seconds") or 0.0) >= MIN_DENSITY_PROFILE_MEASURE_SECONDS
    ]
    density_values = [float(item["density"]) for item in profile_items]
    note_density_values = [
        float(item.get("non_touch_density", item["density"]))
        for item in profile_items
    ]
    note_density_mean = mean(note_density_values)
    note_density_stdev = population_stdev(note_density_values)
    density_mean = mean(density_values)
    density_stdev = population_stdev(density_values)
    note_counts = aggregate_note_counts(selected)
    total_notes = int(note_counts.get("total", 0))
    slide_count = int(note_counts.get("slide", 0))
    charge_count = int(note_counts.get("hold", 0)) + int(note_counts.get("touch_hold", 0))
    slide_ratio = slide_count / total_notes if total_notes else 0.0
    charge_ratio = charge_count / total_notes if total_notes else 0.0
    slide_density = slide_count / duration_seconds if duration_seconds > 0 else 0.0
    charge_density = charge_count / duration_seconds if duration_seconds > 0 else 0.0
    tap_moments = aggregate_tap_moments(selected)
    handtrip_tap_moments = aggregate_handtrip_tap_moments(selected)
    tap_only_moments = aggregate_tap_only_moments(selected)
    slide_paths = aggregate_slide_paths(selected)
    tap_distance = tap_distance_summary(tap_moments, duration_seconds)
    handtrip_tap_distance = tap_distance_summary(
        handtrip_tap_moments,
        duration_seconds,
        use_high_bpm_twelfth_boundary=True,
    )
    slide_movement = slide_movement_summary(slide_paths, duration_seconds)
    four_measure_burst = four_measure_burst_summary(selected)
    return {
        "start_measure": selected[0]["index"],
        "end_measure": selected[-1]["index"],
        "measure_count": len(selected),
        "start_seconds": selected[0]["start_seconds"],
        "end_seconds": selected[-1]["end_seconds"],
        "duration_seconds": duration_seconds,
        "weighted_sum": weighted_sum,
        "density": weighted_sum / duration_seconds if duration_seconds > 0 else 0.0,
        "mean_measure_density": density_mean,
        "max_measure_density": max(density_values) if density_values else 0.0,
        "min_measure_density": min(density_values) if density_values else 0.0,
        "density_curve": [
            {
                "measure": item.get("index"),
                "density": float(item.get("density") or 0.0),
                "non_touch_density": float(item.get("non_touch_density", item.get("density", 0.0)) or 0.0),
                "weighted_sum": float(item.get("weighted_sum") or 0.0),
                "duration_seconds": float(item.get("duration_seconds") or 0.0),
            }
            for item in selected
        ],
        "density_profile": {
            "mean": note_density_mean,
            "q90": percentile(note_density_values, 0.90),
            "stdev": note_density_stdev,
            "coefficient_of_variation": (
                note_density_stdev / note_density_mean if note_density_mean > 0 else 0.0
            ),
            "includes_touch": False,
            "window_count": len(density_values),
            "filtered_short_measure_count": len(selected) - len(profile_items),
            "min_measure_duration_seconds": MIN_DENSITY_PROFILE_MEASURE_SECONDS,
        },
        "four_measure_burst": four_measure_burst,
        "note_counts": note_counts,
        "note_mix": note_counts["note_mix"],
        "special_note_ratios": note_counts["special_note_ratios"],
        "button_distribution": note_counts["button_distribution"],
        "slide_charge_score": {
            "slide": saturated_ratio_density_score(
                ratio_value=slide_ratio,
                density_value=slide_density,
                ratio_98=0.20,
                density_98=1.5,
            ),
            "charge": saturated_ratio_density_score(
                ratio_value=charge_ratio,
                density_value=charge_density,
                ratio_98=0.20,
                density_98=1.5,
            ),
        },
        "tap_distance": tap_distance,
        "handtrip_tap_distance": handtrip_tap_distance,
        "slide_movement": slide_movement,
        "handtrip_movement": combined_handtrip_summary(
            handtrip_tap_distance,
            slide_movement,
            duration_seconds,
        ),
        "same_button_triple_tap": same_button_triple_tap_summary(tap_only_moments),
    }


def metric_ratio(segment: dict[str, Any], family: str, key: str) -> float:
    data = segment.get(family, {})
    if not isinstance(data, dict):
        return 0.0
    item = data.get(key, {})
    if not isinstance(item, dict):
        return 0.0
    return float(item.get("ratio") or 0.0)


def segment_density_curve_points(segment: dict[str, Any]) -> list[dict[str, float]]:
    raw_curve = segment.get("density_curve", [])
    points: list[dict[str, float]] = []
    if isinstance(raw_curve, list):
        for item in raw_curve:
            if isinstance(item, dict):
                density = float(item.get("density") or 0.0)
                duration = float(item.get("duration_seconds") or 0.0)
                weighted_sum = item.get("weighted_sum")
                points.append(
                    {
                        "density": density,
                        "duration_seconds": duration if duration > 0 else 1.0,
                        "weighted_sum": float(weighted_sum) if weighted_sum is not None else math.nan,
                    }
                )
            else:
                points.append(
                    {
                        "density": float(item or 0.0),
                        "duration_seconds": 1.0,
                        "weighted_sum": math.nan,
                    }
                )
    if points:
        return points
    duration = float(segment.get("duration_seconds") or 1.0)
    return [
        {
            "density": float(segment.get("density") or 0.0),
            "duration_seconds": duration if duration > 0 else 1.0,
            "weighted_sum": float(segment["weighted_sum"]) if "weighted_sum" in segment else math.nan,
        }
    ]


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator <= 0:
        return 0.0
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def smoothed_density_values(values: list[float], window_size: int = 3) -> list[float]:
    if len(values) < 3 or window_size <= 1:
        return values
    radius = window_size // 2
    smoothed = []
    for index in range(len(values)):
        window = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        smoothed.append(mean(window))
    return smoothed


def density_state_thresholds(segments: list[dict[str, Any]]) -> dict[str, float]:
    values = [
        point["density"]
        for segment in segments
        for point in segment_density_curve_points(segment)
    ]
    if not values:
        values = [0.0]
    positive_values = [value for value in values if value > 0]
    density_basis = positive_values or values
    q25 = percentile(values, 0.25)
    q75 = percentile(values, 0.75)
    iqr = q75 - q25
    global_stdev = population_stdev(values)
    low_density_max = percentile(density_basis, 0.33)
    high_density_min = percentile(density_basis, 0.67)
    rest_density_max = max(
        0.25,
        min(low_density_max * 0.45, percentile(values, 0.20)),
    )
    if high_density_min < low_density_max:
        high_density_min = low_density_max
    return {
        "rest_density_max": rest_density_max,
        "low_density_max": max(rest_density_max, low_density_max),
        "high_density_min": max(rest_density_max, high_density_min),
        "delta_epsilon": max(0.10, iqr * 0.08),
        "trend_min_change": max(0.50, iqr * 0.35, global_stdev * 0.45),
        "volatility_min_stdev": max(0.50, iqr * 0.40, global_stdev * 0.70),
        "volatility_min_cv": 0.35,
        "volatility_min_range": max(1.00, iqr * 0.90, global_stdev * 1.20),
        "burst_min_lift": max(0.75, iqr * 0.50, global_stdev * 0.60),
        "burst_min_ratio": 1.35,
    }


def density_state_stats(
    segment: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    points = segment_density_curve_points(segment)
    raw_values = [point["density"] for point in points]
    values = smoothed_density_values(raw_values)
    durations = [point["duration_seconds"] for point in points]
    total_duration = sum(durations)
    weighted_mean = (
        sum(value * duration for value, duration in zip(raw_values, durations)) / total_duration
        if total_duration > 0
        else mean(raw_values)
    )
    deltas = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    epsilon = thresholds["delta_epsilon"]
    signs = [1 if delta > epsilon else -1 if delta < -epsilon else 0 for delta in deltas]
    non_flat_signs = [sign for sign in signs if sign != 0]
    sign_changes = sum(
        1
        for left, right in zip(non_flat_signs, non_flat_signs[1:])
        if left != right
    )
    transition_count = len(deltas)
    up_count = sum(1 for sign in signs if sign > 0)
    down_count = sum(1 for sign in signs if sign < 0)
    low_or_empty_duration = 0.0
    for point in points:
        weighted_sum = point["weighted_sum"]
        is_empty = math.isfinite(weighted_sum) and weighted_sum <= 0
        if is_empty or point["density"] <= thresholds["rest_density_max"]:
            low_or_empty_duration += point["duration_seconds"]
    rest_ratio = low_or_empty_duration / total_duration if total_duration > 0 else 0.0
    raw_stdev = population_stdev(raw_values)
    return {
        "raw_values": raw_values,
        "values": values,
        "mean": weighted_mean,
        "stdev": population_stdev(values),
        "cv": population_stdev(values) / weighted_mean if weighted_mean > 0 else 0.0,
        "raw_stdev": raw_stdev,
        "raw_cv": raw_stdev / weighted_mean if weighted_mean > 0 else 0.0,
        "raw_range": (max(raw_values) - min(raw_values)) if raw_values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "range": (max(values) - min(values)) if values else 0.0,
        "slope": linear_slope(values),
        "net_change": (values[-1] - values[0]) if len(values) >= 2 else 0.0,
        "transition_count": transition_count,
        "up_ratio": up_count / transition_count if transition_count else 0.0,
        "down_ratio": down_count / transition_count if transition_count else 0.0,
        "sign_change_ratio": sign_changes / max(1, len(non_flat_signs) - 1),
        "rest_ratio": rest_ratio,
    }


def density_state_burst_event(
    segment: dict[str, Any],
    stats: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    burst = segment.get("four_measure_burst", {})
    if not isinstance(burst, dict) or not burst:
        return {"active": False}
    burst_density = float(burst.get("density") or 0.0)
    mean_density = float(stats["mean"])
    measure_count = int(segment.get("measure_count") or len(stats["values"]))
    window_size = int(burst.get("window_size") or 0)
    lift = burst_density - mean_density
    ratio = burst_density / mean_density if mean_density > 0 else 0.0
    active = (
        measure_count > max(window_size, 1)
        and burst_density >= thresholds["high_density_min"]
        and lift >= thresholds["burst_min_lift"]
        and ratio >= thresholds["burst_min_ratio"]
    )
    return {
        "active": active,
        "density": burst_density,
        "mean_density": mean_density,
        "lift": lift,
        "ratio": ratio,
        "window_size": window_size,
        "start_measure": burst.get("start_measure"),
        "end_measure": burst.get("end_measure"),
        "thresholds": {
            "min_lift": thresholds["burst_min_lift"],
            "min_ratio": thresholds["burst_min_ratio"],
            "min_density": thresholds["high_density_min"],
        },
    }


def classify_density_state(
    stats: dict[str, Any],
    thresholds: dict[str, float],
) -> str:
    mean_density = float(stats["mean"])
    if mean_density <= thresholds["rest_density_max"] or stats["rest_ratio"] >= 0.65:
        return "REST"
    if (
        stats["net_change"] >= thresholds["trend_min_change"]
        and stats["up_ratio"] >= 0.55
        and stats["down_ratio"] <= 0.30
    ):
        return "RISING"
    if (
        stats["net_change"] >= thresholds["trend_min_change"]
        and stats["up_ratio"] >= stats["down_ratio"] + 0.15
        and stats["sign_change_ratio"] <= 0.55
    ):
        return "RISING"
    if (
        -stats["net_change"] >= thresholds["trend_min_change"]
        and stats["down_ratio"] >= 0.55
        and stats["up_ratio"] <= 0.30
    ):
        return "FALLING"
    if (
        -stats["net_change"] >= thresholds["trend_min_change"]
        and stats["down_ratio"] >= stats["up_ratio"] + 0.10
        and stats["sign_change_ratio"] <= 0.55
    ):
        return "FALLING"
    if (
        stats["stdev"] >= thresholds["volatility_min_stdev"]
        and stats["cv"] >= thresholds["volatility_min_cv"]
        and (
            stats["sign_change_ratio"] >= 0.35
            or stats["range"] >= thresholds["volatility_min_range"]
        )
    ):
        return "VOLATILE"
    if (
        stats["range"] >= thresholds["volatility_min_range"]
        and stats["cv"] >= thresholds["volatility_min_cv"]
        and stats["sign_change_ratio"] >= 0.35
    ):
        return "VOLATILE"
    if (
        stats["raw_cv"] >= 0.45
        and stats["raw_range"] >= thresholds["volatility_min_range"]
        and stats["sign_change_ratio"] >= 0.50
    ):
        return "VOLATILE"
    return "STEADY"

def density_profile(segment: dict[str, Any]) -> dict[str, float]:
    profile = segment.get("density_profile", {})
    if not isinstance(profile, dict) or not profile:
        return {
            "mean": float(segment.get("mean_measure_density", segment.get("density", 0.0)) or 0.0),
            "q90": float(segment.get("max_measure_density", segment.get("density", 0.0)) or 0.0),
            "coefficient_of_variation": 0.0,
        }
    return {
        "mean": float(profile.get("mean") or 0.0),
        "q90": float(profile.get("q90") or 0.0),
        "coefficient_of_variation": float(profile.get("coefficient_of_variation") or 0.0),
    }


def compute_peak_note_scores(segments: list[dict[str, Any]]) -> None:
    profiles = [density_profile(segment) for segment in segments]
    means = [profile["mean"] for profile in profiles]
    q90s = [profile["q90"] for profile in profiles]
    cvs = [profile["coefficient_of_variation"] for profile in profiles]
    burst_values = [
        (
            float(segment["four_measure_burst"].get("density") or 0.0)
            if isinstance(segment.get("four_measure_burst"), dict)
            else None
        )
        for segment in segments
    ]
    nonzero_bursts = [value for value in burst_values if value is not None and value > 0]
    burst_density_98 = percentile(nonzero_bursts, 0.98) if nonzero_bursts else 1.0
    alpha_note = 1.65
    lambda_note = 0.35
    note_denominator = 1 + lambda_note / 2

    for segment, profile, burst_density in zip(segments, profiles, burst_values):
        density_peak = normalize_to_q95(profile["q90"], q90s)
        density_note = normalize_to_q95(profile["mean"], means)
        volatility = normalize_percentile(profile["coefficient_of_variation"], cvs)
        centered_volatility = volatility - 0.5
        note_raw = (density_note**alpha_note) * (1 + lambda_note * centered_volatility)
        if burst_density is None:
            burst_normalized = None
            burst_score = None
            peak = 200 * density_peak
            peak_raw = density_peak
        else:
            burst_peak = burst_peak_score(burst_density, burst_density_98)
            burst_normalized = burst_peak["normalized_density"]
            burst_score = burst_peak["score"]
            peak = burst_score
            peak_raw = burst_normalized
        note = clip(note_raw / note_denominator)
        segment["peak_note_score"] = {
            "peak": peak,
            "burst": burst_score,
            "note": note * 200,
            "peak_raw": peak_raw,
            "note_raw": note_raw,
            "normalized": {
                "density_peak_q90": density_peak,
                "density_note_mean": density_note,
                "volatility": volatility,
                "burst_density": burst_normalized,
            },
            "raw_inputs": {
                "density_peak_q90": profile["q90"],
                "density_note_mean": profile["mean"],
                "density_cv": profile["coefficient_of_variation"],
                "burst_density": burst_density,
            },
            "parameters": {
                "alpha_note": alpha_note,
                "lambda_note": lambda_note,
                "burst_density_98": burst_density_98,
                "peak_floor_score": 10.0,
                "peak_alpha": 1.35,
                "normalization": "current_segments_zero_to_q95",
                "mapping": "burst_p98_steeper_floor_to_200",
            },
        }


def compute_handtrip_scores(segments: list[dict[str, Any]]) -> None:
    alpha = 1.7
    values = []
    for segment in segments:
        handtrip_movement = segment.get("handtrip_movement")
        if not isinstance(handtrip_movement, dict) or not handtrip_movement:
            handtrip_movement = segment.get("tap_distance", {})
        values.append(
            float(handtrip_movement.get("distance_per_second") or 0.0)
            if isinstance(handtrip_movement, dict)
            else 0.0
        )
    density_98 = percentile(values, 0.98) if values else 1.0
    if density_98 <= 0:
        density_98 = 1.0
    for segment, value in zip(segments, values):
        normalized_density = clip(value / density_98) if value > 0 and density_98 > 0 else 0.0
        segment["handtrip_score"] = {
            "score": 200 * clip(normalized_density**alpha),
            "density": value,
            "normalized_density": normalized_density,
            "density_98": density_98,
            "alpha": alpha,
            "normalization": "current_segments_p98_power_to_200",
        }


def compute_tricky_scores(
    segments: list[dict[str, Any]],
    *,
    tau_seconds: float = 0.60,
    gamma: float = 2.2,
) -> None:
    shortest_values: list[float | None] = []
    intensities: list[float] = []
    for segment in segments:
        triple_tap = segment.get("same_button_triple_tap", {})
        shortest = None
        if isinstance(triple_tap, dict) and triple_tap.get("shortest_time_seconds") is not None:
            shortest = float(triple_tap["shortest_time_seconds"])
        shortest_values.append(shortest)
        if shortest is not None and shortest > 0:
            intensities.append(1 / shortest)

    intensity_98 = percentile(intensities, 0.98) if intensities else 1.0
    if intensity_98 <= 0:
        intensity_98 = 1.0

    for segment, shortest in zip(segments, shortest_values):
        if shortest is None:
            normalized = 0.001
            base_score = 0.2
        elif shortest <= 0:
            normalized = 1.0
            base_score = 200.0
        else:
            normalized = clip((1 / shortest) / intensity_98)
            base_score = 200 * clip(normalized)
        score = tricky_final_curve(base_score)
        segment["tricky_score"] = {
            "score": score,
            "base_score": base_score,
            "shortest_time_seconds": shortest,
            "tau_seconds": tau_seconds,
            "gamma": gamma,
            "intensity_98": intensity_98,
            "normalized": normalized,
            "final_curve": "steep_flat_steep_hermite",
            "normalization": "current_segments_inverse_time_p98_to_200",
        }


def ensure_slide_charge_scores(segments: list[dict[str, Any]]) -> None:
    metrics: list[dict[str, float]] = []
    for segment in segments:
        note_mix = segment.get("note_mix", {})
        slide_ratio = metric_ratio(segment, "note_mix", "slide")
        charge_ratio = metric_ratio(segment, "note_mix", "hold")
        duration_seconds = float(segment.get("duration_seconds") or 0.0)
        slide_count = 0.0
        charge_count = 0.0
        if isinstance(note_mix, dict):
            slide_item = note_mix.get("slide", {})
            hold_item = note_mix.get("hold", {})
            if isinstance(slide_item, dict):
                slide_count = float(slide_item.get("count") or 0.0)
            if isinstance(hold_item, dict):
                charge_count = float(hold_item.get("count") or 0.0)
        slide_density = slide_count / duration_seconds if duration_seconds > 0 else 0.0
        charge_density = charge_count / duration_seconds if duration_seconds > 0 else 0.0
        metrics.append(
            {
                "slide_ratio": slide_ratio,
                "slide_density": slide_density,
                "charge_ratio": charge_ratio,
                "charge_density": charge_density,
            }
        )

    slide_ratio_98 = percentile([item["slide_ratio"] for item in metrics], 0.98) if metrics else 0.20
    slide_density_98 = percentile([item["slide_density"] for item in metrics], 0.98) if metrics else 1.5
    charge_ratio_98 = percentile([item["charge_ratio"] for item in metrics], 0.98) if metrics else 0.20
    charge_density_98 = percentile([item["charge_density"] for item in metrics], 0.98) if metrics else 1.5
    slide_ratio_98 = slide_ratio_98 if slide_ratio_98 > 0 else 0.20
    slide_density_98 = slide_density_98 if slide_density_98 > 0 else 1.5
    charge_ratio_98 = charge_ratio_98 if charge_ratio_98 > 0 else 0.20
    charge_density_98 = charge_density_98 if charge_density_98 > 0 else 1.5

    for segment, item in zip(segments, metrics):
        segment["slide_charge_score"] = {
            "slide": saturated_ratio_density_score(
                ratio_value=item["slide_ratio"],
                density_value=item["slide_density"],
                ratio_98=slide_ratio_98,
                density_98=slide_density_98,
            ),
            "charge": saturated_ratio_density_score(
                ratio_value=item["charge_ratio"],
                density_value=item["charge_density"],
                ratio_98=charge_ratio_98,
                density_98=charge_density_98,
            ),
        }

def classify_steady_density_tier(
    *,
    density: float,
    standard: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if standard is None:
        return None, None
    if standard.get("low") == "all":
        return "LOW", {
            "rule": "forced_low",
            "standard": standard,
        }
    if standard.get("high") == "all":
        return "HIGH", {
            "rule": "forced_high",
            "standard": standard,
        }

    p33 = standard.get("steady_density_p33")
    p67 = standard.get("steady_density_p67")
    if p33 is None or p67 is None:
        return None, {
            "rule": "missing_threshold",
            "standard": standard,
        }
    p33_value = float(p33)
    p67_value = float(p67)
    if density <= p33_value:
        tier = "LOW"
    elif density >= p67_value:
        tier = "HIGH"
    else:
        tier = "MID"
    return tier, {
        "rule": "level_p33_p67",
        "density": density,
        "p33": p33_value,
        "p67": p67_value,
        "standard": standard,
    }


def annotate_segment_labels(
    segments: list[dict[str, Any]],
    *,
    steady_standards: dict[str, Any] | None = None,
    chart_level: Any = None,
) -> None:
    if not segments:
        return

    thresholds = density_state_thresholds(segments)
    steady_standard = steady_density_standard_for_level(steady_standards, chart_level)

    for segment in segments:
        stats = density_state_stats(segment, thresholds)
        label = classify_density_state(stats, thresholds)
        base_label = label
        steady_tier = None
        steady_tier_reason = None
        if label == "STEADY":
            steady_tier, steady_tier_reason = classify_steady_density_tier(
                density=float(stats["mean"]),
                standard=steady_standard,
            )
            if steady_tier is not None:
                label = f"{steady_tier}_STEADY"
        burst_event = density_state_burst_event(segment, stats, thresholds)
        event_labels = ["BURST"] if burst_event.get("active") else []
        segment["label"] = label
        segment["base_label"] = base_label
        if base_label == "STEADY":
            segment["steady_tier"] = steady_tier
        segment["event_labels"] = event_labels
        segment["has_burst"] = "BURST" in event_labels
        segment["label_reason"] = {
            "classifier": "density_state_v1",
            "density": stats["mean"],
            "density_thresholds": thresholds,
            "trend": {
                "slope": stats["slope"],
                "net_change": stats["net_change"],
                "up_ratio": stats["up_ratio"],
                "down_ratio": stats["down_ratio"],
            },
            "volatility": {
                "stdev": stats["stdev"],
                "cv": stats["cv"],
                "range": stats["range"],
                "sign_change_ratio": stats["sign_change_ratio"],
                "raw_stdev": stats["raw_stdev"],
                "raw_cv": stats["raw_cv"],
                "raw_range": stats["raw_range"],
            },
            "rest_ratio": stats["rest_ratio"],
            "burst": burst_event,
        }
        if steady_tier_reason is not None:
            segment["label_reason"]["steady_density_tier"] = steady_tier_reason


def chart_report_metadata(
    chart_path: Path,
    chart_data: dict[str, Any],
    densities: list[dict[str, Any]],
) -> dict[str, Any]:
    song = chart_data.get("song", {})
    if not isinstance(song, dict):
        song = {}
    chart = chart_data.get("chart", {})
    if not isinstance(chart, dict):
        chart = {}
    timeline = chart_data.get("timeline", {})
    if not isinstance(timeline, dict):
        timeline = {}
    return {
        "source_file": str(chart_path),
        "song_id": song.get("song_id", chart_data.get("song_id")),
        "title": song.get("title", chart_data.get("title")),
        "artist": song.get("artist"),
        "bpm": song.get("bpm"),
        "genre": song.get("genre"),
        "song_version": song.get("version"),
        "cabinet": song.get("cabinet"),
        "chart_kind": chart.get("chart_kind", chart_data.get("chart_kind")),
        "chart_version": chart.get("chart_version"),
        "difficulty_index": chart.get("difficulty_index", chart_data.get("difficulty_index")),
        "difficulty_name": chart.get("difficulty_name", chart_data.get("difficulty_name")),
        "level": chart.get("level", chart_data.get("level")),
        "charter": chart.get("charter"),
        "created_from_file": chart.get("created_from_file"),
        "measure_count": len(densities),
        "duration_seconds": timeline.get("duration_seconds"),
        "start_seconds": timeline.get("start_seconds"),
        "end_seconds": timeline.get("end_seconds"),
    }


def attach_six_dimension_scores(
    segments: list[dict[str, Any]],
    baselines: dict[str, Any],
) -> list[str]:
    from scripts.simai_segment_scorer import SCORE_NAMES, score_segment_summary

    dimensions = list(SCORE_NAMES)
    for segment in segments:
        scored = score_segment_summary(segment, baselines)
        scores = {
            name: round(float(scored["scores"][name]), 6)
            for name in dimensions
        }
        segment["score_vector"] = [scores[name] for name in dimensions]
        segment["scores"] = scores
        segment["dominant_dimension"] = scored["dominant_dimension"]
        segment["score_raw"] = scored["raw"]
        segment["score_normalized"] = scored["normalized"]
    return dimensions


def build_segmentation_report(
    chart_path: Path,
    config: SegmentationConfig,
    *,
    include_measures: bool = False,
    baselines_json: Path | None = DEFAULT_SEGMENT_SCORE_BASELINES,
    baselines: dict[str, Any] | None = None,
    include_score_baselines: bool = True,
    steady_standards_json: Path | None = None,
    steady_standards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chart_data = read_json(chart_path)
    densities = chart_measure_densities(chart_data)
    plan = choose_segmentation(densities, config)
    points = [0, *plan.boundaries, len(densities)]
    segments = [
        summarize_segment(densities, points[index], points[index + 1])
        for index in range(len(points) - 1)
        if points[index] < points[index + 1]
    ]
    chart = chart_data.get("chart", {})
    chart_level = chart.get("level") if isinstance(chart, dict) else None
    if steady_standards is None:
        steady_standards = (
            load_steady_density_standards(steady_standards_json)
            if steady_standards_json is not None
            else None
        )
    steady_standard = steady_density_standard_for_level(steady_standards, chart_level)
    annotate_segment_labels(
        segments,
        steady_standards=steady_standards,
        chart_level=chart_level,
    )
    dimensions: list[str] = []
    if baselines is None and baselines_json is not None:
        from scripts.simai_segment_scorer import load_baselines

        baselines = load_baselines(baselines_json)
    if baselines is not None:
        dimensions = attach_six_dimension_scores(segments, baselines)
    scores = boundary_change_scores(densities, config.window)
    report = {
        "metadata": chart_report_metadata(chart_path, chart_data, densities),
        "source_file": str(chart_path),
        "song": chart_data.get("song"),
        "chart": chart_data.get("chart"),
        "score_dimensions": dimensions,
        "score_baselines": baselines if include_score_baselines else None,
        "config": {
            "min_segments": config.min_segments,
            "max_segments": config.max_segments,
            "fixed_segments": config.fixed_segments,
            "edge_exclusion": config.edge_exclusion,
            "min_segment_measures": config.min_segment_measures,
            "effective_min_segment_measures": effective_min_segment_measures(
                len(densities),
                config,
            ),
            "max_segment_length_factor": config.max_segment_length_factor,
            "window": config.window,
            "steady_standards_json": str(steady_standards_json) if steady_standards_json else None,
            "steady_density_standard": steady_standard,
        },
        "plan": {
            "segment_count": plan.segment_count,
            "boundaries_after_measure": plan.boundaries,
            "score": plan.score,
            "boundary_score_average": plan.boundary_score_average,
            "balance_penalty": plan.balance_penalty,
            "boundary_scores": [
                {"after_measure": boundary, "score": scores.get(boundary, 0.0)}
                for boundary in plan.boundaries
            ],
        },
        "segments": segments,
    }
    if include_measures:
        report["measures"] = densities
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split an exported Simai chart JSON into density-aware sections."
    )
    parser.add_argument("--chart-file", type=Path, help="Path to one exported chart JSON.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Directory containing index.json. Default: outputs/simai_measures",
    )
    parser.add_argument("--song-id", type=int, help="Song id to look up from index.json.")
    parser.add_argument("--title", help="Case-insensitive title substring to look up.")
    parser.add_argument("--chart-kind", choices=["ST", "DX"], help="Chart kind filter.")
    parser.add_argument("--difficulty-index", type=int, help="Difficulty index filter.")
    parser.add_argument(
        "--segments",
        type=int,
        default=5,
        help="Force an exact segment count. Default: 5.",
    )
    parser.add_argument("--min-segments", type=int, default=5, help="Default: 5.")
    parser.add_argument("--max-segments", type=int, default=5, help="Default: 5.")
    parser.add_argument(
        "--edge-exclusion",
        type=int,
        default=8,
        help="Do not place boundaries in the first/last N measures. Default: 8.",
    )
    parser.add_argument(
        "--min-segment-measures",
        type=int,
        default=None,
        help=(
            "Override minimum measures per segment. Default: floor(total measures / 10), "
            "capped at 10."
        ),
    )
    parser.add_argument(
        "--max-segment-length-factor",
        type=float,
        default=1.75,
        help="Hard cap: max segment length as target length multiplier. Default: 1.75.",
    )
    parser.add_argument("--window", type=int, default=4, help="Change-point window. Default: 4.")
    parser.add_argument("--include-measures", action="store_true", help="Include all measure densities.")
    parser.add_argument(
        "--baselines-json",
        type=Path,
        default=DEFAULT_SEGMENT_SCORE_BASELINES,
        help=(
            "Global six-dimension baseline JSON used by simai_segment_scorer. "
            f"Default: {DEFAULT_SEGMENT_SCORE_BASELINES}"
        ),
    )
    parser.add_argument(
        "--omit-score-baselines",
        action="store_true",
        help="Do not embed the full six-dimension baseline payload in the output JSON.",
    )
    parser.add_argument(
        "--steady-standards-json",
        type=Path,
        default=None,
        help=(
            "Optional steady density p33/p67 standards JSON for LOW/MID/HIGH steady labels. "
            f"Typical value: {DEFAULT_STEADY_DENSITY_STANDARDS}"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Write one chart/song report JSON to this path.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chart_path = resolve_chart(
        chart_file=args.chart_file,
        index_dir=args.index_dir,
        song_id=args.song_id,
        title=args.title,
        chart_kind=args.chart_kind,
        difficulty_index=args.difficulty_index,
    )
    config = SegmentationConfig(
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        fixed_segments=args.segments,
        edge_exclusion=args.edge_exclusion,
        min_segment_measures=args.min_segment_measures,
        max_segment_length_factor=args.max_segment_length_factor,
        window=args.window,
    )
    report = build_segmentation_report(
        chart_path,
        config,
        include_measures=args.include_measures,
        baselines_json=args.baselines_json,
        include_score_baselines=not args.omit_score_baselines,
        steady_standards_json=args.steady_standards_json,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
