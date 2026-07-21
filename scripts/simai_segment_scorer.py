#!/usr/bin/env python3
"""Score explicit measure segments with the global six-dimensional scale."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_density_segmenter import (
    burst_peak_score,
    clip,
    saturated_ratio_density_score,
    summarize_segment,
)
from scripts.simai_global_six_dimension_table import SCORE_NAMES, score_handtrip, score_tricky
from scripts.simai_measure_density import chart_measure_densities, read_json, resolve_chart


DEFAULT_BASELINES = Path("outputs/six_dimension/global_six_dimension_intermediates.json")


def load_baselines(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    baselines = payload.get("baselines", payload)
    if not isinstance(baselines, dict):
        raise ValueError(f"baseline file does not contain an object: {path}")
    required = {
        "density_note_mean_p95",
        "density_cv_p05",
        "density_cv_p95",
        "burst_density_p98",
        "slide_density_p98",
        "slide_ratio_p98",
        "charge_density_p98",
        "charge_ratio_p98",
        "handtrip_density_p98",
        "tricky_intensity_p98",
    }
    missing = sorted(required - set(baselines))
    if missing:
        raise ValueError(f"baseline file is missing keys: {', '.join(missing)}")
    return baselines


def normalize_zero_to_high(value: float, high: float) -> float:
    if high <= 0:
        return 0.0
    return clip(value / high)


def normalize_low_high(value: float, low: float, high: float) -> float:
    if math.isclose(low, high):
        return 0.5 if value > 0 else 0.0
    return clip((value - low) / (high - low))


def note_score_from_profile(profile: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    alpha_note = 1.65
    lambda_note = 0.35
    denominator = 1 + lambda_note / 2
    density_mean = float(profile.get("mean") or 0.0)
    density_cv = float(profile.get("coefficient_of_variation") or 0.0)
    density_note = normalize_zero_to_high(
        density_mean,
        float(baselines["density_note_mean_p95"]),
    )
    volatility = normalize_low_high(
        density_cv,
        float(baselines["density_cv_p05"]),
        float(baselines["density_cv_p95"]),
    )
    centered = volatility - 0.5
    raw = (density_note**alpha_note) * (1 + lambda_note * centered)
    return {
        "score": 200 * clip(raw / denominator),
        "density_note_mean": density_mean,
        "density_cv": density_cv,
        "normalized": {
            "density_note_mean": density_note,
            "volatility": volatility,
        },
        "parameters": {
            "density_note_mean_p95": float(baselines["density_note_mean_p95"]),
            "density_cv_p05": float(baselines["density_cv_p05"]),
            "density_cv_p95": float(baselines["density_cv_p95"]),
            "alpha_note": alpha_note,
            "lambda_note": lambda_note,
            "mapping": "global_p95_density_p05_p95_cv_to_200",
        },
    }


def segment_raw_features(summary: dict[str, Any]) -> dict[str, Any]:
    note_counts = summary.get("note_counts", {})
    if not isinstance(note_counts, dict):
        note_counts = {}
    total_notes = int(note_counts.get("total", 0))
    slide_count = int(note_counts.get("slide", 0))
    charge_count = int(note_counts.get("hold", 0)) + int(note_counts.get("touch_hold", 0))
    duration = float(summary.get("duration_seconds") or 0.0)
    profile = summary.get("density_profile", {})
    burst = summary.get("four_measure_burst", {})
    handtrip = summary.get("handtrip_movement", {})
    tricky = summary.get("same_button_triple_tap", {})
    return {
        "duration_seconds": duration,
        "total_notes": total_notes,
        "slide_count": slide_count,
        "charge_count": charge_count,
        "density_note_mean": float(profile.get("mean") or 0.0)
        if isinstance(profile, dict)
        else 0.0,
        "density_peak_q90": float(profile.get("q90") or 0.0)
        if isinstance(profile, dict)
        else 0.0,
        "density_cv": float(profile.get("coefficient_of_variation") or 0.0)
        if isinstance(profile, dict)
        else 0.0,
        "burst_density": float(burst.get("density") or 0.0) if isinstance(burst, dict) else 0.0,
        "burst_note_count": int(burst.get("note_count") or 0) if isinstance(burst, dict) else 0,
        "burst_duration_seconds": float(burst.get("duration_seconds") or 0.0)
        if isinstance(burst, dict)
        else 0.0,
        "burst_start_measure": burst.get("start_measure") if isinstance(burst, dict) else None,
        "burst_end_measure": burst.get("end_measure") if isinstance(burst, dict) else None,
        "slide_ratio": slide_count / total_notes if total_notes else 0.0,
        "slide_density": slide_count / duration if duration > 0 else 0.0,
        "charge_ratio": charge_count / total_notes if total_notes else 0.0,
        "charge_density": charge_count / duration if duration > 0 else 0.0,
        "handtrip_density": float(handtrip.get("distance_per_second") or 0.0)
        if isinstance(handtrip, dict)
        else 0.0,
        "tricky_shortest_time": (
            float(tricky["shortest_time_seconds"])
            if isinstance(tricky, dict) and tricky.get("shortest_time_seconds") is not None
            else None
        ),
    }


def score_segment_summary(summary: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    raw = segment_raw_features(summary)
    profile = summary.get("density_profile", {})
    note = note_score_from_profile(profile if isinstance(profile, dict) else {}, baselines)
    peak = burst_peak_score(
        raw["burst_density"],
        float(baselines["burst_density_p98"]),
    )
    slide = saturated_ratio_density_score(
        ratio_value=raw["slide_ratio"],
        density_value=raw["slide_density"],
        ratio_98=float(baselines["slide_ratio_p98"]),
        density_98=float(baselines["slide_density_p98"]),
    )
    charge = saturated_ratio_density_score(
        ratio_value=raw["charge_ratio"],
        density_value=raw["charge_density"],
        ratio_98=float(baselines["charge_ratio_p98"]),
        density_98=float(baselines["charge_density_p98"]),
    )
    handtrip = score_handtrip(
        raw["handtrip_density"],
        float(baselines["handtrip_density_p98"]),
    )
    tricky = score_tricky(
        raw["tricky_shortest_time"],
        float(baselines["tricky_intensity_p98"]),
    )
    scores = {
        "note": note["score"],
        "peak": peak["score"],
        "charge": charge["score"],
        "slide": slide["score"],
        "handtrip": handtrip["score"],
        "tricky": tricky["score"],
    }
    dominant = max(SCORE_NAMES, key=lambda name: scores[name])
    return {
        "raw": raw,
        "normalized": {
            "note": note["normalized"],
            "peak": {"burst_density": peak["normalized_density"]},
            "slide": {
                "density": slide["normalized_density"],
                "ratio": slide["normalized_ratio"],
            },
            "charge": {
                "density": charge["normalized_density"],
                "ratio": charge["normalized_ratio"],
            },
            "handtrip_density": handtrip["normalized_density"],
            "tricky": tricky["normalized"],
        },
        "scores": scores,
        "dominant_dimension": dominant,
        "score_detail": {
            "note": note,
            "peak": peak,
            "slide": slide,
            "charge": charge,
            "handtrip": handtrip,
            "tricky": tricky,
        },
    }


def parse_measure_range(text: str) -> tuple[int, int]:
    separators = (":", "-", "..")
    for separator in separators:
        if separator in text:
            left, right = text.split(separator, 1)
            start = int(left.strip())
            end = int(right.strip())
            break
    else:
        start = end = int(text.strip())
    if start <= 0 or end <= 0:
        raise argparse.ArgumentTypeError("measure ranges are 1-based and must be positive")
    if end < start:
        raise argparse.ArgumentTypeError("range end must be greater than or equal to start")
    return start, end


def ranges_from_segments_json(path: Path) -> list[tuple[int, int]]:
    payload = read_json(path)
    raw_segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_segments, list):
        raise ValueError("segments json must be a list or contain a 'segments' list")
    ranges: list[tuple[int, int]] = []
    for index, segment in enumerate(raw_segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"segment #{index} is not an object")
        start = segment.get("start_measure")
        end = segment.get("end_measure")
        if start is None or end is None:
            raise ValueError(f"segment #{index} is missing start_measure/end_measure")
        ranges.append(parse_measure_range(f"{start}:{end}"))
    return ranges


def chart_metadata(chart_path: Path, chart_data: dict[str, Any]) -> dict[str, Any]:
    meta = chart_data.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    song = chart_data.get("song", {})
    if not isinstance(song, dict):
        song = {}
    chart = chart_data.get("chart", {})
    if not isinstance(chart, dict):
        chart = {}
    return {
        "source_file": str(chart_path),
        "song_id": chart_data.get("song_id", meta.get("song_id", song.get("song_id"))),
        "title": chart_data.get("title", meta.get("title", song.get("title"))),
        "chart_kind": chart_data.get("chart_kind", meta.get("chart_kind", chart.get("chart_kind"))),
        "difficulty_index": chart_data.get(
            "difficulty_index",
            meta.get("difficulty_index", chart.get("difficulty_index")),
        ),
        "difficulty_name": chart_data.get(
            "difficulty_name",
            meta.get("difficulty_name", chart.get("difficulty_name")),
        ),
        "level": chart_data.get("level", meta.get("level", chart.get("level"))),
    }


def score_chart_segments(
    *,
    chart_path: Path,
    ranges: list[tuple[int, int]],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    chart_data = read_json(chart_path)
    densities = chart_measure_densities(chart_data)
    if not densities:
        raise ValueError(f"chart has no measures: {chart_path}")
    if not ranges:
        ranges = [(1, len(densities))]

    segments = []
    for start_measure, end_measure in ranges:
        if end_measure > len(densities):
            raise ValueError(
                f"range {start_measure}:{end_measure} exceeds chart length {len(densities)}"
            )
        summary = summarize_segment(densities, start_measure - 1, end_measure)
        scored = score_segment_summary(summary, baselines)
        segments.append(
            {
                "start_measure": start_measure,
                "end_measure": end_measure,
                "measure_count": end_measure - start_measure + 1,
                "start_seconds": summary.get("start_seconds"),
                "end_seconds": summary.get("end_seconds"),
                "duration_seconds": summary.get("duration_seconds"),
                **scored,
            }
        )

    return {
        "chart": {
            **chart_metadata(chart_path, chart_data),
            "measure_count": len(densities),
        },
        "dimensions": list(SCORE_NAMES),
        "baselines": baselines,
        "segments": segments,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score explicit measure segments on the global six-dimensional Simai scale."
    )
    parser.add_argument("--chart-file", type=Path, default=None)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Directory containing index.json when resolving by song/title.",
    )
    parser.add_argument("--song-id", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--chart-kind", default=None)
    parser.add_argument("--difficulty-index", type=int, default=None)
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        type=parse_measure_range,
        default=[],
        help="1-based inclusive measure range, e.g. 1:16. Repeatable.",
    )
    parser.add_argument(
        "--segments-json",
        type=Path,
        default=None,
        help="JSON list or object with segments containing start_measure/end_measure.",
    )
    parser.add_argument(
        "--baselines-json",
        type=Path,
        default=DEFAULT_BASELINES,
        help=f"Global baseline/intermediate JSON. Default: {DEFAULT_BASELINES}",
    )
    parser.add_argument("--output", type=Path, default=None)
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
    ranges = list(args.ranges)
    if args.segments_json is not None:
        ranges.extend(ranges_from_segments_json(args.segments_json))
    baselines = load_baselines(args.baselines_json)
    payload = score_chart_segments(chart_path=chart_path, ranges=ranges, baselines=baselines)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
