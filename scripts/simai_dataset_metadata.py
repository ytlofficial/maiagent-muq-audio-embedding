#!/usr/bin/env python3
"""Shared identifiers and metadata for chart, segment, and chunk tables."""

from __future__ import annotations

from typing import Any


SCORE_NAMES = ("note", "peak", "charge", "slide", "handtrip", "tricky")
EXPECTED_SEGMENT_COUNT = 5


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def chart_id_for(song_id: Any, chart_kind: Any, difficulty: Any) -> str:
    parsed_song_id = optional_int(song_id)
    parsed_difficulty = optional_int(difficulty)
    kind = str(chart_kind or "").strip().upper()
    if parsed_song_id is None or parsed_difficulty is None or not kind:
        raise ValueError(
            "chart_id requires song_id, chart_kind, and difficulty_index"
        )
    return f"{parsed_song_id}:{kind}:{parsed_difficulty}"


def level_value(level: Any) -> float | None:
    """Convert display levels to numeric values, using .6 for plus levels."""

    if level is None:
        return None
    text = str(level).strip()
    if not text:
        return None
    plus = text.endswith("+")
    if plus:
        text = text[:-1]
    try:
        return float(text) + (0.6 if plus else 0.0)
    except ValueError:
        return None


def segment_key_for(chart_id: str, segment_id: int) -> str:
    if segment_id < 0 or segment_id >= EXPECTED_SEGMENT_COUNT:
        raise ValueError(f"segment_id must be in 0-4, got {segment_id}")
    return f"{chart_id}:{segment_id}"


def segment_rows_from_report(
    report: dict[str, Any],
    *,
    chart_name: str,
    report_file: str,
) -> list[dict[str, Any]]:
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    segments = report.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"report has no segments list: {report_file}")
    if len(segments) != EXPECTED_SEGMENT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SEGMENT_COUNT} segments, got {len(segments)}: "
            f"{report_file}"
        )

    chart_id = chart_id_for(
        metadata.get("song_id"),
        metadata.get("chart_kind"),
        metadata.get("difficulty_index"),
    )
    rows: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {segment_id} is not an object: {report_file}")
        scores = segment.get("scores", {})
        if not isinstance(scores, dict) or any(name not in scores for name in SCORE_NAMES):
            raise ValueError(
                f"segment {segment_id} is missing six-dimensional scores: {report_file}"
            )
        score_values = [float(scores[name]) for name in SCORE_NAMES]
        rows.append(
            {
                "key": segment_key_for(chart_id, segment_id),
                "chart_id": chart_id,
                "chart_name": chart_name,
                "song_id": int(metadata["song_id"]),
                "segment_id": segment_id,
                "start_measure": int(segment["start_measure"]),
                "end_measure": int(segment["end_measure"]),
                "measure_count": int(segment["end_measure"])
                - int(segment["start_measure"])
                + 1,
                "label": str(segment.get("label") or ""),
                "base_label": str(segment.get("base_label") or ""),
                "note": score_values[0],
                "peak": score_values[1],
                "charge": score_values[2],
                "slide": score_values[3],
                "handtrip": score_values[4],
                "tricky": score_values[5],
                "score_vector": score_values,
                "dominant_dimension": str(segment.get("dominant_dimension") or ""),
                "report_file": report_file,
            }
        )
    return rows
