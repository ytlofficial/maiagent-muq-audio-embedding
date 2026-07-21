#!/usr/bin/env python3
"""Build whole-chart six-dimensional Simai score tables.

Each chart is summarized as one full-chart segment. Global medians and
percentile bands are computed first, then every chart receives six 0-200
scores: note, peak, charge, slide, handtrip, and tricky.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_density_segmenter import (
    burst_peak_score,
    clip,
    normalize_percentile,
    normalize_to_q95,
    percentile,
    saturated_ratio_density_score,
    summarize_segment,
    tricky_final_curve,
)
from scripts.simai_measure_density import chart_measure_densities, read_json


SCORE_NAMES = ("note", "peak", "charge", "slide", "handtrip", "tricky")


def median_nonzero(values: list[float], fallback: float) -> float:
    nonzero = [value for value in values if value > 0]
    return statistics.median(nonzero) if nonzero else fallback


def score_peak_note(
    *,
    density_peak_q90: float,
    density_note_mean: float,
    density_cv: float,
    burst_density: float,
    q90_values: list[float],
    mean_values: list[float],
    cv_values: list[float],
    burst_density_98: float,
) -> dict[str, Any]:
    alpha_note = 1.65
    lambda_note = 0.35
    note_denominator = 1 + lambda_note / 2
    density_peak = normalize_to_q95(density_peak_q90, q90_values)
    density_note = normalize_to_q95(density_note_mean, mean_values)
    volatility = normalize_percentile(density_cv, cv_values)
    centered = volatility - 0.5
    note_raw = (density_note**alpha_note) * (1 + lambda_note * centered)
    burst_peak = burst_peak_score(burst_density, burst_density_98)
    burst_score = burst_peak["score"]
    peak = burst_score
    return {
        "note": 200 * clip(note_raw / note_denominator),
        "peak": peak,
        "burst": burst_score,
        "normalized": {
            "density_peak_q90": density_peak,
            "density_note_mean": density_note,
            "volatility": volatility,
            "burst_density": burst_peak["normalized_density"],
        },
        "parameters": {
            "burst_density_98": burst_density_98,
            "peak_floor_score": burst_peak["floor_score"],
            "peak_alpha": burst_peak["alpha"],
            "lambda_note": lambda_note,
            "mapping": "burst_p98_steeper_floor_to_200",
        },
    }


def score_handtrip(density: float, density_98: float) -> dict[str, float]:
    alpha = 1.7
    normalized = clip(density / density_98) if density > 0 and density_98 > 0 else 0.0
    return {
        "score": 200 * clip(normalized**alpha),
        "density": density,
        "density_98": density_98,
        "normalized_density": normalized,
        "alpha": alpha,
        "mapping": "density_p98_power_to_200",
    }


def score_tricky(
    shortest_time_seconds: float | None,
    intensity_98: float,
    gamma: float = 2.2,
) -> dict[str, float | None]:
    if shortest_time_seconds is None:
        base_score = 0.2
        return {
            "score": tricky_final_curve(base_score),
            "base_score": base_score,
            "shortest_time_seconds": None,
            "intensity_98": intensity_98,
            "gamma": gamma,
            "normalized": 0.001,
            "final_curve": "steep_flat_steep_hermite",
            "normalization": "inverse_time_p98_to_200",
        }
    if shortest_time_seconds <= 0:
        normalized = 1.0
    else:
        normalized = clip((1 / shortest_time_seconds) / intensity_98) if intensity_98 > 0 else 1.0
    base_score = 200 * clip(normalized)
    return {
        "score": tricky_final_curve(base_score),
        "base_score": base_score,
        "shortest_time_seconds": shortest_time_seconds,
        "intensity_98": intensity_98,
        "gamma": gamma,
        "normalized": normalized,
        "final_curve": "steep_flat_steep_hermite",
        "normalization": "inverse_time_p98_to_200",
    }


def chart_summary_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    chart_path = Path(entry["file"])
    chart_data = read_json(chart_path)
    densities = chart_measure_densities(chart_data)
    summary = summarize_segment(densities, 0, len(densities))
    return {
        "source_file": str(chart_path),
        "song_id": entry.get("song_id"),
        "title": entry.get("title"),
        "chart_kind": entry.get("chart_kind"),
        "difficulty_index": entry.get("difficulty_index"),
        "difficulty_name": entry.get("difficulty_name"),
        "level": entry.get("level"),
        "measure_count": entry.get("measure_count"),
        "duration_seconds": summary["duration_seconds"],
        "summary": summary,
    }


def raw_feature_row(chart_summary: dict[str, Any]) -> dict[str, Any]:
    summary = chart_summary["summary"]
    note_counts = summary.get("note_counts", {})
    total_notes = int(note_counts.get("total", 0)) if isinstance(note_counts, dict) else 0
    slide_count = int(note_counts.get("slide", 0)) if isinstance(note_counts, dict) else 0
    charge_count = (
        int(note_counts.get("hold", 0)) + int(note_counts.get("touch_hold", 0))
        if isinstance(note_counts, dict)
        else 0
    )
    duration = float(summary.get("duration_seconds") or 0.0)
    density_profile = summary.get("density_profile", {})
    burst = summary.get("four_measure_burst", {})
    handtrip = summary.get("handtrip_movement", {})
    tricky = summary.get("same_button_triple_tap", {})
    return {
        **{
            key: chart_summary.get(key)
            for key in (
                "source_file",
                "song_id",
                "title",
                "chart_kind",
                "difficulty_index",
                "difficulty_name",
                "level",
                "measure_count",
            )
        },
        "duration_seconds": duration,
        "total_notes": total_notes,
        "slide_count": slide_count,
        "charge_count": charge_count,
        "density_note_mean": float(density_profile.get("mean") or 0.0)
        if isinstance(density_profile, dict)
        else 0.0,
        "density_peak_q90": float(density_profile.get("q90") or 0.0)
        if isinstance(density_profile, dict)
        else 0.0,
        "density_cv": float(density_profile.get("coefficient_of_variation") or 0.0)
        if isinstance(density_profile, dict)
        else 0.0,
        "burst_density": float(burst.get("density") or 0.0)
        if isinstance(burst, dict)
        else 0.0,
        "burst_note_count": int(burst.get("note_count") or 0)
        if isinstance(burst, dict)
        else 0,
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


def compute_global_baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tricky_times = [
        float(row["tricky_shortest_time"])
        for row in rows
        if row.get("tricky_shortest_time") is not None
    ]
    tricky_intensities = [1 / value for value in tricky_times if value > 0]
    return {
        "chart_count": len(rows),
        "density_note_mean_p95": percentile([row["density_note_mean"] for row in rows], 0.95),
        "density_peak_q90_p95": percentile([row["density_peak_q90"] for row in rows], 0.95),
        "density_cv_p05": percentile([row["density_cv"] for row in rows], 0.05),
        "density_cv_p95": percentile([row["density_cv"] for row in rows], 0.95),
        "burst_density_p98": percentile([row.get("burst_density", 0.0) for row in rows], 0.98),
        "slide_density_p98": percentile([row["slide_density"] for row in rows], 0.98),
        "slide_ratio_p98": percentile([row["slide_ratio"] for row in rows], 0.98),
        "charge_density_p98": percentile([row["charge_density"] for row in rows], 0.98),
        "charge_ratio_p98": percentile([row["charge_ratio"] for row in rows], 0.98),
        "handtrip_density_p98": percentile([row["handtrip_density"] for row in rows], 0.98),
        "tricky_intensity_p98": percentile(tricky_intensities, 0.98) if tricky_intensities else 1.0,
    }


def score_rows(rows: list[dict[str, Any]], baselines: dict[str, Any]) -> list[dict[str, Any]]:
    mean_values = [row["density_note_mean"] for row in rows]
    q90_values = [row["density_peak_q90"] for row in rows]
    cv_values = [row["density_cv"] for row in rows]
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        peak_note = score_peak_note(
            density_peak_q90=row["density_peak_q90"],
            density_note_mean=row["density_note_mean"],
            density_cv=row["density_cv"],
            burst_density=float(row.get("burst_density") or 0.0),
            q90_values=q90_values,
            mean_values=mean_values,
            cv_values=cv_values,
            burst_density_98=float(baselines["burst_density_p98"]),
        )
        slide = saturated_ratio_density_score(
            ratio_value=row["slide_ratio"],
            density_value=row["slide_density"],
            ratio_98=float(baselines["slide_ratio_p98"]),
            density_98=float(baselines["slide_density_p98"]),
        )
        charge = saturated_ratio_density_score(
            ratio_value=row["charge_ratio"],
            density_value=row["charge_density"],
            ratio_98=float(baselines["charge_ratio_p98"]),
            density_98=float(baselines["charge_density_p98"]),
        )
        handtrip = score_handtrip(
            row["handtrip_density"],
            float(baselines["handtrip_density_p98"]),
        )
        tricky = score_tricky(
            row["tricky_shortest_time"],
            float(baselines["tricky_intensity_p98"]),
        )
        scores = {
            "note": peak_note["note"],
            "peak": peak_note["peak"],
            "charge": charge["score"],
            "slide": slide["score"],
            "handtrip": handtrip["score"],
            "tricky": tricky["score"],
        }
        dominant = max(SCORE_NAMES, key=lambda name: scores[name])
        scored_rows.append(
            {
                **row,
                **{f"{name}_score": scores[name] for name in SCORE_NAMES},
                "dominant_dimension": dominant,
                "score_detail": {
                    "peak_note": peak_note,
                    "slide": slide,
                    "charge": charge,
                    "handtrip": handtrip,
                    "tricky": tricky,
                },
            }
        )
    return scored_rows


def build_table(index_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    index = read_json(index_dir / "index.json")
    entries = index.get("charts", [])
    if limit is not None:
        entries = entries[:limit]
    chart_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for entry in entries:
        try:
            chart_summaries.append(chart_summary_from_entry(entry))
        except Exception as exc:  # Keep whole-library scoring resilient to bad source rows.
            errors.append(
                {
                    "song_id": entry.get("song_id"),
                    "title": entry.get("title"),
                    "chart_kind": entry.get("chart_kind"),
                    "difficulty_index": entry.get("difficulty_index"),
                    "difficulty_name": entry.get("difficulty_name"),
                    "level": entry.get("level"),
                    "file": entry.get("file"),
                    "error": str(exc),
                }
            )
    raw_rows = [raw_feature_row(summary) for summary in chart_summaries]
    baselines = compute_global_baselines(raw_rows)
    rows = score_rows(raw_rows, baselines)
    return {
        "source_index": str(index_dir / "index.json"),
        "baselines": baselines,
        "dimensions": list(SCORE_NAMES),
        "error_count": len(errors),
        "errors": errors,
        "rows": rows,
    }


def write_outputs(table: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "song_id",
        "title",
        "chart_kind",
        "difficulty_index",
        "difficulty_name",
        "level",
        "measure_count",
        "duration_seconds",
        "total_notes",
        "burst_density",
        "burst_note_count",
        "burst_duration_seconds",
        "burst_start_measure",
        "burst_end_measure",
        "note_score",
        "peak_score",
        "charge_score",
        "slide_score",
        "handtrip_score",
        "tricky_score",
        "dominant_dimension",
        "source_file",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in table["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def intermediate_row(row: dict[str, Any]) -> dict[str, Any]:
    peak_note = row.get("score_detail", {}).get("peak_note", {})
    slide = row.get("score_detail", {}).get("slide", {})
    charge = row.get("score_detail", {}).get("charge", {})
    handtrip = row.get("score_detail", {}).get("handtrip", {})
    tricky = row.get("score_detail", {}).get("tricky", {})
    return {
        "song_id": row.get("song_id"),
        "title": row.get("title"),
        "chart_kind": row.get("chart_kind"),
        "difficulty_index": row.get("difficulty_index"),
        "difficulty_name": row.get("difficulty_name"),
        "level": row.get("level"),
        "source_file": row.get("source_file"),
        "measure_count": row.get("measure_count"),
        "duration_seconds": row.get("duration_seconds"),
        "total_notes": row.get("total_notes"),
        "raw": {
            "density_note_mean": row.get("density_note_mean"),
            "density_peak_q90": row.get("density_peak_q90"),
            "density_cv": row.get("density_cv"),
            "burst_density": row.get("burst_density"),
            "burst_note_count": row.get("burst_note_count"),
            "burst_duration_seconds": row.get("burst_duration_seconds"),
            "burst_start_measure": row.get("burst_start_measure"),
            "burst_end_measure": row.get("burst_end_measure"),
            "slide_ratio": row.get("slide_ratio"),
            "slide_density": row.get("slide_density"),
            "charge_ratio": row.get("charge_ratio"),
            "charge_density": row.get("charge_density"),
            "handtrip_density": row.get("handtrip_density"),
            "tricky_shortest_time": row.get("tricky_shortest_time"),
        },
        "normalized": {
            "peak_note": peak_note.get("normalized") if isinstance(peak_note, dict) else None,
            "slide": {
                "density": slide.get("normalized_density"),
                "ratio": slide.get("normalized_ratio"),
            }
            if isinstance(slide, dict)
            else None,
            "charge": {
                "density": charge.get("normalized_density"),
                "ratio": charge.get("normalized_ratio"),
            }
            if isinstance(charge, dict)
            else None,
            "handtrip_density": (
                handtrip.get("normalized_density") if isinstance(handtrip, dict) else None
            ),
            "tricky": tricky.get("normalized") if isinstance(tricky, dict) else None,
        },
        "component_scores": {
            "burst": peak_note.get("burst") if isinstance(peak_note, dict) else None,
            "tricky_base": tricky.get("base_score") if isinstance(tricky, dict) else None,
        },
        "scores": {name: row.get(f"{name}_score") for name in SCORE_NAMES},
        "dominant_dimension": row.get("dominant_dimension"),
    }


def write_intermediates(table: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_index": table.get("source_index"),
        "baselines": table.get("baselines"),
        "dimensions": table.get("dimensions"),
        "error_count": table.get("error_count"),
        "errors": table.get("errors"),
        "rows": [intermediate_row(row) for row in table.get("rows", [])],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a whole-library six-dimensional Simai score table."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Directory containing index.json. Default: outputs/simai_measures",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/six_dimension/global_six_dimension_table.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/six_dimension/global_six_dimension_table.csv"),
    )
    parser.add_argument(
        "--output-intermediates",
        type=Path,
        default=Path("outputs/six_dimension/global_six_dimension_intermediates.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process first N charts.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    table = build_table(args.index_dir, limit=args.limit)
    write_outputs(table, args.output_json, args.output_csv)
    write_intermediates(table, args.output_intermediates)
    print(json.dumps({
        "json": str(args.output_json),
        "csv": str(args.output_csv),
        "intermediates": str(args.output_intermediates),
        "chart_count": table["baselines"]["chart_count"],
        "error_count": table["error_count"],
        "baselines": table["baselines"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
