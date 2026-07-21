#!/usr/bin/env python3
"""Export density segmentation reports for every compiled Simai chart."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_density_segmenter import (
    DEFAULT_SEGMENT_SCORE_BASELINES,
    DEFAULT_STEADY_DENSITY_STANDARDS,
    SegmentationConfig,
    build_segmentation_report,
    load_steady_density_standards,
)
from scripts.simai_segment_scorer import load_baselines


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def report_output_name(chart_file: Path) -> str:
    return chart_file.name


def segment_label_counts(report: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(segment.get("label", "UNKNOWN") for segment in report.get("segments", [])))


def segment_tier_counts(report: dict[str, Any]) -> dict[str, int]:
    return dict(
        Counter(
            segment.get("steady_tier")
            for segment in report.get("segments", [])
            if segment.get("steady_tier") is not None
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-export density segmentation reports for compiled Simai charts."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Directory containing index.json and chart JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/segmentation_reports_p33p67"),
        help="Output directory. Default: outputs/segmentation_reports_p33p67",
    )
    parser.add_argument("--segments", type=int, default=5, help="Force an exact segment count. Default: 5.")
    parser.add_argument("--min-segments", type=int, default=5, help="Default: 5.")
    parser.add_argument("--max-segments", type=int, default=5, help="Default: 5.")
    parser.add_argument("--edge-exclusion", type=int, default=8, help="Default: 8.")
    parser.add_argument("--min-segment-measures", type=int, default=None)
    parser.add_argument("--max-segment-length-factor", type=float, default=1.75)
    parser.add_argument("--window", type=int, default=4, help="Default: 4.")
    parser.add_argument(
        "--baselines-json",
        type=Path,
        default=DEFAULT_SEGMENT_SCORE_BASELINES,
        help=f"Six-dimension baseline JSON. Default: {DEFAULT_SEGMENT_SCORE_BASELINES}",
    )
    parser.add_argument(
        "--steady-standards-json",
        type=Path,
        default=DEFAULT_STEADY_DENSITY_STANDARDS,
        help=f"Steady p33/p67 standards JSON. Default: {DEFAULT_STEADY_DENSITY_STANDARDS}",
    )
    parser.add_argument("--include-measures", action="store_true", help="Include per-measure densities in each report.")
    parser.add_argument(
        "--omit-density-curve",
        action="store_true",
        help="Remove each segment's density_curve while keeping summary and six-dimension fields.",
    )
    parser.add_argument(
        "--embed-score-baselines",
        action="store_true",
        help="Embed the full six-dimension baseline payload in every report. Usually too large for batch output.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    index_path = args.index_dir / "index.json"
    index_payload = read_json(index_path)
    charts = index_payload.get("charts", [])
    if not isinstance(charts, list):
        raise ValueError(f"{index_path} does not contain a charts list")

    config = SegmentationConfig(
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        fixed_segments=args.segments,
        edge_exclusion=args.edge_exclusion,
        min_segment_measures=args.min_segment_measures,
        max_segment_length_factor=args.max_segment_length_factor,
        window=args.window,
    )
    baselines = load_baselines(args.baselines_json)
    steady_standards = load_steady_density_standards(args.steady_standards_json)

    reports_dir = args.out / "charts"
    label_counts: Counter[str] = Counter()
    steady_tier_counts: Counter[str] = Counter()
    report_index: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_index": str(index_path),
        "output_directory": str(args.out),
        "chart_count": 0,
        "error_count": 0,
        "config": {
            "min_segments": args.min_segments,
            "max_segments": args.max_segments,
            "fixed_segments": args.segments,
            "edge_exclusion": args.edge_exclusion,
            "min_segment_measures": args.min_segment_measures,
            "max_segment_length_factor": args.max_segment_length_factor,
            "window": args.window,
            "baselines_json": str(args.baselines_json),
            "steady_standards_json": str(args.steady_standards_json),
            "include_measures": args.include_measures,
            "omit_density_curve": args.omit_density_curve,
            "embed_score_baselines": args.embed_score_baselines,
        },
        "reports": [],
        "errors": [],
    }

    for index, chart_entry in enumerate(charts, start=1):
        if not isinstance(chart_entry, dict):
            continue
        chart_file = Path(str(chart_entry.get("file")))
        output_path = reports_dir / report_output_name(chart_file)
        try:
            report = build_segmentation_report(
                chart_file,
                config,
                include_measures=args.include_measures,
                baselines_json=None,
                baselines=baselines,
                include_score_baselines=args.embed_score_baselines,
                steady_standards_json=args.steady_standards_json,
                steady_standards=steady_standards,
            )
            if args.omit_density_curve:
                for segment in report.get("segments", []):
                    if isinstance(segment, dict):
                        segment.pop("density_curve", None)
        except Exception as exc:  # noqa: BLE001 - batch index should preserve all chart failures.
            report_index["error_count"] += 1
            report_index["errors"].append(
                {
                    "source_file": str(chart_file),
                    "song_id": chart_entry.get("song_id"),
                    "title": chart_entry.get("title"),
                    "chart_kind": chart_entry.get("chart_kind"),
                    "difficulty_index": chart_entry.get("difficulty_index"),
                    "difficulty_name": chart_entry.get("difficulty_name"),
                    "level": chart_entry.get("level"),
                    "error": str(exc),
                }
            )
            continue

        write_json(output_path, report)
        labels = segment_label_counts(report)
        tiers = segment_tier_counts(report)
        label_counts.update(labels)
        steady_tier_counts.update(tiers)
        report_index["chart_count"] += 1
        report_index["reports"].append(
            {
                "source_file": str(chart_file),
                "report_file": str(output_path),
                "song_id": chart_entry.get("song_id"),
                "title": chart_entry.get("title"),
                "chart_kind": chart_entry.get("chart_kind"),
                "difficulty_index": chart_entry.get("difficulty_index"),
                "difficulty_name": chart_entry.get("difficulty_name"),
                "level": chart_entry.get("level"),
                "segment_count": len(report.get("segments", [])),
                "label_counts": labels,
                "steady_tier_counts": tiers,
            }
        )
        if index % 100 == 0:
            print(f"processed {index}/{len(charts)} charts", flush=True)

    report_index["label_counts"] = dict(label_counts)
    report_index["steady_tier_counts"] = dict(steady_tier_counts)
    write_json(args.out / "index.json", report_index)
    print(f"exported_reports: {report_index['chart_count']}")
    print(f"errors: {report_index['error_count']}")
    print(f"output: {args.out}")
    return 1 if report_index["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
