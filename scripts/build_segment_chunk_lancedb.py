#!/usr/bin/env python3
"""Build a LanceDB index of Simai segment chunk embeddings.

Chunk ranges are delegated to ``export_segment_chunk_ranges.py`` so this script
uses the same four-measure chunking policy as the existing segmentation export.
For each chunk, the matching measures are read from ``outputs/simai_measures``,
embedded with ``simai_pattern_embedding.py``, and stored in LanceDB.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_segment_chunk_ranges import chunk_specs_from_report
from scripts.simai_dataset_metadata import (
    chart_id_for,
    level_value,
    segment_key_for,
    segment_rows_from_report,
)
from scripts.simai_pattern_embedding import build_embedding_from_measures, select_measures


DEFAULT_REPORTS_DIR = REPO_ROOT / "outputs" / "segmentation_reports" / "charts"
DEFAULT_MEASURES_DIR = REPO_ROOT / "outputs" / "simai_measures" / "charts"
DEFAULT_DB_PATH = REPO_ROOT / "outputs" / "lancedb" / "simai_pattern_chunks"
DEFAULT_TABLE_NAME = "simai_pattern_chunks"
DEFAULT_INDEX_TABLE_NAME = "simai_pattern_chunk_index"
DEFAULT_SEGMENT_TABLE_NAME = "simai_segments"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chart_name_for_path(path: Path) -> str:
    return path.stem


def normalize_text(value: Any) -> str:
    return str(value or "").casefold()


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def db_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def db_int(value: Any) -> int:
    parsed = optional_int(value)
    return parsed if parsed is not None else -1


def db_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def matches_selector(
    report: dict[str, Any],
    *,
    song_id: int | None,
    title: str | None,
    chart_kind: str | None,
    difficulty_index: int | None,
    difficulty_name: str | None,
) -> bool:
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    if song_id is not None and optional_int(metadata.get("song_id")) != song_id:
        return False
    if title is not None and normalize_text(title) not in normalize_text(metadata.get("title")):
        return False
    if chart_kind is not None and normalize_text(metadata.get("chart_kind")) != normalize_text(chart_kind):
        return False
    if difficulty_index is not None and optional_int(metadata.get("difficulty_index")) != difficulty_index:
        return False
    if difficulty_name is not None and normalize_text(metadata.get("difficulty_name")) != normalize_text(difficulty_name):
        return False
    return True


def iter_report_paths(args: argparse.Namespace) -> Iterator[Path]:
    if args.report_file is not None:
        yield args.report_file
        return

    for report_path in sorted(args.reports_dir.glob("*.json")):
        if any(
            selector is not None
            for selector in (
                args.song_id,
                args.title,
                args.chart_kind,
                args.difficulty_index,
                args.difficulty_name,
            )
        ):
            report = read_json(report_path)
            if not matches_selector(
                report,
                song_id=args.song_id,
                title=args.title,
                chart_kind=args.chart_kind,
                difficulty_index=args.difficulty_index,
                difficulty_name=args.difficulty_name,
            ):
                continue
        yield report_path


def measure_path_for_report(report_path: Path, measures_dir: Path) -> Path:
    return measures_dir / report_path.name


def chart_metadata(
    report_path: Path,
    report: dict[str, Any],
    measures: dict[str, Any],
    measures_path: Path,
) -> dict[str, Any]:
    report_metadata = report.get("metadata", {})
    if not isinstance(report_metadata, dict):
        report_metadata = {}
    song = measures.get("song", {}) if isinstance(measures.get("song"), dict) else {}
    chart = measures.get("chart", {}) if isinstance(measures.get("chart"), dict) else {}
    timeline = measures.get("timeline", {}) if isinstance(measures.get("timeline"), dict) else {}

    return {
        "chart_name": chart_name_for_path(report_path),
        "song_id": report_metadata.get("song_id", song.get("song_id")),
        "title": report_metadata.get("title", song.get("title")),
        "artist": report_metadata.get("artist", song.get("artist")),
        "bpm": report_metadata.get("bpm", song.get("bpm")),
        "genre": report_metadata.get("genre", song.get("genre")),
        "cabinet": report_metadata.get("cabinet", song.get("cabinet")),
        "chart_kind": report_metadata.get("chart_kind", chart.get("chart_kind")),
        "chart_version": report_metadata.get("chart_version", chart.get("chart_version")),
        "difficulty_index": report_metadata.get("difficulty_index", chart.get("difficulty_index")),
        "difficulty_name": report_metadata.get("difficulty_name", chart.get("difficulty_name")),
        "level": report_metadata.get("level", chart.get("level")),
        "charter": report_metadata.get("charter", chart.get("charter")),
        "measure_count": report_metadata.get("measure_count", timeline.get("measure_count")),
        "report_file": str(report_path),
        "measure_file": str(measures_path),
    }


def row_key(chart_name: str, start_measure: int, end_measure: int) -> str:
    return f"{chart_name}:{start_measure}-{end_measure}"


def compact_segments(report: dict[str, Any]) -> list[dict[str, Any]]:
    segments = report.get("segments", [])
    if not isinstance(segments, list):
        return []

    compact: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        compact.append(
            {
                "segment_id": segment_id,
                "index": segment.get("index", segment_id + 1),
                "label": segment.get("label"),
                "base_label": segment.get("base_label"),
                "start_measure": segment.get("start_measure"),
                "end_measure": segment.get("end_measure"),
                "event_labels": segment.get("event_labels", []),
                "steady_tier": segment.get("steady_tier"),
            }
        )
    return compact


def build_rows_and_index_for_report(
    report_path: Path,
    *,
    measures_dir: Path,
    include_sparse_features: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = read_json(report_path)
    measures_path = measure_path_for_report(report_path, measures_dir)
    if not measures_path.exists():
        raise FileNotFoundError(f"missing measure file for {report_path.name}: {measures_path}")

    measures_data = read_json(measures_path)
    metadata = chart_metadata(report_path, report, measures_data, measures_path)
    chart_name = str(metadata["chart_name"])
    chart_id = chart_id_for(
        metadata["song_id"], metadata["chart_kind"], metadata["difficulty_index"]
    )
    chunk_specs = chunk_specs_from_report(report)
    ranges = [spec.range for spec in chunk_specs]

    rows: list[dict[str, Any]] = []
    for chunk_index, chunk_spec in enumerate(chunk_specs, start=1):
        start_measure, end_measure = chunk_spec.start_measure, chunk_spec.end_measure
        selected = select_measures(measures_data, (start_measure, end_measure))
        overlap_measure_slots = chunk_spec.overlap_measure_slots
        embedding = build_embedding_from_measures(
            selected,
            require_expected_measures=True,
            overlap_measure_slots=overlap_measure_slots,
            include_sparse_features=include_sparse_features,
        )

        row: dict[str, Any] = {
            "key": row_key(chart_name, start_measure, end_measure),
            "chart_id": chart_id,
            "chart_name": chart_name,
            "title": db_text(metadata["title"]),
            "song_id": db_int(metadata["song_id"]),
            "artist": db_text(metadata["artist"]),
            "chart_kind": db_text(metadata["chart_kind"]),
            "difficulty_index": db_int(metadata["difficulty_index"]),
            "difficulty_name": db_text(metadata["difficulty_name"]),
            "level": db_text(metadata["level"]),
            "segment_id": chunk_spec.segment_id,
            "segment_key": segment_key_for(chart_id, chunk_spec.segment_id),
            "chunk_index": chunk_index,
            "start_measure": start_measure,
            "end_measure": end_measure,
            "measure_range": f"{start_measure}-{end_measure}",
            "body_start_measure": chunk_spec.body_start_measure,
            "body_end_measure": chunk_spec.body_end_measure,
            "body_range": f"{chunk_spec.body_start_measure}-{chunk_spec.body_end_measure}",
            "overlap_measure_slots_json": json.dumps(overlap_measure_slots, separators=(",", ":")),
            "overlap_measures_json": json.dumps(chunk_spec.overlap_measure_ranges, separators=(",", ":")),
            "measure_weights_json": json.dumps(
                embedding.get("measure_weights", []),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "measure_count": len(selected),
            "vector": embedding["embedding"],
            "embedding_dimension": db_int(embedding["dimension"]),
            "event_count": db_int(embedding["event_count"]),
            "weighted_event_count": db_float(embedding.get("weighted_event_count")),
            "total_beats": db_float(embedding["total_beats"]),
            "rhythm_bpm": db_float(embedding.get("rhythm_bpm")),
            "rhythm_tick_count": db_int(embedding.get("rhythm_tick_count")),
            "rhythm_ticks_per_measure": db_int(embedding.get("rhythm_ticks_per_measure")),
            "report_file": db_text(metadata["report_file"]),
            "measure_file": db_text(metadata["measure_file"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if include_sparse_features:
            row["nonzero_features_json"] = json.dumps(
                embedding.get("nonzero_features", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        rows.append(row)

    plan = report.get("plan", {})
    if not isinstance(plan, dict):
        plan = {}
    row_keys = [str(row["key"]) for row in rows]
    chunk_body_ranges = [
        [spec.body_start_measure, spec.body_end_measure]
        for spec in chunk_specs
    ]
    chunk_overlap_slots = [spec.overlap_measure_slots for spec in chunk_specs]
    chunk_overlap_measures = [spec.overlap_measure_ranges for spec in chunk_specs]
    chunk_measure_weights = [
        json.loads(str(row["measure_weights_json"]))
        for row in rows
    ]
    index_row = {
        "chart_id": chart_id,
        "chart_name": chart_name,
        "song_id": db_int(metadata["song_id"]),
        "title": db_text(metadata["title"]),
        "artist": db_text(metadata["artist"]),
        "bpm": db_text(metadata["bpm"]),
        "genre": db_text(metadata["genre"]),
        "cabinet": db_text(metadata["cabinet"]),
        "chart_kind": db_text(metadata["chart_kind"]),
        "chart_version": db_text(metadata["chart_version"]),
        "difficulty_index": db_int(metadata["difficulty_index"]),
        "difficulty": db_int(metadata["difficulty_index"]),
        "difficulty_name": db_text(metadata["difficulty_name"]),
        "level": db_text(metadata["level"]),
        "level_value": db_float(level_value(metadata["level"])),
        "charter": db_text(metadata["charter"]),
        "measure_count": db_int(metadata["measure_count"]),
        "segment_count": db_int(plan.get("segment_count")),
        "boundaries_after_measure_json": json.dumps(
            plan.get("boundaries_after_measure", []),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "segments_json": json.dumps(
            compact_segments(report),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "chunk_count": len(rows),
        "chunk_ranges_json": json.dumps(ranges, ensure_ascii=False, separators=(",", ":")),
        "chunk_body_ranges_json": json.dumps(chunk_body_ranges, ensure_ascii=False, separators=(",", ":")),
        "chunk_overlap_measure_slots_json": json.dumps(
            chunk_overlap_slots,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "chunk_overlap_measures_json": json.dumps(
            chunk_overlap_measures,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "chunk_measure_weights_json": json.dumps(
            chunk_measure_weights,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "chunk_keys_json": json.dumps(row_keys, ensure_ascii=False, separators=(",", ":")),
        "first_chunk_key": row_keys[0] if row_keys else "",
        "last_chunk_key": row_keys[-1] if row_keys else "",
        "embedding_dimension": db_int(rows[0]["embedding_dimension"] if rows else None),
        "rhythm_tick_count": db_int(rows[0]["rhythm_tick_count"] if rows else None),
        "rhythm_ticks_per_measure": db_int(rows[0]["rhythm_ticks_per_measure"] if rows else None),
        "report_file": db_text(metadata["report_file"]),
        "measure_file": db_text(metadata["measure_file"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return rows, index_row


def build_rows_for_report(
    report_path: Path,
    *,
    measures_dir: Path,
    include_sparse_features: bool,
) -> list[dict[str, Any]]:
    rows, _ = build_rows_and_index_for_report(
        report_path,
        measures_dir=measures_dir,
        include_sparse_features=include_sparse_features,
    )
    return rows


def batched(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    processed_reports = 0
    for report_path in iter_report_paths(args):
        if args.limit_reports is not None and processed_reports >= args.limit_reports:
            break
        rows = build_rows_for_report(
            report_path,
            measures_dir=args.measures_dir,
            include_sparse_features=args.include_sparse_features,
        )
        processed_reports += 1
        for row in rows:
            yield row


def iter_report_payloads(args: argparse.Namespace) -> Iterator[tuple[Path, list[dict[str, Any]], dict[str, Any]]]:
    processed_reports = 0
    for report_path in iter_report_paths(args):
        if args.limit_reports is not None and processed_reports >= args.limit_reports:
            break
        rows, index_row = build_rows_and_index_for_report(
            report_path,
            measures_dir=args.measures_dir,
            include_sparse_features=args.include_sparse_features,
        )
        processed_reports += 1
        yield report_path, rows, index_row


def write_rows_to_lancedb(
    rows: Iterable[dict[str, Any]],
    *,
    db_path: Path,
    table_name: str,
    mode: str,
    batch_size: int,
) -> dict[str, Any]:
    import lancedb

    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))
    table = None
    row_count = 0
    dimensions: set[int] = set()

    for batch in batched(rows, batch_size):
        row_count += len(batch)
        dimensions.update(int(row["embedding_dimension"]) for row in batch)
        if table is None:
            if mode == "append" and table_name in db.table_names():
                table = db.open_table(table_name)
                table.add(batch)
            else:
                table = db.create_table(table_name, data=batch, mode="overwrite")
        else:
            table.add(batch)

    if row_count == 0:
        return {
            "db_path": str(db_path),
            "table_name": table_name,
            "row_count": 0,
            "dimensions": [],
        }

    return {
        "db_path": str(db_path),
        "table_name": table_name,
        "row_count": row_count,
        "dimensions": sorted(dimensions),
    }


def write_payloads_to_lancedb(
    payloads: Iterable[tuple[Path, list[dict[str, Any]], dict[str, Any]]],
    *,
    db_path: Path,
    table_name: str,
    index_table_name: str,
    mode: str,
    batch_size: int,
    write_index_table: bool,
    progress_every: int | None,
    segment_table_name: str = DEFAULT_SEGMENT_TABLE_NAME,
    write_segment_table: bool = True,
) -> dict[str, Any]:
    import lancedb

    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    vector_table = None
    index_table = None
    segment_table = None
    chart_count = 0
    vector_row_count = 0
    segment_row_count = 0
    dimensions: set[int] = set()

    for report_path, rows, index_row in payloads:
        chart_count += 1
        vector_row_count += len(rows)
        dimensions.update(int(row["embedding_dimension"]) for row in rows)
        if progress_every is not None and chart_count % progress_every == 0:
            print(
                f"processed_charts={chart_count} vector_rows={vector_row_count}",
                file=sys.stderr,
                flush=True,
            )

        for batch in batched(rows, batch_size):
            if vector_table is None:
                if mode == "append" and table_name in db.table_names():
                    vector_table = db.open_table(table_name)
                    vector_table.add(batch)
                else:
                    vector_table = db.create_table(table_name, data=batch, mode="overwrite")
            else:
                vector_table.add(batch)

        if write_index_table:
            if index_table is None:
                if mode == "append" and index_table_name in db.table_names():
                    index_table = db.open_table(index_table_name)
                    index_table.add([index_row])
                else:
                    index_table = db.create_table(index_table_name, data=[index_row], mode="overwrite")
            else:
                index_table.add([index_row])

        if write_segment_table:
            segment_rows = segment_rows_from_report(
                read_json(report_path),
                chart_name=str(index_row["chart_name"]),
                report_file=str(report_path),
            )
            segment_row_count += len(segment_rows)
            if segment_table is None:
                if mode == "append" and segment_table_name in db.table_names():
                    segment_table = db.open_table(segment_table_name)
                    segment_table.add(segment_rows)
                else:
                    segment_table = db.create_table(
                        segment_table_name,
                        data=segment_rows,
                        mode="overwrite",
                    )
            else:
                segment_table.add(segment_rows)

    summary: dict[str, Any] = {
        "db_path": str(db_path),
        "table_name": table_name,
        "index_table_name": index_table_name if write_index_table else None,
        "chart_count": chart_count,
        "row_count": vector_row_count,
        "dimensions": sorted(dimensions),
        "segment_table_name": segment_table_name if write_segment_table else None,
        "segment_row_count": segment_row_count,
    }
    if write_index_table:
        summary["index_row_count"] = chart_count
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read segmentation reports, derive four-measure chunks with "
            "export_segment_chunk_ranges.py, embed matching Simai measures, "
            "and store vectors in LanceDB."
        )
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Segmentation report directory. Default: {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument(
        "--measures-dir",
        type=Path,
        default=DEFAULT_MEASURES_DIR,
        help=f"Compiled Simai measure directory. Default: {DEFAULT_MEASURES_DIR}",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Process exactly one segmentation report JSON.",
    )
    parser.add_argument("--song-id", type=int, default=None, help="Process charts for one song id.")
    parser.add_argument("--title", default=None, help="Case-insensitive title substring filter.")
    parser.add_argument("--chart-kind", default=None, help="Chart kind filter, e.g. ST or DX.")
    parser.add_argument("--difficulty-index", type=int, default=None, help="Difficulty index filter.")
    parser.add_argument("--difficulty-name", default=None, help="Difficulty name filter.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"LanceDB directory. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help=f"LanceDB table name. Default: {DEFAULT_TABLE_NAME}",
    )
    parser.add_argument(
        "--index-table",
        default=DEFAULT_INDEX_TABLE_NAME,
        help=f"LanceDB chart index table name. Default: {DEFAULT_INDEX_TABLE_NAME}",
    )
    parser.add_argument(
        "--no-index-table",
        action="store_true",
        help="Do not write the chart-level index table.",
    )
    parser.add_argument(
        "--segment-table",
        default=DEFAULT_SEGMENT_TABLE_NAME,
        help=f"LanceDB segment table name. Default: {DEFAULT_SEGMENT_TABLE_NAME}",
    )
    parser.add_argument(
        "--no-segment-table",
        action="store_true",
        help="Do not write the five-row-per-chart segment table.",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="Overwrite the table or append rows. Default: overwrite.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Rows per LanceDB add batch.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many charts. Use 0 to disable. Default: 100.",
    )
    parser.add_argument(
        "--limit-reports",
        type=int,
        default=None,
        help="Debugging helper: stop after this many report files.",
    )
    parser.add_argument(
        "--include-sparse-features",
        action="store_true",
        help="Store nonzero feature previews as JSON strings. Vectors are always stored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build embeddings and print a summary without writing LanceDB.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.limit_reports is not None and args.limit_reports <= 0:
        raise SystemExit("--limit-reports must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")

    if args.dry_run:
        payloads = list(iter_report_payloads(args))
        rows = [row for _, report_rows, _ in payloads for row in report_rows]
        index_rows = [index_row for _, _, index_row in payloads]
        segment_rows = [
            row
            for report_path, _, index_row in payloads
            for row in segment_rows_from_report(
                read_json(report_path),
                chart_name=str(index_row["chart_name"]),
                report_file=str(report_path),
            )
        ]
        dimensions = sorted({int(row["embedding_dimension"]) for row in rows})
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "chart_count": len(index_rows),
                    "row_count": len(rows),
                    "index_row_count": len(index_rows),
                    "segment_row_count": len(segment_rows),
                    "dimensions": dimensions,
                    "first_key": rows[0]["key"] if rows else None,
                    "last_key": rows[-1]["key"] if rows else None,
                    "first_chart": index_rows[0]["chart_name"] if index_rows else None,
                    "last_chart": index_rows[-1]["chart_name"] if index_rows else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    summary = write_payloads_to_lancedb(
        iter_report_payloads(args),
        db_path=args.db_path,
        table_name=args.table,
        index_table_name=args.index_table,
        mode=args.mode,
        batch_size=args.batch_size,
        write_index_table=not args.no_index_table,
        progress_every=args.progress_every or None,
        segment_table_name=args.segment_table,
        write_segment_table=not args.no_segment_table,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
