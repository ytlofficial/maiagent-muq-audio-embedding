#!/usr/bin/env python3
"""Add chart/segment metadata to an existing Simai LanceDB without re-embedding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_segment_chunk_lancedb import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_TABLE_NAME,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SEGMENT_TABLE_NAME,
    DEFAULT_TABLE_NAME,
    read_json,
    row_key,
)
from scripts.export_segment_chunk_ranges import chunk_specs_from_report
from scripts.simai_dataset_metadata import (
    EXPECTED_SEGMENT_COUNT,
    chart_id_for,
    level_value,
    segment_key_for,
    segment_rows_from_report,
)


DEFAULT_AUDIO_TABLE_NAME = "simai_audio_chunks"
DEFAULT_AUDIO_INDEX_TABLE_NAME = "simai_audio_chunk_index"


def collect_metadata(
    reports_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chart_updates: list[dict[str, Any]] = []
    chunk_updates: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for report_path in sorted(reports_dir.glob("*.json")):
        report = read_json(report_path)
        metadata = report.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"report metadata is not an object: {report_path}")
        chart_name = report_path.stem
        chart_id = chart_id_for(
            metadata.get("song_id"),
            metadata.get("chart_kind"),
            metadata.get("difficulty_index"),
        )
        difficulty = int(metadata["difficulty_index"])
        numeric_level = level_value(metadata.get("level"))
        if numeric_level is None:
            raise ValueError(f"invalid level {metadata.get('level')!r}: {report_path}")
        chart_updates.append(
            {
                "chart_name": chart_name,
                "chart_id": chart_id,
                "difficulty": difficulty,
                "level_value": numeric_level,
            }
        )
        segment_rows.extend(
            segment_rows_from_report(
                report,
                chart_name=chart_name,
                report_file=str(report_path),
            )
        )
        for spec in chunk_specs_from_report(report):
            chunk_updates.append(
                {
                    "key": row_key(chart_name, spec.start_measure, spec.end_measure),
                    "chart_id": chart_id,
                    "segment_id": spec.segment_id,
                    "segment_key": segment_key_for(chart_id, spec.segment_id),
                }
            )

    return chart_updates, chunk_updates, segment_rows


def add_missing_columns(table: Any, fields: list[Any]) -> list[str]:
    missing = [field for field in fields if field.name not in table.schema.names]
    if missing:
        table.add_columns(missing)
    return [field.name for field in missing]


def merge_updates(table: Any, on: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    import pyarrow as pa

    result = (
        table.merge_insert(on)
        .when_matched_update_all()
        .execute(pa.Table.from_pylist(rows))
    )
    return {
        "updated": int(result.num_updated_rows),
        "inserted": int(result.num_inserted_rows),
        "deleted": int(result.num_deleted_rows),
    }


def migrate(
    *,
    db_path: Path,
    reports_dir: Path,
    chart_tables: Iterable[str],
    chunk_tables: Iterable[str],
    segment_table_name: str,
) -> dict[str, Any]:
    import lancedb
    import pyarrow as pa

    chart_updates, chunk_updates, segment_rows = collect_metadata(reports_dir)
    db = lancedb.connect(str(db_path))
    table_names = set(db.table_names())
    summary: dict[str, Any] = {
        "db_path": str(db_path),
        "charts": len(chart_updates),
        "chunks": len(chunk_updates),
        "segments": len(segment_rows),
        "tables": {},
    }

    chart_fields = [
        pa.field("chart_id", pa.string()),
        pa.field("difficulty", pa.int64()),
        pa.field("level_value", pa.float64()),
    ]
    for table_name in chart_tables:
        if table_name not in table_names:
            continue
        table = db.open_table(table_name)
        added = add_missing_columns(table, chart_fields)
        merge = merge_updates(table, "chart_name", chart_updates)
        summary["tables"][table_name] = {"added_columns": added, **merge}

    chunk_fields = [
        pa.field("chart_id", pa.string()),
        pa.field("segment_id", pa.int64()),
        pa.field("segment_key", pa.string()),
    ]
    for table_name in chunk_tables:
        if table_name not in table_names:
            continue
        table = db.open_table(table_name)
        added = add_missing_columns(table, chunk_fields)
        merge = merge_updates(table, "key", chunk_updates)
        summary["tables"][table_name] = {"added_columns": added, **merge}

    db.create_table(segment_table_name, data=segment_rows, mode="overwrite")
    summary["tables"][segment_table_name] = {"rows": len(segment_rows)}
    return summary


def verify(
    *,
    db_path: Path,
    chart_table_name: str,
    chunk_table_names: Iterable[str],
    segment_table_name: str,
) -> dict[str, Any]:
    import lancedb

    db = lancedb.connect(str(db_path))
    chart_table = db.open_table(chart_table_name)
    chart_rows = chart_table.to_arrow().select(
        ["chart_id", "difficulty", "level", "level_value"]
    ).to_pylist()
    chart_ids = {str(row["chart_id"]) for row in chart_rows}
    if len(chart_ids) != len(chart_rows):
        raise RuntimeError("chart_id is not unique in chart table")
    if any(int(row["difficulty"]) not in (5, 6) for row in chart_rows):
        raise RuntimeError("chart difficulty contains values outside 5/6")

    segment_table = db.open_table(segment_table_name)
    segment_rows = segment_table.to_arrow().select(
        [
            "key",
            "chart_id",
            "segment_id",
            "start_measure",
            "end_measure",
            "note",
            "peak",
            "charge",
            "slide",
            "handtrip",
            "tricky",
            "score_vector",
        ]
    ).to_pylist()
    segment_keys = {str(row["key"]) for row in segment_rows}
    if len(segment_rows) != len(chart_rows) * EXPECTED_SEGMENT_COUNT:
        raise RuntimeError("segment table does not contain exactly five rows per chart")
    if {int(row["segment_id"]) for row in segment_rows} != set(range(5)):
        raise RuntimeError("segment_id values are not exactly 0-4")
    score_names = ("note", "peak", "charge", "slide", "handtrip", "tricky")
    score_errors = sum(
        1
        for row in segment_rows
        if len(row["score_vector"]) != len(score_names)
        or any(not 0.0 <= float(row[name]) <= 200.0 for name in score_names)
        or any(
            abs(float(row[name]) - float(row["score_vector"][index])) > 1e-4
            for index, name in enumerate(score_names)
        )
    )
    if score_errors:
        raise RuntimeError(f"segment table has {score_errors} invalid score vectors")
    segment_ranges = {
        (str(row["chart_id"]), int(row["segment_id"])): (
            int(row["start_measure"]),
            int(row["end_measure"]),
        )
        for row in segment_rows
    }

    chunks: dict[str, Any] = {}
    chunk_mappings: dict[str, dict[str, tuple[str, int, str]]] = {}
    for table_name in chunk_table_names:
        if table_name not in db.table_names():
            continue
        table = db.open_table(table_name)
        rows = table.to_arrow().select(
            [
                "key",
                "chart_id",
                "segment_id",
                "segment_key",
                "body_start_measure",
                "body_end_measure",
            ]
        ).to_pylist()
        bad_fk = sum(
            1
            for row in rows
            if str(row["chart_id"]) not in chart_ids
            or str(row["segment_key"]) not in segment_keys
            or int(row["segment_id"]) not in range(5)
        )
        if bad_fk:
            raise RuntimeError(f"{table_name} has {bad_fk} invalid chart/segment links")
        bad_body_ranges = sum(
            1
            for row in rows
            if not (
                segment_ranges[(str(row["chart_id"]), int(row["segment_id"]))][0]
                <= int(row["body_start_measure"])
                <= int(row["body_end_measure"])
                <= segment_ranges[(str(row["chart_id"]), int(row["segment_id"]))][1]
            )
        )
        if bad_body_ranges:
            raise RuntimeError(
                f"{table_name} has {bad_body_ranges} chunks outside their segment body range"
            )
        chunk_mappings[table_name] = {
            str(row["key"]): (
                str(row["chart_id"]),
                int(row["segment_id"]),
                str(row["segment_key"]),
            )
            for row in rows
        }
        chunks[table_name] = {
            "rows": len(rows),
            "invalid_links": bad_fk,
            "invalid_body_ranges": bad_body_ranges,
        }

    mapping_values = list(chunk_mappings.values())
    chunk_mapping_mismatches = 0
    if len(mapping_values) > 1:
        baseline = mapping_values[0]
        for mapping in mapping_values[1:]:
            all_keys = set(baseline) | set(mapping)
            chunk_mapping_mismatches += sum(
                baseline.get(key) != mapping.get(key) for key in all_keys
            )
    if chunk_mapping_mismatches:
        raise RuntimeError(
            f"chunk tables have {chunk_mapping_mismatches} chart/segment mapping mismatches"
        )

    plus_examples = sorted(
        {
            (str(row["level"]), float(row["level_value"]))
            for row in chart_rows
            if str(row["level"]).endswith("+")
        }
    )
    return {
        "chart_rows": len(chart_rows),
        "unique_chart_ids": len(chart_ids),
        "segment_rows": len(segment_rows),
        "unique_segment_keys": len(segment_keys),
        "score_errors": score_errors,
        "plus_level_values": plus_examples,
        "chunk_mapping_mismatches": chunk_mapping_mismatches,
        "chunks": chunks,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--chart-table", default=DEFAULT_INDEX_TABLE_NAME)
    parser.add_argument("--audio-chart-table", default=DEFAULT_AUDIO_INDEX_TABLE_NAME)
    parser.add_argument("--chunk-table", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--audio-chunk-table", default=DEFAULT_AUDIO_TABLE_NAME)
    parser.add_argument("--segment-table", default=DEFAULT_SEGMENT_TABLE_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    chart_updates, chunk_updates, segment_rows = collect_metadata(args.reports_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "charts": len(chart_updates),
                    "chunks": len(chunk_updates),
                    "segments": len(segment_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    migration = migrate(
        db_path=args.db_path,
        reports_dir=args.reports_dir,
        chart_tables=(args.chart_table, args.audio_chart_table),
        chunk_tables=(args.chunk_table, args.audio_chunk_table),
        segment_table_name=args.segment_table,
    )
    verification = verify(
        db_path=args.db_path,
        chart_table_name=args.chart_table,
        chunk_table_names=(args.chunk_table, args.audio_chunk_table),
        segment_table_name=args.segment_table,
    )
    print(
        json.dumps(
            {"migration": migration, "verification": verification},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
