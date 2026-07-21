#!/usr/bin/env python3
"""One-shot full rebuild for the Simai pattern embedding LanceDB.

This script overwrites the previous vector table and chart-index table, embeds
all segmentation chunks with the current embedding algorithm, and verifies that
the rebuilt LanceDB is internally consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_segment_chunk_lancedb import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_TABLE_NAME,
    DEFAULT_MEASURES_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_TABLE_NAME,
    iter_report_payloads,
    write_payloads_to_lancedb,
)


def build_payload_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        reports_dir=args.reports_dir,
        measures_dir=args.measures_dir,
        report_file=None,
        song_id=None,
        title=None,
        chart_kind=None,
        difficulty_index=None,
        difficulty_name=None,
        limit_reports=args.limit_reports,
        include_sparse_features=args.include_sparse_features,
    )


def count_reports(reports_dir: Path, limit_reports: int | None) -> int:
    count = len(list(reports_dir.glob("*.json")))
    if limit_reports is not None:
        return min(count, limit_reports)
    return count


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def with_progress(
    payloads: Iterable[tuple[Path, list[dict[str, Any]], dict[str, Any]]],
    *,
    total_reports: int,
    progress_every: int | None,
) -> Iterator[tuple[Path, list[dict[str, Any]], dict[str, Any]]]:
    start_time = time.monotonic()
    processed = 0
    vector_rows = 0

    if progress_every is not None:
        print(
            f"rebuild_started total_charts={total_reports}",
            file=sys.stderr,
            flush=True,
        )

    for payload in payloads:
        _report_path, rows, _index_row = payload
        yield payload
        processed += 1
        vector_rows += len(rows)

        if progress_every is None:
            continue
        if processed % progress_every != 0 and processed != total_reports:
            continue

        elapsed = time.monotonic() - start_time
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(total_reports - processed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        percent = (processed / total_reports * 100.0) if total_reports else 100.0
        print(
            "progress "
            f"charts={processed}/{total_reports} "
            f"percent={percent:.1f}% "
            f"vector_rows={vector_rows} "
            f"elapsed={format_duration(elapsed)} "
            f"rate={rate:.2f}_charts/s "
            f"eta={format_duration(eta)}",
            file=sys.stderr,
            flush=True,
        )


def verify_lancedb(db_path: Path, table_name: str, index_table_name: str) -> dict[str, Any]:
    import lancedb

    db = lancedb.connect(str(db_path))
    vector_table = db.open_table(table_name)
    index_table = db.open_table(index_table_name)

    vector_rows = vector_table.count_rows()
    index_rows = index_table.count_rows()
    index_data = index_table.to_arrow().select(
        ["chart_name", "chunk_count", "embedding_dimension"]
    ).to_pylist()
    chunk_count_sum = sum(int(row["chunk_count"]) for row in index_data)
    dimensions = sorted({int(row["embedding_dimension"]) for row in index_data})

    if vector_rows != chunk_count_sum:
        raise RuntimeError(
            f"vector row count mismatch: vector_rows={vector_rows}, "
            f"index chunk_count sum={chunk_count_sum}"
        )
    if index_rows != len(index_data):
        raise RuntimeError(
            f"index row count mismatch: index_rows={index_rows}, scanned={len(index_data)}"
        )
    if not dimensions:
        raise RuntimeError("rebuilt index has no embedding dimensions")

    return {
        "db_path": str(db_path),
        "table_name": table_name,
        "index_table_name": index_table_name,
        "vector_rows": vector_rows,
        "index_rows": index_rows,
        "chunk_count_sum": chunk_count_sum,
        "embedding_dimensions": dimensions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overwrite and fully rebuild the Simai pattern embedding LanceDB "
            "from segmentation reports and compiled Simai measures."
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
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"LanceDB directory to overwrite. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help=f"Vector table name. Default: {DEFAULT_TABLE_NAME}",
    )
    parser.add_argument(
        "--index-table",
        default=DEFAULT_INDEX_TABLE_NAME,
        help=f"Chart index table name. Default: {DEFAULT_INDEX_TABLE_NAME}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Rows per LanceDB add batch. Default: 256.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many charts. Use 0 to disable. Default: 100.",
    )
    parser.add_argument(
        "--include-sparse-features",
        action="store_true",
        help="Store sparse nonzero feature previews as JSON strings.",
    )
    parser.add_argument(
        "--limit-reports",
        type=int,
        default=None,
        help="Debug helper: rebuild only the first N reports.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the final LanceDB consistency check.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    if args.limit_reports is not None and args.limit_reports <= 0:
        raise SystemExit("--limit-reports must be positive")

    payload_args = build_payload_args(args)
    total_reports = count_reports(args.reports_dir, args.limit_reports)
    payloads = with_progress(
        iter_report_payloads(payload_args),
        total_reports=total_reports,
        progress_every=args.progress_every or None,
    )
    summary = write_payloads_to_lancedb(
        payloads,
        db_path=args.db_path,
        table_name=args.table,
        index_table_name=args.index_table,
        mode="overwrite",
        batch_size=args.batch_size,
        write_index_table=True,
        progress_every=None,
    )

    result: dict[str, Any] = {"rebuild": summary}
    if not args.skip_verify:
        result["verify"] = verify_lancedb(args.db_path, args.table, args.index_table)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
