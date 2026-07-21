#!/usr/bin/env python3
"""One-shot full rebuild for 512-d SVD Simai pattern embeddings.

The pipeline is:
1. Build the current four-measure structural embeddings.
2. Fit sklearn.decomposition.TruncatedSVD over all raw vectors.
3. Store normalized 512-dimensional vectors back into LanceDB with overwrite.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
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
from scripts.rebuild_full_pattern_embedding_lancedb import verify_lancedb


DEFAULT_COMPONENTS = 512


@dataclass
class PayloadGroup:
    report_path: Path
    rows: list[dict[str, Any]]
    index_row: dict[str, Any]
    start_row: int
    row_count: int


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


def collect_raw_payloads(
    args: argparse.Namespace,
    *,
    total_reports: int,
) -> tuple[list[PayloadGroup], Any, dict[str, Any]]:
    import numpy as np
    from scipy.sparse import csr_matrix

    start_time = time.monotonic()
    processed = 0
    row_count = 0
    raw_dimension: int | None = None
    data_chunks: list[Any] = []
    index_chunks: list[Any] = []
    indptr = [0]
    groups: list[PayloadGroup] = []

    if args.progress_every:
        print(
            f"collect_started total_charts={total_reports}",
            file=sys.stderr,
            flush=True,
        )

    for report_path, rows, index_row in iter_report_payloads(build_payload_args(args)):
        group_start = row_count
        for row in rows:
            vector = row.pop("vector")
            vector_array = np.asarray(vector, dtype=np.float32)
            dimension = int(row["embedding_dimension"])
            if raw_dimension is None:
                raw_dimension = dimension
            elif dimension != raw_dimension:
                raise RuntimeError(
                    f"mixed raw dimensions: expected {raw_dimension}, got {dimension}"
                )

            nonzero = np.flatnonzero(vector_array)
            data_chunks.append(vector_array[nonzero])
            index_chunks.append(nonzero.astype(np.int32, copy=False))
            indptr.append(indptr[-1] + int(nonzero.size))
            row["raw_embedding_dimension"] = dimension
            row_count += 1

        groups.append(
            PayloadGroup(
                report_path=report_path,
                rows=rows,
                index_row=index_row,
                start_row=group_start,
                row_count=len(rows),
            )
        )
        processed += 1

        if args.progress_every and (
            processed % args.progress_every == 0 or processed == total_reports
        ):
            elapsed = time.monotonic() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = max(total_reports - processed, 0)
            eta = remaining / rate if rate > 0 else 0.0
            percent = (processed / total_reports * 100.0) if total_reports else 100.0
            print(
                "collect_progress "
                f"charts={processed}/{total_reports} "
                f"percent={percent:.1f}% "
                f"raw_rows={row_count} "
                f"elapsed={format_duration(elapsed)} "
                f"rate={rate:.2f}_charts/s "
                f"eta={format_duration(eta)}",
                file=sys.stderr,
                flush=True,
            )

    if raw_dimension is None or row_count == 0:
        raise RuntimeError("no embedding rows were generated")

    data = (
        np.concatenate(data_chunks).astype(np.float32, copy=False)
        if data_chunks
        else np.asarray([], dtype=np.float32)
    )
    indices = (
        np.concatenate(index_chunks).astype(np.int32, copy=False)
        if index_chunks
        else np.asarray([], dtype=np.int32)
    )
    matrix = csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(row_count, raw_dimension),
        dtype=np.float32,
    )

    summary = {
        "chart_count": processed,
        "raw_row_count": row_count,
        "raw_embedding_dimension": raw_dimension,
        "raw_nonzero_values": int(data.size),
        "raw_density": float(data.size / (row_count * raw_dimension)),
    }
    return groups, matrix, summary


def fit_transform_svd(
    matrix: Any,
    *,
    components: int,
    algorithm: str,
    n_iter: int,
    random_state: int,
    normalize_output: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    max_components = min(matrix.shape)
    if algorithm == "arpack":
        max_components -= 1
    if components <= 0:
        raise ValueError("--components must be positive")
    if components > max_components:
        raise ValueError(
            f"--components={components} is too high for matrix shape {matrix.shape}; "
            f"max for algorithm={algorithm} is {max_components}"
        )

    print(
        f"svd_started rows={matrix.shape[0]} raw_dim={matrix.shape[1]} "
        f"components={components} algorithm={algorithm}",
        file=sys.stderr,
        flush=True,
    )
    start_time = time.monotonic()
    svd = TruncatedSVD(
        n_components=components,
        algorithm=algorithm,
        n_iter=n_iter,
        random_state=random_state,
    )
    reduced = svd.fit_transform(matrix).astype(np.float32, copy=False)

    if normalize_output:
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        reduced[nonzero] /= norms[nonzero]

    elapsed = time.monotonic() - start_time
    explained = float(np.sum(svd.explained_variance_ratio_))
    print(
        f"svd_done elapsed={format_duration(elapsed)} "
        f"explained_variance_ratio_sum={explained:.6f}",
        file=sys.stderr,
        flush=True,
    )

    summary = {
        "components": components,
        "algorithm": algorithm,
        "n_iter": n_iter,
        "random_state": random_state,
        "normalize_output": normalize_output,
        "explained_variance_ratio_sum": explained,
        "singular_values": [float(value) for value in svd.singular_values_],
    }
    return reduced, svd, summary


def save_svd_model(
    model_path: Path,
    *,
    svd: Any,
    raw_summary: dict[str, Any],
    svd_summary: dict[str, Any],
) -> dict[str, str]:
    from joblib import dump

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "svd": svd,
        "raw_summary": raw_summary,
        "svd_summary": svd_summary,
    }
    dump(payload, model_path)

    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata = {
        "model_path": str(model_path),
        "raw_summary": raw_summary,
        "svd_summary": {
            key: value
            for key, value in svd_summary.items()
            if key != "singular_values"
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"model_path": str(model_path), "metadata_path": str(metadata_path)}


def transformed_payloads(
    groups: Iterable[PayloadGroup],
    reduced: Any,
    *,
    raw_dimension: int,
    output_dimension: int,
    model_path: Path,
    svd_summary: dict[str, Any],
) -> Iterator[tuple[Path, list[dict[str, Any]], dict[str, Any]]]:
    cursor = 0
    explained = float(svd_summary["explained_variance_ratio_sum"])
    normalize_output = bool(svd_summary["normalize_output"])
    for group in groups:
        for row in group.rows:
            row["vector"] = reduced[cursor].astype(float).tolist()
            row["raw_embedding_dimension"] = raw_dimension
            row["embedding_dimension"] = output_dimension
            row["svd_components"] = output_dimension
            row["svd_model_path"] = str(model_path)
            row["svd_normalize_output"] = normalize_output
            row["svd_explained_variance_ratio_sum"] = explained
            cursor += 1

        index_row = dict(group.index_row)
        index_row["raw_embedding_dimension"] = raw_dimension
        index_row["embedding_dimension"] = output_dimension
        index_row["svd_components"] = output_dimension
        index_row["svd_model_path"] = str(model_path)
        index_row["svd_normalize_output"] = normalize_output
        index_row["svd_explained_variance_ratio_sum"] = explained
        yield group.report_path, group.rows, index_row


def verify_vector_dimension(db_path: Path, table_name: str, expected_dimension: int) -> dict[str, Any]:
    import lancedb

    table = lancedb.connect(str(db_path)).open_table(table_name)
    if table.count_rows() == 0:
        raise RuntimeError("vector table has no rows")
    row = table.to_arrow().select(["vector", "embedding_dimension"]).slice(0, 1).to_pylist()[0]
    actual_dimension = len(row["vector"])
    stored_dimension = int(row["embedding_dimension"])
    if actual_dimension != expected_dimension or stored_dimension != expected_dimension:
        raise RuntimeError(
            f"expected {expected_dimension}-d vectors, got len(vector)={actual_dimension}, "
            f"embedding_dimension={stored_dimension}"
        )
    return {
        "sample_vector_dimension": actual_dimension,
        "sample_embedding_dimension": stored_dimension,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overwrite the Simai pattern LanceDB with TruncatedSVD-compressed "
            "512-dimensional vectors."
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
        "--components",
        type=int,
        default=DEFAULT_COMPONENTS,
        help=f"TruncatedSVD output dimensions. Default: {DEFAULT_COMPONENTS}.",
    )
    parser.add_argument(
        "--svd-algorithm",
        choices=("randomized", "arpack"),
        default="randomized",
        help="TruncatedSVD algorithm. Default: randomized.",
    )
    parser.add_argument(
        "--svd-n-iter",
        type=int,
        default=7,
        help="Power iterations for randomized SVD. Default: 7.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=20260707,
        help="Random seed for TruncatedSVD. Default: 20260707.",
    )
    parser.add_argument(
        "--no-normalize-svd-output",
        action="store_true",
        help="Store raw SVD projections instead of L2-normalized 512-d vectors.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Where to save the fitted joblib SVD model. Default: <db-path>/<table>_svd512.joblib.",
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

    model_path = args.model_output
    if model_path is None:
        model_path = args.db_path / f"{args.table}_svd{args.components}.joblib"

    total_reports = count_reports(args.reports_dir, args.limit_reports)
    groups, raw_matrix, raw_summary = collect_raw_payloads(args, total_reports=total_reports)
    reduced, svd, svd_summary = fit_transform_svd(
        raw_matrix,
        components=args.components,
        algorithm=args.svd_algorithm,
        n_iter=args.svd_n_iter,
        random_state=args.random_state,
        normalize_output=not args.no_normalize_svd_output,
    )
    model_summary = save_svd_model(
        model_path,
        svd=svd,
        raw_summary=raw_summary,
        svd_summary=svd_summary,
    )

    write_summary = write_payloads_to_lancedb(
        transformed_payloads(
            groups,
            reduced,
            raw_dimension=int(raw_summary["raw_embedding_dimension"]),
            output_dimension=args.components,
            model_path=model_path,
            svd_summary=svd_summary,
        ),
        db_path=args.db_path,
        table_name=args.table,
        index_table_name=args.index_table,
        mode="overwrite",
        batch_size=args.batch_size,
        write_index_table=True,
        progress_every=args.progress_every or None,
    )

    result: dict[str, Any] = {
        "raw": raw_summary,
        "svd": {
            key: value
            for key, value in svd_summary.items()
            if key != "singular_values"
        },
        "model": model_summary,
        "rebuild": write_summary,
    }
    if not args.skip_verify:
        result["verify"] = verify_lancedb(args.db_path, args.table, args.index_table)
        result["verify"].update(
            verify_vector_dimension(args.db_path, args.table, args.components)
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
