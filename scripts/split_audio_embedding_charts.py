#!/usr/bin/env python3
"""Select 1600 charts and create song-disjoint 1000/300/300 splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_segment_chunk_lancedb import DEFAULT_DB_PATH, DEFAULT_INDEX_TABLE_NAME


DEFAULT_SPLIT_SIZES = {"train": 1000, "validation": 300, "test": 300}
DEFAULT_OUT_PREFIX = REPO_ROOT / "datasets" / "audio_embedding_charts_1000_300_300"
DEFAULT_SEED = 20260715

CSV_FIELDS = (
    "split",
    "selection_reason",
    "chart_id",
    "chart_name",
    "song_id",
    "title",
    "artist",
    "chart_kind",
    "chart_version",
    "difficulty",
    "difficulty_name",
    "level",
    "level_value",
    "segment_count",
    "chunk_count",
    "first_chunk_key",
    "last_chunk_key",
)


def version_number(version: Any) -> int:
    text = str(version or "").strip()
    prefix = text.split(".", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(f"version does not begin with a number: {version!r}") from exc


def load_chart_rows(db_path: Path, table_name: str) -> list[dict[str, Any]]:
    import lancedb

    columns = [field for field in CSV_FIELDS if field not in ("split", "selection_reason")]
    table = lancedb.connect(str(db_path)).open_table(table_name)
    rows = table.to_arrow().select(columns).to_pylist()
    return sorted(
        rows,
        key=lambda row: (
            int(row["song_id"]),
            str(row["chart_kind"]),
            int(row["difficulty"]),
        ),
    )


def select_charts(
    rows: list[dict[str, Any]],
    *,
    selected_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if selected_size <= 0 or selected_size > len(rows):
        raise ValueError("selected_size must be between 1 and the source row count")
    exclude_count = len(rows) - selected_size
    low_level = [row for row in rows if float(row["level_value"]) < 13.0]
    if len(low_level) < exclude_count:
        raise ValueError(
            f"need {exclude_count} exclusions but only {len(low_level)} charts are below level 13"
        )

    rng = random.Random(seed)
    tie_breakers = {str(row["chart_id"]): rng.random() for row in rows}
    exclusion_order = sorted(
        low_level,
        key=lambda row: (
            version_number(row["chart_version"]),
            float(row["level_value"]),
            tie_breakers[str(row["chart_id"])],
        ),
    )
    excluded_ids = {str(row["chart_id"]) for row in exclusion_order[:exclude_count]}

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        if str(row["chart_id"]) in excluded_ids:
            row["split"] = "excluded"
            row["selection_reason"] = "early_version_below_level_13"
            excluded.append(row)
        else:
            row["selection_reason"] = "selected"
            selected.append(row)
    return selected, excluded


def category_functions() -> dict[str, Callable[[dict[str, Any]], str]]:
    return {
        "version": lambda row: str(row["chart_version"]),
        "difficulty": lambda row: str(row["difficulty_name"]),
        "level": lambda row: str(row["level"]),
        "version_difficulty": lambda row: (
            f"{row['chart_version']}|{row['difficulty_name']}"
        ),
        "chart_kind": lambda row: str(row["chart_kind"]),
    }


def assign_song_disjoint_splits(
    rows: list[dict[str, Any]],
    *,
    split_sizes: dict[str, int],
    seed: int,
    time_limit_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    total_size = sum(split_sizes.values())
    if total_size != len(rows):
        raise ValueError(f"split sizes sum to {total_size}, but selected rows contain {len(rows)}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["song_id"])].append(row)
    song_ids = sorted(grouped)
    splits = list(split_sizes)
    categories = category_functions()

    category_totals = {
        feature: Counter(function(row) for row in rows)
        for feature, function in categories.items()
    }
    group_category_counts = {
        song_id: {
            feature: Counter(function(row) for row in grouped[song_id])
            for feature, function in categories.items()
        }
        for song_id in song_ids
    }

    costs: list[float] = []
    integrality: list[int] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_variable(cost: float, *, integer: bool, upper: float) -> int:
        index = len(costs)
        costs.append(cost)
        integrality.append(1 if integer else 0)
        lower_bounds.append(0.0)
        upper_bounds.append(upper)
        return index

    rng = random.Random(seed)
    assignment_variables: dict[tuple[int, str], int] = {}
    for song_id in song_ids:
        for split in splits:
            assignment_variables[(song_id, split)] = add_variable(
                rng.random() * 1e-8,
                integer=True,
                upper=1.0,
            )

    constraint_rows: list[dict[int, float]] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(coefficients: dict[int, float], target: float) -> None:
        constraint_rows.append(coefficients)
        constraint_lower.append(target)
        constraint_upper.append(target)

    for song_id in song_ids:
        add_constraint(
            {assignment_variables[(song_id, split)]: 1.0 for split in splits},
            1.0,
        )

    for split, target_size in split_sizes.items():
        add_constraint(
            {
                assignment_variables[(song_id, split)]: float(len(grouped[song_id]))
                for song_id in song_ids
            },
            float(target_size),
        )

    feature_weights = {
        "version": 8.0,
        "difficulty": 12.0,
        "level": 6.0,
        "version_difficulty": 3.0,
        "chart_kind": 2.0,
    }
    for feature, total_counts in category_totals.items():
        for category, category_total in sorted(total_counts.items()):
            slack_cost = feature_weights[feature] / max(float(category_total), 10.0)
            for split, split_size in split_sizes.items():
                positive_slack = add_variable(slack_cost, integer=False, upper=np.inf)
                negative_slack = add_variable(slack_cost, integer=False, upper=np.inf)
                coefficients = {
                    assignment_variables[(song_id, split)]: float(
                        group_category_counts[song_id][feature][category]
                    )
                    for song_id in song_ids
                    if group_category_counts[song_id][feature][category]
                }
                coefficients[positive_slack] = -1.0
                coefficients[negative_slack] = 1.0
                target = category_total * split_size / total_size
                add_constraint(coefficients, float(target))

    total_chunks = sum(int(row["chunk_count"]) for row in rows)
    for split, split_size in split_sizes.items():
        positive_slack = add_variable(4.0 / total_chunks, integer=False, upper=np.inf)
        negative_slack = add_variable(4.0 / total_chunks, integer=False, upper=np.inf)
        coefficients = {
            assignment_variables[(song_id, split)]: float(
                sum(int(row["chunk_count"]) for row in grouped[song_id])
            )
            for song_id in song_ids
        }
        coefficients[positive_slack] = -1.0
        coefficients[negative_slack] = 1.0
        add_constraint(coefficients, total_chunks * split_size / total_size)

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    for row_index, coefficients in enumerate(constraint_rows):
        for column_index, value in coefficients.items():
            matrix_rows.append(row_index)
            matrix_columns.append(column_index)
            matrix_values.append(value)
    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(constraint_rows), len(costs)),
    ).tocsr()

    result = milp(
        c=np.asarray(costs, dtype=np.float64),
        integrality=np.asarray(integrality, dtype=np.int8),
        bounds=Bounds(
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=np.float64),
            np.asarray(constraint_upper, dtype=np.float64),
        ),
        options={
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if result.x is None:
        raise RuntimeError(f"split optimizer failed: {result.message}")

    song_split: dict[int, str] = {}
    for song_id in song_ids:
        values = {
            split: float(result.x[assignment_variables[(song_id, split)]])
            for split in splits
        }
        selected_split = max(values, key=values.get)
        if values[selected_split] < 0.5:
            raise RuntimeError(f"optimizer returned fractional assignment for song {song_id}")
        song_split[song_id] = selected_split

    assigned: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        row["split"] = song_split[int(row["song_id"])]
        assigned.append(row)
    return assigned


def validate_assignment(
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    *,
    source_count: int,
    split_sizes: dict[str, int],
) -> None:
    if len(selected) + len(excluded) != source_count:
        raise RuntimeError("selected and excluded rows do not cover the source charts")
    counts = Counter(str(row["split"]) for row in selected)
    if counts != Counter(split_sizes):
        raise RuntimeError(f"split counts do not match targets: {counts}")
    songs_by_split = {
        split: {int(row["song_id"]) for row in selected if row["split"] == split}
        for split in split_sizes
    }
    for left_index, left in enumerate(split_sizes):
        for right in list(split_sizes)[left_index + 1 :]:
            overlap = songs_by_split[left] & songs_by_split[right]
            if overlap:
                raise RuntimeError(f"song leakage between {left} and {right}: {len(overlap)}")
    if any(float(row["level_value"]) >= 13.0 for row in excluded):
        raise RuntimeError("exclusion set contains charts at level 13 or above")


def count_values(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def percentage_point_spreads(
    rows: list[dict[str, Any]],
    split_sizes: dict[str, int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature, function in category_functions().items():
        if feature == "version_difficulty":
            continue
        categories = sorted({function(row) for row in rows})
        spreads: dict[str, float] = {}
        for category in categories:
            percentages = [
                100.0
                * sum(
                    1
                    for row in rows
                    if row["split"] == split and function(row) == category
                )
                / split_sizes[split]
                for split in split_sizes
            ]
            spreads[category] = max(percentages) - min(percentages)
        output[feature] = {
            "max_percentage_point_spread": max(spreads.values(), default=0.0),
            "by_category": dict(sorted(spreads.items())),
        }
    return output


def build_summary(
    source_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    *,
    split_sizes: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split in split_sizes:
        rows = [row for row in selected if row["split"] == split]
        split_summary[split] = {
            "charts": len(rows),
            "songs": len({int(row["song_id"]) for row in rows}),
            "chunks": sum(int(row["chunk_count"]) for row in rows),
            "versions": count_values(rows, "chart_version"),
            "difficulties": count_values(rows, "difficulty_name"),
            "levels": count_values(rows, "level"),
            "chart_kinds": count_values(rows, "chart_kind"),
        }
    return {
        "seed": seed,
        "source_charts": len(source_rows),
        "selected_charts": len(selected),
        "excluded_charts": len(excluded),
        "selection_policy": (
            "Exclude charts below level 13 in ascending chart-version order; "
            "within a version, lower levels are excluded first."
        ),
        "song_disjoint": True,
        "selected_songs": len({int(row["song_id"]) for row in selected}),
        "selected_below_level_13": sum(
            float(row["level_value"]) < 13.0 for row in selected
        ),
        "excluded_below_level_13": sum(
            float(row["level_value"]) < 13.0 for row in excluded
        ),
        "excluded_versions": count_values(excluded, "chart_version"),
        "excluded_levels": count_values(excluded, "level"),
        "splits": split_summary,
        "distribution_spreads": percentage_point_spreads(selected, split_sizes),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item.get("split", "")),
                version_number(item["chart_version"]),
                int(item["song_id"]),
                str(item["chart_kind"]),
                int(item["difficulty"]),
            ),
        ):
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}{suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--table", default=DEFAULT_INDEX_TABLE_NAME)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--optimizer-time-limit", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    split_sizes = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    if any(size <= 0 for size in split_sizes.values()):
        raise SystemExit("all split sizes must be positive")

    source_rows = load_chart_rows(args.db_path, args.table)
    selected, excluded = select_charts(
        source_rows,
        selected_size=sum(split_sizes.values()),
        seed=args.seed,
    )
    assigned = assign_song_disjoint_splits(
        selected,
        split_sizes=split_sizes,
        seed=args.seed,
        time_limit_seconds=args.optimizer_time_limit,
    )
    validate_assignment(
        assigned,
        excluded,
        source_count=len(source_rows),
        split_sizes=split_sizes,
    )
    summary = build_summary(
        source_rows,
        assigned,
        excluded,
        split_sizes=split_sizes,
        seed=args.seed,
    )

    write_csv(output_path(args.out_prefix, ".csv"), assigned)
    for split in split_sizes:
        write_csv(
            output_path(args.out_prefix, f"_{split}.csv"),
            [row for row in assigned if row["split"] == split],
        )
    write_csv(output_path(args.out_prefix, "_excluded.csv"), excluded)
    summary_path = output_path(args.out_prefix, "_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written_prefix={args.out_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
