#!/usr/bin/env python3
"""Select a balanced 300-chart dataset from maimai_charts.db."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path


SPLIT_SIZES = {
    "train": 200,
    "validate": 50,
    "test": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select chart dataset splits.")
    parser.add_argument("--db", default="maimai_charts.db", type=Path)
    parser.add_argument("--out", default="datasets/chart_dataset_300.csv", type=Path)
    parser.add_argument("--seed", default=20260630, type=int)
    return parser.parse_args()


def charter_bucket(row: sqlite3.Row) -> str:
    if row["charter_category"] == "多人合作谱":
        return "多人合作谱"
    names = json.loads(row["charter_main_names"] or "[]")
    return names[0] if names else "未署名"


def load_rows(db_path: Path) -> list[dict[str, object]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT
            s.song_id,
            s.title,
            s.artist,
            s.version AS song_version,
            COALESCE(s.genre, '未标注') AS genre,
            c.chart_kind,
            c.chart_version,
            c.difficulty_index,
            c.difficulty_name,
            c.level,
            c.charter,
            c.charter_category,
            c.charter_main_names,
            c.content_hash,
            c.created_from_file
        FROM charts c
        JOIN songs s USING (song_id)
        ORDER BY s.song_id, c.chart_kind, c.difficulty_index
        """
    ).fetchall()
    result: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["chart_id"] = f"{row['song_id']}:{row['chart_kind']}:{row['difficulty_index']}"
        item["charter_bucket"] = charter_bucket(row)
        result.append(item)
    con.close()
    return result


def coverage_key(row: dict[str, object], kind: str) -> str:
    if kind == "version":
        return str(row["chart_version"])
    if kind == "genre":
        return str(row["genre"])
    if kind == "charter":
        return str(row["charter_bucket"])
    raise ValueError(kind)


def target_kind_counts(rows: list[dict[str, object]], total: int) -> dict[str, int]:
    counts = Counter(str(row["chart_kind"]) for row in rows)
    dx_target = round(total * counts["DX"] / len(rows))
    return {"DX": dx_target, "ST": total - dx_target}


def select_rows(rows: list[dict[str, object]], total: int, rng: random.Random) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    target_kinds = target_kind_counts(rows, total)
    kind_counts: Counter[str] = Counter()

    all_values = {
        kind: {coverage_key(row, kind) for row in rows}
        for kind in ("version", "genre", "charter")
    }
    covered = {kind: set() for kind in all_values}

    def add(row: dict[str, object]) -> None:
        chart_id = str(row["chart_id"])
        selected.append(row)
        selected_ids.add(chart_id)
        kind_counts[str(row["chart_kind"])] += 1
        for kind in covered:
            covered[kind].add(coverage_key(row, kind))

    while any(covered[kind] != all_values[kind] for kind in all_values):
        best_rows: list[dict[str, object]] = []
        best_score: tuple[float, float] | None = None
        for row in rows:
            if row["chart_id"] in selected_ids:
                continue
            new_cover = sum(
                1
                for kind in all_values
                if coverage_key(row, kind) not in covered[kind]
            )
            if not new_cover:
                continue
            chart_kind = str(row["chart_kind"])
            kind_deficit = target_kinds[chart_kind] - kind_counts[chart_kind]
            score = (float(new_cover), float(kind_deficit))
            if best_score is None or score > best_score:
                best_score = score
                best_rows = [row]
            elif score == best_score:
                best_rows.append(row)
        if not best_rows:
            raise RuntimeError("Could not satisfy mandatory coverage.")
        add(rng.choice(best_rows))

    version_total = Counter(str(row["chart_version"]) for row in rows)
    genre_total = Counter(str(row["genre"]) for row in rows)
    charter_total = Counter(str(row["charter_bucket"]) for row in rows)
    selected_version = Counter(str(row["chart_version"]) for row in selected)
    selected_genre = Counter(str(row["genre"]) for row in selected)
    selected_charter = Counter(str(row["charter_bucket"]) for row in selected)

    while len(selected) < total:
        best_rows = []
        best_score = None
        for row in rows:
            if row["chart_id"] in selected_ids:
                continue
            chart_kind = str(row["chart_kind"])
            version = str(row["chart_version"])
            genre = str(row["genre"])
            charter = str(row["charter_bucket"])
            kind_ratio = kind_counts[chart_kind] / max(target_kinds[chart_kind], 1)
            version_ratio = selected_version[version] / max(version_total[version], 1)
            genre_ratio = selected_genre[genre] / max(genre_total[genre], 1)
            charter_ratio = selected_charter[charter] / max(charter_total[charter], 1)
            score = -(
                2.0 * kind_ratio
                + 1.0 * version_ratio
                + 1.0 * genre_ratio
                + 0.75 * charter_ratio
            )
            if best_score is None or score > best_score:
                best_score = score
                best_rows = [row]
            elif score == best_score:
                best_rows.append(row)
        row = rng.choice(best_rows)
        add(row)
        selected_version[str(row["chart_version"])] += 1
        selected_genre[str(row["genre"])] += 1
        selected_charter[str(row["charter_bucket"])] += 1

    return selected


def assign_splits(rows: list[dict[str, object]], rng: random.Random) -> list[dict[str, object]]:
    rows = rows[:]
    split_counts = Counter()
    kind_targets = {
        split: target_kind_counts(rows, size)
        for split, size in SPLIT_SIZES.items()
    }
    split_kind_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_SIZES}
    split_version_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_SIZES}
    split_genre_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_SIZES}

    def assign(row: dict[str, object], split: str) -> None:
        row["split"] = split
        split_counts[split] += 1
        split_kind_counts[split][str(row["chart_kind"])] += 1
        split_version_counts[split][str(row["chart_version"])] += 1
        split_genre_counts[split][str(row["genre"])] += 1

    for kind in ("version", "genre"):
        values = sorted({coverage_key(row, kind) for row in rows})
        for value in values:
            candidates = [row for row in rows if "split" not in row and coverage_key(row, kind) == value]
            if not candidates:
                continue
            for split in SPLIT_SIZES:
                if split_counts[split] >= SPLIT_SIZES[split]:
                    continue
                candidates = [row for row in candidates if "split" not in row]
                if not candidates:
                    break
                assign(rng.choice(candidates), split)

    while any(split_counts[split] < size for split, size in SPLIT_SIZES.items()):
        unassigned = [row for row in rows if "split" not in row]
        best_choice = None
        best_score = None
        for row in unassigned:
            for split, size in SPLIT_SIZES.items():
                if split_counts[split] >= size:
                    continue
                chart_kind = str(row["chart_kind"])
                kind_deficit = kind_targets[split][chart_kind] - split_kind_counts[split][chart_kind]
                version_new = 1 if split_version_counts[split][str(row["chart_version"])] == 0 else 0
                genre_new = 1 if split_genre_counts[split][str(row["genre"])] == 0 else 0
                fill_deficit = size - split_counts[split]
                score = (kind_deficit, version_new, genre_new, fill_deficit)
                if best_score is None or score > best_score:
                    best_score = score
                    best_choice = (row, split)
        if best_choice is None:
            raise RuntimeError("Could not assign splits.")
        assign(*best_choice)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "chart_id",
        "song_id",
        "title",
        "artist",
        "difficulty_index",
        "difficulty_name",
        "level",
        "chart_kind",
        "chart_version",
        "song_version",
        "charter",
        "charter_category",
        "charter_bucket",
        "charter_main_names",
        "content_hash",
        "created_from_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r["split"]), str(r["chart_version"]), int(r["song_id"]), str(r["chart_kind"]), int(r["difficulty_index"]))):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_split_csvs(path: Path, rows: list[dict[str, object]]) -> None:
    for split in SPLIT_SIZES:
        split_path = path.with_name(f"{path.stem}_{split}{path.suffix}")
        write_csv(split_path, [row for row in rows if row["split"] == split])


def print_summary(rows: list[dict[str, object]], all_rows: list[dict[str, object]]) -> None:
    print(f"selected={len(rows)}")
    print(f"all_versions={len({row['chart_version'] for row in all_rows})}")
    print(f"selected_versions={len({row['chart_version'] for row in rows})}")
    print(f"all_genres={len({row['genre'] for row in all_rows})}")
    print(f"selected_genres={len({row['genre'] for row in rows})}")
    print(f"all_charter_buckets={len({row['charter_bucket'] for row in all_rows})}")
    print(f"selected_charter_buckets={len({row['charter_bucket'] for row in rows})}")
    print("split_counts")
    for split in SPLIT_SIZES:
        split_rows = [row for row in rows if row["split"] == split]
        print(f"{split}|{len(split_rows)}|DX={sum(row['chart_kind']=='DX' for row in split_rows)}|ST={sum(row['chart_kind']=='ST' for row in split_rows)}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    all_rows = load_rows(args.db)
    selected = select_rows(all_rows, 300, rng)
    assigned = assign_splits(selected, rng)
    write_csv(args.out, assigned)
    write_split_csvs(args.out, assigned)
    print_summary(assigned, all_rows)
    print(f"written={args.out}")


if __name__ == "__main__":
    main()
