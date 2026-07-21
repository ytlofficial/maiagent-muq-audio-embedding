#!/usr/bin/env python3
"""Count songs whose chart text contains musical dividers greater than a threshold."""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DIVIDER_RE = re.compile(r"\{(?!#)(\d+(?:\.\d+)?)\}")


@dataclass(frozen=True)
class MatchRow:
    song_id: int
    title: str
    difficulty_name: str
    level: str | None
    max_divider: float
    dividers: tuple[float, ...]


def find_dividers(chart_content: str, threshold: float) -> tuple[float, ...]:
    values = {
        float(match.group(1))
        for match in DIVIDER_RE.finditer(chart_content)
        if float(match.group(1)) > threshold
    }
    return tuple(sorted(values))


def query_matches(db_path: Path, threshold: float) -> list[MatchRow]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                s.song_id,
                s.title,
                c.difficulty_name,
                c.level,
                c.chart_content
            FROM charts c
            JOIN songs s USING(song_id)
            WHERE c.has_chart = 1
              AND TRIM(c.chart_content) <> ''
            ORDER BY s.title, c.difficulty_index
            """
        ).fetchall()
    finally:
        conn.close()

    matches: list[MatchRow] = []
    for row in rows:
        dividers = find_dividers(row["chart_content"], threshold)
        if not dividers:
            continue
        matches.append(
            MatchRow(
                song_id=row["song_id"],
                title=row["title"],
                difficulty_name=row["difficulty_name"],
                level=row["level"],
                max_divider=max(dividers),
                dividers=dividers,
            )
        )
    return matches


def format_divider(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count songs/charts containing Simai {divider} values above a threshold."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("maimai_charts.db"),
        help="SQLite database path. Default: maimai_charts.db",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=384.0,
        help="Count dividers strictly greater than this value. Default: 384.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of sample matching charts to print. Default: 20.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    matches = query_matches(args.db, args.threshold)
    song_ids = {match.song_id for match in matches}

    print(f"threshold: > {format_divider(args.threshold)}")
    print(f"matching_songs: {len(song_ids)}")
    print(f"matching_charts: {len(matches)}")
    print()
    print("samples:")
    for match in matches[: args.limit]:
        dividers = ", ".join(format_divider(value) for value in match.dividers)
        print(
            f"- {match.title} [{match.difficulty_name}] "
            f"Lv.{match.level or '-'} max={format_divider(match.max_divider)} "
            f"dividers={dividers}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
