#!/usr/bin/env python3
"""Export compiled Simai measure segments for every chart in the database."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_measure_compiler import (
    SimaiTimelineError,
    compile_chart,
    first_seconds_for,
    measure_to_dict,
    parse_maidata,
)


SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ChartRow:
    song_id: int
    title: str
    artist: str | None
    bpm: str | None
    genre: str | None
    cabinet: str | None
    version: str | None
    chart_kind: str
    chart_version: str
    difficulty_index: int
    difficulty_name: str
    level: str | None
    charter: str | None
    chart_content: str
    created_from_file: str


def safe_filename(value: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "chart"


def query_charts(db_path: Path) -> list[ChartRow]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                s.song_id,
                s.title,
                s.artist,
                s.bpm,
                s.genre,
                s.cabinet,
                s.version,
                c.chart_kind,
                c.chart_version,
                c.difficulty_index,
                c.difficulty_name,
                c.level,
                c.charter,
                c.chart_content,
                c.created_from_file
            FROM charts c
            JOIN songs s USING(song_id)
            WHERE c.has_chart = 1
              AND TRIM(c.chart_content) <> ''
            ORDER BY s.song_id, c.chart_kind, c.difficulty_index
            """
        ).fetchall()
    finally:
        conn.close()

    return [ChartRow(**dict(row)) for row in rows]


def chart_first_seconds(row: ChartRow) -> tuple[float, str]:
    maidata_path = Path(row.created_from_file)
    if not maidata_path.exists():
        return 0.0, "default_0_missing_maidata"

    try:
        fields = parse_maidata(maidata_path)
    except OSError:
        return 0.0, "default_0_unreadable_maidata"

    difficulty = str(row.difficulty_index)
    first = first_seconds_for(fields, difficulty)
    if fields.get(f"first_{difficulty}") or fields.get("first"):
        return first, "maidata_first"
    return first, "default_0_no_first"


def chart_output_name(row: ChartRow) -> str:
    base = (
        f"{row.song_id:05d}_"
        f"{safe_filename(row.chart_kind)}_"
        f"{row.difficulty_index}_"
        f"{safe_filename(row.difficulty_name)}"
    )
    return f"{base}.json"


def build_chart_export(row: ChartRow) -> dict[str, Any]:
    first_seconds, first_source = chart_first_seconds(row)
    compiled = compile_chart(row.chart_content, first_seconds=first_seconds)

    return {
        "song": {
            "song_id": row.song_id,
            "title": row.title,
            "artist": row.artist,
            "bpm": row.bpm,
            "genre": row.genre,
            "cabinet": row.cabinet,
            "version": row.version,
        },
        "chart": {
            "chart_kind": row.chart_kind,
            "chart_version": row.chart_version,
            "difficulty_index": row.difficulty_index,
            "difficulty_name": row.difficulty_name,
            "level": row.level,
            "charter": row.charter,
            "created_from_file": row.created_from_file,
        },
        "timeline": {
            "first_seconds": round(first_seconds, 6),
            "first_seconds_source": first_source,
            "measure_beats": compiled.measure_beats,
            "slot_count": len(compiled.slots),
            "measure_count": len(compiled.measures),
            "duration_seconds": round(
                compiled.slots[-1].end_seconds - compiled.first_seconds
                if compiled.slots
                else 0.0,
                6,
            ),
            "start_seconds": round(compiled.first_seconds, 6),
            "end_seconds": round(compiled.slots[-1].end_seconds, 6)
            if compiled.slots
            else round(compiled.first_seconds, 6),
        },
        "normalization_notes": compiled.normalization_notes,
        "measures": [
            measure_to_dict(measure, include_slots=False, include_raw=True)
            for measure in compiled.measures
        ],
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export measure-level Simai chart timelines for every chart in the database."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("maimai_charts.db"),
        help="SQLite chart database. Default: maimai_charts.db",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Fixed output directory. Default: outputs/simai_measures",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = args.out
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    rows = query_charts(args.db)
    index: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.db),
        "output_directory": str(out_dir),
        "chart_count": 0,
        "error_count": 0,
        "charts": [],
        "errors": [],
    }

    for row in rows:
        output_name = chart_output_name(row)
        output_path = charts_dir / output_name
        try:
            chart_export = build_chart_export(row)
        except (SimaiTimelineError, ValueError, OSError) as exc:
            index["error_count"] += 1
            index["errors"].append(
                {
                    "song_id": row.song_id,
                    "title": row.title,
                    "chart_kind": row.chart_kind,
                    "difficulty_index": row.difficulty_index,
                    "difficulty_name": row.difficulty_name,
                    "error": str(exc),
                }
            )
            continue

        write_json(output_path, chart_export)
        index["chart_count"] += 1
        index["charts"].append(
            {
                "song_id": row.song_id,
                "title": row.title,
                "chart_kind": row.chart_kind,
                "difficulty_index": row.difficulty_index,
                "difficulty_name": row.difficulty_name,
                "level": row.level,
                "file": str(output_path),
                "measure_count": chart_export["timeline"]["measure_count"],
                "start_seconds": chart_export["timeline"]["start_seconds"],
                "end_seconds": chart_export["timeline"]["end_seconds"],
                "duration_seconds": chart_export["timeline"]["duration_seconds"],
            }
        )

    write_json(out_dir / "index.json", index)
    print(f"exported_charts: {index['chart_count']}")
    print(f"errors: {index['error_count']}")
    print(f"output: {out_dir}")
    return 1 if index["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
