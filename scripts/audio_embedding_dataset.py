#!/usr/bin/env python3
"""Build audio-to-chart training records from the existing LanceDB tables."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCORE_NAMES = ("note", "peak", "charge", "slide", "handtrip", "tricky")
PATTERN_TABLE_NAME = "simai_pattern_chunks"
AUDIO_TABLE_NAME = "simai_audio_chunks"
SEGMENT_TABLE_NAME = "simai_segments"
TEACHER_DIMENSION = 512


@dataclass(frozen=True)
class AudioEmbeddingRecord:
    key: str
    chart_id: str
    song_id: int
    difficulty: int
    difficulty_id: int
    level_value: float
    segment_id: int
    segment_key: str
    audio_file: str
    teacher_vector: np.ndarray
    numeric_conditions: np.ndarray


def load_split_rows(csv_path: Path) -> dict[str, dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"split CSV is empty: {csv_path}")

    required = {"chart_id", "song_id", "difficulty", "level_value"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"split CSV is missing columns: {sorted(missing)}")

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        chart_id = str(row["chart_id"])
        if chart_id in result:
            raise ValueError(f"duplicate chart_id in split CSV: {chart_id}")
        difficulty = int(row["difficulty"])
        if difficulty not in (5, 6):
            raise ValueError(f"unsupported difficulty {difficulty} for {chart_id}")
        result[chart_id] = {
            **row,
            "song_id": int(row["song_id"]),
            "difficulty": difficulty,
            "level_value": float(row["level_value"]),
        }
    return result


def read_lance_columns(table: Any, columns: Sequence[str]) -> Any:
    """Read only needed columns while retaining every table row."""
    return table.search().select(list(columns)).limit(None).to_arrow()


def fixed_size_vectors(column: Any, dimension: int) -> np.ndarray:
    array = column.combine_chunks()
    if len(array) == 0:
        return np.empty((0, dimension), dtype=np.float32)
    values = array.values.to_numpy(zero_copy_only=False)
    vectors = np.asarray(values, dtype=np.float32).reshape(len(array), dimension)
    return vectors


def audio_path_candidates(raw_path: str, data_root: Path) -> list[Path]:
    raw = Path(raw_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(data_root / raw)

    normalized = raw_path.replace("\\", "/")
    for marker in ("outputs/audio_chunks/", "audio_chunks/"):
        marker_index = normalized.find(marker)
        if marker_index < 0:
            continue
        suffix = Path(normalized[marker_index:])
        candidates.append(data_root / suffix)
        if marker.startswith("outputs/"):
            candidates.append(data_root / Path(str(suffix)[len("outputs/") :]))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_audio_path(raw_path: str, data_root: Path, *, require_exists: bool) -> Path:
    candidates = audio_path_candidates(raw_path, data_root)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    if require_exists:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"audio chunk is missing; checked: {checked}")
    return candidates[-1]


def numeric_conditions(level_value: float, scores: Sequence[float]) -> np.ndarray:
    if len(scores) != len(SCORE_NAMES):
        raise ValueError(f"expected {len(SCORE_NAMES)} segment scores, got {len(scores)}")
    # Levels top out around 15 and segment scores use a nominal 0-200 scale.
    values = [max(0.0, min(level_value / 15.0, 1.5))]
    values.extend(max(0.0, min(float(score) / 200.0, 2.0)) for score in scores)
    return np.asarray(values, dtype=np.float32)


def normalize_teacher_vector(vector: np.ndarray, key: str) -> np.ndarray:
    if vector.shape != (TEACHER_DIMENSION,):
        raise ValueError(f"teacher vector for {key} has shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"teacher vector for {key} contains NaN or infinity")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"teacher vector for {key} has invalid norm {norm}")
    return np.asarray(vector / norm, dtype=np.float32)


def load_audio_embedding_records(
    *,
    db_path: Path,
    split_csv: Path,
    data_root: Path,
    pattern_table_name: str = PATTERN_TABLE_NAME,
    audio_table_name: str = AUDIO_TABLE_NAME,
    segment_table_name: str = SEGMENT_TABLE_NAME,
    require_audio: bool = True,
) -> list[AudioEmbeddingRecord]:
    import lancedb

    split_rows = load_split_rows(split_csv)
    selected_chart_ids = set(split_rows)
    db = lancedb.connect(str(db_path))

    pattern_table = read_lance_columns(
        db.open_table(pattern_table_name),
        ("key", "chart_id", "song_id", "segment_id", "segment_key", "vector"),
    )
    pattern_vectors = fixed_size_vectors(pattern_table.column("vector"), TEACHER_DIMENSION)
    pattern_columns = {
        name: pattern_table.column(name).to_pylist()
        for name in ("key", "chart_id", "song_id", "segment_id", "segment_key")
    }

    patterns: dict[str, dict[str, Any]] = {}
    charts_with_patterns: set[str] = set()
    for row_index, chart_id_value in enumerate(pattern_columns["chart_id"]):
        chart_id = str(chart_id_value)
        if chart_id not in selected_chart_ids:
            continue
        key = str(pattern_columns["key"][row_index])
        if key in patterns:
            raise ValueError(f"duplicate pattern chunk key: {key}")
        patterns[key] = {
            "chart_id": chart_id,
            "song_id": int(pattern_columns["song_id"][row_index]),
            "segment_id": int(pattern_columns["segment_id"][row_index]),
            "segment_key": str(pattern_columns["segment_key"][row_index]),
            "vector": normalize_teacher_vector(pattern_vectors[row_index], key),
        }
        charts_with_patterns.add(chart_id)

    missing_charts = selected_chart_ids - charts_with_patterns
    if missing_charts:
        preview = ", ".join(sorted(missing_charts)[:10])
        raise ValueError(f"{len(missing_charts)} selected charts have no pattern chunks: {preview}")

    audio_table = read_lance_columns(
        db.open_table(audio_table_name),
        ("key", "chart_id", "audio_file"),
    )
    audio_by_key: dict[str, str] = {}
    for row in audio_table.to_pylist():
        if str(row["chart_id"]) not in selected_chart_ids:
            continue
        key = str(row["key"])
        if key in audio_by_key:
            raise ValueError(f"duplicate audio chunk key: {key}")
        audio_by_key[key] = str(row["audio_file"])

    segment_table = read_lance_columns(
        db.open_table(segment_table_name),
        ("key", "chart_id", *SCORE_NAMES),
    )
    scores_by_segment: dict[str, tuple[float, ...]] = {}
    for row in segment_table.to_pylist():
        if str(row["chart_id"]) not in selected_chart_ids:
            continue
        key = str(row["key"])
        scores_by_segment[key] = tuple(float(row[name]) for name in SCORE_NAMES)

    pattern_keys = set(patterns)
    audio_keys = set(audio_by_key)
    if pattern_keys != audio_keys:
        missing_audio = sorted(pattern_keys - audio_keys)
        extra_audio = sorted(audio_keys - pattern_keys)
        raise ValueError(
            "pattern/audio key mismatch: "
            f"missing_audio={len(missing_audio)} extra_audio={len(extra_audio)} "
            f"examples={missing_audio[:3] or extra_audio[:3]}"
        )

    records: list[AudioEmbeddingRecord] = []
    for key in sorted(patterns):
        pattern = patterns[key]
        chart = split_rows[pattern["chart_id"]]
        if pattern["song_id"] != chart["song_id"]:
            raise ValueError(f"song_id mismatch for {key}")
        segment_id = int(pattern["segment_id"])
        if segment_id not in range(5):
            raise ValueError(f"invalid segment_id {segment_id} for {key}")
        segment_key = str(pattern["segment_key"])
        if segment_key not in scores_by_segment:
            raise ValueError(f"missing segment row {segment_key} for {key}")
        audio_file = resolve_audio_path(
            audio_by_key[key],
            data_root,
            require_exists=require_audio,
        )
        difficulty = int(chart["difficulty"])
        records.append(
            AudioEmbeddingRecord(
                key=key,
                chart_id=pattern["chart_id"],
                song_id=int(chart["song_id"]),
                difficulty=difficulty,
                difficulty_id=difficulty - 5,
                level_value=float(chart["level_value"]),
                segment_id=segment_id,
                segment_key=segment_key,
                audio_file=str(audio_file),
                teacher_vector=pattern["vector"],
                numeric_conditions=numeric_conditions(
                    float(chart["level_value"]),
                    scores_by_segment[segment_key],
                ),
            )
        )
    return records


def limit_records(
    records: Sequence[AudioEmbeddingRecord],
    maximum: int,
    *,
    seed: int,
) -> list[AudioEmbeddingRecord]:
    if maximum <= 0 or maximum >= len(records):
        return list(records)
    rng = random.Random(seed)
    selected = rng.sample(list(records), maximum)
    return sorted(selected, key=lambda record: record.key)


def dataset_summary(records: Iterable[AudioEmbeddingRecord]) -> dict[str, Any]:
    materialized = list(records)
    chart_ids = {record.chart_id for record in materialized}
    song_ids = {record.song_id for record in materialized}
    return {
        "record_count": len(materialized),
        "chart_count": len(chart_ids),
        "song_count": len(song_ids),
        "difficulty_counts": dict(
            sorted(Counter(record.difficulty for record in materialized).items())
        ),
        "segment_counts": dict(
            sorted(Counter(record.segment_id for record in materialized).items())
        ),
        "minimum_level": min((record.level_value for record in materialized), default=None),
        "maximum_level": max((record.level_value for record in materialized), default=None),
    }


def print_dataset_summary(name: str, records: Sequence[AudioEmbeddingRecord]) -> None:
    print(json.dumps({name: dataset_summary(records)}, ensure_ascii=False, indent=2))
