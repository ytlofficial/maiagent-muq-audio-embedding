#!/usr/bin/env python3
"""Build a LanceDB table of audio chunks aligned to Simai segment chunks.

The chunk ranges are produced by ``export_segment_chunk_ranges.py`` so the row
keys match ``build_segment_chunk_lancedb.py`` exactly:
``{chart_name}:{start_measure}-{end_measure}``.

Audio is cut from ``chartdata-rebuilt/<song>/track.mp3`` by default and written
to ``outputs/audio_chunks/simai_audio_chunks``. If ``track.mp3`` is missing but
``track.ogg`` exists, the source is first converted into a cached MP3 and then
chunked. The LanceDB rows store the same chunk index plus audio file metadata;
use ``--store-audio-bytes`` when the table should also contain the encoded
audio bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_segment_chunk_lancedb import (
    DEFAULT_DB_PATH,
    DEFAULT_MEASURES_DIR,
    batched,
    chart_metadata,
    compact_segments,
    db_float,
    db_int,
    db_text,
    matches_selector,
    measure_path_for_report,
    read_json,
    row_key,
)
from scripts.export_segment_chunk_ranges import chunk_specs_from_report
from scripts.simai_dataset_metadata import chart_id_for, level_value, segment_key_for


DEFAULT_RANGES_DIR = REPO_ROOT / "outputs" / "segmentation_ranges" / "charts"
DEFAULT_CHARTDATA_ROOT = REPO_ROOT / "chartdata-rebuilt"
DEFAULT_AUDIO_OUT_DIR = REPO_ROOT / "outputs" / "audio_chunks" / "simai_audio_chunks"
DEFAULT_TABLE_NAME = "simai_audio_chunks"
DEFAULT_INDEX_TABLE_NAME = "simai_audio_chunk_index"
DEFAULT_AUDIO_FILENAME = "track.mp3"
DEFAULT_FALLBACK_AUDIO_FILENAMES = ("track.ogg",)
DEFAULT_AUDIO_SUFFIX = ".mp3"
DEFAULT_CONVERTED_SOURCE_DIR = REPO_ROOT / "outputs" / "audio_chunks" / "converted_sources"
CHARTDATA_MARKERS = ("chartdata-rebuilt", "chartdata", "chartdata--old")


@dataclass(frozen=True)
class PreparedSourceAudio:
    original_file: Path
    working_file: Path
    original_format: str
    working_format: str
    was_converted: bool


class SkippedSourceAudioFormat(RuntimeError):
    """Raised when a chart's resolved source audio format is intentionally skipped."""


def selector_is_active(args: argparse.Namespace) -> bool:
    return any(
        selector is not None
        for selector in (
            args.song_id,
            args.title,
            args.chart_kind,
            args.difficulty_index,
            args.difficulty_name,
        )
    )


def iter_range_paths(args: argparse.Namespace) -> Iterator[Path]:
    if args.range_file is not None:
        yield args.range_file
        return

    for range_path in sorted(args.ranges_dir.glob("*.json")):
        if selector_is_active(args):
            report = read_json(range_path)
            if not matches_selector(
                report,
                song_id=args.song_id,
                title=args.title,
                chart_kind=args.chart_kind,
                difficulty_index=args.difficulty_index,
                difficulty_name=args.difficulty_name,
            ):
                continue
        yield range_path


def created_from_file_for(report: dict[str, Any], measures: dict[str, Any]) -> str:
    chart = measures.get("chart", {}) if isinstance(measures.get("chart"), dict) else {}
    value = chart.get("created_from_file")
    if value:
        return str(value)

    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    value = metadata.get("created_from_file")
    if value:
        return str(value)

    raise ValueError("missing chart.created_from_file/metadata.created_from_file")


def relative_song_dir_from_created_file(created_from_file: str) -> Path:
    path = Path(created_from_file)
    parts = path.parts
    for marker in CHARTDATA_MARKERS:
        if marker in parts:
            marker_index = parts.index(marker)
            relative_parts = parts[marker_index + 1 : -1]
            return Path(*relative_parts) if relative_parts else Path()
    return path.parent


def resolve_source_audio_file(
    created_from_file: str,
    *,
    chartdata_root: Path,
    audio_filename: str,
) -> Path:
    return chartdata_root / relative_song_dir_from_created_file(created_from_file) / audio_filename


def source_audio_candidates(
    created_from_file: str,
    *,
    chartdata_root: Path,
    audio_filename: str,
    fallback_audio_filenames: Iterable[str],
) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for filename in [audio_filename, *fallback_audio_filenames]:
        path = resolve_source_audio_file(
            created_from_file,
            chartdata_root=chartdata_root,
            audio_filename=filename,
        )
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def resolve_existing_source_audio_file(
    created_from_file: str,
    *,
    chartdata_root: Path,
    audio_filename: str,
    fallback_audio_filenames: Iterable[str],
) -> Path:
    candidates = source_audio_candidates(
        created_from_file,
        chartdata_root=chartdata_root,
        audio_filename=audio_filename,
        fallback_audio_filenames=fallback_audio_filenames,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"missing source audio. Tried: {joined}")


def source_audio_format(path: Path) -> str:
    return path.suffix.removeprefix(".").casefold()


def normalize_audio_formats(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    formats: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            normalized = item.strip().removeprefix(".").casefold()
            if normalized:
                formats.add(normalized)
    return formats


def chunk_time_range(
    measures_data: dict[str, Any],
    start_measure: int,
    end_measure: int,
) -> tuple[float, float]:
    measures = measures_data.get("measures", [])
    if not isinstance(measures, list):
        raise ValueError("measure JSON does not contain a measures list")

    by_index: dict[int, dict[str, Any]] = {}
    for measure in measures:
        if isinstance(measure, dict) and measure.get("index") is not None:
            by_index[int(measure["index"])] = measure

    start = by_index.get(start_measure)
    end = by_index.get(end_measure)
    if start is None or end is None:
        raise ValueError(f"missing measure time range for {start_measure}:{end_measure}")

    return float(start["start_seconds"]), float(end["end_seconds"])


def chunk_audio_path(
    audio_out_dir: Path,
    chart_name: str,
    chunk_index: int,
    start_measure: int,
    end_measure: int,
    *,
    suffix: str,
) -> Path:
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    filename = f"{chunk_index:03d}_{start_measure:04d}-{end_measure:04d}{suffix}"
    return audio_out_dir / chart_name / filename


def resolve_ffmpeg(ffmpeg: str) -> str:
    if Path(ffmpeg).name != ffmpeg:
        if Path(ffmpeg).exists():
            return ffmpeg
        raise FileNotFoundError(f"ffmpeg executable does not exist: {ffmpeg}")

    resolved = shutil.which(ffmpeg)
    if resolved is not None:
        return resolved

    local_venv_ffmpeg = REPO_ROOT / ".venv" / "bin" / ffmpeg
    if local_venv_ffmpeg.exists():
        return str(local_venv_ffmpeg)

    raise FileNotFoundError(
        f"missing ffmpeg executable: {ffmpeg}. Install ffmpeg or pass --ffmpeg /path/to/ffmpeg."
    )


def extract_audio_chunk(
    source_audio_file: Path,
    audio_file: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    ffmpeg: str,
    mp3_quality: int,
    reuse_existing_audio: bool,
) -> bool:
    if reuse_existing_audio and audio_file.exists() and audio_file.stat().st_size > 0:
        return False

    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError(
            f"invalid audio duration for {audio_file}: start={start_seconds}, end={end_seconds}"
        )

    audio_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(source_audio_file),
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        str(mp3_quality),
        str(audio_file),
    ]
    subprocess.run(command, check=True)

    if not audio_file.exists() or audio_file.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a non-empty audio chunk: {audio_file}")
    return True


def converted_source_audio_path(
    converted_source_dir: Path,
    created_from_file: str,
    original_audio_file: Path,
) -> Path:
    song_dir = relative_song_dir_from_created_file(created_from_file)
    stem = original_audio_file.stem or "track"
    return converted_source_dir / song_dir / f"{stem}.mp3"


def convert_source_audio_to_mp3(
    source_audio_file: Path,
    converted_audio_file: Path,
    *,
    ffmpeg: str,
    mp3_quality: int,
    reuse_existing_audio: bool,
) -> bool:
    if reuse_existing_audio and converted_audio_file.exists() and converted_audio_file.stat().st_size > 0:
        return False

    converted_audio_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_audio_file),
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        str(mp3_quality),
        str(converted_audio_file),
    ]
    subprocess.run(command, check=True)

    if not converted_audio_file.exists() or converted_audio_file.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a non-empty converted source: {converted_audio_file}")
    return True


def prepare_source_audio(
    source_audio_file: Path,
    *,
    created_from_file: str,
    converted_source_dir: Path,
    ffmpeg: str | None,
    mp3_quality: int,
    reuse_existing_audio: bool,
    convert_audio: bool,
) -> tuple[PreparedSourceAudio, bool]:
    original_format = source_audio_format(source_audio_file)
    if original_format == "mp3":
        return (
            PreparedSourceAudio(
                original_file=source_audio_file,
                working_file=source_audio_file,
                original_format=original_format,
                working_format="mp3",
                was_converted=False,
            ),
            False,
        )

    converted_audio_file = converted_source_audio_path(
        converted_source_dir,
        created_from_file,
        source_audio_file,
    )
    converted = False
    if convert_audio:
        if ffmpeg is None:
            raise ValueError("ffmpeg is required when converting source audio")
        converted = convert_source_audio_to_mp3(
            source_audio_file,
            converted_audio_file,
            ffmpeg=ffmpeg,
            mp3_quality=mp3_quality,
            reuse_existing_audio=reuse_existing_audio,
        )

    return (
        PreparedSourceAudio(
            original_file=source_audio_file,
            working_file=converted_audio_file,
            original_format=original_format,
            working_format="mp3",
            was_converted=True,
        ),
        converted,
    )


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_file_metadata(path: Path) -> tuple[int, str]:
    if not path.exists():
        return 0, ""
    return path.stat().st_size, sha1_file(path)


def build_audio_rows_and_index_for_report(
    range_path: Path,
    *,
    measures_dir: Path,
    chartdata_root: Path,
    audio_filename: str,
    fallback_audio_filenames: Iterable[str],
    audio_out_dir: Path,
    audio_suffix: str,
    converted_source_dir: Path,
    extract_audio: bool,
    ffmpeg: str | None,
    mp3_quality: int,
    reuse_existing_audio: bool,
    store_audio_bytes: bool,
    only_source_audio_formats: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = read_json(range_path)
    measures_path = measure_path_for_report(range_path, measures_dir)
    if not measures_path.exists():
        raise FileNotFoundError(f"missing measure file for {range_path.name}: {measures_path}")

    measures_data = read_json(measures_path)
    metadata = chart_metadata(range_path, report, measures_data, measures_path)
    chart_name = str(metadata["chart_name"])
    chart_id = chart_id_for(
        metadata["song_id"], metadata["chart_kind"], metadata["difficulty_index"]
    )
    created_from_file = created_from_file_for(report, measures_data)
    original_source_audio_file = resolve_existing_source_audio_file(
        created_from_file,
        chartdata_root=chartdata_root,
        audio_filename=audio_filename,
        fallback_audio_filenames=fallback_audio_filenames,
    )
    original_source_audio_format = source_audio_format(original_source_audio_file)
    if only_source_audio_formats and original_source_audio_format not in only_source_audio_formats:
        allowed = ",".join(sorted(only_source_audio_formats))
        raise SkippedSourceAudioFormat(
            f"source format {original_source_audio_format} not in {allowed}: {original_source_audio_file}"
        )
    prepared_source, converted_source_count = prepare_source_audio(
        original_source_audio_file,
        created_from_file=created_from_file,
        converted_source_dir=converted_source_dir,
        ffmpeg=ffmpeg,
        mp3_quality=mp3_quality,
        reuse_existing_audio=reuse_existing_audio,
        convert_audio=extract_audio,
    )
    source_audio_file = prepared_source.working_file
    if extract_audio and not source_audio_file.exists():
        raise FileNotFoundError(f"missing converted source audio for {chart_name}: {source_audio_file}")

    chunk_specs = chunk_specs_from_report(report)
    rows: list[dict[str, Any]] = []
    extracted_count = 0

    for chunk_index, chunk_spec in enumerate(chunk_specs, start=1):
        start_measure = chunk_spec.start_measure
        end_measure = chunk_spec.end_measure
        audio_start_seconds, audio_end_seconds = chunk_time_range(
            measures_data,
            start_measure,
            end_measure,
        )
        audio_file = chunk_audio_path(
            audio_out_dir,
            chart_name,
            chunk_index,
            start_measure,
            end_measure,
            suffix=audio_suffix,
        )

        if extract_audio:
            if ffmpeg is None:
                raise ValueError("ffmpeg is required when extract_audio=True")
            if extract_audio_chunk(
                source_audio_file,
                audio_file,
                start_seconds=audio_start_seconds,
                end_seconds=audio_end_seconds,
                ffmpeg=ffmpeg,
                mp3_quality=mp3_quality,
                reuse_existing_audio=reuse_existing_audio,
            ):
                extracted_count += 1

        audio_size_bytes, audio_sha1 = audio_file_metadata(audio_file)
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
            "overlap_measure_slots_json": json.dumps(
                chunk_spec.overlap_measure_slots,
                separators=(",", ":"),
            ),
            "overlap_measures_json": json.dumps(
                chunk_spec.overlap_measure_ranges,
                separators=(",", ":"),
            ),
            "audio_start_seconds": audio_start_seconds,
            "audio_end_seconds": audio_end_seconds,
            "audio_duration_seconds": audio_end_seconds - audio_start_seconds,
            "source_audio_file": str(source_audio_file),
            "original_source_audio_file": str(prepared_source.original_file),
            "source_audio_original_format": prepared_source.original_format,
            "source_audio_working_format": prepared_source.working_format,
            "source_audio_was_converted": prepared_source.was_converted,
            "audio_file": str(audio_file),
            "audio_format": audio_suffix.removeprefix("."),
            "audio_size_bytes": audio_size_bytes,
            "audio_sha1": audio_sha1,
            "created_from_file": created_from_file,
            "report_file": db_text(metadata["report_file"]),
            "measure_file": db_text(metadata["measure_file"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if store_audio_bytes:
            row["audio_bytes"] = audio_file.read_bytes() if audio_file.exists() else b""
        rows.append(row)

    plan = report.get("plan", {}) if isinstance(report.get("plan"), dict) else {}
    row_keys = [str(row["key"]) for row in rows]
    audio_files = [str(row["audio_file"]) for row in rows]
    ranges = [spec.range for spec in chunk_specs]
    chunk_body_ranges = [[spec.body_start_measure, spec.body_end_measure] for spec in chunk_specs]
    chunk_overlap_slots = [spec.overlap_measure_slots for spec in chunk_specs]
    chunk_overlap_measures = [spec.overlap_measure_ranges for spec in chunk_specs]

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
        "chunk_body_ranges_json": json.dumps(
            chunk_body_ranges,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
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
        "chunk_keys_json": json.dumps(row_keys, ensure_ascii=False, separators=(",", ":")),
        "audio_files_json": json.dumps(audio_files, ensure_ascii=False, separators=(",", ":")),
        "first_chunk_key": row_keys[0] if row_keys else "",
        "last_chunk_key": row_keys[-1] if row_keys else "",
        "source_audio_file": str(source_audio_file),
        "original_source_audio_file": str(prepared_source.original_file),
        "source_audio_original_format": prepared_source.original_format,
        "source_audio_working_format": prepared_source.working_format,
        "source_audio_was_converted": prepared_source.was_converted,
        "converted_source_audio_count": int(converted_source_count),
        "audio_out_dir": str(audio_out_dir / chart_name),
        "created_from_file": created_from_file,
        "extracted_chunk_count": extracted_count,
        "report_file": db_text(metadata["report_file"]),
        "measure_file": db_text(metadata["measure_file"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return rows, index_row


def iter_audio_report_payloads(
    args: argparse.Namespace,
    *,
    extract_audio: bool,
    ffmpeg: str | None,
) -> Iterator[tuple[Path, list[dict[str, Any]], dict[str, Any]]]:
    processed_reports = 0
    only_source_audio_formats = normalize_audio_formats(args.only_source_audio_format)
    for range_path in iter_range_paths(args):
        if args.limit_reports is not None and processed_reports >= args.limit_reports:
            break
        try:
            rows, index_row = build_audio_rows_and_index_for_report(
                range_path,
                measures_dir=args.measures_dir,
                chartdata_root=args.chartdata_root,
                audio_filename=args.audio_filename,
                fallback_audio_filenames=args.fallback_audio_filename,
                audio_out_dir=args.audio_out_dir,
                audio_suffix=args.audio_suffix,
                converted_source_dir=args.converted_source_dir,
                extract_audio=extract_audio,
                ffmpeg=ffmpeg,
                mp3_quality=args.mp3_quality,
                reuse_existing_audio=args.reuse_existing_audio,
                store_audio_bytes=args.store_audio_bytes,
                only_source_audio_formats=only_source_audio_formats,
            )
        except SkippedSourceAudioFormat as exc:
            if args.log_source_format_skips:
                print(f"skip_source_audio_format {range_path.name}: {exc}", file=sys.stderr, flush=True)
            continue
        except FileNotFoundError as exc:
            if not args.skip_missing_audio:
                raise
            print(f"skip_missing_audio {range_path.name}: {exc}", file=sys.stderr, flush=True)
            continue

        processed_reports += 1
        yield range_path, rows, index_row


def write_audio_payloads_to_lancedb(
    payloads: Iterable[tuple[Path, list[dict[str, Any]], dict[str, Any]]],
    *,
    db_path: Path,
    table_name: str,
    index_table_name: str,
    mode: str,
    batch_size: int,
    write_index_table: bool,
    progress_every: int | None,
    store_audio_bytes: bool,
) -> dict[str, Any]:
    import lancedb

    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    audio_table = None
    index_table = None
    chart_count = 0
    audio_row_count = 0
    extracted_count = 0
    converted_source_count = 0

    for _range_path, rows, index_row in payloads:
        chart_count += 1
        audio_row_count += len(rows)
        extracted_count += int(index_row.get("extracted_chunk_count", 0))
        converted_source_count += int(index_row.get("converted_source_audio_count", 0))
        if progress_every is not None and chart_count % progress_every == 0:
            print(
                f"processed_charts={chart_count} audio_rows={audio_row_count}",
                file=sys.stderr,
                flush=True,
            )

        for batch in batched(rows, batch_size):
            if audio_table is None:
                if mode == "append" and table_name in db.table_names():
                    audio_table = db.open_table(table_name)
                    audio_table.add(batch)
                else:
                    audio_table = db.create_table(table_name, data=batch, mode="overwrite")
            else:
                audio_table.add(batch)

        if write_index_table:
            if index_table is None:
                if mode == "append" and index_table_name in db.table_names():
                    index_table = db.open_table(index_table_name)
                    index_table.add([index_row])
                else:
                    index_table = db.create_table(index_table_name, data=[index_row], mode="overwrite")
            else:
                index_table.add([index_row])

    summary: dict[str, Any] = {
        "db_path": str(db_path),
        "table_name": table_name,
        "index_table_name": index_table_name if write_index_table else None,
        "chart_count": chart_count,
        "row_count": audio_row_count,
        "extracted_chunk_count": extracted_count,
        "converted_source_audio_count": converted_source_count,
        "stores_audio_bytes": store_audio_bytes,
    }
    if write_index_table:
        summary["index_row_count"] = chart_count
    return summary


def dry_run_summary(
    payloads: list[tuple[Path, list[dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    rows = [row for _, report_rows, _ in payloads for row in report_rows]
    index_rows = [index_row for _, _, index_row in payloads]
    source_audio_files = sorted({row["source_audio_file"] for row in rows})
    return {
        "dry_run": True,
        "chart_count": len(index_rows),
        "row_count": len(rows),
        "index_row_count": len(index_rows),
        "first_key": rows[0]["key"] if rows else None,
        "last_key": rows[-1]["key"] if rows else None,
        "first_chart": index_rows[0]["chart_name"] if index_rows else None,
        "last_chart": index_rows[-1]["chart_name"] if index_rows else None,
        "source_audio_file_count": len(source_audio_files),
        "first_source_audio_file": source_audio_files[0] if source_audio_files else None,
        "first_audio_file": rows[0]["audio_file"] if rows else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cut chartdata-rebuilt track.mp3/track.ogg files into four-measure "
            "segment chunks and store audio rows in a LanceDB table keyed like "
            "simai_pattern_chunks."
        )
    )
    parser.add_argument(
        "--ranges-dir",
        "--reports-dir",
        dest="ranges_dir",
        type=Path,
        default=DEFAULT_RANGES_DIR,
        help=f"Segmentation ranges directory. Default: {DEFAULT_RANGES_DIR}",
    )
    parser.add_argument(
        "--range-file",
        "--report-file",
        dest="range_file",
        type=Path,
        default=None,
        help="Process exactly one segmentation range/report JSON.",
    )
    parser.add_argument(
        "--measures-dir",
        type=Path,
        default=DEFAULT_MEASURES_DIR,
        help=f"Compiled Simai measure directory. Default: {DEFAULT_MEASURES_DIR}",
    )
    parser.add_argument(
        "--chartdata-root",
        type=Path,
        default=DEFAULT_CHARTDATA_ROOT,
        help=f"Audio source root. Default: {DEFAULT_CHARTDATA_ROOT}",
    )
    parser.add_argument(
        "--audio-filename",
        default=DEFAULT_AUDIO_FILENAME,
        help=f"Source audio filename inside each chartdata song folder. Default: {DEFAULT_AUDIO_FILENAME}",
    )
    parser.add_argument(
        "--fallback-audio-filename",
        action="append",
        default=list(DEFAULT_FALLBACK_AUDIO_FILENAMES),
        help=(
            "Fallback source audio filename when --audio-filename is missing. "
            "Can be repeated. Default: track.ogg."
        ),
    )
    parser.add_argument(
        "--audio-out-dir",
        type=Path,
        default=DEFAULT_AUDIO_OUT_DIR,
        help=f"Directory for cut audio chunks. Default: {DEFAULT_AUDIO_OUT_DIR}",
    )
    parser.add_argument(
        "--audio-suffix",
        default=DEFAULT_AUDIO_SUFFIX,
        help=f"Output audio suffix/format. Default: {DEFAULT_AUDIO_SUFFIX}",
    )
    parser.add_argument(
        "--converted-source-dir",
        type=Path,
        default=DEFAULT_CONVERTED_SOURCE_DIR,
        help=(
            "Directory for converted source MP3 files when chartdata contains track.ogg. "
            f"Default: {DEFAULT_CONVERTED_SOURCE_DIR}"
        ),
    )
    parser.add_argument(
        "--only-source-audio-format",
        action="append",
        default=None,
        help=(
            "Only process charts whose resolved source audio has this extension, "
            "for example ogg. Can be repeated or comma-separated. Other formats are skipped."
        ),
    )
    parser.add_argument(
        "--log-source-format-skips",
        action="store_true",
        help="Print one stderr line for each chart skipped by --only-source-audio-format.",
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
        help=f"LanceDB audio table name. Default: {DEFAULT_TABLE_NAME}",
    )
    parser.add_argument(
        "--index-table",
        default=DEFAULT_INDEX_TABLE_NAME,
        help=f"LanceDB audio index table name. Default: {DEFAULT_INDEX_TABLE_NAME}",
    )
    parser.add_argument(
        "--no-index-table",
        action="store_true",
        help="Do not write the chart-level audio index table.",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="Overwrite the audio table or append rows. Default: overwrite.",
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
        help="Debugging helper: stop after this many range files.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable name/path used to cut MP3 chunks. Default: ffmpeg.",
    )
    parser.add_argument(
        "--mp3-quality",
        type=int,
        default=2,
        help="libmp3lame -q:a value for chunk export. Default: 2.",
    )
    parser.add_argument(
        "--reuse-existing-audio",
        action="store_true",
        help="Reuse existing non-empty chunk files instead of cutting them again.",
    )
    parser.add_argument(
        "--store-audio-bytes",
        action="store_true",
        help="Also store encoded audio bytes in LanceDB. Without this, rows store audio paths and hashes.",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip charts whose chartdata-rebuilt track.mp3 is missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the aligned audio rows and print a summary without cutting audio or writing LanceDB.",
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
    if args.mp3_quality < 0:
        raise SystemExit("--mp3-quality must be non-negative")

    ffmpeg = None if args.dry_run else resolve_ffmpeg(args.ffmpeg)
    payloads = iter_audio_report_payloads(args, extract_audio=not args.dry_run, ffmpeg=ffmpeg)

    if args.dry_run:
        print(json.dumps(dry_run_summary(list(payloads)), ensure_ascii=False, indent=2))
        return 0

    summary = write_audio_payloads_to_lancedb(
        payloads,
        db_path=args.db_path,
        table_name=args.table,
        index_table_name=args.index_table,
        mode=args.mode,
        batch_size=args.batch_size,
        write_index_table=not args.no_index_table,
        progress_every=args.progress_every or None,
        store_audio_bytes=args.store_audio_bytes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
