#!/usr/bin/env python3
"""Export four-measure chunk ranges from compact segmentation range JSON files.

For each segment, three-measure bodies are created inside that segment. Each
body is expanded to four measures by appending one overlap measure after it;
overlap may cross segment boundaries. At the end of a chart, the missing
overlap is taken from before the body instead. Output JSON is intentionally only
a nested list of
``[start_measure, end_measure]`` ranges.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RANGES_DIR = Path("outputs/segmentation_ranges/charts")
BODY_MEASURES = 3
CHUNK_MEASURES = 4


@dataclass(frozen=True)
class ChunkSpec:
    start_measure: int
    end_measure: int
    body_start_measure: int
    body_end_measure: int
    segment_id: int = -1

    @property
    def range(self) -> list[int]:
        return [self.start_measure, self.end_measure]

    @property
    def body_range(self) -> list[int]:
        return [self.body_start_measure, self.body_end_measure]

    @property
    def overlap_measure_slots(self) -> list[int]:
        return [
            measure - self.start_measure
            for measure in range(self.start_measure, self.end_measure + 1)
            if measure < self.body_start_measure or measure > self.body_end_measure
        ]

    @property
    def overlap_measure_ranges(self) -> list[int]:
        return [
            measure
            for measure in range(self.start_measure, self.end_measure + 1)
            if measure < self.body_start_measure or measure > self.body_end_measure
        ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")


def source_measure_count(payload: dict[str, Any]) -> int:
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("measure_count") is not None:
        return int(metadata["measure_count"])

    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("input has no segments and no metadata.measure_count")
    return max(int(segment["end_measure"]) for segment in segments if isinstance(segment, dict))


def segment_ranges(payload: dict[str, Any]) -> list[tuple[int, int]]:
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("input does not contain a segments list")

    ranges: list[tuple[int, int]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"segment #{index} is not an object")
        start = int(segment["start_measure"])
        end = int(segment["end_measure"])
        if start <= 0 or end < start:
            raise ValueError(f"invalid segment range: {start}:{end}")
        ranges.append((start, end))
    return ranges


def body_ranges_for_segment(start: int, end: int) -> list[tuple[int, int]]:
    """Return body ranges whose normal size is three measures within a segment."""

    length = end - start + 1
    if length <= BODY_MEASURES:
        return [(start, end)]

    bodies: list[tuple[int, int]] = []
    cursor = start
    while cursor + BODY_MEASURES - 1 <= end:
        bodies.append((cursor, cursor + BODY_MEASURES - 1))
        cursor += BODY_MEASURES

    if bodies[-1][1] < end:
        final_start = end - BODY_MEASURES + 1
        final_body = (final_start, end)
        if final_body != bodies[-1]:
            bodies.append(final_body)
    return bodies


def expand_body_to_chunk(body_start: int, body_end: int, measure_count: int) -> list[int]:
    """Expand one body range to exactly four measures.

    Normal case is a single overlap measure after the body. At the chart end,
    the missing overlap is shifted before the body.
    """

    if measure_count < CHUNK_MEASURES:
        raise ValueError(
            f"chart has only {measure_count} measures; cannot build {CHUNK_MEASURES}-measure chunks"
        )

    start = body_start
    end = body_end

    deficit = CHUNK_MEASURES - (end - start + 1)
    add_after = min(deficit, measure_count - end)
    end += add_after
    deficit -= add_after

    if deficit > 0:
        start -= deficit

    if start < 1:
        end += 1 - start
        start = 1
    if end > measure_count:
        start -= end - measure_count
        end = measure_count

    if end - start + 1 != CHUNK_MEASURES:
        raise ValueError(f"failed to expand body {body_start}:{body_end} to four measures")
    return [start, end]


def expand_body_to_chunk_spec(
    body_start: int,
    body_end: int,
    measure_count: int,
    *,
    segment_id: int = -1,
) -> ChunkSpec:
    start, end = expand_body_to_chunk(body_start, body_end, measure_count)
    return ChunkSpec(
        start_measure=start,
        end_measure=end,
        body_start_measure=body_start,
        body_end_measure=body_end,
        segment_id=segment_id,
    )


def chunk_specs_from_segmentation_ranges(payload: dict[str, Any]) -> list[ChunkSpec]:
    measure_count = source_measure_count(payload)
    chunks: list[ChunkSpec] = []
    for segment_id, (start, end) in enumerate(segment_ranges(payload)):
        for body_start, body_end in body_ranges_for_segment(start, end):
            chunk = expand_body_to_chunk_spec(
                body_start,
                body_end,
                measure_count,
                segment_id=segment_id,
            )
            if chunks and chunks[-1].range == chunk.range:
                chunks[-1] = chunk
            else:
                chunks.append(chunk)
    return chunks


def chunk_ranges_from_segmentation_ranges(payload: dict[str, Any]) -> list[list[int]]:
    return [chunk.range for chunk in chunk_specs_from_segmentation_ranges(payload)]


def chunk_ranges_from_report(report: dict[str, Any]) -> list[list[int]]:
    """Backward-compatible alias for older imports/tests."""

    return chunk_ranges_from_segmentation_ranges(report)


def chunk_specs_from_report(report: dict[str, Any]) -> list[ChunkSpec]:
    return chunk_specs_from_segmentation_ranges(report)


def output_name_for_source(source_file: Path) -> str:
    return source_file.name


def export_directory(ranges_dir: Path, out_dir: Path) -> int:
    count = 0
    for range_file in sorted(ranges_dir.glob("*.json")):
        payload = read_json(range_file)
        chunks = chunk_ranges_from_segmentation_ranges(payload)
        write_json(out_dir / output_name_for_source(range_file), chunks)
        count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create four-measure chunk range lists from compact segmentation range JSON files."
    )
    parser.add_argument(
        "--ranges-dir",
        type=Path,
        default=DEFAULT_RANGES_DIR,
        help=f"Segmentation ranges directory. Default: {DEFAULT_RANGES_DIR}",
    )
    parser.add_argument(
        "--range-file",
        type=Path,
        default=None,
        help="Process one segmentation ranges JSON and print only its nested list to stdout, unless --output is set.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write one report's nested list to this file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Batch mode: write one nested-list JSON per report to this directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    range_file = args.range_file or args.report_file
    ranges_dir = args.reports_dir or args.ranges_dir

    if range_file is not None:
        chunks = chunk_ranges_from_segmentation_ranges(read_json(range_file))
        if args.output is not None:
            write_json(args.output, chunks)
        else:
            print(json.dumps(chunks, ensure_ascii=False))
        return 0

    if args.out_dir is None:
        raise SystemExit("provide --range-file for stdout output, or --out-dir for batch export")

    count = export_directory(ranges_dir, args.out_dir)
    print(f"exported {count} chunk range files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
