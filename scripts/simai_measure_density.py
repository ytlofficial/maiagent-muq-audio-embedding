#!/usr/bin/env python3
"""Compute density from exported Simai measure JSON files."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_measure_compiler import parse_slot_prefix, split_chart_slots
from scripts.simai_note_counter import (
    CountResult,
    count_simai_notes,
    slide_lane_paths_from_note_text,
    strip_simai_comments,
    tap_lanes_from_note_text,
    tap_or_hold_lanes_from_note_text,
)


SIXTEENTH_NOTE_BEATS = 0.25
TWELFTH_NOTE_BEATS = 1 / 3
HANDTRIP_HIGH_BPM_THRESHOLD = 200.0


@dataclass(frozen=True)
class ChartRef:
    path: Path
    song_id: int | None
    title: str | None
    chart_kind: str | None
    difficulty_index: int | None
    difficulty_name: str | None
    level: str | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chart_ref_from_index_entry(entry: dict[str, Any]) -> ChartRef:
    return ChartRef(
        path=Path(entry["file"]),
        song_id=entry.get("song_id"),
        title=entry.get("title"),
        chart_kind=entry.get("chart_kind"),
        difficulty_index=entry.get("difficulty_index"),
        difficulty_name=entry.get("difficulty_name"),
        level=entry.get("level"),
    )


def load_chart_index(index_dir: Path) -> list[ChartRef]:
    index = read_json(index_dir / "index.json")
    return [chart_ref_from_index_entry(entry) for entry in index.get("charts", [])]


def resolve_chart(
    *,
    chart_file: Path | None = None,
    index_dir: Path = Path("outputs/simai_measures"),
    song_id: int | None = None,
    title: str | None = None,
    chart_kind: str | None = None,
    difficulty_index: int | None = None,
) -> Path:
    if chart_file is not None:
        return chart_file

    matches = load_chart_index(index_dir)
    if song_id is not None:
        matches = [match for match in matches if match.song_id == song_id]
    if title:
        title_lower = title.lower()
        matches = [match for match in matches if match.title and title_lower in match.title.lower()]
    if chart_kind:
        matches = [
            match
            for match in matches
            if match.chart_kind and match.chart_kind.lower() == chart_kind.lower()
        ]
    if difficulty_index is not None:
        matches = [match for match in matches if match.difficulty_index == difficulty_index]

    if not matches:
        raise ValueError("no chart matched the provided filters")
    if len(matches) > 1:
        preview = "\n".join(
            f"- {match.song_id} {match.title} {match.chart_kind} "
            f"{match.difficulty_index} {match.difficulty_name}: {match.path}"
            for match in matches[:20]
        )
        raise ValueError(f"{len(matches)} charts matched; narrow the filters:\n{preview}")
    return matches[0].path


def measure_chart_fragment(measure: dict[str, Any]) -> str:
    simai = strip_simai_comments(str(measure.get("simai", "")))
    bpm = measure.get("bpm")
    prefix = f"({bpm:g})" if isinstance(bpm, (int, float)) and bpm > 0 else ""
    fragment = f"{prefix}{simai}"
    if not fragment.rstrip().endswith(","):
        fragment = f"{fragment},"
    return f"{fragment}E"


def density_from_count(count_result: CountResult, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return count_result.counts.density_weight / duration_seconds


def non_touch_density_from_count(count_result: CountResult, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return count_result.counts.non_touch_density_weight / duration_seconds


def circular_key_distance(left: float, right: float) -> float:
    distance = abs(left - right) % 8
    return min(distance, 8 - distance)


def circular_lane_center(lanes: list[int]) -> float:
    if not lanes:
        return 0.0
    if len(lanes) == 1:
        return float(lanes[0])

    best_unwrapped: list[int] | None = None
    best_range: int | None = None
    for anchor in lanes:
        unwrapped = [lane + 8 if lane < anchor else lane for lane in lanes]
        lane_range = max(unwrapped) - min(unwrapped)
        if best_range is None or lane_range < best_range:
            best_range = lane_range
            best_unwrapped = unwrapped

    assert best_unwrapped is not None
    return sum(best_unwrapped) / len(best_unwrapped)


def tap_moment_distance(left_lanes: list[int], right_lanes: list[int]) -> float:
    left = sorted(left_lanes)
    right = sorted(right_lanes)
    if not left or not right:
        return 0.0
    if len(left) == 1 and len(right) == 1:
        return circular_key_distance(left[0], right[0])
    if len(left) == 1:
        return min(circular_key_distance(left[0], lane) for lane in right)
    if len(right) == 1:
        return min(circular_key_distance(lane, right[0]) for lane in left)
    if len(left) == len(right):
        near_shift_distances = [
            max(circular_key_distance(left_lane, right_lane) for left_lane, right_lane in zip(left, candidate))
            for candidate in itertools.permutations(right)
        ]
        best_near_shift = min(near_shift_distances)
        if best_near_shift <= 1:
            return best_near_shift
    return abs(circular_lane_center(left) - circular_lane_center(right))


def tap_distance_summary(
    moments: list[dict[str, Any]],
    total_duration_seconds: float,
    *,
    use_high_bpm_twelfth_boundary: bool = False,
) -> dict[str, Any]:
    ordered = sorted(moments, key=lambda item: (float(item["time_seconds"]), float(item["beat"])))
    total_distance = 0.0
    pair_count = 0
    skipped_short_interval_count = 0
    high_bpm_boundary_pair_count = 0
    max_pair_distance = 0.0

    for previous, current in zip(ordered, ordered[1:]):
        interval_beats = float(current["beat"]) - float(previous["beat"])
        previous_bpm = float(previous.get("bpm") or 0.0)
        current_bpm = float(current.get("bpm") or 0.0)
        use_twelfth_boundary = (
            use_high_bpm_twelfth_boundary
            and max(previous_bpm, current_bpm) > HANDTRIP_HIGH_BPM_THRESHOLD
        )
        min_interval_beats = TWELFTH_NOTE_BEATS if use_twelfth_boundary else SIXTEENTH_NOTE_BEATS
        if use_twelfth_boundary:
            high_bpm_boundary_pair_count += 1
        if interval_beats <= min_interval_beats:
            skipped_short_interval_count += 1
            continue
        distance = tap_moment_distance(
            list(previous.get("lanes", [])),
            list(current.get("lanes", [])),
        )
        total_distance += distance
        max_pair_distance = max(max_pair_distance, distance)
        pair_count += 1

    return {
        "moment_count": len(ordered),
        "pair_count": pair_count,
        "skipped_short_interval_count": skipped_short_interval_count,
        "min_interval_beats": SIXTEENTH_NOTE_BEATS,
        "high_bpm_min_interval_beats": TWELFTH_NOTE_BEATS,
        "high_bpm_threshold": HANDTRIP_HIGH_BPM_THRESHOLD,
        "uses_high_bpm_twelfth_boundary": use_high_bpm_twelfth_boundary,
        "high_bpm_boundary_pair_count": high_bpm_boundary_pair_count,
        "total_distance": total_distance,
        "distance_per_second": (
            total_distance / total_duration_seconds if total_duration_seconds > 0 else 0.0
        ),
        "average_pair_distance": total_distance / pair_count if pair_count else 0.0,
        "max_pair_distance": max_pair_distance,
    }


def slide_path_distance(path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(
        circular_key_distance(float(left), float(right))
        for left, right in zip(path, path[1:])
    )


def slide_movement_summary(
    paths: list[list[int]],
    total_duration_seconds: float,
) -> dict[str, Any]:
    path_distances = [slide_path_distance(path) for path in paths]
    total_distance = sum(path_distances)
    return {
        "path_count": len(paths),
        "total_distance": total_distance,
        "distance_per_second": (
            total_distance / total_duration_seconds if total_duration_seconds > 0 else 0.0
        ),
        "average_path_distance": total_distance / len(paths) if paths else 0.0,
        "max_path_distance": max(path_distances) if path_distances else 0.0,
        "paths": [
            {"lanes": path, "distance": distance}
            for path, distance in zip(paths, path_distances)
        ],
    }


def combined_handtrip_summary(
    tap_distance: dict[str, Any],
    slide_movement: dict[str, Any],
    total_duration_seconds: float,
) -> dict[str, float]:
    tap_total = float(tap_distance.get("total_distance") or 0.0)
    slide_total = float(slide_movement.get("total_distance") or 0.0)
    total = tap_total + slide_total
    return {
        "tap_total_distance": tap_total,
        "slide_total_distance": slide_total,
        "total_distance": total,
        "distance_per_second": (
            total / total_duration_seconds if total_duration_seconds > 0 else 0.0
        ),
    }


def lane_moments_for_measure(
    measure: dict[str, Any],
    *,
    include_holds: bool,
    include_slide_heads: bool = True,
) -> list[dict[str, Any]]:
    fragment = measure_chart_fragment(measure)
    slots, _ = split_chart_slots(fragment)
    current_bpm = float(measure.get("bpm") or 0) or None
    current_divider: float | None = None
    current_exact_seconds: float | None = None
    cursor_seconds = 0.0
    cursor_beats = 0.0
    measure_start_seconds = float(measure.get("start_seconds") or 0.0)
    moments: list[dict[str, Any]] = []

    for raw_slot in slots:
        directives, note_text = parse_slot_prefix(raw_slot)
        for directive in directives:
            if directive.kind == "bpm":
                current_bpm = directive.value
            elif directive.kind == "divider":
                current_divider = directive.value
                current_exact_seconds = None
            elif directive.kind == "exact_divider_seconds":
                current_exact_seconds = directive.value

        bpm_fraction = Fraction(str(current_bpm)) if current_bpm else None
        lanes = (
            tap_or_hold_lanes_from_note_text(
                note_text,
                bpm_fraction,
                include_slide_heads=include_slide_heads,
            )
            if include_holds
            else tap_lanes_from_note_text(note_text, bpm_fraction)
        )
        if lanes:
            moments.append(
                {
                    "time_seconds": measure_start_seconds + cursor_seconds,
                    "local_seconds": cursor_seconds,
                    "beat": cursor_beats,
                    "local_beat": cursor_beats,
                    "bpm": current_bpm,
                    "lanes": lanes,
                    "center": circular_lane_center(lanes),
                }
            )

        if current_exact_seconds is not None:
            duration_seconds = current_exact_seconds
            beat_length = duration_seconds * current_bpm / 60.0 if current_bpm else 0.0
        else:
            if current_bpm is None:
                current_bpm = float(measure.get("bpm") or 120.0)
            if current_divider is None:
                current_divider = 4.0
            duration_seconds = 240.0 / current_bpm / current_divider
            beat_length = 4.0 / current_divider

        cursor_seconds += duration_seconds
        cursor_beats += beat_length

    return moments


def slide_lane_paths_for_measure(measure: dict[str, Any]) -> list[list[int]]:
    fragment = measure_chart_fragment(measure)
    slots, _ = split_chart_slots(fragment)
    current_bpm = float(measure.get("bpm") or 0) or None
    current_divider: float | None = None
    current_exact_seconds: float | None = None
    paths: list[list[int]] = []

    for raw_slot in slots:
        directives, note_text = parse_slot_prefix(raw_slot)
        for directive in directives:
            if directive.kind == "bpm":
                current_bpm = directive.value
            elif directive.kind == "divider":
                current_divider = directive.value
                current_exact_seconds = None
            elif directive.kind == "exact_divider_seconds":
                current_exact_seconds = directive.value
        _ = current_divider, current_exact_seconds
        bpm_fraction = Fraction(str(current_bpm)) if current_bpm else None
        paths.extend(slide_lane_paths_from_note_text(note_text, bpm_fraction))

    return paths


def tap_moments_for_measure(measure: dict[str, Any]) -> list[dict[str, Any]]:
    return lane_moments_for_measure(measure, include_holds=True)


def handtrip_tap_moments_for_measure(measure: dict[str, Any]) -> list[dict[str, Any]]:
    return lane_moments_for_measure(
        measure,
        include_holds=True,
        include_slide_heads=False,
    )


def tap_only_moments_for_measure(measure: dict[str, Any]) -> list[dict[str, Any]]:
    return lane_moments_for_measure(measure, include_holds=False)


def same_button_triple_tap_summary(moments: list[dict[str, Any]]) -> dict[str, Any]:
    times_by_lane: dict[int, list[float]] = {lane: [] for lane in range(1, 9)}
    seen_lane_beats: set[tuple[int, float]] = set()
    deduplicated_occurrence_count = 0
    for moment in moments:
        time_seconds = float(moment.get("time_seconds", 0.0))
        beat = round(float(moment.get("beat", 0.0)), 9)
        for lane in sorted(set(int(lane) for lane in moment.get("lanes", []))):
            if 1 <= lane <= 8:
                lane_beat = (lane, beat)
                if lane_beat in seen_lane_beats:
                    deduplicated_occurrence_count += 1
                    continue
                seen_lane_beats.add(lane_beat)
                times_by_lane[lane].append(time_seconds)

    best: dict[str, Any] | None = None
    window_count = 0
    for lane, times in times_by_lane.items():
        ordered_times = sorted(times)
        for index in range(0, len(ordered_times) - 2):
            window_count += 1
            start_seconds = ordered_times[index]
            end_seconds = ordered_times[index + 2]
            duration_seconds = end_seconds - start_seconds
            if best is None or duration_seconds < float(best["shortest_time_seconds"]):
                best = {
                    "lane": lane,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "shortest_time_seconds": duration_seconds,
                }

    return {
        "found": best is not None,
        "shortest_time_seconds": best["shortest_time_seconds"] if best else None,
        "lane": best["lane"] if best else None,
        "start_seconds": best["start_seconds"] if best else None,
        "end_seconds": best["end_seconds"] if best else None,
        "tap_occurrences_by_lane": {
            str(lane): len(times)
            for lane, times in times_by_lane.items()
        },
        "candidate_window_count": window_count,
        "deduplicated_occurrence_count": deduplicated_occurrence_count,
    }


def measure_density(measure: dict[str, Any]) -> dict[str, Any]:
    duration_seconds = float(measure.get("duration_seconds") or 0)
    count_result = count_simai_notes(measure_chart_fragment(measure), include_parsed_notes=False)
    tap_moments = tap_moments_for_measure(measure)
    handtrip_tap_moments = handtrip_tap_moments_for_measure(measure)
    tap_only_moments = tap_only_moments_for_measure(measure)
    slide_paths = slide_lane_paths_for_measure(measure)
    tap_distance = tap_distance_summary(tap_moments, duration_seconds)
    handtrip_tap_distance = tap_distance_summary(
        handtrip_tap_moments,
        duration_seconds,
        use_high_bpm_twelfth_boundary=True,
    )
    slide_movement = slide_movement_summary(slide_paths, duration_seconds)
    return {
        "index": measure.get("index"),
        "start_seconds": measure.get("start_seconds"),
        "end_seconds": measure.get("end_seconds"),
        "duration_seconds": duration_seconds,
        "bpm": measure.get("bpm"),
        "beats": measure.get("beats"),
        "weighted_sum": count_result.counts.density_weight,
        "non_touch_weighted_sum": count_result.counts.non_touch_density_weight,
        "density": density_from_count(count_result, duration_seconds),
        "non_touch_density": non_touch_density_from_count(count_result, duration_seconds),
        "note_counts": count_result.to_dict(include_notes=False),
        "tap_moments": tap_moments,
        "tap_distance": tap_distance,
        "handtrip_tap_moments": handtrip_tap_moments,
        "handtrip_tap_distance": handtrip_tap_distance,
        "slide_movement": slide_movement,
        "handtrip_movement": combined_handtrip_summary(
            handtrip_tap_distance,
            slide_movement,
            duration_seconds,
        ),
        "tap_only_moments": tap_only_moments,
        "same_button_triple_tap": same_button_triple_tap_summary(tap_only_moments),
    }


def chart_measure_densities(chart_data: dict[str, Any]) -> list[dict[str, Any]]:
    densities: list[dict[str, Any]] = []
    beat_cursor = 0.0
    for measure in chart_data.get("measures", []):
        density = measure_density(measure)
        for key in ("tap_moments", "handtrip_tap_moments", "tap_only_moments"):
            for moment in density[key]:
                moment["beat"] = beat_cursor + float(moment["local_beat"])
        densities.append(density)
        beat_cursor += float(measure.get("beats") or 0.0)
    return densities


def select_measure_range(
    densities: list[dict[str, Any]],
    *,
    measure: int | None = None,
    start_measure: int | None = None,
    end_measure: int | None = None,
) -> list[dict[str, Any]]:
    if measure is not None:
        start_measure = measure
        end_measure = measure
    if start_measure is None:
        start_measure = 1
    if end_measure is None:
        end_measure = len(densities)
    if start_measure < 1 or end_measure < start_measure:
        raise ValueError("invalid measure range")
    return [
        density
        for density in densities
        if start_measure <= int(density["index"]) <= end_measure
    ]


def summarize_density_range(selected: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_sum = sum(float(item["weighted_sum"]) for item in selected)
    duration_seconds = sum(float(item["duration_seconds"]) for item in selected)
    if not selected:
        return {
            "start_measure": None,
            "end_measure": None,
            "measure_count": 0,
            "weighted_sum": 0,
            "duration_seconds": 0,
            "density": 0,
        }
    return {
        "start_measure": selected[0]["index"],
        "end_measure": selected[-1]["index"],
        "measure_count": len(selected),
        "weighted_sum": weighted_sum,
        "duration_seconds": duration_seconds,
        "density": weighted_sum / duration_seconds if duration_seconds > 0 else 0,
    }


def build_density_report(
    chart_path: Path,
    *,
    measure: int | None = None,
    start_measure: int | None = None,
    end_measure: int | None = None,
    include_measures: bool = True,
) -> dict[str, Any]:
    chart_data = read_json(chart_path)
    densities = chart_measure_densities(chart_data)
    selected = select_measure_range(
        densities,
        measure=measure,
        start_measure=start_measure,
        end_measure=end_measure,
    )
    report = {
        "source_file": str(chart_path),
        "song": chart_data.get("song"),
        "chart": chart_data.get("chart"),
        "range": summarize_density_range(selected),
    }
    if include_measures:
        report["measures"] = selected
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute weighted Simai chart density from exported measure JSON."
    )
    parser.add_argument("--chart-file", type=Path, help="Path to one exported chart JSON.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Directory containing index.json. Default: outputs/simai_measures",
    )
    parser.add_argument("--song-id", type=int, help="Song id to look up from index.json.")
    parser.add_argument("--title", help="Case-insensitive title substring to look up.")
    parser.add_argument("--chart-kind", choices=["ST", "DX"], help="Chart kind filter.")
    parser.add_argument("--difficulty-index", type=int, help="Difficulty index filter.")
    parser.add_argument("--measure", type=int, help="Single measure index to compute.")
    parser.add_argument("--start-measure", type=int, help="First measure index, inclusive.")
    parser.add_argument("--end-measure", type=int, help="Last measure index, inclusive.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the selected range summary, not per-measure rows.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chart_path = resolve_chart(
        chart_file=args.chart_file,
        index_dir=args.index_dir,
        song_id=args.song_id,
        title=args.title,
        chart_kind=args.chart_kind,
        difficulty_index=args.difficulty_index,
    )
    report = build_density_report(
        chart_path,
        measure=args.measure,
        start_measure=args.start_measure,
        end_measure=args.end_measure,
        include_measures=not args.summary_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
