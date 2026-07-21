#!/usr/bin/env python3
"""Build a rule-based embedding for four-measure Simai fragments.

The embedding is intentionally structural rather than textual. It keeps
absolute 1-8 button information, adds topology-only features, and adds a D8
rotation/mirror pooled branch so rotated or mirrored configurations remain
similar without becoming identical.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simai_measure_compiler import (
    compile_chart,
    compiled_to_dict,
    parse_slot_prefix,
    split_chart_slots,
)
from scripts.simai_measure_density import read_json, resolve_chart
from scripts.simai_note_counter import (
    SimaiNoteParseError,
    combined_track_duration,
    duration_body,
    expand_each_tokens,
    parse_note_token,
    parse_sensor,
    parse_shape,
    parse_duration_fraction,
    read_duration,
    split_top_level,
    strip_ignored_whitespace,
    strip_simai_comments,
)


LANES = tuple(range(1, 9))
SHAPES = ("-", ">", "<", "^", "v", "p", "q", "s", "z", "pp", "qq", "V", "w")
SOFT_SHAPES = ("-", ">", "<", "^", "v", "p", "q", "s", "z", "pp", "qq")
KINDS = ("tap", "hold", "slide", "touch", "touch_hold")
BUTTON_KINDS = ("tap", "hold", "slide_start", "slide_end", "slide_middle")
START_TAP_MODES = ("star", "normal", "omitted_fade", "omitted_sudden")
TOUCH_FAMILIES = ("A", "B", "C", "D", "E")
TOUCH_KINDS = ("touch", "touch_hold")
TOPOLOGY_DIRS = ("same", "cw", "ccw", "opposite")
HAND_NAMES = ("left", "right")
LANE_SIDES = ("left", "right")
SEGMENT_BINS = ("1", "2", "3", "4", "5", "6_plus")
TRACK_BINS = ("1", "2", "3", "4_plus")
PATH_LENGTH_BINS = tuple(str(index) for index in range(0, 13))
BEAT_BINS = 96
EXPECTED_MEASURES = 4
OVERLAP_MEASURE_WEIGHT = 0.5
RHYTHM_TICKS_PER_BEAT = 24
RHYTHM_TICKS_PER_MEASURE = 96
RHYTHM_EXPECTED_TICKS = EXPECTED_MEASURES * RHYTHM_TICKS_PER_MEASURE
RHYTHM_FFT_BINS = RHYTHM_EXPECTED_TICKS // 2 + 1

MIRROR_SHAPE = {
    ">": "<",
    "<": ">",
    "p": "q",
    "q": "p",
    "pp": "qq",
    "qq": "pp",
    "s": "z",
    "z": "s",
}

BLOCK_WEIGHTS = {
    "time": 0.10,
    "absolute": 0.18,
    "topology": 0.22,
    "slide": 0.18,
    "touch": 0.10,
    "d8_mean": 0.14,
    "d8_max": 0.06,
    "hand_balance": 0.10,
    "rhythm_fft": 0.16,
}

BLOCK_ORDER = (
    "time",
    "absolute",
    "topology",
    "slide",
    "touch",
    "d8_mean",
    "d8_max",
    "hand_balance",
    "rhythm_fft",
)
SIGNED_BLOCKS = {"hand_balance", "rhythm_fft"}

LANE_RIGHT_HAND_PROBABILITY = {
    # 1/4 and 8/5 sit near the vertical seams, so their preference is softer.
    1: 0.75,
    2: 0.90,
    3: 0.90,
    4: 0.75,
    5: 0.25,
    6: 0.10,
    7: 0.10,
    8: 0.25,
}

SLIDE_SHAPE_FEATURE_WEIGHT = 0.45
SLIDE_SHAPE_ENDPOINT_FEATURE_WEIGHT = 0.55
SLIDE_PATH_LENGTH_FEATURE_WEIGHT = 0.35
STAR_SLIDE_DELAY_BEATS = 1.0
STAR_EACH_GUIDE_WINDOW_BEATS = (0.75, 1.25)
STAR_SLIDE_DEFAULT_DURATION_BEATS = 1.0


@dataclass(frozen=True)
class SlideSegment:
    shape: str
    start: int
    end: int
    middle: int | None = None


@dataclass(frozen=True)
class SlideTrack:
    segments: tuple[SlideSegment, ...]
    break_track: bool = False
    duration_beats: float | None = None


@dataclass(frozen=True)
class PatternEvent:
    kind: str
    measure_slot: int
    global_beat: float
    local_beat: float
    group_id: str
    group_size: int
    order: int
    weight: float = 1.0
    lane: int | None = None
    modifiers: frozenset[str] = frozenset()
    sensor_family: str | None = None
    sensor_index: int | None = None
    slide_start: int | None = None
    slide_start_tap: str | None = None
    slide_start_break: bool = False
    slide_start_ex: bool = False
    slide_tracks: tuple[SlideTrack, ...] = ()
    raw: str = ""


@dataclass(frozen=True)
class HandAssignedEvent:
    event: PatternEvent
    hand: str
    lane: int
    weight: float


def circular_distance(left: int, right: int) -> int:
    delta = abs(left - right) % 8
    return min(delta, 8 - delta)


def cw_steps(start: int, end: int) -> int:
    return (end - start) % 8


def ccw_steps(start: int, end: int) -> int:
    return (start - end) % 8


def direction_label(start: int, end: int) -> str:
    cw = cw_steps(start, end)
    ccw = ccw_steps(start, end)
    if cw == 0:
        return "same"
    if cw == ccw:
        return "opposite"
    return "cw" if cw < ccw else "ccw"


def bin_name(value: int, bins: tuple[str, ...]) -> str:
    if value < len(bins):
        return bins[value - 1]
    return bins[-1]


def lane_index_to_point(index: float, radius: float = 1.0) -> tuple[float, float]:
    angle = math.pi / 2 - 2 * math.pi * index / 8
    return (radius * math.cos(angle), radius * math.sin(angle))


def lane_point(lane: int, radius: float = 1.0) -> tuple[float, float]:
    return lane_index_to_point(float(lane - 1), radius)


def lerp(left: tuple[float, float], right: tuple[float, float], t: float) -> tuple[float, float]:
    return (left[0] + (right[0] - left[0]) * t, left[1] + (right[1] - left[1]) * t)


def quad_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    a = lerp(p0, p1, t)
    b = lerp(p1, p2, t)
    return lerp(a, b, t)


def polyline_sample(points: list[tuple[float, float]], sample_count: int) -> list[tuple[float, float]]:
    if len(points) == 1:
        return points * sample_count
    lengths = [
        math.dist(left, right)
        for left, right in zip(points, points[1:])
    ]
    total = sum(lengths)
    if total <= 0:
        return [points[0]] * sample_count
    samples: list[tuple[float, float]] = []
    for index in range(sample_count):
        target = total * index / (sample_count - 1) if sample_count > 1 else 0.0
        cursor = 0.0
        for segment_index, segment_length in enumerate(lengths):
            if cursor + segment_length >= target or segment_index == len(lengths) - 1:
                local = (target - cursor) / segment_length if segment_length > 0 else 0.0
                samples.append(lerp(points[segment_index], points[segment_index + 1], local))
                break
            cursor += segment_length
    return samples


def arc_sample(start: int, end: int, *, clockwise: bool, sample_count: int) -> list[tuple[float, float]]:
    delta = cw_steps(start, end) if clockwise else ccw_steps(start, end)
    if delta == 0:
        delta = 8
    start_index = start - 1
    samples = []
    for index in range(sample_count):
        t = index / (sample_count - 1) if sample_count > 1 else 0.0
        offset = delta * t
        lane_index = start_index + offset if clockwise else start_index - offset
        samples.append(lane_index_to_point(lane_index))
    return samples


def shortest_arc_sample(start: int, end: int, sample_count: int) -> list[tuple[float, float]]:
    return arc_sample(start, end, clockwise=cw_steps(start, end) <= ccw_steps(start, end), sample_count=sample_count)


def curve_control_point(
    start: int,
    end: int,
    *,
    sign: float,
    scale: float,
) -> tuple[float, float]:
    p0 = lane_point(start)
    p2 = lane_point(end)
    middle = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
    dx = p2[0] - p0[0]
    dy = p2[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    normal = (-dy / length, dx / length)
    return (middle[0] + normal[0] * sign * scale, middle[1] + normal[1] * sign * scale)


def curved_sample(
    start: int,
    end: int,
    *,
    sign: float,
    scale: float,
    sample_count: int,
) -> list[tuple[float, float]]:
    p0 = lane_point(start)
    p2 = lane_point(end)
    control = curve_control_point(start, end, sign=sign, scale=scale)
    return [
        quad_bezier(p0, control, p2, index / (sample_count - 1) if sample_count > 1 else 0.0)
        for index in range(sample_count)
    ]


def fan_sample(start: int, end: int, sample_count: int) -> list[tuple[float, float]]:
    branch_ends = [((end - 2) % 8) + 1, end, (end % 8) + 1]
    branches = [
        polyline_sample([lane_point(start), lane_point(branch_end)], sample_count)
        for branch_end in branch_ends
    ]
    return [
        (
            sum(branch[index][0] for branch in branches) / len(branches),
            sum(branch[index][1] for branch in branches) / len(branches),
        )
        for index in range(sample_count)
    ]


def sample_slide_segment(segment: SlideSegment, sample_count: int = 16) -> list[tuple[float, float]]:
    start = segment.start
    end = segment.end
    shape = segment.shape
    if shape == "-":
        return polyline_sample([lane_point(start), lane_point(end)], sample_count)
    if shape == ">":
        return arc_sample(start, end, clockwise=True, sample_count=sample_count)
    if shape == "<":
        return arc_sample(start, end, clockwise=False, sample_count=sample_count)
    if shape == "^":
        return shortest_arc_sample(start, end, sample_count)
    if shape == "v":
        return polyline_sample([lane_point(start), (0.0, 0.0), lane_point(end)], sample_count)
    if shape == "p":
        return curved_sample(start, end, sign=1.0, scale=0.55, sample_count=sample_count)
    if shape == "q":
        return curved_sample(start, end, sign=-1.0, scale=0.55, sample_count=sample_count)
    if shape == "pp":
        return curved_sample(start, end, sign=1.0, scale=1.05, sample_count=sample_count)
    if shape == "qq":
        return curved_sample(start, end, sign=-1.0, scale=1.05, sample_count=sample_count)
    if shape == "s":
        return polyline_sample(
            [
                lane_point(start),
                curve_control_point(start, end, sign=1.0, scale=0.70),
                curve_control_point(start, end, sign=-1.0, scale=0.70),
                lane_point(end),
            ],
            sample_count,
        )
    if shape == "z":
        return polyline_sample(
            [
                lane_point(start),
                curve_control_point(start, end, sign=-1.0, scale=0.70),
                curve_control_point(start, end, sign=1.0, scale=0.70),
                lane_point(end),
            ],
            sample_count,
        )
    if shape == "V":
        middle = segment.middle if segment.middle is not None else end
        return polyline_sample([lane_point(start), lane_point(middle), lane_point(end)], sample_count)
    if shape == "w":
        return fan_sample(start, end, sample_count)
    return polyline_sample([lane_point(start), lane_point(end)], sample_count)


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def path_distance(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    if len(left) != len(right):
        size = min(len(left), len(right))
        left = left[:size]
        right = right[:size]
    if not left:
        return 0.0
    return sum(math.dist(a, b) for a, b in zip(left, right)) / len(left)


def slide_segment_distance(left: SlideSegment, right: SlideSegment) -> float:
    """Continuous geometry distance for two shape-aware slide segments."""

    left_path = sample_slide_segment(left)
    right_path = sample_slide_segment(right)
    geometry = path_distance(left_path, right_path)
    start_end = (
        circular_distance(left.start, right.start)
        + circular_distance(left.end, right.end)
    ) / 8
    middle = 0.0
    if left.middle is not None or right.middle is not None:
        if left.middle is None or right.middle is None:
            middle = 1.0
        else:
            middle = circular_distance(left.middle, right.middle) / 4
    shape_penalty = 0.0 if left.shape == right.shape else 0.25
    return geometry + 0.25 * start_end + 0.10 * middle + shape_penalty


def soft_shape_weights(segment: SlideSegment) -> dict[str, float]:
    actual = sample_slide_segment(segment)
    weights: dict[str, float] = {}
    for shape in SOFT_SHAPES:
        candidate = SlideSegment(shape=shape, start=segment.start, end=segment.end)
        distance = path_distance(actual, sample_slide_segment(candidate))
        weights[shape] = math.exp(-distance / 0.42)
    return weights


def strip_duration_brackets(token: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", token)


def parse_slide_token(
    token: str,
    current_bpm: Fraction | None = None,
) -> tuple[int, str, bool, bool, tuple[SlideTrack, ...]]:
    kind, _ = parse_note_token(token, current_bpm)
    if kind != "slide" or not token or token[0] not in "12345678":
        raise SimaiNoteParseError(f"not a slide token: {token!r}")

    start_lane = int(token[0])
    position = 1
    start_modifiers = []
    while position < len(token) and token[position] in "bx@":
        start_modifiers.append(token[position])
        position += 1

    start_tap = "star"
    if "@" in start_modifiers:
        start_tap = "normal"
    if position < len(token) and token[position] in "?!":
        start_tap = "omitted_fade" if token[position] == "?" else "omitted_sudden"
        position += 1

    tracks: list[SlideTrack] = []
    while position < len(token):
        track, position = parse_slide_track(token, position, start_lane, current_bpm)
        tracks.append(track)
        if position < len(token) and token[position] == "*":
            position += 1
            continue
        break

    return start_lane, start_tap, "b" in start_modifiers, "x" in start_modifiers, tuple(tracks)


def parse_slide_track(
    token: str,
    position: int,
    start_lane: int,
    current_bpm: Fraction | None,
) -> tuple[SlideTrack, int]:
    segments: list[SlideSegment] = []
    current_lane = start_lane
    break_track = False
    durations: list[Fraction | None] = []

    while position < len(token):
        if token[position] == "[":
            start = position
            position = read_duration(token, position)
            durations.append(parse_duration_fraction(duration_body(token, start, position), current_bpm))
            continue
        if token[position] == "b":
            if position + 1 < len(token) and token[position + 1] == "[":
                break_track = True
                position += 1
                continue
            if position + 1 == len(token) or token[position + 1] == "*":
                break_track = True
                position += 1
                break
            if parse_shape(token, position + 1) is not None:
                break_track = True
                position += 1
                continue

        parsed = parse_shape(token, position)
        if parsed is None:
            break
        shape, next_position = parsed
        if shape == "V":
            middle = int(token[position + 1])
            end = int(token[position + 2])
            segments.append(SlideSegment(shape=shape, start=current_lane, middle=middle, end=end))
        else:
            endpoint_position = position + len(shape)
            end = int(token[endpoint_position])
            segments.append(SlideSegment(shape=shape, start=current_lane, end=end))
        current_lane = end
        position = next_position

    if not segments:
        raise SimaiNoteParseError(f"slide track has no segments in {token!r}")
    duration = combined_track_duration(durations) if durations else None
    duration_beats = float(duration * 4) if duration is not None else None
    return SlideTrack(tuple(segments), break_track=break_track, duration_beats=duration_beats), position


def parse_touch_token(token: str) -> tuple[str, int | None, frozenset[str]]:
    parsed = parse_sensor(token)
    if parsed is None:
        raise SimaiNoteParseError(f"not a touch token: {token!r}")
    sensor, position = parsed
    sensor = "C" if sensor in {"C1", "C2"} else sensor
    family = sensor[0]
    sensor_index = int(sensor[1]) if len(sensor) > 1 else None
    suffix = strip_duration_brackets(token[position:])
    modifiers = set()
    if "h" in suffix:
        modifiers.add("h")
    if "f" in suffix:
        modifiers.add("f")
    if "x" in suffix:
        modifiers.add("x")
    return family, sensor_index, frozenset(modifiers)


def parse_button_modifiers(token: str) -> frozenset[str]:
    suffix = strip_duration_brackets(token[1:])
    modifiers = set()
    if "h" in suffix:
        modifiers.add("h")
    if "b" in suffix:
        modifiers.add("b")
    if "x" in suffix:
        modifiers.add("x")
    if "$$" in suffix:
        modifiers.add("rotating_star")
    elif "$" in suffix:
        modifiers.add("star")
    return frozenset(modifiers)


def measure_chart_fragment(measure: dict[str, Any]) -> str:
    simai = strip_simai_comments(str(measure.get("simai", "")))
    bpm = measure.get("bpm")
    prefix = f"({bpm:g})" if isinstance(bpm, (int, float)) and bpm > 0 else ""
    fragment = f"{prefix}{simai}"
    if not fragment.rstrip().endswith(","):
        fragment = f"{fragment},"
    return f"{fragment}E"


def slot_beat_length(
    *,
    current_bpm: float | None,
    current_divider: float | None,
    current_exact_seconds: float | None,
) -> float:
    if current_exact_seconds is not None:
        bpm = current_bpm if current_bpm and current_bpm > 0 else 120.0
        return current_exact_seconds * bpm / 60.0
    divider = current_divider if current_divider and current_divider > 0 else 4.0
    return 4.0 / divider


def events_from_measure(
    measure: dict[str, Any],
    *,
    measure_slot: int,
    beat_cursor: float,
    event_weight: float = 1.0,
) -> list[PatternEvent]:
    fragment = measure_chart_fragment(measure)
    slots, _ = split_chart_slots(fragment)
    current_bpm = float(measure.get("bpm") or 0.0) or None
    current_divider: float | None = None
    current_exact_seconds: float | None = None
    local_beat = 0.0
    events: list[PatternEvent] = []
    order = 0

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

        if note_text:
            bpm_fraction = Fraction(str(current_bpm)) if current_bpm else None
            slot_events = events_from_note_text(
                note_text,
                bpm_fraction,
                measure_slot=measure_slot,
                global_beat=beat_cursor + local_beat,
                local_beat=local_beat,
                order_start=order,
                event_weight=event_weight,
            )
            events.extend(slot_events)
            order += len(slot_events)

        local_beat += slot_beat_length(
            current_bpm=current_bpm,
            current_divider=current_divider,
            current_exact_seconds=current_exact_seconds,
        )

    return events


def events_from_note_text(
    note_text: str,
    current_bpm: Fraction | None,
    *,
    measure_slot: int,
    global_beat: float,
    local_beat: float,
    order_start: int,
    event_weight: float = 1.0,
) -> list[PatternEvent]:
    note_text = strip_ignored_whitespace(note_text)
    if not note_text:
        return []

    events: list[PatternEvent] = []
    pseudo_groups = split_top_level(note_text, "`")
    order = order_start
    for pseudo_index, pseudo_group in enumerate(pseudo_groups):
        if not pseudo_group:
            continue
        each_tokens = split_top_level(pseudo_group, "/")
        expanded_tokens = expand_each_tokens(each_tokens)
        group_id = f"{measure_slot}:{global_beat:.9f}:{pseudo_index}"
        for token in expanded_tokens:
            kind, _ = parse_note_token(token, current_bpm)
            event_beat = global_beat + pseudo_index * 1e-4
            if kind == "slide":
                start, start_tap, start_break, start_ex, tracks = parse_slide_token(token, current_bpm)
                events.append(
                    PatternEvent(
                        kind="slide",
                        measure_slot=measure_slot,
                        global_beat=event_beat,
                        local_beat=local_beat,
                        group_id=group_id,
                        group_size=len(expanded_tokens),
                        order=order,
                        weight=event_weight,
                        lane=start,
                        slide_start=start,
                        slide_start_tap=start_tap,
                        slide_start_break=start_break,
                        slide_start_ex=start_ex,
                        slide_tracks=tracks,
                        raw=token,
                    )
                )
            elif kind in {"touch", "touch_hold"}:
                family, sensor_index, modifiers = parse_touch_token(token)
                events.append(
                    PatternEvent(
                        kind=kind,
                        measure_slot=measure_slot,
                        global_beat=event_beat,
                        local_beat=local_beat,
                        group_id=group_id,
                        group_size=len(expanded_tokens),
                        order=order,
                        weight=event_weight,
                        modifiers=modifiers,
                        sensor_family=family,
                        sensor_index=sensor_index,
                        raw=token,
                    )
                )
            else:
                lane = int(token[0])
                events.append(
                    PatternEvent(
                        kind=kind,
                        measure_slot=measure_slot,
                        global_beat=event_beat,
                        local_beat=local_beat,
                        group_id=group_id,
                        group_size=len(expanded_tokens),
                        order=order,
                        weight=event_weight,
                        lane=lane,
                        modifiers=parse_button_modifiers(token),
                        raw=token,
                    )
                )
            order += 1
    return events


def events_from_measures(
    measures: list[dict[str, Any]],
    measure_weights: list[float] | None = None,
) -> tuple[list[PatternEvent], float]:
    events: list[PatternEvent] = []
    beat_cursor = 0.0
    for measure_slot, measure in enumerate(measures):
        event_weight = measure_weights[measure_slot] if measure_weights is not None else 1.0
        events.extend(
            events_from_measure(
                measure,
                measure_slot=measure_slot,
                beat_cursor=beat_cursor,
                event_weight=event_weight,
            )
        )
        beat_cursor += float(measure.get("beats") or 4.0)
    return events, beat_cursor if beat_cursor > 0 else EXPECTED_MEASURES * 4.0


def canonical_sensor_name(family: str | None, index: int | None) -> str | None:
    if family is None:
        return None
    if family == "C":
        return "C"
    if index is None:
        return None
    return f"{family}{index}"


def representative_lane(event: PatternEvent) -> int | None:
    if event.lane is not None:
        return event.lane
    if event.sensor_index is not None:
        return event.sensor_index
    return None


def lane_side(lane: int) -> str:
    return "right" if 1 <= lane <= 4 else "left"


def lane_hand_probabilities(lane: int) -> dict[str, float]:
    right = LANE_RIGHT_HAND_PROBABILITY.get(lane, 0.5)
    return {"left": 1.0 - right, "right": right}


def transform_lane(lane: int | None, *, rotation: int, mirror: bool) -> int | None:
    if lane is None:
        return None
    index = lane - 1
    if mirror:
        index = (-index) % 8
    return ((index + rotation) % 8) + 1


def transform_shape(shape: str, *, mirror: bool) -> str:
    if not mirror:
        return shape
    return MIRROR_SHAPE.get(shape, shape)


def transform_segment(segment: SlideSegment, *, rotation: int, mirror: bool) -> SlideSegment:
    return SlideSegment(
        shape=transform_shape(segment.shape, mirror=mirror),
        start=transform_lane(segment.start, rotation=rotation, mirror=mirror) or segment.start,
        end=transform_lane(segment.end, rotation=rotation, mirror=mirror) or segment.end,
        middle=transform_lane(segment.middle, rotation=rotation, mirror=mirror),
    )


def transform_event(event: PatternEvent, *, rotation: int, mirror: bool) -> PatternEvent:
    transformed_tracks = tuple(
        SlideTrack(
            tuple(
                transform_segment(segment, rotation=rotation, mirror=mirror)
                for segment in track.segments
            ),
            break_track=track.break_track,
            duration_beats=track.duration_beats,
        )
        for track in event.slide_tracks
    )
    return replace(
        event,
        lane=transform_lane(event.lane, rotation=rotation, mirror=mirror),
        sensor_index=transform_lane(event.sensor_index, rotation=rotation, mirror=mirror),
        slide_start=transform_lane(event.slide_start, rotation=rotation, mirror=mirror),
        slide_tracks=transformed_tracks,
    )


def schema_for_time() -> list[str]:
    names = []
    names.extend(f"event_kind_{kind}" for kind in KINDS)
    names.extend(f"measure_{measure}_event_count" for measure in range(EXPECTED_MEASURES))
    names.extend(
        f"measure_{measure}_{kind}_count"
        for measure in range(EXPECTED_MEASURES)
        for kind in KINDS
    )
    names.extend(f"beat_bin_{index}" for index in range(BEAT_BINS))
    return names


def schema_for_absolute() -> list[str]:
    names = []
    names.extend(
        f"lane_{lane}_{kind}"
        for lane in LANES
        for kind in BUTTON_KINDS
    )
    names.extend(
        f"lane_{lane}_{modifier}"
        for lane in LANES
        for modifier in ("break", "ex", "star_tap", "rotating_star_tap", "slide_starless")
    )
    names.extend(
        f"transition_{left}_to_{right}"
        for left in LANES
        for right in LANES
    )
    names.extend(
        f"measure_{measure}_lane_{lane}"
        for measure in range(EXPECTED_MEASURES)
        for lane in LANES
    )
    return names


def schema_for_topology() -> list[str]:
    names = []
    names.extend(f"transition_dist_{distance}" for distance in range(0, 5))
    names.extend(f"transition_dir_{direction}" for direction in TOPOLOGY_DIRS)
    names.extend(f"each_size_{size}" for size in ("1", "2", "3", "4", "5_plus"))
    names.extend(f"each_pair_dist_{distance}" for distance in range(0, 5))
    names.extend(
        f"kind_bigram_{left}_to_{right}"
        for left in KINDS
        for right in KINDS
    )
    return names


def schema_for_slide() -> list[str]:
    names = []
    names.extend(f"shape_{shape}" for shape in SHAPES)
    names.extend(
        f"shape_endpoint_dist_{shape}_{distance}"
        for shape in SHAPES
        for distance in range(0, 5)
    )
    names.extend(
        f"shape_start_end_{shape}_{start}_{end}"
        for shape in SHAPES
        for start in LANES
        for end in LANES
    )
    names.extend(
        f"shape_middle_{shape}_{middle}"
        for shape in ("V",)
        for middle in LANES
    )
    names.extend(f"soft_shape_{shape}" for shape in SOFT_SHAPES)
    names.extend(f"path_length_bin_{name}" for name in PATH_LENGTH_BINS)
    names.extend(f"segment_count_{name}" for name in SEGMENT_BINS)
    names.extend(f"track_count_{name}" for name in TRACK_BINS)
    names.extend(f"start_tap_{mode}" for mode in START_TAP_MODES)
    names.extend(f"slide_modifier_{name}" for name in ("start_break", "start_ex", "track_break"))
    names.extend(f"slide_direction_{direction}" for direction in TOPOLOGY_DIRS)
    names.extend(f"slide_declared_distance_{distance}" for distance in range(0, 5))
    return names


def schema_for_touch() -> list[str]:
    sensors = [
        *(f"{family}{index}" for family in ("A", "B", "D", "E") for index in LANES),
        "C",
    ]
    names = []
    names.extend(
        f"sensor_{sensor}_{kind}"
        for sensor in sensors
        for kind in TOUCH_KINDS
    )
    names.extend(
        f"sensor_family_{family}_{kind}"
        for family in TOUCH_FAMILIES
        for kind in TOUCH_KINDS
    )
    names.extend(f"touch_index_{index}" for index in LANES)
    names.extend(f"touch_modifier_{name}" for name in ("firework", "ex"))
    names.append("touch_center_count")
    return names


def schema_for_hand_balance() -> list[str]:
    names = [
        "left_load",
        "right_load",
        "total_load",
        "load_balance_abs",
        "left_minus_right_load",
        "preferred_hand_load",
        "cross_hand_load",
        "forced_each_pair_count",
        "soft_single_event_count",
        "multi_each_group_count",
        "star_head_default_count",
        "star_each_guided_count",
        "star_slide_overlap_forced_count",
        "star_slide_overlap_conflict_count",
    ]
    names.extend(
        f"measure_{measure}_{hand}_load"
        for measure in range(EXPECTED_MEASURES)
        for hand in HAND_NAMES
    )
    names.extend(f"measure_{measure}_balance_abs" for measure in range(EXPECTED_MEASURES))
    names.extend(
        f"{hand}_lane_{lane}_load"
        for hand in HAND_NAMES
        for lane in LANES
    )
    names.extend(
        f"{hand}_side_{side}_load"
        for hand in HAND_NAMES
        for side in LANE_SIDES
    )
    names.extend(
        f"{hand}_transition_dist_{distance}"
        for hand in HAND_NAMES
        for distance in range(0, 5)
    )
    names.extend(
        f"{hand}_transition_dir_{direction}"
        for hand in HAND_NAMES
        for direction in TOPOLOGY_DIRS
    )
    names.extend(f"{hand}_movement_total_distance" for hand in HAND_NAMES)
    names.extend(f"each_lr_pair_dist_{distance}" for distance in range(0, 5))
    names.extend(f"each_lr_pair_side_{side}" for side in ("split", "left", "right"))
    return names


def schema_for_rhythm_fft() -> list[str]:
    names = []
    names.extend(f"fft_real_{index}" for index in range(RHYTHM_FFT_BINS))
    names.extend(f"fft_imag_{index}" for index in range(RHYTHM_FFT_BINS))
    names.extend(("bpm_scaled", "bpm_log2_ratio_120", "onset_density"))
    return names


def base_schema() -> dict[str, list[str]]:
    return {
        "time": schema_for_time(),
        "absolute": schema_for_absolute(),
        "topology": schema_for_topology(),
        "slide": schema_for_slide(),
        "touch": schema_for_touch(),
    }


def effective_bpm_from_measures(measures: list[dict[str, Any]], total_beats: float) -> float:
    durations = [
        float(measure.get("duration_seconds") or 0.0)
        for measure in measures
    ]
    total_seconds = sum(duration for duration in durations if duration > 0)
    if total_seconds > 0 and total_beats > 0:
        return total_beats * 60.0 / total_seconds

    weighted_sum = 0.0
    weight = 0.0
    for measure in measures:
        bpm = float(measure.get("bpm") or 0.0)
        beats = float(measure.get("beats") or 4.0)
        if bpm > 0 and beats > 0:
            weighted_sum += bpm * beats
            weight += beats
    if weight > 0:
        return weighted_sum / weight
    return 120.0


def rhythm_tick_series(events: list[PatternEvent], total_beats: float) -> list[float]:
    series = [0.0] * RHYTHM_EXPECTED_TICKS
    max_beat = max(total_beats, 1e-9)
    for event in events:
        if event.kind not in KINDS:
            continue
        if event.global_beat < 0 or event.global_beat >= max_beat:
            continue
        tick = int(round(event.global_beat * RHYTHM_TICKS_PER_BEAT))
        if 0 <= tick < RHYTHM_EXPECTED_TICKS:
            series[tick] = min(series[tick] + event.weight, 1.0)
    return series


def rfft_ortho(series: list[float]) -> list[complex]:
    try:
        import numpy as np
    except ImportError:
        norm = math.sqrt(len(series))
        return [
            sum(
                value * cmath.exp(-2j * math.pi * bin_index * sample_index / len(series))
                for sample_index, value in enumerate(series)
            ) / norm
            for bin_index in range(len(series) // 2 + 1)
        ]

    values = np.asarray(series, dtype=float)
    return [complex(value) for value in np.fft.rfft(values, norm="ortho")]


def extract_rhythm_fft_block(
    events: list[PatternEvent],
    total_beats: float,
    rhythm_bpm: float | None,
) -> dict[str, float]:
    series = rhythm_tick_series(events, total_beats)
    fft_values = rfft_ortho(series)
    bpm = float(rhythm_bpm or 0.0)

    block = {name: 0.0 for name in schema_for_rhythm_fft()}
    for index, value in enumerate(fft_values[:RHYTHM_FFT_BINS]):
        block[f"fft_real_{index}"] = value.real
        block[f"fft_imag_{index}"] = value.imag
    block["bpm_scaled"] = bpm / 300.0 if bpm > 0 else 0.0
    block["bpm_log2_ratio_120"] = math.log2(bpm / 120.0) if bpm > 0 else 0.0
    block["onset_density"] = sum(series) / len(series) if series else 0.0
    return block


def add(features: dict[str, float], name: str, value: float = 1.0) -> None:
    if name in features:
        features[name] += value


def extract_base_blocks(events: list[PatternEvent], total_beats: float) -> dict[str, dict[str, float]]:
    schema = base_schema()
    blocks = {block: {name: 0.0 for name in names} for block, names in schema.items()}
    ordered = sorted(events, key=lambda item: (item.global_beat, item.order))

    for event in ordered:
        weight = event.weight
        add(blocks["time"], f"event_kind_{event.kind}", weight)
        if 0 <= event.measure_slot < EXPECTED_MEASURES:
            add(blocks["time"], f"measure_{event.measure_slot}_event_count", weight)
            add(blocks["time"], f"measure_{event.measure_slot}_{event.kind}_count", weight)
        bin_index = int(max(0.0, min(0.999999, event.global_beat / max(total_beats, 1e-9))) * BEAT_BINS)
        add(blocks["time"], f"beat_bin_{bin_index}", weight)

        lane = representative_lane(event)
        if lane is not None and 1 <= lane <= 8 and 0 <= event.measure_slot < EXPECTED_MEASURES:
            add(blocks["absolute"], f"measure_{event.measure_slot}_lane_{lane}", weight)

        if event.kind in {"tap", "hold"} and event.lane is not None:
            add(blocks["absolute"], f"lane_{event.lane}_{event.kind}", weight)
            if "b" in event.modifiers:
                add(blocks["absolute"], f"lane_{event.lane}_break", weight)
            if "x" in event.modifiers:
                add(blocks["absolute"], f"lane_{event.lane}_ex", weight)
            if "star" in event.modifiers:
                add(blocks["absolute"], f"lane_{event.lane}_star_tap", weight)
            if "rotating_star" in event.modifiers:
                add(blocks["absolute"], f"lane_{event.lane}_rotating_star_tap", weight)
        elif event.kind == "slide" and event.slide_start is not None:
            add(blocks["absolute"], f"lane_{event.slide_start}_slide_start", weight)
            if event.slide_start_tap in {"omitted_fade", "omitted_sudden"}:
                add(blocks["absolute"], f"lane_{event.slide_start}_slide_starless", weight)
            if event.slide_start_break:
                add(blocks["absolute"], f"lane_{event.slide_start}_break", weight)
            if event.slide_start_ex:
                add(blocks["absolute"], f"lane_{event.slide_start}_ex", weight)
            add_slide_features(blocks, event)
        elif event.kind in {"touch", "touch_hold"}:
            add_touch_features(blocks, event)

    add_transition_features(blocks, ordered)
    add_each_features(blocks, ordered)
    return blocks


def add_slide_features(blocks: dict[str, dict[str, float]], event: PatternEvent) -> None:
    weight = event.weight
    if event.slide_start_tap:
        add(blocks["slide"], f"start_tap_{event.slide_start_tap}", weight)
    if event.slide_start_break:
        add(blocks["slide"], "slide_modifier_start_break", weight)
    if event.slide_start_ex:
        add(blocks["slide"], "slide_modifier_start_ex", weight)
    add(blocks["slide"], f"track_count_{bin_name(len(event.slide_tracks), TRACK_BINS)}", weight)

    for track in event.slide_tracks:
        if track.break_track:
            add(blocks["slide"], "slide_modifier_track_break", weight)
        add(blocks["slide"], f"segment_count_{bin_name(len(track.segments), SEGMENT_BINS)}", weight)
        for segment in track.segments:
            shape_weight = weight * SLIDE_SHAPE_FEATURE_WEIGHT
            shape_endpoint_weight = weight * SLIDE_SHAPE_ENDPOINT_FEATURE_WEIGHT
            path_length_weight = weight * SLIDE_PATH_LENGTH_FEATURE_WEIGHT
            add(blocks["slide"], f"shape_{segment.shape}", shape_weight)
            distance = circular_distance(segment.start, segment.end)
            add(blocks["slide"], f"shape_endpoint_dist_{segment.shape}_{distance}", shape_endpoint_weight)
            add(blocks["slide"], f"shape_start_end_{segment.shape}_{segment.start}_{segment.end}", shape_weight)
            add(blocks["slide"], f"slide_declared_distance_{distance}", weight)
            add(blocks["slide"], f"slide_direction_{direction_label(segment.start, segment.end)}", weight)
            if event.slide_start is not None:
                add(blocks["absolute"], f"lane_{event.slide_start}_slide_start", weight)
            add(blocks["absolute"], f"lane_{segment.end}_slide_end", weight)
            if segment.middle is not None:
                add(blocks["absolute"], f"lane_{segment.middle}_slide_middle", weight)
                add(blocks["slide"], f"shape_middle_V_{segment.middle}", shape_weight)
            for shape, soft_weight in soft_shape_weights(segment).items():
                add(blocks["slide"], f"soft_shape_{shape}", soft_weight * shape_weight)
            length = path_length(sample_slide_segment(segment))
            length_bin = str(min(int(length), int(PATH_LENGTH_BINS[-1])))
            add(blocks["slide"], f"path_length_bin_{length_bin}", path_length_weight)


def add_touch_features(blocks: dict[str, dict[str, float]], event: PatternEvent) -> None:
    weight = event.weight
    sensor = canonical_sensor_name(event.sensor_family, event.sensor_index)
    if sensor is not None:
        add(blocks["touch"], f"sensor_{sensor}_{event.kind}", weight)
    if event.sensor_family is not None:
        add(blocks["touch"], f"sensor_family_{event.sensor_family}_{event.kind}", weight)
    if event.sensor_index is not None:
        add(blocks["touch"], f"touch_index_{event.sensor_index}", weight)
    if event.sensor_family == "C":
        add(blocks["touch"], "touch_center_count", weight)
    if "f" in event.modifiers:
        add(blocks["touch"], "touch_modifier_firework", weight)
    if "x" in event.modifiers:
        add(blocks["touch"], "touch_modifier_ex", weight)


def add_transition_features(blocks: dict[str, dict[str, float]], ordered: list[PatternEvent]) -> None:
    lane_events = [
        event for event in ordered
        if representative_lane(event) is not None
    ]
    for previous, current in zip(lane_events, lane_events[1:]):
        left = representative_lane(previous)
        right = representative_lane(current)
        if left is None or right is None:
            continue
        weight = min(previous.weight, current.weight)
        add(blocks["absolute"], f"transition_{left}_to_{right}", weight)
        distance = circular_distance(left, right)
        add(blocks["topology"], f"transition_dist_{distance}", weight)
        add(blocks["topology"], f"transition_dir_{direction_label(left, right)}", weight)
        add(blocks["topology"], f"kind_bigram_{previous.kind}_to_{current.kind}", weight)


def add_each_features(blocks: dict[str, dict[str, float]], ordered: list[PatternEvent]) -> None:
    groups: dict[str, list[PatternEvent]] = {}
    for event in ordered:
        groups.setdefault(event.group_id, []).append(event)
    for group in groups.values():
        size = len(group)
        size_name = str(size) if size < 5 else "5_plus"
        group_weight = min(event.weight for event in group)
        add(blocks["topology"], f"each_size_{size_name}", group_weight)
        lanes = [
            (event, lane) for event in group
            if (lane := representative_lane(event)) is not None
        ]
        for index, (left_event, left) in enumerate(lanes):
            for right_event, right in lanes[index + 1:]:
                pair_weight = min(left_event.weight, right_event.weight)
                add(blocks["topology"], f"each_pair_dist_{circular_distance(left, right)}", pair_weight)


def add_hand_assignment_features(
    block: dict[str, float],
    assignments: list[HandAssignedEvent],
) -> None:
    measure_loads = {
        measure: {hand: 0.0 for hand in HAND_NAMES}
        for measure in range(EXPECTED_MEASURES)
    }
    by_hand = {hand: [] for hand in HAND_NAMES}

    for assignment in assignments:
        event = assignment.event
        hand = assignment.hand
        lane = assignment.lane
        weight = assignment.weight
        if weight <= 0:
            continue

        side = lane_side(lane)
        add(block, f"{hand}_load", weight)
        add(block, "total_load", weight)
        add(block, f"{hand}_lane_{lane}_load", weight)
        add(block, f"{hand}_side_{side}_load", weight)
        if hand == side:
            add(block, "preferred_hand_load", weight)
        else:
            add(block, "cross_hand_load", weight)
        if 0 <= event.measure_slot < EXPECTED_MEASURES:
            measure_loads[event.measure_slot][hand] += weight
            add(block, f"measure_{event.measure_slot}_{hand}_load", weight)
        by_hand[hand].append(assignment)

    left_load = block["left_load"]
    right_load = block["right_load"]
    total_load = left_load + right_load
    if total_load > 0:
        block["load_balance_abs"] = abs(left_load - right_load) / total_load
        block["left_minus_right_load"] = (left_load - right_load) / total_load

    for measure, loads in measure_loads.items():
        measure_total = loads["left"] + loads["right"]
        if measure_total > 0:
            block[f"measure_{measure}_balance_abs"] = (
                abs(loads["left"] - loads["right"]) / measure_total
            )

    for hand, hand_assignments in by_hand.items():
        ordered = sorted(
            hand_assignments,
            key=lambda item: (item.event.global_beat, item.event.order, item.lane),
        )
        for previous, current in zip(ordered, ordered[1:]):
            weight = min(previous.weight, current.weight)
            if weight <= 0:
                continue
            distance = circular_distance(previous.lane, current.lane)
            add(block, f"{hand}_transition_dist_{distance}", weight)
            add(block, f"{hand}_transition_dir_{direction_label(previous.lane, current.lane)}", weight)
            add(block, f"{hand}_movement_total_distance", distance * weight)


def force_lr_each_assignments(group: list[PatternEvent]) -> list[HandAssignedEvent] | None:
    lane_events = [
        event for event in sorted(group, key=lambda item: item.order)
        if (representative_lane(event) is not None)
    ]
    if len(lane_events) != 2 or any(event.group_size != 2 for event in lane_events):
        return None

    left, right = lane_events
    left_lane = representative_lane(left)
    right_lane = representative_lane(right)
    if left_lane is None or right_lane is None:
        return None

    score_first_left = (
        lane_hand_probabilities(left_lane)["left"]
        + lane_hand_probabilities(right_lane)["right"]
    )
    score_second_left = (
        lane_hand_probabilities(right_lane)["left"]
        + lane_hand_probabilities(left_lane)["right"]
    )
    if score_second_left > score_first_left:
        left, right = right, left
        left_lane, right_lane = right_lane, left_lane

    return [
        HandAssignedEvent(left, "left", left_lane, left.weight),
        HandAssignedEvent(right, "right", right_lane, right.weight),
    ]


def add_lr_each_pair_features(block: dict[str, float], assignments: list[HandAssignedEvent]) -> None:
    if len(assignments) != 2:
        return
    left = next((item for item in assignments if item.hand == "left"), None)
    right = next((item for item in assignments if item.hand == "right"), None)
    if left is None or right is None:
        return

    weight = min(left.weight, right.weight)
    add(block, "forced_each_pair_count", weight)
    add(block, f"each_lr_pair_dist_{circular_distance(left.lane, right.lane)}", weight)
    left_side = lane_side(left.lane)
    right_side = lane_side(right.lane)
    if left_side != right_side:
        add(block, "each_lr_pair_side_split", weight)
    else:
        add(block, f"each_lr_pair_side_{left_side}", weight)


def opposite_hand(hand: str) -> str:
    return "right" if hand == "left" else "left"


def dominant_hand_for_lane(lane: int) -> str:
    probabilities = lane_hand_probabilities(lane)
    return "right" if probabilities["right"] >= probabilities["left"] else "left"


def is_star_slide(event: PatternEvent) -> bool:
    return (
        event.kind == "slide"
        and event.slide_start is not None
        and event.slide_start_tap == "star"
    )


def star_slide_duration_beats(event: PatternEvent) -> float:
    durations = [
        track.duration_beats
        for track in event.slide_tracks
        if track.duration_beats is not None and track.duration_beats > 0
    ]
    if durations:
        return max(durations)
    return STAR_SLIDE_DEFAULT_DURATION_BEATS


def star_slide_trace_interval(event: PatternEvent) -> tuple[float, float]:
    start = event.global_beat + STAR_SLIDE_DELAY_BEATS
    end = start + max(star_slide_duration_beats(event), 1e-6)
    return start, end


def guiding_each_hand_for_star(
    event: PatternEvent,
    forced_each_assignments: dict[str, list[HandAssignedEvent]],
) -> str | None:
    lane = event.slide_start
    if lane is None:
        return None

    window_start, window_end = STAR_EACH_GUIDE_WINDOW_BEATS
    candidates: list[tuple[float, str]] = []
    for group_assignments in forced_each_assignments.values():
        if not group_assignments:
            continue
        group_beat = group_assignments[0].event.global_beat
        delta = group_beat - event.global_beat
        if delta < window_start or delta > window_end:
            continue
        for assignment in group_assignments:
            if assignment.lane == lane:
                candidates.append((abs(delta - STAR_SLIDE_DELAY_BEATS), assignment.hand))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def active_star_hands_at(
    beat: float,
    active_star_assignments: list[tuple[PatternEvent, str, float, float]],
) -> set[str]:
    return {
        hand
        for _event, hand, start, end in active_star_assignments
        if start <= beat < end
    }


def register_active_star_assignment(
    active_star_assignments: list[tuple[PatternEvent, str, float, float]],
    assignment: HandAssignedEvent,
) -> None:
    if not is_star_slide(assignment.event):
        return
    start, end = star_slide_trace_interval(assignment.event)
    active_star_assignments.append((assignment.event, assignment.hand, start, end))


def assign_star_slide_hand(
    event: PatternEvent,
    block: dict[str, float],
    forced_each_assignments: dict[str, list[HandAssignedEvent]],
    active_star_assignments: list[tuple[PatternEvent, str, float, float]],
) -> HandAssignedEvent:
    lane = event.slide_start
    if lane is None:
        lane = representative_lane(event)
    if lane is None:
        raise ValueError("star slide has no representative lane")

    hand = dominant_hand_for_lane(lane)
    source = "head"

    guided_hand = guiding_each_hand_for_star(event, forced_each_assignments)
    if guided_hand is not None:
        hand = guided_hand
        source = "each"

    trace_start, _ = star_slide_trace_interval(event)
    active_hands = active_star_hands_at(trace_start, active_star_assignments)
    if len(active_hands) == 1:
        hand = opposite_hand(next(iter(active_hands)))
        source = "overlap"
    elif len(active_hands) > 1:
        add(block, "star_slide_overlap_conflict_count", event.weight)

    if source == "head":
        add(block, "star_head_default_count", event.weight)
    elif source == "each":
        add(block, "star_each_guided_count", event.weight)
    elif source == "overlap":
        add(block, "star_slide_overlap_forced_count", event.weight)

    return HandAssignedEvent(event, hand, lane, event.weight)


def extract_hand_balance_block(events: list[PatternEvent]) -> dict[str, float]:
    block = {name: 0.0 for name in schema_for_hand_balance()}
    groups: dict[str, list[PatternEvent]] = {}
    ordered = sorted(events, key=lambda item: (item.global_beat, item.order))
    for event in ordered:
        if representative_lane(event) is not None:
            groups.setdefault(event.group_id, []).append(event)

    forced_each_assignments = {
        group_id: forced
        for group_id, group in groups.items()
        if (forced := force_lr_each_assignments(group)) is not None
    }
    active_star_assignments: list[tuple[PatternEvent, str, float, float]] = []
    assignments: list[HandAssignedEvent] = []
    for group_id, group in groups.items():
        forced = forced_each_assignments.get(group_id)
        if forced is not None:
            assignments.extend(forced)
            add_lr_each_pair_features(block, forced)
            for assignment in forced:
                register_active_star_assignment(active_star_assignments, assignment)
            continue

        lane_events = [
            event for event in sorted(group, key=lambda item: item.order)
            if (representative_lane(event) is not None)
        ]
        if len(lane_events) > 1:
            add(block, "multi_each_group_count", min(event.weight for event in lane_events))
        else:
            add(block, "soft_single_event_count", lane_events[0].weight if lane_events else 0.0)

        for event in lane_events:
            lane = representative_lane(event)
            if lane is None:
                continue
            if is_star_slide(event):
                assignment = assign_star_slide_hand(
                    event,
                    block,
                    forced_each_assignments,
                    active_star_assignments,
                )
                assignments.append(assignment)
                register_active_star_assignment(active_star_assignments, assignment)
                continue
            probabilities = lane_hand_probabilities(lane)
            for hand, probability in probabilities.items():
                assignments.append(HandAssignedEvent(event, hand, lane, event.weight * probability))

    add_hand_assignment_features(block, assignments)
    return block


def d8_pooled_blocks(events: list[PatternEvent], total_beats: float) -> dict[str, dict[str, float]]:
    schema = base_schema()
    pooled_names = schema["absolute"] + schema["topology"] + schema["slide"] + schema["touch"]
    all_vectors: list[dict[str, float]] = []
    for mirror in (False, True):
        for rotation in range(8):
            transformed = [
                transform_event(event, rotation=rotation, mirror=mirror)
                for event in events
            ]
            blocks = extract_base_blocks(transformed, total_beats)
            merged: dict[str, float] = {}
            for block in ("absolute", "topology", "slide", "touch"):
                merged.update(blocks[block])
            all_vectors.append(merged)

    mean_values = {name: 0.0 for name in pooled_names}
    max_values = {name: 0.0 for name in pooled_names}
    for name in pooled_names:
        values = [vector.get(name, 0.0) for vector in all_vectors]
        mean_values[name] = sum(values) / len(values)
        max_values[name] = max(values)
    return {"d8_mean": mean_values, "d8_max": max_values}


def full_schema() -> dict[str, list[str]]:
    schema = base_schema()
    pooled_names = schema["absolute"] + schema["topology"] + schema["slide"] + schema["touch"]
    schema["d8_mean"] = list(pooled_names)
    schema["d8_max"] = list(pooled_names)
    schema["hand_balance"] = schema_for_hand_balance()
    schema["rhythm_fft"] = schema_for_rhythm_fft()
    return schema


def normalize_block(values: list[float]) -> list[float]:
    transformed = [math.log1p(max(0.0, value)) for value in values]
    norm = math.sqrt(sum(value * value for value in transformed))
    if norm <= 0:
        return transformed
    return [value / norm for value in transformed]


def normalize_signed_block(values: list[float]) -> list[float]:
    transformed = [
        math.copysign(math.log1p(abs(value)), value)
        for value in values
    ]
    norm = math.sqrt(sum(value * value for value in transformed))
    if norm <= 0:
        return transformed
    return [value / norm for value in transformed]


def vectorize_blocks(blocks: dict[str, dict[str, float]]) -> tuple[list[float], list[str], dict[str, dict[str, int]]]:
    schema = full_schema()
    vector: list[float] = []
    names: list[str] = []
    slices: dict[str, dict[str, int]] = {}
    cursor = 0
    for block in BLOCK_ORDER:
        block_names = schema[block]
        raw_values = [blocks.get(block, {}).get(name, 0.0) for name in block_names]
        normalized = normalize_signed_block(raw_values) if block in SIGNED_BLOCKS else normalize_block(raw_values)
        weight = BLOCK_WEIGHTS[block]
        weighted = [value * weight for value in normalized]
        vector.extend(weighted)
        names.extend(f"{block}:{name}" for name in block_names)
        slices[block] = {"start": cursor, "end": cursor + len(block_names)}
        cursor += len(block_names)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector, names, slices


def sparse_nonzero_features(blocks: dict[str, dict[str, float]], *, limit: int = 200) -> dict[str, list[dict[str, float]]]:
    sparse: dict[str, list[dict[str, float]]] = {}
    for block, values in blocks.items():
        items = [
            {"name": name, "value": value}
            for name, value in values.items()
            if abs(value) > 1e-12
        ]
        items.sort(key=lambda item: (-abs(item["value"]), item["name"]))
        sparse[block] = items[:limit]
    return sparse


def build_embedding_from_events(
    events: list[PatternEvent],
    *,
    total_beats: float,
    rhythm_bpm: float | None = None,
    measure_weights: list[float] | None = None,
    include_sparse_features: bool = True,
) -> dict[str, Any]:
    blocks = extract_base_blocks(events, total_beats)
    blocks.update(d8_pooled_blocks(events, total_beats))
    blocks["hand_balance"] = extract_hand_balance_block(events)
    blocks["rhythm_fft"] = extract_rhythm_fft_block(events, total_beats, rhythm_bpm)
    vector, feature_names, block_slices = vectorize_blocks(blocks)
    payload: dict[str, Any] = {
        "embedding": vector,
        "dimension": len(vector),
        "feature_names": feature_names,
        "block_slices": block_slices,
        "block_weights": BLOCK_WEIGHTS,
        "event_count": len(events),
        "weighted_event_count": sum(event.weight for event in events),
        "total_beats": total_beats,
        "rhythm_bpm": rhythm_bpm,
        "measure_weights": measure_weights or [],
        "overlap_measure_weight": OVERLAP_MEASURE_WEIGHT,
        "rhythm_ticks_per_measure": RHYTHM_TICKS_PER_MEASURE,
        "rhythm_tick_count": RHYTHM_EXPECTED_TICKS,
    }
    if include_sparse_features:
        payload["nonzero_features"] = sparse_nonzero_features(blocks)
    return payload


def weights_for_overlap_slots(
    measure_count: int,
    overlap_measure_slots: Iterable[int] | None,
) -> list[float]:
    weights = [1.0] * measure_count
    slots = list(overlap_measure_slots) if overlap_measure_slots is not None else []
    if not slots and measure_count == EXPECTED_MEASURES:
        slots = [measure_count - 1]
    for slot in slots:
        if 0 <= int(slot) < measure_count:
            weights[int(slot)] = OVERLAP_MEASURE_WEIGHT
    return weights


def build_embedding_from_measures(
    measures: list[dict[str, Any]],
    *,
    require_expected_measures: bool | None = None,
    require_six_measures: bool | None = None,
    overlap_measure_slots: Iterable[int] | None = None,
    include_sparse_features: bool = True,
) -> dict[str, Any]:
    if require_expected_measures is None:
        require_expected_measures = True if require_six_measures is None else require_six_measures
    if require_expected_measures and len(measures) != EXPECTED_MEASURES:
        raise ValueError(f"expected exactly {EXPECTED_MEASURES} measures, got {len(measures)}")
    measure_weights = weights_for_overlap_slots(len(measures), overlap_measure_slots)
    events, total_beats = events_from_measures(measures, measure_weights=measure_weights)
    rhythm_bpm = effective_bpm_from_measures(measures, total_beats)
    return build_embedding_from_events(
        events,
        total_beats=total_beats,
        rhythm_bpm=rhythm_bpm,
        measure_weights=measure_weights,
        include_sparse_features=include_sparse_features,
    )


def parse_measure_range(text: str) -> tuple[int, int]:
    for separator in ("..", ":", "-"):
        if separator in text:
            left, right = text.split(separator, 1)
            start = int(left.strip())
            end = int(right.strip())
            break
    else:
        start = end = int(text.strip())
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError("range must be a positive 1-based inclusive range")
    return start, end


def select_measures(chart_data: dict[str, Any], measure_range: tuple[int, int]) -> list[dict[str, Any]]:
    measures = chart_data.get("measures", [])
    if not isinstance(measures, list):
        raise ValueError("chart JSON does not contain a measures list")
    start, end = measure_range
    if end > len(measures):
        raise ValueError(f"range {start}:{end} exceeds chart length {len(measures)}")
    return measures[start - 1:end]


def chart_metadata(chart_path: Path | None, chart_data: dict[str, Any]) -> dict[str, Any]:
    song = chart_data.get("song", {}) if isinstance(chart_data.get("song"), dict) else {}
    chart = chart_data.get("chart", {}) if isinstance(chart_data.get("chart"), dict) else {}
    return {
        "source_file": str(chart_path) if chart_path else None,
        "song_id": song.get("song_id"),
        "title": song.get("title"),
        "chart_kind": chart.get("chart_kind"),
        "difficulty_index": chart.get("difficulty_index"),
        "difficulty_name": chart.get("difficulty_name"),
        "level": chart.get("level"),
    }


def chart_data_from_simai(source: str) -> dict[str, Any]:
    compiled = compile_chart(source)
    return compiled_to_dict(compiled, include_slots=False, include_raw=True)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a four-measure Simai fragment into a rule-based structural embedding."
    )
    parser.add_argument("--chart-file", type=Path, default=None)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("outputs/simai_measures"),
        help="Directory containing index.json when resolving by song/title.",
    )
    parser.add_argument("--song-id", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--chart-kind", default=None)
    parser.add_argument("--difficulty-index", type=int, default=None)
    parser.add_argument("--simai", default=None, help="Raw Simai chart text or fragment.")
    parser.add_argument("--simai-file", type=Path, default=None, help="Read raw Simai chart text from a file.")
    parser.add_argument(
        "--range",
        type=parse_measure_range,
        default=(1, EXPECTED_MEASURES),
        help="1-based inclusive measure range. Default: 1:4.",
    )
    parser.add_argument(
        "--allow-non-six",
        action="store_true",
        help="Allow ranges that are not exactly four measures.",
    )
    parser.add_argument(
        "--overlap-slot",
        type=int,
        action="append",
        default=None,
        help="Zero-based measure slot to weight as overlap. Default: last slot.",
    )
    parser.add_argument(
        "--no-sparse-features",
        action="store_true",
        help="Do not include the nonzero feature preview in JSON output.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chart_path: Path | None = None
    if args.simai_file is not None:
        chart_data = chart_data_from_simai(args.simai_file.read_text(encoding="utf-8-sig"))
    elif args.simai is not None:
        chart_data = chart_data_from_simai(args.simai)
    else:
        chart_path = resolve_chart(
            chart_file=args.chart_file,
            index_dir=args.index_dir,
            song_id=args.song_id,
            title=args.title,
            chart_kind=args.chart_kind,
            difficulty_index=args.difficulty_index,
        )
        chart_data = read_json(chart_path)

    measures = select_measures(chart_data, args.range)
    embedding = build_embedding_from_measures(
        measures,
        require_expected_measures=not args.allow_non_six,
        overlap_measure_slots=args.overlap_slot,
        include_sparse_features=not args.no_sparse_features,
    )
    payload = {
        "chart": chart_metadata(chart_path, chart_data),
        "range": {
            "start_measure": args.range[0],
            "end_measure": args.range[1],
            "measure_count": len(measures),
        },
        **embedding,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
