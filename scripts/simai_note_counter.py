#!/usr/bin/env python3
"""Count Simai note types from official-core chart text.

The counter follows the machine-readable Simai core grammar stored in
``simai_machine_readable_spec``. It is intentionally focused on note
recognition/counting rather than endpoint validation or timing expansion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable


NUMBER_RE = r"\d+(?:\.\d+)?"
DIRECTIVE_RE = re.compile(
    rf"(?:\({NUMBER_RE}\)|\{{#{NUMBER_RE}\}}|\{{{NUMBER_RE}\}})"
)
SIMPLE_SHAPES = ("pp", "qq", "-", ">", "<", "^", "v", "p", "q", "s", "z", "w")


class SimaiNoteParseError(ValueError):
    """Raised when a Simai note token cannot be recognized."""


@dataclass
class NoteCounts:
    tap: int = 0
    hold: int = 0
    slide: int = 0
    touch: int = 0
    touch_hold: int = 0
    button_1: int = 0
    button_2: int = 0
    button_3: int = 0
    button_4: int = 0
    button_5: int = 0
    button_6: int = 0
    button_7: int = 0
    button_8: int = 0
    break_notes: int = 0
    break_taps: int = 0
    break_holds: int = 0
    break_slides: int = 0
    zetsuan_notes: int = 0
    ex_notes: int = 0
    protected_notes: int = 0
    firework_touches: int = 0
    star_taps: int = 0
    slide_start_taps: int = 0
    starless_slides: int = 0
    each_groups: int = 0
    pseudo_each_groups: int = 0
    slots: int = 0
    empty_slots: int = 0
    duration_unknowns: int = 0
    density_weight: int = 0
    non_touch_density_weight: int = 0

    @property
    def total(self) -> int:
        return self.tap + self.hold + self.slide + self.touch + self.touch_hold

    def add(self, other: "NoteCounts") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


@dataclass(frozen=True)
class ParsedNote:
    kind: str
    raw: str
    slot_index: int
    pseudo_each_index: int
    each_index: int
    contributes: dict[str, int]
    duration_values: list[str]
    density_weight: int


@dataclass
class CountResult:
    counts: NoteCounts
    notes: list[ParsedNote] = field(default_factory=list)

    def to_dict(self, *, include_notes: bool = False) -> dict[str, object]:
        data: dict[str, object] = asdict(self.counts)
        data["total"] = self.counts.total
        data["note_mix"] = note_mix_from_counts(data)
        data["special_note_ratios"] = special_ratios_from_counts(data)
        data["button_distribution"] = button_distribution_from_counts(data)
        if include_notes:
            data["notes"] = [asdict(note) for note in self.notes]
        return data


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def note_mix_from_counts(counts: dict[str, object]) -> dict[str, object]:
    tap = int(counts.get("tap", 0))
    slide = int(counts.get("slide", 0))
    hold = int(counts.get("hold", 0)) + int(counts.get("touch_hold", 0))
    touch = int(counts.get("touch", 0))
    total = tap + slide + hold + touch
    return {
        "total": total,
        "tap": {"count": tap, "ratio": ratio(tap, total)},
        "slide": {"count": slide, "ratio": ratio(slide, total)},
        "hold": {"count": hold, "ratio": ratio(hold, total)},
        "touch": {"count": touch, "ratio": ratio(touch, total)},
    }


def special_ratios_from_counts(counts: dict[str, object]) -> dict[str, object]:
    total = int(counts.get("total", 0))
    break_notes = int(counts.get("break_notes", 0))
    protected_notes = int(counts.get("protected_notes", counts.get("ex_notes", 0)))
    return {
        "break": {"count": break_notes, "ratio": ratio(break_notes, total)},
        "protected": {"count": protected_notes, "ratio": ratio(protected_notes, total)},
    }


def button_distribution_from_counts(counts: dict[str, object]) -> dict[str, object]:
    lane_counts = {
        str(button): int(counts.get(f"button_{button}", 0))
        for button in range(1, 9)
    }
    values = list(lane_counts.values())
    total = sum(values)
    mean_value = total / 8 if values else 0.0
    variance = (
        sum((value - mean_value) ** 2 for value in values) / len(values)
        if values
        else 0.0
    )
    stdev = math.sqrt(variance)
    return {
        "counts": lane_counts,
        "total": total,
        "mean": mean_value,
        "variance": variance,
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean_value if mean_value else 0.0,
        "normalized_variance": variance / (mean_value**2) if mean_value else 0.0,
        "min_lane_count": min(values) if values else 0,
        "max_lane_count": max(values) if values else 0,
    }


def strip_simai_comments(source: str) -> str:
    return "\n".join(line.split("||", 1)[0] for line in source.splitlines())


def strip_ignored_whitespace(source: str) -> str:
    return re.sub(r"\s+", "", strip_simai_comments(source))


def split_chart_slots(chart: str) -> list[str]:
    chart = strip_ignored_whitespace(chart)
    if not chart:
        raise SimaiNoteParseError("empty chart")

    bracket_depth = 0
    slots: list[str] = []
    current: list[str] = []
    index = 0

    while index < len(chart):
        char = chart[index]
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise SimaiNoteParseError(f"unmatched ']' at offset {index}")
        elif char == "," and bracket_depth == 0:
            slots.append("".join(current))
            current = []
            index += 1
            continue
        elif char == "E" and bracket_depth == 0 and not current and not chart[index + 1 :]:
            return slots

        current.append(char)
        index += 1

    raise SimaiNoteParseError("missing terminating E")


def strip_timing_directives(slot: str) -> str:
    position = 0
    while True:
        match = DIRECTIVE_RE.match(slot, position)
        if not match:
            return slot[position:]
        position = match.end()


def strip_timing_directives_with_bpm(slot: str, current_bpm: Fraction | None) -> tuple[str, Fraction | None]:
    position = 0
    while True:
        match = DIRECTIVE_RE.match(slot, position)
        if not match:
            return slot[position:], current_bpm
        raw = match.group(0)
        if raw.startswith("("):
            current_bpm = Fraction(raw[1:-1])
        position = match.end()


def split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    bracket_depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise SimaiNoteParseError(f"unmatched ']' in {text!r}")
        elif char == separator and bracket_depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def read_duration(token: str, position: int) -> int:
    if position >= len(token) or token[position] != "[":
        raise SimaiNoteParseError(f"expected duration at offset {position} in {token!r}")
    end = token.find("]", position + 1)
    if end < 0:
        raise SimaiNoteParseError(f"missing closing ']' in {token!r}")
    if end == position + 1:
        raise SimaiNoteParseError(f"empty duration in {token!r}")
    return end + 1


def duration_body(token: str, start: int, end: int) -> str:
    return token[start + 1 : end - 1]


def parse_musical_or_exact_part(part: str, current_bpm: Fraction | None) -> Fraction | None:
    if ":" in part:
        if "#" in part:
            _, part = part.split("#", 1)
        divider, count = part.split(":", 1)
        return Fraction(count) / Fraction(divider)

    if "#" in part:
        bpm_text, seconds_text = part.split("#", 1)
        return Fraction(seconds_text) * Fraction(bpm_text) / 240

    if current_bpm is None:
        return None
    return Fraction(part) * current_bpm / 240


def parse_duration_fraction(body: str, current_bpm: Fraction | None) -> Fraction | None:
    if body.startswith("#"):
        if current_bpm is None:
            return None
        return Fraction(body[1:]) * current_bpm / 240

    if "##" in body:
        _, trace_part = body.split("##", 1)
        return parse_musical_or_exact_part(trace_part, current_bpm)

    return parse_musical_or_exact_part(body, current_bpm)


def weight_for_hold_like(duration: Fraction | None) -> tuple[int, bool]:
    if duration is None:
        return 1, True
    if duration <= Fraction(1, 16):
        return 1, False
    if duration <= Fraction(1, 2):
        return 2, False
    return 2 + math.ceil(float((duration - Fraction(1, 2)) / Fraction(1, 2))), False


def weight_for_slide(duration: Fraction | None) -> tuple[int, bool]:
    if duration is None:
        return 2, True
    if duration < Fraction(1, 2):
        return 2, False
    return 2 + math.ceil(float((duration - Fraction(1, 2)) / Fraction(1, 2))), False


def combined_track_duration(durations: list[Fraction | None]) -> Fraction | None:
    if any(duration is None for duration in durations):
        return None
    if len(durations) == 1:
        return durations[0]
    return sum(durations, Fraction(0))


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass
class NoteMetrics:
    counts: NoteCounts
    duration_values: list[Fraction] = field(default_factory=list)
    density_weight: int = 0
    touch_density_weight: int = 0


def parse_shape(token: str, position: int) -> tuple[str, int] | None:
    for shape in SIMPLE_SHAPES:
        if token.startswith(shape, position):
            next_position = position + len(shape)
            if next_position < len(token) and token[next_position] in "12345678":
                return shape, next_position + 1
            return None

    if token.startswith("V", position):
        if (
            position + 2 < len(token)
            and token[position + 1] in "12345678"
            and token[position + 2] in "12345678"
        ):
            return "V", position + 3
    return None


def parse_slide_track(token: str, position: int, current_bpm: Fraction | None) -> tuple[int, list[Fraction | None], bool]:
    segment_count = 0
    durations: list[Fraction | None] = []
    track_break = False

    while position < len(token):
        if token[position] == "[":
            start = position
            position = read_duration(token, position)
            durations.append(parse_duration_fraction(duration_body(token, start, position), current_bpm))
            continue
        if token[position] == "b":
            if position + 1 < len(token) and token[position + 1] == "[":
                track_break = True
                position += 1
                continue
            if position + 1 == len(token) or token[position + 1] == "*":
                track_break = True
                return position + 1, durations, track_break
            if parse_shape(token, position + 1) is not None:
                track_break = True
                position += 1
                continue
            raise SimaiNoteParseError(f"slide BREAK marker appears before chain end in {token!r}")
        parsed_shape = parse_shape(token, position)
        if parsed_shape is None:
            break
        _, position = parsed_shape
        segment_count += 1

    if segment_count == 0 or not durations:
        raise SimaiNoteParseError(f"invalid slide track in {token!r}")
    return position, durations, track_break


def try_parse_slide(token: str, current_bpm: Fraction | None) -> NoteMetrics | None:
    if not token or token[0] not in "12345678":
        return None

    position = 1
    while position < len(token) and token[position] in "bx@":
        position += 1

    starless = False
    if position < len(token) and token[position] in "?!":
        starless = True
        position += 1

    if parse_shape(token, position) is None:
        return None

    counts = NoteCounts()
    counts.slide = 1
    density_weight = 0
    if starless:
        counts.starless_slides = 1
    else:
        counts.tap = 1
        counts.slide_start_taps = 1
        setattr(counts, f"button_{token[0]}", 1)
        density_weight += 1

    start_modifiers = token[1:position]
    if "b" in start_modifiers:
        counts.break_notes += 1
        counts.break_taps += 1
        counts.zetsuan_notes += 1
    if "x" in start_modifiers:
        counts.ex_notes += 1
        counts.protected_notes += 1

    duration_values: list[Fraction] = []
    position, durations, track_break = parse_slide_track(token, position, current_bpm)
    if track_break:
        counts.break_notes += 1
        counts.break_slides += 1
        counts.zetsuan_notes += 1
    duration = combined_track_duration(durations)
    weight, unknown = weight_for_slide(duration)
    density_weight += weight
    if unknown:
        counts.duration_unknowns += 1
    else:
        duration_values.append(duration)
    while position < len(token) and token[position] == "*":
        position += 1
        position, durations, track_break = parse_slide_track(token, position, current_bpm)
        counts.slide += 1
        if track_break:
            counts.break_notes += 1
            counts.break_slides += 1
            counts.zetsuan_notes += 1
        duration = combined_track_duration(durations)
        weight, unknown = weight_for_slide(duration)
        density_weight += weight
        if unknown:
            counts.duration_unknowns += 1
        else:
            duration_values.append(duration)

    if position != len(token):
        raise SimaiNoteParseError(f"unparsed slide suffix {token[position:]!r} in {token!r}")
    return NoteMetrics(counts=counts, duration_values=duration_values, density_weight=density_weight)


def parse_sensor(token: str) -> tuple[str, int] | None:
    if not token:
        return None
    if token[0] == "C":
        if len(token) >= 2 and token[1].isdigit():
            if token[1] in "12":
                return "C", 2
            raise SimaiNoteParseError(f"invalid C sensor alias in {token!r}")
        return "C", 1
    if len(token) >= 2 and token[0] in "ABDE" and token[1] in "12345678":
        return token[:2], 2
    return None


def parse_touch_or_touch_hold(token: str, current_bpm: Fraction | None) -> NoteMetrics | None:
    parsed_sensor = parse_sensor(token)
    if parsed_sensor is None:
        return None

    _, position = parsed_sensor
    modifiers_start = position
    while position < len(token) and token[position] in "hfx":
        position += 1
    modifiers = token[modifiers_start:position]
    duration: Fraction | None = None
    if position < len(token) and token[position] == "[":
        start = position
        position = read_duration(token, position)
        duration = parse_duration_fraction(duration_body(token, start, position), current_bpm)
    if position < len(token) and token[position] == "f":
        modifiers += "f"
        position += 1
    if position != len(token):
        raise SimaiNoteParseError(f"unparsed touch suffix {token[position:]!r} in {token!r}")

    counts = NoteCounts()
    duration_values: list[Fraction] = []
    if "h" in modifiers:
        counts.touch_hold = 1
        if duration is None:
            duration = Fraction(1, 1280)
        touch_density_weight, unknown = weight_for_hold_like(duration)
        if unknown:
            counts.duration_unknowns = 1
        else:
            duration_values.append(duration)
    else:
        counts.touch = 1
        touch_density_weight = 1
    if "f" in modifiers:
        counts.firework_touches = 1
    if "x" in modifiers:
        counts.ex_notes = 1
        counts.protected_notes = 1
    return NoteMetrics(
        counts=counts,
        duration_values=duration_values,
        density_weight=0,
        touch_density_weight=touch_density_weight,
    )


def parse_button_tap_or_hold(token: str, current_bpm: Fraction | None) -> NoteMetrics | None:
    if not token or token[0] not in "12345678":
        return None

    position = 1
    modifiers: list[str] = []
    while position < len(token):
        if token.startswith("$$", position):
            modifiers.append("$$")
            position += 2
        elif token[position] in "hbx$":
            modifiers.append(token[position])
            position += 1
        else:
            break

    has_duration = position < len(token) and token[position] == "["
    duration: Fraction | None = None
    if has_duration:
        start = position
        position = read_duration(token, position)
        duration = parse_duration_fraction(duration_body(token, start, position), current_bpm)
    while position < len(token):
        if token.startswith("$$", position):
            modifiers.append("$$")
            position += 2
        elif token[position] in "bx$":
            modifiers.append(token[position])
            position += 1
        else:
            break
    if position != len(token):
        raise SimaiNoteParseError(f"unparsed button note suffix {token[position:]!r} in {token!r}")

    counts = NoteCounts()
    setattr(counts, f"button_{token[0]}", 1)
    duration_values: list[Fraction] = []
    if "h" in modifiers:
        counts.hold = 1
        if duration is None:
            duration = Fraction(1, 1280)
        density_weight, unknown = weight_for_hold_like(duration)
        if unknown:
            counts.duration_unknowns = 1
        else:
            duration_values.append(duration)
    else:
        if has_duration:
            raise SimaiNoteParseError(f"button duration requires hold modifier in {token!r}")
        counts.tap = 1
        density_weight = 1
    if "b" in modifiers:
        counts.break_notes = 1
        counts.zetsuan_notes = 1
        if "h" in modifiers:
            counts.break_holds = 1
        else:
            counts.break_taps = 1
    if "x" in modifiers:
        counts.ex_notes = 1
        counts.protected_notes = 1
    if "$" in modifiers or "$$" in modifiers:
        counts.star_taps = 1
    return NoteMetrics(counts=counts, duration_values=duration_values, density_weight=density_weight)


def parse_note_token(token: str, current_bpm: Fraction | None) -> tuple[str, NoteMetrics]:
    if not token:
        raise SimaiNoteParseError("empty note token")

    slide_metrics = try_parse_slide(token, current_bpm)
    if slide_metrics is not None:
        return "slide", slide_metrics

    touch_metrics = parse_touch_or_touch_hold(token, current_bpm)
    if touch_metrics is not None:
        return ("touch_hold" if touch_metrics.counts.touch_hold else "touch"), touch_metrics

    button_metrics = parse_button_tap_or_hold(token, current_bpm)
    if button_metrics is not None:
        return ("hold" if button_metrics.counts.hold else "tap"), button_metrics

    raise SimaiNoteParseError(f"unknown note token {token!r}")


def expand_compact_button_taps(group: str) -> list[str]:
    if len(group) >= 2 and all(char in "12345678" for char in group):
        return list(group)

    tokens: list[str] = []
    position = 0
    while position < len(group):
        if group[position] not in "12345678":
            return [group]
        start = position
        position += 1
        modifier_start = position
        while position < len(group):
            if group.startswith("$$", position):
                position += 2
            elif group[position] in "bx$":
                position += 1
            else:
                break
        if position == modifier_start:
            return [group]
        tokens.append(group[start:position])
        if position < len(group) and group[position] not in "12345678":
            return [group]
    return tokens if len(tokens) >= 2 else [group]


def expand_each_tokens(each_tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in each_tokens:
        expanded.extend(expand_compact_button_taps(token))
    return expanded


def count_simai_notes(source: str, *, include_parsed_notes: bool = True) -> CountResult:
    """Parse a Simai chart fragment and count note types.

    A normal slide contributes one ``tap`` for the approaching star head and
    one ``slide`` for the trace. Same-start multi-slides count one shared start
    tap plus one slide per ``*`` track. Starless ``?``/``!`` slides do not add
    that start tap.
    """

    result = CountResult(counts=NoteCounts())
    slots = split_chart_slots(source)
    result.counts.slots = len(slots)
    current_bpm: Fraction | None = None

    for slot_index, raw_slot in enumerate(slots):
        note_text, current_bpm = strip_timing_directives_with_bpm(raw_slot, current_bpm)
        if not note_text:
            result.counts.empty_slots += 1
            continue

        pseudo_groups = split_top_level(note_text, "`")
        if len(pseudo_groups) > 1:
            result.counts.pseudo_each_groups += len(pseudo_groups)

        for pseudo_index, pseudo_group in enumerate(pseudo_groups):
            if not pseudo_group:
                raise SimaiNoteParseError(f"empty pseudo-EACH group in slot {slot_index}")
            each_tokens = split_top_level(pseudo_group, "/")
            expanded_tokens = expand_each_tokens(each_tokens)

            if len(expanded_tokens) > 1:
                result.counts.each_groups += 1

            touch_density_bucket = 0
            parsed_for_group: list[tuple[int, str, str, NoteMetrics]] = []
            for each_index, token in enumerate(expanded_tokens):
                kind, metrics = parse_note_token(token, current_bpm)
                result.counts.add(metrics.counts)
                result.counts.density_weight += metrics.density_weight
                result.counts.non_touch_density_weight += metrics.density_weight
                touch_density_bucket += metrics.touch_density_weight
                if include_parsed_notes:
                    parsed_for_group.append((each_index, token, kind, metrics))

            capped_touch_density = min(3, touch_density_bucket)
            result.counts.density_weight += capped_touch_density

            if not include_parsed_notes:
                continue

            remaining_touch_density = capped_touch_density
            for each_index, token, kind, metrics in parsed_for_group:
                note_touch_density = min(metrics.touch_density_weight, remaining_touch_density)
                remaining_touch_density -= note_touch_density
                result.notes.append(
                    ParsedNote(
                        kind=kind,
                        raw=token,
                        slot_index=slot_index,
                        pseudo_each_index=pseudo_index,
                        each_index=each_index,
                        contributes=asdict(metrics.counts),
                        duration_values=[format_fraction(value) for value in metrics.duration_values],
                        density_weight=metrics.density_weight + note_touch_density,
                    )
                )

    return result


def tap_or_hold_lanes_from_note_text(
    note_text: str,
    current_bpm: Fraction | None = None,
    *,
    include_holds: bool = True,
    include_slide_heads: bool = True,
) -> list[int]:
    """Return 1-8 button lanes for tap/hold notes in one timing slot.

    Normal slide star heads are treated as tap lanes. Starless slides are
    excluded because they do not create a start tap.
    """

    if not note_text:
        return []
    note_text = strip_ignored_whitespace(note_text)
    if not note_text:
        return []

    lanes: list[int] = []
    pseudo_groups = split_top_level(note_text, "`")
    for pseudo_group in pseudo_groups:
        if not pseudo_group:
            continue
        each_tokens = split_top_level(pseudo_group, "/")
        expanded_tokens = expand_each_tokens(each_tokens)
        for token in expanded_tokens:
            kind, metrics = parse_note_token(token, current_bpm)
            allowed_kinds = {"tap", "hold"} if include_holds else {"tap"}
            if kind in allowed_kinds and token and token[0] in "12345678":
                lanes.append(int(token[0]))
            elif (
                include_slide_heads
                and kind == "slide"
                and metrics.counts.slide_start_taps
                and token
                and token[0] in "12345678"
            ):
                lanes.append(int(token[0]))
    return lanes


def tap_lanes_from_note_text(
    note_text: str,
    current_bpm: Fraction | None = None,
) -> list[int]:
    """Return 1-8 button lanes for ordinary tap notes in one timing slot."""

    return tap_or_hold_lanes_from_note_text(
        note_text,
        current_bpm,
        include_holds=False,
    )


def slide_lane_paths_from_token(token: str, current_bpm: Fraction | None = None) -> list[list[int]]:
    kind, _ = parse_note_token(token, current_bpm)
    if kind != "slide" or not token or token[0] not in "12345678":
        return []

    start_lane = int(token[0])
    position = 1
    while position < len(token) and token[position] in "bx@":
        position += 1
    if position < len(token) and token[position] in "?!":
        position += 1

    paths: list[list[int]] = []
    while position < len(token):
        current_path = [start_lane]
        while position < len(token):
            if token[position] == "[":
                position = read_duration(token, position)
                continue
            if token[position] == "b":
                if position + 1 < len(token) and token[position + 1] == "[":
                    position += 1
                    continue
                if position + 1 == len(token) or token[position + 1] == "*":
                    position += 1
                    break
                if parse_shape(token, position + 1) is not None:
                    position += 1
                    continue
            parsed_shape = parse_shape(token, position)
            if parsed_shape is None:
                break
            shape, next_position = parsed_shape
            if shape == "V":
                current_path.extend([int(token[position + 1]), int(token[position + 2])])
            else:
                endpoint_position = position + len(shape)
                current_path.append(int(token[endpoint_position]))
            position = next_position
        if len(current_path) > 1:
            paths.append(current_path)
        if position < len(token) and token[position] == "*":
            position += 1
            continue
        break
    return paths


def slide_lane_paths_from_note_text(
    note_text: str,
    current_bpm: Fraction | None = None,
) -> list[list[int]]:
    if not note_text:
        return []
    note_text = strip_ignored_whitespace(note_text)
    if not note_text:
        return []

    paths: list[list[int]] = []
    pseudo_groups = split_top_level(note_text, "`")
    for pseudo_group in pseudo_groups:
        if not pseudo_group:
            continue
        each_tokens = split_top_level(pseudo_group, "/")
        expanded_tokens = expand_each_tokens(each_tokens)
        for token in expanded_tokens:
            paths.extend(slide_lane_paths_from_token(token, current_bpm))
    return paths


def count_maidata_charts(source: str) -> dict[str, dict[str, object]]:
    """Count every ``&inote_*`` chart field in a legacy maidata document."""

    charts: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in source.splitlines():
        match = re.match(r"^&([^=\s]+)=(.*)$", line)
        if match:
            current_key = match.group(1)
            if current_key.startswith("inote_"):
                charts[current_key] = [match.group(2)]
            continue
        if current_key in charts:
            charts[current_key].append(line)

    return {
        key: count_simai_notes("\n".join(lines)).to_dict(include_notes=False)
        for key, lines in charts.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count TAP/HOLD/SLIDE/TOUCH notes in Simai text.")
    parser.add_argument("input", nargs="?", help="Simai chart text, or a file path when --file is used.")
    parser.add_argument("--file", type=Path, help="Read Simai or maidata text from this file.")
    parser.add_argument(
        "--maidata",
        action="store_true",
        help="Treat input as a maidata file and count every &inote_* field.",
    )
    parser.add_argument("--include-notes", action="store_true", help="Include recognized note tokens.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.file:
        source = args.file.read_text(encoding="utf-8-sig")
    elif args.input:
        source = args.input
    else:
        raise SystemExit("provide Simai text as an argument or use --file")

    if args.maidata:
        data = count_maidata_charts(source)
    else:
        data = count_simai_notes(source).to_dict(include_notes=args.include_notes)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
