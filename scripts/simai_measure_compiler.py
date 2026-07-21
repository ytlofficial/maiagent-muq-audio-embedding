#!/usr/bin/env python3
"""Split Simai / maidata charts into measures and compute music time ranges.

This module intentionally focuses on Simai's timeline layer: BPM directives,
divider directives, comma-delimited slots, and the terminating ``E``. It keeps
note text opaque, so it remains useful for real-world maidata files that contain
simulator-specific note extensions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable


FIELD_RE = re.compile(r"^&([^=\s]+)=(.*)$")
DIRECTIVE_RE = re.compile(
    r"""
    \s*
    (?:
        \((?P<bpm>\d+(?:\.\d+)?)\)
      | \{\#(?P<exact>\d+(?:\.\d+)?)\}
      | \{(?P<divider>\d+(?:\.\d+)?)\}
    )
    """,
    re.VERBOSE,
)
NORMALIZATION_TARGET_DIVIDERS = (
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
)
MAX_NORMALIZED_BEAT_ERROR = Fraction(1, 96)
MAX_CUMULATIVE_NORMALIZED_BEAT_ERROR = Fraction(1, 32)
STRICT_NORMALIZED_BEAT_ERROR = Fraction(1, 384)


class SimaiTimelineError(ValueError):
    """Raised when a chart cannot be compiled into a timeline."""


@dataclass(frozen=True)
class TimingDirective:
    kind: str
    value: float
    raw: str


@dataclass
class Slot:
    index: int
    raw: str
    note_text: str
    directives: list[TimingDirective]
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    beat_length: float
    bpm: float | None
    divider: float | None
    exact_divider_seconds: float | None


@dataclass
class Measure:
    index: int
    start_seconds: float
    end_seconds: float
    start_slot: int
    end_slot: int
    beats: float
    bpm: float | None
    ended_by: str
    slots: list[Slot] = field(default_factory=list)


@dataclass
class CompiledChart:
    source_kind: str
    difficulty: str | None
    first_seconds: float
    measure_beats: float
    slots: list[Slot]
    measures: list[Measure]
    terminator: str
    normalization_notes: list[str] = field(default_factory=list)


def parse_maidata(path: Path) -> dict[str, str]:
    """Parse a legacy maidata ``&key=value`` envelope."""

    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = FIELD_RE.match(line)
        if match:
            if current_key is not None:
                fields[current_key] = "\n".join(current_lines).strip()
            current_key = match.group(1)
            current_lines = [match.group(2)]
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        fields[current_key] = "\n".join(current_lines).strip()

    return fields


def parse_number(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SimaiTimelineError(f"invalid {context}: {value!r}") from exc
    if parsed <= 0:
        raise SimaiTimelineError(f"{context} must be positive: {value!r}")
    return parsed


def first_seconds_for(fields: dict[str, str], difficulty: str | None) -> float:
    values: list[str | None] = []
    if difficulty:
        values.append(fields.get(f"first_{difficulty}"))
    values.append(fields.get("first"))

    for value in values:
        if value is None or not value.strip():
            continue
        return float(value.strip())
    return 0.0


def ensure_terminator(chart: str) -> tuple[str, list[str]]:
    """Append an ``E`` terminator when a chart omits it."""
    stripped = chart.strip()
    if not stripped:
        raise SimaiTimelineError("empty chart")

    try:
        split_chart_slots(stripped)
    except SimaiTimelineError as exc:
        if str(exc) != "missing terminating E":
            raise
    else:
        return stripped, []

    if stripped.endswith(","):
        return f"{stripped}E", ["appended missing E terminator"]
    return f"{stripped},E", ["appended missing E terminator"]


def split_chart_slots(chart: str) -> tuple[list[str], str]:
    """Split a raw Simai chart into comma slots and the final terminator."""

    stripped = chart.strip()
    if not stripped:
        raise SimaiTimelineError("empty chart")

    bracket_depth = 0
    slots: list[str] = []
    current: list[str] = []
    index = 0

    while index < len(stripped):
        char = stripped[index]
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "," and bracket_depth == 0:
            slots.append("".join(current))
            current = []
            index += 1
            continue
        elif char == "E" and bracket_depth == 0 and not "".join(current).strip():
            tail = stripped[index + 1 :].strip()
            if tail:
                current.append(char)
                index += 1
                continue
            return slots, "E"

        current.append(char)
        index += 1

    raise SimaiTimelineError("missing terminating E")


def parse_slot_prefix(raw: str) -> tuple[list[TimingDirective], str]:
    directives: list[TimingDirective] = []
    position = 0

    while True:
        match = DIRECTIVE_RE.match(raw, position)
        if not match:
            break
        raw_directive = match.group(0).strip()
        if match.group("bpm") is not None:
            directives.append(
                TimingDirective("bpm", parse_number(match.group("bpm"), "BPM"), raw_directive)
            )
        elif match.group("exact") is not None:
            directives.append(
                TimingDirective(
                    "exact_divider_seconds",
                    parse_number(match.group("exact"), "exact divider seconds"),
                    raw_directive,
                )
            )
        else:
            directives.append(
                TimingDirective(
                    "divider",
                    parse_number(match.group("divider"), "divider"),
                    raw_directive,
                )
            )
        position = match.end()

    return directives, raw[position:].strip()


def replace_divider_directive(
    raw: str,
    target_divider: int,
    *,
    source_divider: float | None = None,
) -> str:
    replaced = False

    def replace_match(match: re.Match[str]) -> str:
        nonlocal replaced
        if replaced or match.group("divider") is None:
            return match.group(0)
        if source_divider is not None and float(match.group("divider")) != source_divider:
            return match.group(0)
        replaced = True
        return match.group(0).replace(match.group(0).strip(), f"{{{target_divider}}}")

    return DIRECTIVE_RE.sub(replace_match, raw, count=0)


def slot_has_bpm_change(slot: str) -> bool:
    directives, _ = parse_slot_prefix(slot)
    return any(directive.kind == "bpm" for directive in directives)


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def directive_for_slot_timing(slot: Slot) -> str:
    if slot.exact_divider_seconds is not None:
        return f"{{#{format_number(slot.exact_divider_seconds)}}}"
    if slot.divider is not None:
        return f"{{{format_number(slot.divider)}}}"
    return ""


def raw_starts_with_slot_timing(raw: str) -> bool:
    directives, _ = parse_slot_prefix(raw)
    return bool(directives and directives[-1].kind in {"divider", "exact_divider_seconds"})


def normalized_measure_simai(measure: Measure) -> str:
    if not measure.slots:
        return ""

    parts = [f"{slot.raw}," for slot in measure.slots]
    first_slot = measure.slots[0]
    if raw_starts_with_slot_timing(first_slot.raw):
        return "".join(parts)

    timing_directive = directive_for_slot_timing(first_slot)
    if not timing_directive:
        return "".join(parts)

    directives, note_text = parse_slot_prefix(first_slot.raw)
    bpm_prefix = "".join(directive.raw for directive in directives if directive.kind == "bpm")
    other_prefix = "".join(
        directive.raw for directive in directives if directive.kind != "bpm"
    )
    if bpm_prefix:
        parts[0] = f"{bpm_prefix}{timing_directive}{other_prefix}{note_text},"
    else:
        parts[0] = f"{timing_directive}{first_slot.raw},"
    return "".join(parts)


def slot_has_any_timing_directive(raw: str) -> bool:
    directives, _ = parse_slot_prefix(raw)
    return bool(directives)


def high_divider_for_slot(raw: str, threshold: float) -> float | None:
    directives, _ = parse_slot_prefix(raw)
    dividers = [
        directive.value
        for directive in directives
        if directive.kind == "divider" and directive.value > threshold
    ]
    if len(dividers) != 1:
        return None
    return dividers[0]


def fraction_from_number(value: float) -> Fraction:
    return Fraction(format_number(value))


def large_divider_conversion_plan(
    value: float,
    *,
    default_target_divider: int,
) -> tuple[float, int, str | None]:
    if value in {639.0, 641.0}:
        return 640.0, 32, "{639}/{641}->{640}"
    if value == 983.0:
        return 984.0, 24, "{983}->{984}"
    if value == 625.0:
        return 624.0, 24, "{625}->{624}"
    if value == 995.0:
        return 996.0, 12, "{995}->{996}"
    if value == 989.0:
        return 984.0, 24, "{989}->{984}"
    if value == 937.0:
        return 936.0, 8, "{937}->{936}"
    if value == 673.0:
        return 672.0, 16, "{673}->{672}"
    if value == 512.0:
        return 512.0, 64, "{512}->nearest {64}"
    if value == 500.0:
        return 500.0, 16, "{500}->nearest {16}"
    if value == 621.0:
        return 624.0, 32, "{621}->{624}"
    if value == 613.0:
        return 613.0, 32, "{613}->nearest {32}"
    if value == 637.0:
        return 640.0, 32, "{637}->{640}"
    if value == 993.0:
        return 992.0, 8, "{993}->{992}"
    if value == 828.0:
        return 832.0, 64, "{828}->{832}"
    if value == 833.0:
        return 832.0, 4, "{833}->{832}"
    return value, default_target_divider, None


def large_divider_conversion_plan_for_run(
    value: float,
    run_length: int,
    *,
    default_target_divider: int,
) -> tuple[float, int, str | None]:
    if value == 977.0 and 280 <= run_length <= 290:
        return 977.0, 24, "{977}->nearest {24}"
    if value == 673.0 and 6 <= run_length <= 30:
        return 672.0, 96, "{673}->{672}, nearest {96}"
    if value == 844.0:
        if 30 <= run_length <= 36 or 95 <= run_length <= 105:
            return 844.0, 128, "{844}->nearest {128}"
        if 60 <= run_length <= 70 or 140 <= run_length <= 150 or 270 <= run_length <= 285:
            return 844.0, 64, "{844}->nearest {64}"
    if value == 641.0:
        if 4 <= run_length <= 6:
            return 640.0, 128, "{641}->{640}, nearest {128}"
        if 28 <= run_length <= 32:
            return 640.0, 64, "{641}->{640}, nearest {64}"
    if value == 649.0:
        if 68 <= run_length <= 74 or 138 <= run_length <= 146 or 208 <= run_length <= 218:
            return 649.0, 64, "{649}->nearest {64}"
    if value == 643.0:
        if 64 <= run_length <= 70 or 130 <= run_length <= 138:
            return 643.0, 48, "{643}->nearest {48}"
        if 120 <= run_length <= 130:
            return 643.0, 128, "{643}->nearest {128}"
    if value == 922.0:
        if 115 <= run_length <= 119:
            return 922.0, 8, "{922}->nearest {8}"
        if 37 <= run_length <= 41:
            return 922.0, 24, "{922}->nearest {24}"
    if value == 400.0:
        if 99 <= run_length <= 101 or 398 <= run_length <= 402:
            return 400.0, 4, "{400}->nearest {4}"
    if value == 657.0:
        if 74 <= run_length <= 80:
            return 657.0, 128, "{657}->nearest {128}"
        if 150 <= run_length <= 158:
            return 657.0, 64, "{657}->nearest {64}"
    if value == 417.0:
        if 50 <= run_length <= 54:
            return 417.0, 8, "{417}->nearest {8}"
        if 102 <= run_length <= 106:
            return 417.0, 4, "{417}->nearest {4}"
        if 206 <= run_length <= 211:
            return 417.0, 4, "{417}->nearest {4}"
        if 311 <= run_length <= 315:
            return 417.0, 4, "{417}->nearest {4}"
        if 415 <= run_length <= 419:
            return 417.0, 4, "{417}->nearest {4}"
    if value == 768.0:
        if 1 <= run_length <= 2:
            return 768.0, 256, "{768}->nearest {256}"
        if 3 <= run_length <= 4:
            return 768.0, 256, "{768}->nearest {256}"
        if 5 <= run_length <= 6:
            return 768.0, 192, "{768}->nearest {192}"
        if 7 <= run_length <= 8:
            return 768.0, 96, "{768}->nearest {96}"
        if 9 <= run_length <= 17:
            return 768.0, 64, "{768}->nearest {64}"
    if value == 999.0:
        if 123 <= run_length <= 127:
            return 999.0, 8, "{999}->nearest {8}"
        if 198 <= run_length <= 202:
            return 999.0, 256, "{999}->nearest {256}"
        if 248 <= run_length <= 252:
            return 999.0, 4, "{999}->nearest {4}"
        if 497 <= run_length <= 502:
            return 999.0, 4, "{999}->nearest {4}"
    if value == 909.0:
        if 4 <= run_length <= 5:
            return 909.0, 192, "{909}->nearest {192}"
        if 6 <= run_length <= 8:
            return 909.0, 128, "{909}->nearest {128}"
        if 9 <= run_length <= 17:
            return 909.0, 256, "{909}->nearest {256}"
    if value == 622.0 and 12 <= run_length <= 14:
        return 622.0, 48, "{622}->nearest {48}"
    if value == 601.0:
        if 35 <= run_length <= 37:
            return 601.0, 384, "{601}->nearest {384}"
        if 106 <= run_length <= 110:
            return 601.0, 128, "{601}->nearest {128}"
        if 142 <= run_length <= 146:
            return 601.0, 96, "{601}->nearest {96}"
        if 214 <= run_length <= 218:
            return 601.0, 64, "{601}->nearest {64}"
    if value == 617.0:
        if 51 <= run_length <= 55:
            return 617.0, 128, "{617}->nearest {128}"
        if 133 <= run_length <= 137:
            return 617.0, 32, "{617}->nearest {32}"
        if 239 <= run_length <= 243:
            return 617.0, 64, "{617}->nearest {64}"
    if value == 631.0:
        if 90 <= run_length <= 94:
            return 631.0, 48, "{631}->nearest {48}"
        if 136 <= run_length <= 140:
            return 631.0, 32, "{631}->nearest {32}"
        if 182 <= run_length <= 186:
            return 631.0, 24, "{631}->nearest {24}"
    if value == 832.0:
        if 49 <= run_length <= 53:
            return 832.0, 16, "{832}->nearest {16}"
        if 205 <= run_length <= 209:
            return 832.0, 4, "{832}->nearest {4}"
        if 830 <= run_length <= 834:
            return 832.0, 4, "{832}->nearest {4}"
    if value == 921.0:
        if 6 <= run_length <= 8:
            return 921.0, 128, "{921}->nearest {128}"
        if 13 <= run_length <= 15:
            return 921.0, 64, "{921}->nearest {64}"
        if 919 <= run_length <= 923:
            return 921.0, 4, "{921}->nearest {4}"
    if value != 1000.0:
        return large_divider_conversion_plan(
            value,
            default_target_divider=default_target_divider,
        )

    if 39 <= run_length <= 42:
        return 1000.0, 24, "{1000}->nearest {24}"
    if 73 <= run_length <= 77:
        return 1000.0, 64, "{1000}->nearest {64}"
    if 124 <= run_length <= 128 or 370 <= run_length <= 390:
        return 1000.0, 8, "{1000}->nearest {8}"
    if 249 <= run_length <= 252 or 730 <= run_length <= 760:
        return 1000.0, 4, "{1000}->nearest {4}"
    return 1000.0, 4, "{1000}->nearest {4}"


def nearest_nonnegative_int(value: Fraction) -> int:
    rounded = (value.numerator * 2 + value.denominator) // (2 * value.denominator)
    return max(0, rounded)


def nearest_positive_int(value: Fraction) -> int:
    return max(1, nearest_nonnegative_int(value))


def choose_target_count(
    target_count_fraction: Fraction,
    *,
    source_beats: Fraction,
    target_divider: int,
    cumulative_beat_error: Fraction | None,
) -> int:
    if cumulative_beat_error is None:
        return nearest_nonnegative_int(target_count_fraction)

    floor_count = max(1, target_count_fraction.numerator // target_count_fraction.denominator)
    candidates = {
        floor_count,
        floor_count + 1,
        nearest_positive_int(target_count_fraction),
    }

    def count_key(count: int) -> tuple[Fraction, Fraction, int]:
        delta = Fraction(4 * count, target_divider) - source_beats
        return abs(cumulative_beat_error + delta), abs(delta), count

    return min(candidates, key=count_key)


def should_round_large_divider_grid(compatibility_rule: str | None) -> bool:
    return compatibility_rule is not None


def beat_error_for_target(
    source_fraction: Fraction,
    run_length: int,
    target_divider: int,
) -> Fraction:
    return abs(beat_delta_for_target(source_fraction, run_length, target_divider))


def beat_delta_for_target(
    source_fraction: Fraction,
    run_length: int,
    target_divider: int,
) -> Fraction:
    target_count = nearest_nonnegative_int(Fraction(run_length * target_divider, 1) / source_fraction)
    if target_count <= 0:
        return Fraction(10**9)
    source_beats = Fraction(4 * run_length, 1) / source_fraction
    target_beats = Fraction(4 * target_count, target_divider)
    return target_beats - source_beats


def refine_target_divider_for_error(
    source_fraction: Fraction,
    run_length: int,
    target_divider: int,
    compatibility_rule: str | None,
    max_beat_error: Fraction = MAX_NORMALIZED_BEAT_ERROR,
    cumulative_beat_error: Fraction | None = None,
) -> int:
    if not should_round_large_divider_grid(compatibility_rule):
        return target_divider
    initial_count = nearest_nonnegative_int(Fraction(run_length * target_divider, 1) / source_fraction)
    if initial_count <= 0:
        return target_divider
    if beat_error_for_target(source_fraction, run_length, target_divider) <= max_beat_error:
        return target_divider

    candidates = [
        candidate
        for candidate in NORMALIZATION_TARGET_DIVIDERS
        if candidate >= target_divider
    ]
    if target_divider not in candidates:
        candidates.append(target_divider)

    def candidate_key(candidate: int) -> tuple[bool, Fraction | int, int]:
        error = beat_error_for_target(source_fraction, run_length, candidate)
        if cumulative_beat_error is not None:
            source_beats = Fraction(4 * run_length, 1) / source_fraction
            target_count_fraction = Fraction(run_length * candidate, 1) / source_fraction
            target_count = choose_target_count(
                target_count_fraction,
                source_beats=source_beats,
                target_divider=candidate,
                cumulative_beat_error=cumulative_beat_error,
            )
            delta = Fraction(4 * target_count, candidate) - source_beats
            cumulative_error = abs(cumulative_beat_error + delta)
            return (
                cumulative_error > MAX_CUMULATIVE_NORMALIZED_BEAT_ERROR,
                cumulative_error,
                candidate,
            )
        if error <= max_beat_error:
            return False, candidate, candidate
        return True, error, candidate

    return min(candidates, key=candidate_key)


def collect_run_notes(run: list[str]) -> list[str] | None:
    notes: list[str] = []
    for index, raw in enumerate(run):
        directives, note_text = parse_slot_prefix(raw)
        if directives and index != 0:
            return None
        if note_text:
            notes.append(note_text)
    return notes


def merge_notes_into_previous_slot(slots: list[str], notes: list[str]) -> bool:
    if not notes:
        return False

    for index in range(len(slots) - 1, -1, -1):
        directives, note_text = parse_slot_prefix(slots[index])
        if not note_text:
            continue
        prefix = slots[index][: len(slots[index]) - len(note_text)]
        slots[index] = f"{prefix}{note_text}/{'/'.join(notes)}"
        return True

    return False


def merge_note_into_slot(slot: str, note_text: str) -> str | None:
    if not note_text:
        return None
    _, existing_note_text = parse_slot_prefix(slot)
    if not existing_note_text:
        return f"{slot}{note_text}"
    prefix = slot[: len(slot) - len(existing_note_text)]
    return f"{prefix}{existing_note_text}/{note_text}"


def try_convert_large_divider_run(
    raw_slots: list[str],
    start: int,
    *,
    threshold: float,
    default_target_divider: int,
    max_beat_error: Fraction = MAX_NORMALIZED_BEAT_ERROR,
    cumulative_beat_error: Fraction | None = None,
) -> tuple[list[str] | None, int, int, str | None]:
    source_divider = high_divider_for_slot(raw_slots[start], threshold)
    if source_divider is None:
        return None, start + 1, default_target_divider, None

    end = start + 1
    while end < len(raw_slots) and not slot_has_any_timing_directive(raw_slots[end]):
        end += 1

    run = raw_slots[start:end]
    mapped_source_divider, target_divider, compatibility_rule = (
        large_divider_conversion_plan_for_run(
            source_divider,
            len(run),
            default_target_divider=default_target_divider,
        )
    )
    source_fraction = fraction_from_number(mapped_source_divider)
    target_divider = refine_target_divider_for_error(
        source_fraction,
        len(run),
        target_divider,
        compatibility_rule,
        max_beat_error=max_beat_error,
        cumulative_beat_error=cumulative_beat_error,
    )
    source_beats = Fraction(4 * len(run), 1) / source_fraction
    target_count_fraction = Fraction(len(run) * target_divider, 1) / source_fraction
    if target_count_fraction.denominator != 1:
        if not should_round_large_divider_grid(compatibility_rule):
            return None, end, target_divider, compatibility_rule
        target_count = choose_target_count(
            target_count_fraction,
            source_beats=source_beats,
            target_divider=target_divider,
            cumulative_beat_error=cumulative_beat_error,
        )
    else:
        target_count = target_count_fraction.numerator

    if target_count <= 0:
        return None, end, target_divider, compatibility_rule

    target_slots = [""] * target_count
    for source_index, raw in enumerate(run):
        directives, note_text = parse_slot_prefix(raw)
        if source_index == 0:
            target_index_fraction = Fraction(0)
            output_raw = replace_divider_directive(
                raw,
                target_divider,
                source_divider=source_divider,
            )
        elif not note_text:
            continue
        else:
            target_index_fraction = Fraction(source_index * target_divider, 1) / source_fraction
            output_raw = note_text

        if target_index_fraction.denominator != 1:
            if not should_round_large_divider_grid(compatibility_rule):
                return None, end, target_divider, compatibility_rule
            target_index = nearest_nonnegative_int(target_index_fraction)
        else:
            target_index = target_index_fraction.numerator
        if target_index >= len(target_slots):
            if not should_round_large_divider_grid(compatibility_rule):
                return None, end, target_divider, compatibility_rule
            clamped_count = len(target_slots)
            extended_count = target_index + 1
            clamped_error = abs(Fraction(4 * clamped_count, target_divider) - source_beats)
            extended_error = abs(Fraction(4 * extended_count, target_divider) - source_beats)
            if extended_error < clamped_error:
                target_slots.extend([""] * (target_index - len(target_slots) + 1))
            else:
                target_index = len(target_slots) - 1
        if target_slots[target_index]:
            if not should_round_large_divider_grid(compatibility_rule):
                return None, end, target_divider, compatibility_rule
            merged_slot = merge_note_into_slot(target_slots[target_index], note_text)
            if merged_slot is None:
                return None, end, target_divider, compatibility_rule
            target_slots[target_index] = merged_slot
            continue
        if directives and source_index != 0:
            return None, end, target_divider, compatibility_rule
        target_slots[target_index] = output_raw

    return target_slots, end, target_divider, compatibility_rule


def normalize_large_divider_runs(
    raw_slots: list[str],
    *,
    threshold: float = 384.0,
    default_target_divider: int = 32,
) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    converted_targets: Counter[int] = Counter()
    compatibility_rules: Counter[str] = Counter()
    merged_micro_runs = 0
    removed_empty_declarations = Counter[int]()
    skipped = 0
    cumulative_beat_error = Fraction(0)
    index = 0

    while index < len(raw_slots):
        source_divider = high_divider_for_slot(raw_slots[index], threshold)
        if source_divider is None:
            normalized.append(raw_slots[index])
            index += 1
            continue

        converted_run, next_index, target_divider, compatibility_rule = (
            try_convert_large_divider_run(
                raw_slots,
                index,
                threshold=threshold,
                default_target_divider=default_target_divider,
            )
        )
        if converted_run is not None and compatibility_rule:
            mapped_source_divider, _, _ = large_divider_conversion_plan_for_run(
                source_divider,
                next_index - index,
                default_target_divider=default_target_divider,
            )
            source_beats = Fraction(4 * (next_index - index), 1) / fraction_from_number(
                mapped_source_divider
            )
            converted_beats = Fraction(4 * len(converted_run), target_divider)
            beat_error = converted_beats - source_beats
            if (
                abs(cumulative_beat_error + beat_error)
                > MAX_CUMULATIVE_NORMALIZED_BEAT_ERROR
            ):
                converted_run, next_index, target_divider, compatibility_rule = (
                    try_convert_large_divider_run(
                        raw_slots,
                        index,
                        threshold=threshold,
                        default_target_divider=default_target_divider,
                        max_beat_error=STRICT_NORMALIZED_BEAT_ERROR,
                        cumulative_beat_error=cumulative_beat_error,
                    )
                )
        if converted_run is None:
            run = raw_slots[index:next_index]
            notes_to_merge = collect_run_notes(run) if len(run) <= 2 else None
            if source_divider == 833.0 and notes_to_merge == []:
                removed_empty_declarations[833] += 1
            elif notes_to_merge is not None and merge_notes_into_previous_slot(
                normalized,
                notes_to_merge,
            ):
                merged_micro_runs += 1
            else:
                normalized.extend(run)
                skipped += 1
        else:
            if compatibility_rule:
                mapped_source_divider, _, _ = large_divider_conversion_plan_for_run(
                    source_divider,
                    next_index - index,
                    default_target_divider=default_target_divider,
                )
                source_beats = Fraction(4 * (next_index - index), 1) / fraction_from_number(
                    mapped_source_divider
                )
                converted_beats = Fraction(4 * len(converted_run), target_divider)
                cumulative_beat_error += converted_beats - source_beats
            normalized.extend(converted_run)
            converted_targets[target_divider] += 1
            if compatibility_rule:
                compatibility_rules[compatibility_rule] += 1
        index = next_index

    notes: list[str] = []
    for target_divider, count in sorted(converted_targets.items()):
        notes.append(f"converted {count} large divider run(s) to {{{target_divider}}}")
    for rule, count in sorted(compatibility_rules.items()):
        notes.append(f"treated {count} large divider run(s) with {rule} before converting")
    if merged_micro_runs:
        notes.append(f"merged {merged_micro_runs} micro large-divider run(s) into previous EACH")
    for divider, count in sorted(removed_empty_declarations.items()):
        notes.append(f"removed {count} empty {{{divider}}} declaration run(s)")
    if skipped:
        notes.append(
            "kept "
            f"{skipped} large divider run(s) because they do not align to normalized grids"
        )
    return normalized, notes


def compile_chart(
    chart: str,
    *,
    first_seconds: float = 0.0,
    measure_beats: float = 4.0,
    initial_bpm: float | None = None,
    initial_divider: float | None = None,
) -> CompiledChart:
    """Compile a raw Simai chart into slots and measure ranges."""

    if measure_beats <= 0:
        raise SimaiTimelineError("measure_beats must be positive")

    chart, normalization_notes = ensure_terminator(chart)
    raw_slots, terminator = split_chart_slots(chart)
    raw_slots, large_divider_notes = normalize_large_divider_runs(raw_slots)
    normalization_notes.extend(large_divider_notes)
    current_bpm = initial_bpm
    current_divider = initial_divider
    current_exact_seconds: float | None = None
    cursor = float(first_seconds)
    slots: list[Slot] = []
    measures: list[Measure] = []
    current_measure_slots: list[Slot] = []
    current_measure_start = cursor
    current_measure_beats = 0.0
    current_measure_bpm = current_bpm

    def close_measure(end_seconds: float, ended_by: str) -> None:
        nonlocal current_measure_slots, current_measure_start, current_measure_beats
        nonlocal current_measure_bpm
        if not current_measure_slots:
            current_measure_start = end_seconds
            current_measure_beats = 0.0
            current_measure_bpm = current_bpm
            return
        measures.append(
            Measure(
                index=len(measures) + 1,
                start_seconds=current_measure_start,
                end_seconds=end_seconds,
                start_slot=current_measure_slots[0].index,
                end_slot=current_measure_slots[-1].index,
                beats=current_measure_beats,
                bpm=current_measure_bpm,
                ended_by=ended_by,
                slots=current_measure_slots,
            )
        )
        current_measure_slots = []
        current_measure_start = end_seconds
        current_measure_beats = 0.0
        current_measure_bpm = current_bpm

    for raw_index, raw_slot in enumerate(raw_slots, start=1):
        if slot_has_bpm_change(raw_slot):
            close_measure(cursor, "bpm_change")

        directives, note_text = parse_slot_prefix(raw_slot)
        for directive in directives:
            if directive.kind == "bpm":
                current_bpm = directive.value
                current_measure_bpm = current_measure_bpm or current_bpm
                if current_divider is not None and current_exact_seconds is None:
                    pass
            elif directive.kind == "divider":
                current_divider = directive.value
                current_exact_seconds = None
            elif directive.kind == "exact_divider_seconds":
                current_exact_seconds = directive.value

        if current_exact_seconds is not None:
            duration_seconds = current_exact_seconds
            if current_bpm is not None:
                beat_length = duration_seconds * current_bpm / 60.0
            else:
                beat_length = 0.0
        else:
            if current_bpm is None:
                raise SimaiTimelineError(
                    f"slot {raw_index} needs a BPM before musical timing can be used"
                )
            if current_divider is None:
                current_divider = 4.0
                normalization_notes.append("inserted default initial {4} divider")
            duration_seconds = 240.0 / current_bpm / current_divider
            beat_length = 4.0 / current_divider

        start_seconds = cursor
        end_seconds = cursor + duration_seconds
        slot = Slot(
            index=raw_index,
            raw=raw_slot,
            note_text=note_text,
            directives=directives,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration_seconds,
            beat_length=beat_length,
            bpm=current_bpm,
            divider=current_divider,
            exact_divider_seconds=current_exact_seconds,
        )
        slots.append(slot)

        if not current_measure_slots:
            current_measure_start = start_seconds
            current_measure_bpm = current_bpm
        current_measure_slots.append(slot)
        cursor = end_seconds
        current_measure_beats += beat_length

        while current_measure_beats >= measure_beats - 1e-9:
            overflow = current_measure_beats - measure_beats
            if overflow > 1e-9:
                seconds_per_beat = duration_seconds / beat_length
                end_at_boundary = cursor - overflow * seconds_per_beat
                current_measure_beats = measure_beats
                close_measure(end_at_boundary, "measure_full")
                current_measure_slots = [slot]
                current_measure_start = end_at_boundary
                current_measure_bpm = current_bpm
                current_measure_beats = overflow
                continue
            current_measure_beats = measure_beats
            close_measure(cursor, "measure_full")

    close_measure(cursor, "chart_end")

    return CompiledChart(
        source_kind="raw_simai",
        difficulty=None,
        first_seconds=float(first_seconds),
        measure_beats=float(measure_beats),
        slots=slots,
        measures=measures,
        terminator=terminator,
        normalization_notes=list(dict.fromkeys(normalization_notes)),
    )


def compile_maidata(
    path: Path,
    *,
    difficulty: str,
    first_seconds: float | None = None,
    measure_beats: float = 4.0,
) -> CompiledChart:
    fields = parse_maidata(path)
    chart_key = f"inote_{difficulty}"
    chart = fields.get(chart_key, "").strip()
    if not chart:
        raise SimaiTimelineError(f"{path} does not contain &{chart_key}=")

    compiled = compile_chart(
        chart,
        first_seconds=first_seconds
        if first_seconds is not None
        else first_seconds_for(fields, difficulty),
        measure_beats=measure_beats,
    )
    compiled.source_kind = "maidata"
    compiled.difficulty = difficulty
    return compiled


def slot_to_dict(slot: Slot, *, include_raw: bool) -> dict[str, object]:
    data: dict[str, object] = {
        "index": slot.index,
        "start_seconds": round(slot.start_seconds, 6),
        "end_seconds": round(slot.end_seconds, 6),
        "duration_seconds": round(slot.duration_seconds, 6),
        "beat_length": round(slot.beat_length, 9),
        "bpm": slot.bpm,
        "divider": slot.divider,
        "exact_divider_seconds": slot.exact_divider_seconds,
        "note_text": slot.note_text,
        "directives": [
            {"kind": directive.kind, "value": directive.value, "raw": directive.raw}
            for directive in slot.directives
        ],
    }
    if include_raw:
        data["raw"] = slot.raw
    return data


def measure_to_dict(measure: Measure, *, include_slots: bool, include_raw: bool) -> dict[str, object]:
    data: dict[str, object] = {
        "index": measure.index,
        "start_seconds": round(measure.start_seconds, 6),
        "end_seconds": round(measure.end_seconds, 6),
        "duration_seconds": round(measure.end_seconds - measure.start_seconds, 6),
        "start_slot": measure.start_slot,
        "end_slot": measure.end_slot,
        "beats": round(measure.beats, 9),
        "bpm": measure.bpm,
        "ended_by": measure.ended_by,
    }
    if include_raw:
        data["simai"] = normalized_measure_simai(measure)
    if include_slots:
        data["slots"] = [slot_to_dict(slot, include_raw=include_raw) for slot in measure.slots]
    return data


def compiled_to_dict(
    compiled: CompiledChart,
    *,
    include_slots: bool = True,
    include_raw: bool = True,
) -> dict[str, object]:
    data: dict[str, object] = {
        "source_kind": compiled.source_kind,
        "difficulty": compiled.difficulty,
        "first_seconds": compiled.first_seconds,
        "measure_beats": compiled.measure_beats,
        "slot_count": len(compiled.slots),
        "measure_count": len(compiled.measures),
        "duration_seconds": round(
            compiled.slots[-1].end_seconds - compiled.first_seconds if compiled.slots else 0.0,
            6,
        ),
        "normalization_notes": compiled.normalization_notes,
        "measures": [
            measure_to_dict(measure, include_slots=include_slots, include_raw=include_raw)
            for measure in compiled.measures
        ],
    }
    return data


def render_measure_summary(measures: Iterable[Measure]) -> str:
    lines = ["measure,start_seconds,end_seconds,duration_seconds,start_slot,end_slot,beats,bpm,ended_by"]
    for measure in measures:
        lines.append(
            ",".join(
                [
                    str(measure.index),
                    f"{measure.start_seconds:.6f}",
                    f"{measure.end_seconds:.6f}",
                    f"{measure.end_seconds - measure.start_seconds:.6f}",
                    str(measure.start_slot),
                    str(measure.end_slot),
                    f"{measure.beats:.9f}",
                    "" if measure.bpm is None else f"{measure.bpm:g}",
                    measure.ended_by,
                ]
            )
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile Simai / maidata timeline information into 4-beat measures "
            "with concrete music time ranges."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--maidata", type=Path, help="Path to a maidata.txt file.")
    source.add_argument("--chart", help="Raw Simai chart string.")
    parser.add_argument(
        "--difficulty",
        default="5",
        help="maidata difficulty index to read, for example 2, 3, 4, 5, or 6. Default: 5.",
    )
    parser.add_argument(
        "--first",
        type=float,
        default=None,
        help="Override chart start time in music seconds. Defaults to &first[_n] or 0.",
    )
    parser.add_argument(
        "--measure-beats",
        type=float,
        default=4.0,
        help="Beats per measure. Default: 4 for 4/4.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument(
        "--no-slots",
        action="store_true",
        help="Only output measure ranges, omitting per-slot details in JSON.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Omit raw slot text from JSON output.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.maidata is not None:
            compiled = compile_maidata(
                args.maidata,
                difficulty=args.difficulty,
                first_seconds=args.first,
                measure_beats=args.measure_beats,
            )
        else:
            compiled = compile_chart(
                args.chart,
                first_seconds=args.first or 0.0,
                measure_beats=args.measure_beats,
            )
    except SimaiTimelineError as exc:
        parser.exit(2, f"simai timeline error: {exc}\n")

    if args.format == "csv":
        print(render_measure_summary(compiled.measures))
    else:
        print(
            json.dumps(
                compiled_to_dict(
                    compiled,
                    include_slots=not args.no_slots,
                    include_raw=not args.no_raw,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
