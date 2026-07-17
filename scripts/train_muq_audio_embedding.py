#!/usr/bin/env python3
"""Train a frozen-MuQ audio tower against the 512-D chart embedding space."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audio_embedding_dataset import (
    AUDIO_TABLE_NAME,
    PATTERN_TABLE_NAME,
    SEGMENT_TABLE_NAME,
    dataset_summary,
    limit_records,
    load_audio_embedding_records,
)


DEFAULT_DB_PATH = Path("outputs/lancedb/simai_pattern_chunks")
DEFAULT_TRAIN_CSV = Path("datasets/audio_embedding_charts_1000_300_300_train.csv")
DEFAULT_VALIDATION_CSV = Path(
    "datasets/audio_embedding_charts_1000_300_300_validation.csv"
)
DEFAULT_TEST_CSV = Path("datasets/audio_embedding_charts_1000_300_300_test.csv")

BOOLEAN_CONFIG_KEYS = {
    "skip_audio_file_check",
    "dry_run",
    "smoke_test",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Freeze MuQ, pool its 24 kHz frame features, condition on chart metadata, "
            "and train a 512-D audio projection with InfoNCE and cosine distillation."
        )
    )
    result.add_argument(
        "--config",
        type=Path,
        help="YAML configuration file. Explicit CLI options override its values.",
    )
    result.add_argument("--data-root", type=Path, default=REPO_ROOT)
    result.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    result.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    result.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    result.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    result.add_argument("--pattern-table", default=PATTERN_TABLE_NAME)
    result.add_argument("--audio-table", default=AUDIO_TABLE_NAME)
    result.add_argument("--segment-table", default=SEGMENT_TABLE_NAME)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--model-id", default="OpenMuQ/MuQ-large-msd-iter")
    result.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ["MUQ_MODEL_PATH"]) if os.environ.get("MUQ_MODEL_PATH") else None,
        help="Local MuQ snapshot. Falls back to --model-id when absent.",
    )
    result.add_argument("--device", default="auto")
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--validation-batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--temperature-learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--initial-temperature", type=float, default=0.07)
    result.add_argument("--distillation-weight", type=float, default=0.2)
    result.add_argument("--hidden-dimension", type=int, default=1024)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--validation-every", type=int, default=1)
    result.add_argument("--retrieval-block-size", type=int, default=256)
    result.add_argument("--max-train-samples", type=int, default=0)
    result.add_argument("--max-validation-samples", type=int, default=0)
    result.add_argument("--max-test-samples", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260715)
    result.add_argument("--resume", type=Path)
    result.add_argument(
        "--skip-audio-file-check",
        action="store_true",
        help="Build metadata even when audio files are not mounted.",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all joins and files without importing PyTorch or MuQ.",
    )
    result.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch with 32 train and 16 validation chunks.",
    )
    return result


def load_config_defaults(config_path: Path, argument_parser: argparse.ArgumentParser) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc

    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a YAML mapping")

    actions = {
        action.dest: action
        for action in argument_parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(payload) - set(actions))
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")

    defaults: dict[str, Any] = {}
    for key, value in payload.items():
        action = actions[key]
        if key in BOOLEAN_CONFIG_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"configuration key {key!r} must be true or false")
            defaults[key] = value
        elif value is not None and action.type is not None:
            defaults[key] = action.type(value)
        else:
            defaults[key] = value
    return defaults


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args(argv)
    argument_parser = parser()
    if preliminary.config:
        argument_parser.set_defaults(
            **load_config_defaults(preliminary.config, argument_parser)
        )
    return argument_parser.parse_args(argv)


def resolve_input(path: Path, data_root: Path) -> Path:
    if path.exists():
        return path.resolve()
    candidate = data_root / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"input not found: {path} (also checked {candidate})")


def resolve_output(path: Path | None, data_root: Path) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return data_root / "outputs" / "audio_embedding_runs" / f"muq_statistical_{stamp}"
    return path if path.is_absolute() else data_root / path


def ensure_splits_song_disjoint(**splits: Any) -> None:
    names = list(splits)
    song_ids = {
        name: {record.song_id for record in records}
        for name, records in splits.items()
    }
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = song_ids[left_name] & song_ids[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} song leakage: {sorted(overlap)[:10]}"
                )


def validate_arguments(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "epochs",
        "batch_size",
        "validation_batch_size",
        "validation_every",
        "retrieval_block_size",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    for name in ("max_train_samples", "max_validation_samples", "max_test_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0.01 <= args.initial_temperature <= 1.0:
        raise ValueError("--initial-temperature must be between 0.01 and 1.0")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in the range [0, 1)")
    if args.learning_rate <= 0.0 or args.temperature_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0.0 or args.distillation_weight < 0.0:
        raise ValueError("loss weights cannot be negative")


def main() -> int:
    args = parse_arguments()
    validate_arguments(args)
    data_root = args.data_root.resolve()
    db_path = resolve_input(args.db_path, data_root)
    train_csv = resolve_input(args.train_csv, data_root)
    validation_csv = resolve_input(args.validation_csv, data_root)
    test_csv = resolve_input(args.test_csv, data_root)

    train_records = load_audio_embedding_records(
        db_path=db_path,
        split_csv=train_csv,
        data_root=data_root,
        pattern_table_name=args.pattern_table,
        audio_table_name=args.audio_table,
        segment_table_name=args.segment_table,
        require_audio=not args.skip_audio_file_check,
    )
    validation_records = load_audio_embedding_records(
        db_path=db_path,
        split_csv=validation_csv,
        data_root=data_root,
        pattern_table_name=args.pattern_table,
        audio_table_name=args.audio_table,
        segment_table_name=args.segment_table,
        require_audio=not args.skip_audio_file_check,
    )
    test_records = load_audio_embedding_records(
        db_path=db_path,
        split_csv=test_csv,
        data_root=data_root,
        pattern_table_name=args.pattern_table,
        audio_table_name=args.audio_table,
        segment_table_name=args.segment_table,
        require_audio=not args.skip_audio_file_check,
    )
    ensure_splits_song_disjoint(
        train=train_records,
        validation=validation_records,
        test=test_records,
    )

    max_train = 32 if args.smoke_test and args.max_train_samples == 0 else args.max_train_samples
    max_validation = (
        16
        if args.smoke_test and args.max_validation_samples == 0
        else args.max_validation_samples
    )
    max_test = 16 if args.smoke_test and args.max_test_samples == 0 else args.max_test_samples
    train_records = limit_records(train_records, max_train, seed=args.seed)
    validation_records = limit_records(validation_records, max_validation, seed=args.seed + 1)
    test_records = limit_records(test_records, max_test, seed=args.seed + 2)
    summary = {
        "db_path": str(db_path),
        "train_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "test_csv": str(test_csv),
        "train": dataset_summary(train_records),
        "validation": dataset_summary(validation_records),
        "test": dataset_summary(test_records),
    }
    print(json.dumps({"dataset": summary}, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("dry_run_complete")
        return 0

    # Delay heavyweight imports so metadata-only checks work outside the CUDA image.
    from scripts.muq_audio_training import TrainingConfig, run_training

    output_dir = resolve_output(args.output_dir, data_root).resolve()
    model_path = args.model_path
    if model_path and not model_path.is_absolute():
        model_path = data_root / model_path
    resume = args.resume
    if resume and not resume.is_absolute():
        resume = data_root / resume
    epochs = 1 if args.smoke_test else args.epochs
    num_workers = 0 if args.smoke_test else args.num_workers
    unique_train_songs = len({record.song_id for record in train_records})
    batch_size = min(args.batch_size, unique_train_songs)
    if args.smoke_test:
        batch_size = min(batch_size, 4)

    config = TrainingConfig(
        output_dir=output_dir,
        model_id=args.model_id,
        model_path=model_path,
        device=args.device,
        epochs=epochs,
        batch_size=batch_size,
        validation_batch_size=min(args.validation_batch_size, len(validation_records)),
        num_workers=num_workers,
        learning_rate=args.learning_rate,
        temperature_learning_rate=args.temperature_learning_rate,
        weight_decay=args.weight_decay,
        initial_temperature=args.initial_temperature,
        distillation_weight=args.distillation_weight,
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
        gradient_clip=args.gradient_clip,
        validation_every=args.validation_every,
        retrieval_block_size=args.retrieval_block_size,
        seed=args.seed,
        resume=resume,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "arguments": vars(args),
        "resolved_dataset": summary,
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(config).items()
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    result = run_training(
        train_records,
        validation_records,
        config,
        test_records=test_records,
    )
    print(json.dumps({"training_complete": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
