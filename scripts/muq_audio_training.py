#!/usr/bin/env python3
"""Frozen-MuQ audio tower training implementation."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from scripts.audio_embedding_dataset import AudioEmbeddingRecord


SAMPLE_RATE = 24_000


@dataclass
class TrainingConfig:
    output_dir: Path
    model_id: str = "OpenMuQ/MuQ-large-msd-iter"
    model_path: Path | None = None
    device: str = "auto"
    epochs: int = 20
    batch_size: int = 16
    validation_batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-3
    temperature_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    initial_temperature: float = 0.07
    distillation_weight: float = 0.2
    hidden_dimension: int = 1024
    dropout: float = 0.1
    gradient_clip: float = 1.0
    validation_every: int = 1
    retrieval_block_size: int = 256
    seed: int = 20260715
    resume: Path | None = None


class AudioChunkDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[AudioEmbeddingRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, _ = librosa.load(
            record.audio_file,
            sr=SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"invalid waveform for {record.key}: {waveform.shape}")
        if not np.isfinite(waveform).all():
            raise ValueError(f"waveform contains NaN or infinity: {record.audio_file}")
        return {
            "key": record.key,
            "song_id": record.song_id,
            "waveform": waveform,
            "teacher_vector": record.teacher_vector,
            "difficulty_id": record.difficulty_id,
            "segment_id": record.segment_id,
            "numeric_conditions": record.numeric_conditions,
        }


def collate_audio_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([sample["waveform"].shape[0] for sample in samples], dtype=torch.long)
    maximum = int(lengths.max().item())
    waveforms = torch.zeros((len(samples), maximum), dtype=torch.float32)
    attention_mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
    for row_index, sample in enumerate(samples):
        length = int(lengths[row_index].item())
        waveforms[row_index, :length] = torch.from_numpy(sample["waveform"])
        attention_mask[row_index, :length] = True
    return {
        "keys": [str(sample["key"]) for sample in samples],
        "song_ids": torch.tensor([sample["song_id"] for sample in samples], dtype=torch.long),
        "waveforms": waveforms,
        "waveform_lengths": lengths,
        "attention_mask": attention_mask,
        "teacher_vectors": torch.from_numpy(
            np.stack([sample["teacher_vector"] for sample in samples]).astype(np.float32)
        ),
        "difficulty_ids": torch.tensor(
            [sample["difficulty_id"] for sample in samples], dtype=torch.long
        ),
        "segment_ids": torch.tensor(
            [sample["segment_id"] for sample in samples], dtype=torch.long
        ),
        "numeric_conditions": torch.from_numpy(
            np.stack([sample["numeric_conditions"] for sample in samples]).astype(np.float32)
        ),
    }


class SongDisjointBatchSampler(Sampler[list[int]]):
    """Yield batches containing at most one chunk from each song."""

    def __init__(
        self,
        records: Sequence[AudioEmbeddingRecord],
        batch_size: int,
        *,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 2:
            raise ValueError("contrastive training requires batch_size >= 2")
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.groups[record.song_id].append(index)
        if len(self.groups) < batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds the {len(self.groups)} unique training songs"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        queues = {song_id: list(indices) for song_id, indices in self.groups.items()}
        for indices in queues.values():
            rng.shuffle(indices)
        maximum_rounds = max(len(indices) for indices in queues.values())
        for round_index in range(maximum_rounds):
            active_songs = [
                song_id for song_id, indices in queues.items() if round_index < len(indices)
            ]
            rng.shuffle(active_songs)
            for start in range(0, len(active_songs), self.batch_size):
                song_batch = active_songs[start : start + self.batch_size]
                if len(song_batch) < self.batch_size and self.drop_last:
                    continue
                yield [queues[song_id][round_index] for song_id in song_batch]

    def __len__(self) -> int:
        maximum_rounds = max(len(indices) for indices in self.groups.values())
        total = 0
        for round_index in range(maximum_rounds):
            active = sum(round_index < len(indices) for indices in self.groups.values())
            if self.drop_last:
                total += active // self.batch_size
            else:
                total += math.ceil(active / self.batch_size)
        return total


class StatisticalPooling(nn.Module):
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"expected [B,T,D] features, got {tuple(features.shape)}")
        if mask.shape != features.shape[:2]:
            raise ValueError(
                f"feature mask {tuple(mask.shape)} does not match {tuple(features.shape[:2])}"
            )
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every sample must contain at least one valid MuQ frame")

        valid = mask.unsqueeze(-1).to(features.dtype)
        count = valid.sum(dim=1).clamp_min(1.0)
        mean = (features * valid).sum(dim=1) / count
        maximum = features.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        variance = (((features - mean.unsqueeze(1)) ** 2) * valid).sum(dim=1) / count
        std = torch.sqrt(variance.clamp_min(0.0) + 1e-6)
        return torch.cat((mean, maximum, std), dim=-1)


class ConditionedAudioProjection(nn.Module):
    def __init__(
        self,
        *,
        audio_feature_dimension: int,
        hidden_dimension: int,
        output_dimension: int = 512,
        numeric_condition_dimension: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        difficulty_dimension = 32
        segment_dimension = 16
        self.pooling = StatisticalPooling()
        self.difficulty_embedding = nn.Embedding(2, difficulty_dimension)
        self.segment_embedding = nn.Embedding(5, segment_dimension)
        fusion_dimension = (
            audio_feature_dimension * 3
            + difficulty_dimension
            + segment_dimension
            + numeric_condition_dimension
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(fusion_dimension),
            nn.Linear(fusion_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        difficulty_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        numeric_conditions: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pooling(features, feature_mask)
        conditions = torch.cat(
            (
                self.difficulty_embedding(difficulty_ids),
                self.segment_embedding(segment_ids),
                numeric_conditions,
            ),
            dim=-1,
        )
        return F.normalize(self.projector(torch.cat((pooled, conditions), dim=-1)), dim=-1)


class ContrastiveObjective(nn.Module):
    def __init__(self, initial_temperature: float, distillation_weight: float) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())
        self.distillation_weight = distillation_weight

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        logits = audio_embeddings @ teacher_embeddings.transpose(0, 1) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
        )
        positive_cosine = (audio_embeddings * teacher_embeddings).sum(dim=-1)
        distillation = (1.0 - positive_cosine).mean()
        total = contrastive + self.distillation_weight * distillation
        return total, {
            "contrastive_loss": contrastive.detach(),
            "distillation_loss": distillation.detach(),
            "positive_cosine": positive_cosine.mean().detach(),
            "temperature": temperature.detach(),
        }


def muq_feature_mask(attention_mask: torch.Tensor, feature_length: int) -> torch.Tensor:
    if feature_length <= 0:
        raise ValueError(f"invalid MuQ feature length: {feature_length}")
    skip = max(1, int(attention_mask.shape[1] / feature_length))
    mask = attention_mask[:, ::skip]
    if mask.shape[1] < feature_length:
        mask = F.pad(mask, (0, feature_length - mask.shape[1]), value=False)
    return mask[:, :feature_length].bool()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_muq(config: TrainingConfig, device: torch.device) -> Any:
    from muq import MuQ

    reference = str(config.model_path) if config.model_path and config.model_path.exists() else config.model_id
    print(f"loading_muQ={reference}")
    model = MuQ.from_pretrained(reference).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "waveforms": batch["waveforms"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "teacher_vectors": batch["teacher_vectors"].to(device, non_blocking=True),
        "difficulty_ids": batch["difficulty_ids"].to(device, non_blocking=True),
        "segment_ids": batch["segment_ids"].to(device, non_blocking=True),
        "numeric_conditions": batch["numeric_conditions"].to(device, non_blocking=True),
    }


def encode_batch(
    muq_model: Any,
    projection: ConditionedAudioProjection,
    batch: dict[str, Any],
) -> torch.Tensor:
    # MuQ explicitly recommends FP32 inference; only the projection head receives gradients.
    with torch.no_grad():
        output = muq_model(
            batch["waveforms"].float(),
            attention_mask=batch["attention_mask"],
            output_hidden_states=False,
        )
        features = output.last_hidden_state.float()
    feature_mask = muq_feature_mask(batch["attention_mask"], features.shape[1])
    return projection(
        features,
        feature_mask,
        batch["difficulty_ids"],
        batch["segment_ids"],
        batch["numeric_conditions"],
    )


def aggregate_running_metrics(
    totals: dict[str, float], metrics: dict[str, torch.Tensor], batch_size: int
) -> None:
    for name, value in metrics.items():
        totals[name] += float(value.item()) * batch_size


def train_epoch(
    *,
    epoch: int,
    loader: DataLoader[dict[str, Any]],
    sampler: SongDisjointBatchSampler,
    muq_model: Any,
    projection: ConditionedAudioProjection,
    objective: ContrastiveObjective,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    projection.train()
    objective.train()
    muq_model.eval()
    sampler.set_epoch(epoch)
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    progress = tqdm(loader, desc=f"train epoch {epoch + 1}", dynamic_ncols=True)
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        audio_embeddings = encode_batch(muq_model, projection, batch)
        loss, metrics = objective(audio_embeddings, batch["teacher_vectors"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss for keys {batch['keys'][:3]}")
        loss.backward()
        parameters = list(projection.parameters()) + list(objective.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=gradient_clip)
        optimizer.step()

        batch_size = int(audio_embeddings.shape[0])
        sample_count += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        aggregate_running_metrics(totals, metrics, batch_size)
        progress.set_postfix(loss=f"{totals['loss'] / sample_count:.4f}")
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def retrieval_ranks(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    *,
    device: torch.device,
    block_size: int,
) -> torch.Tensor:
    candidates_device = F.normalize(candidates, dim=-1).to(device)
    ranks: list[torch.Tensor] = []
    for start in range(0, queries.shape[0], block_size):
        end = min(start + block_size, queries.shape[0])
        query_block = F.normalize(queries[start:end], dim=-1).to(device)
        scores = query_block @ candidates_device.transpose(0, 1)
        targets = torch.arange(start, end, device=device)
        target_scores = scores[torch.arange(end - start, device=device), targets]
        ranks.append((scores > target_scores.unsqueeze(1)).sum(dim=1).add(1).cpu())
    return torch.cat(ranks)


def rank_metrics(ranks: torch.Tensor, prefix: str) -> dict[str, float]:
    ranks_float = ranks.float()
    return {
        f"{prefix}_recall_at_1": float((ranks <= 1).float().mean().item()),
        f"{prefix}_recall_at_5": float((ranks <= 5).float().mean().item()),
        f"{prefix}_recall_at_10": float((ranks <= 10).float().mean().item()),
        f"{prefix}_mrr": float((1.0 / ranks_float).mean().item()),
        f"{prefix}_median_rank": float(ranks_float.median().item()),
    }


@torch.no_grad()
def evaluate(
    *,
    loader: DataLoader[dict[str, Any]],
    muq_model: Any,
    projection: ConditionedAudioProjection,
    objective: ContrastiveObjective,
    device: torch.device,
    retrieval_block_size: int,
    description: str = "validation",
) -> dict[str, float]:
    projection.eval()
    objective.eval()
    muq_model.eval()
    total_loss = 0.0
    sample_count = 0
    audio_batches: list[torch.Tensor] = []
    teacher_batches: list[torch.Tensor] = []
    progress = tqdm(loader, desc=description, dynamic_ncols=True)
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        audio_embeddings = encode_batch(muq_model, projection, batch)
        loss, _ = objective(audio_embeddings, batch["teacher_vectors"])
        batch_size = int(audio_embeddings.shape[0])
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        audio_batches.append(audio_embeddings.cpu())
        teacher_batches.append(F.normalize(batch["teacher_vectors"], dim=-1).cpu())

    audio = torch.cat(audio_batches)
    teacher = torch.cat(teacher_batches)
    a2c_ranks = retrieval_ranks(
        audio, teacher, device=device, block_size=retrieval_block_size
    )
    c2a_ranks = retrieval_ranks(
        teacher, audio, device=device, block_size=retrieval_block_size
    )
    metrics = {"loss": total_loss / max(sample_count, 1)}
    metrics.update(rank_metrics(a2c_ranks, "audio_to_chart"))
    metrics.update(rank_metrics(c2a_ranks, "chart_to_audio"))
    metrics["positive_cosine"] = float((audio * teacher).sum(dim=-1).mean().item())
    if len(audio) > 1:
        metrics["shifted_negative_cosine"] = float(
            (audio * teacher.roll(shifts=1, dims=0)).sum(dim=-1).mean().item()
        )
    return metrics


def make_loader(
    dataset: AudioChunkDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    batch_sampler: Sampler[list[int]] | None = None,
) -> DataLoader[dict[str, Any]]:
    options: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": num_workers,
        "collate_fn": collate_audio_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    if batch_sampler is not None:
        options["batch_sampler"] = batch_sampler
    else:
        options["batch_size"] = batch_size
        options["shuffle"] = False
    return DataLoader(**options)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    projection: ConditionedAudioProjection,
    objective: ContrastiveObjective,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    validation_metrics: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "projection": projection.state_dict(),
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "validation_metrics": validation_metrics,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_history(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    train_records: Sequence[AudioEmbeddingRecord],
    validation_records: Sequence[AudioEmbeddingRecord],
    config: TrainingConfig,
    *,
    test_records: Sequence[AudioEmbeddingRecord] | None = None,
) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and torch.version.cuda is None:
        raise RuntimeError("CUDA device selected with a CPU-only PyTorch build")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    environment = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "test_records": len(test_records or ()),
    }
    print(json.dumps({"environment": environment}, ensure_ascii=False, indent=2))

    muq_model = load_frozen_muq(config, device)
    feature_dimension = int(getattr(muq_model.config, "encoder_dim", 1024))
    projection = ConditionedAudioProjection(
        audio_feature_dimension=feature_dimension,
        hidden_dimension=config.hidden_dimension,
        dropout=config.dropout,
    ).to(device)
    objective = ContrastiveObjective(
        config.initial_temperature,
        config.distillation_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (
            {
                "params": projection.parameters(),
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": objective.parameters(),
                "lr": config.temperature_learning_rate,
                "weight_decay": 0.0,
            },
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )

    start_epoch = 0
    best_recall = -1.0
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        projection.load_state_dict(checkpoint["projection"])
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scheduler.T_max = max(config.epochs, 1)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_recall = float(
            checkpoint.get("validation_metrics", {}).get("audio_to_chart_recall_at_10", -1.0)
        )
        existing_best_path = config.output_dir / "best.pt"
        if existing_best_path.is_file():
            existing_best = torch.load(
                existing_best_path,
                map_location="cpu",
                weights_only=False,
            )
            best_recall = max(
                best_recall,
                float(
                    existing_best.get("validation_metrics", {}).get(
                        "audio_to_chart_recall_at_10", -1.0
                    )
                ),
            )

    train_dataset = AudioChunkDataset(train_records)
    validation_dataset = AudioChunkDataset(validation_records)
    train_sampler = SongDisjointBatchSampler(
        train_records,
        config.batch_size,
        seed=config.seed,
        drop_last=True,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        batch_sampler=train_sampler,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = None
    if test_records:
        test_loader = make_loader(
            AudioChunkDataset(test_records),
            batch_size=config.validation_batch_size,
            num_workers=config.num_workers,
            device=device,
        )

    history_path = config.output_dir / "history.jsonl"
    started_at = time.time()
    last_validation: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    tested_checkpoint: str | None = None
    try:
        for epoch in range(start_epoch, config.epochs):
            train_metrics = train_epoch(
                epoch=epoch,
                loader=train_loader,
                sampler=train_sampler,
                muq_model=muq_model,
                projection=projection,
                objective=objective,
                optimizer=optimizer,
                device=device,
                gradient_clip=config.gradient_clip,
            )
            scheduler.step()
            should_validate = (
                (epoch + 1) % config.validation_every == 0 or epoch + 1 == config.epochs
            )
            if should_validate:
                last_validation = evaluate(
                    loader=validation_loader,
                    muq_model=muq_model,
                    projection=projection,
                    objective=objective,
                    device=device,
                    retrieval_block_size=config.retrieval_block_size,
                    description="validation",
                )

            epoch_payload = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - started_at,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": last_validation if should_validate else None,
            }
            print(json.dumps(epoch_payload, ensure_ascii=False, sort_keys=True))
            append_history(history_path, epoch_payload)
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch + 1)
            if should_validate:
                for name, value in last_validation.items():
                    writer.add_scalar(f"validation/{name}", value, epoch + 1)

            save_checkpoint(
                config.output_dir / "last.pt",
                epoch=epoch,
                projection=projection,
                objective=objective,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                validation_metrics=last_validation,
            )
            recall = last_validation.get("audio_to_chart_recall_at_10", -1.0)
            if should_validate and recall > best_recall:
                best_recall = recall
                save_checkpoint(
                    config.output_dir / "best.pt",
                    epoch=epoch,
                    projection=projection,
                    objective=objective,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    validation_metrics=last_validation,
                )
            writer.flush()

        if test_loader is not None:
            checkpoint_candidates = (
                config.output_dir / "best.pt",
                config.output_dir / "last.pt",
                config.resume,
            )
            checkpoint_path = next(
                (
                    candidate
                    for candidate in checkpoint_candidates
                    if candidate is not None and candidate.is_file()
                ),
                None,
            )
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "no checkpoint is available for final test evaluation"
                )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            projection.load_state_dict(checkpoint["projection"])
            objective.load_state_dict(checkpoint["objective"])
            test_metrics = evaluate(
                loader=test_loader,
                muq_model=muq_model,
                projection=projection,
                objective=objective,
                device=device,
                retrieval_block_size=config.retrieval_block_size,
                description="test",
            )
            tested_checkpoint = str(checkpoint_path)
            test_payload = {
                "checkpoint": tested_checkpoint,
                "record_count": len(test_records or ()),
                "metrics": test_metrics,
            }
            with (config.output_dir / "test_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(test_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            for name, value in test_metrics.items():
                writer.add_scalar(f"test/{name}", value, config.epochs)
            writer.flush()
    finally:
        writer.close()

    return {
        "output_dir": str(config.output_dir),
        "best_audio_to_chart_recall_at_10": best_recall,
        "last_validation": last_validation,
        "tested_checkpoint": tested_checkpoint,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - started_at,
    }
