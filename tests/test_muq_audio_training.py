import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.audio_embedding_dataset import AudioEmbeddingRecord

try:
    import torch

    from scripts.muq_audio_training import (
        ConditionedAudioProjection,
        ContrastiveObjective,
        SongDisjointBatchSampler,
        StatisticalPooling,
        TrainingConfig,
        muq_feature_mask,
        run_training,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is installed in the CUDA training image")
class MuQAudioTrainingTest(unittest.TestCase):
    def record(self, key, song_id):
        return AudioEmbeddingRecord(
            key=key,
            chart_id=f"chart-{song_id}",
            song_id=song_id,
            difficulty=5,
            difficulty_id=0,
            level_value=13.0,
            segment_id=0,
            segment_key=f"segment-{song_id}",
            audio_file="unused.mp3",
            teacher_vector=np.ones(512, dtype=np.float32),
            numeric_conditions=np.ones(7, dtype=np.float32),
        )

    def test_sampler_never_repeats_a_song_inside_batch(self):
        records = [
            self.record(f"{song_id}-{chunk}", song_id)
            for song_id in range(4)
            for chunk in range(3)
        ]
        sampler = SongDisjointBatchSampler(
            records,
            batch_size=3,
            seed=7,
            drop_last=False,
        )
        yielded = []
        for batch in sampler:
            songs = [records[index].song_id for index in batch]
            self.assertEqual(len(songs), len(set(songs)))
            yielded.extend(batch)
        self.assertEqual(sorted(yielded), list(range(len(records))))

    def test_masked_statistical_pooling_ignores_padding(self):
        features = torch.tensor([[[1.0], [3.0], [100.0]]])
        mask = torch.tensor([[True, True, False]])
        pooled = StatisticalPooling()(features, mask)
        expected = torch.tensor([[2.0, 3.0, 1.0]])
        torch.testing.assert_close(pooled, expected, atol=1e-5, rtol=1e-5)

    def test_feature_mask_matches_muq_stride_rule(self):
        mask = torch.tensor(
            [[True] * 800 + [False] * 200, [True] * 1000],
            dtype=torch.bool,
        )
        result = muq_feature_mask(mask, feature_length=10)
        self.assertEqual(tuple(result.shape), (2, 10))
        self.assertEqual(int(result[0].sum()), 8)
        self.assertEqual(int(result[1].sum()), 10)

    def test_projection_and_objective_produce_normalized_trainable_output(self):
        model = ConditionedAudioProjection(
            audio_feature_dimension=4,
            hidden_dimension=16,
            dropout=0.0,
        )
        features = torch.randn(3, 5, 4)
        mask = torch.ones(3, 5, dtype=torch.bool)
        output = model(
            features,
            mask,
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 2, 4]),
            torch.randn(3, 7),
        )
        self.assertEqual(tuple(output.shape), (3, 512))
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))

        target = torch.nn.functional.normalize(torch.randn(3, 512), dim=-1)
        objective = ContrastiveObjective(0.07, 0.2)
        loss, metrics = objective(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("contrastive_loss", metrics)

    def test_training_selects_best_checkpoint_and_evaluates_test_split(self):
        import soundfile as sf

        class FakeMuQ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(encoder_dim=4)

            def forward(self, waveforms, **_kwargs):
                pooled = torch.nn.functional.adaptive_avg_pool1d(
                    waveforms.unsqueeze(1), 4
                ).transpose(1, 2)
                return SimpleNamespace(last_hidden_state=pooled.repeat(1, 1, 4))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio_file = root / "audio.wav"
            sf.write(audio_file, np.linspace(-0.1, 0.1, 2400, dtype=np.float32), 24000)

            def split_records(start_song, count):
                records = []
                for offset in range(count):
                    record = self.record(f"{start_song + offset}-0", start_song + offset)
                    teacher = np.zeros(512, dtype=np.float32)
                    teacher[offset] = 1.0
                    records.append(
                        replace(
                            record,
                            audio_file=str(audio_file),
                            teacher_vector=teacher,
                        )
                    )
                return records

            output_dir = root / "run"
            config = TrainingConfig(
                output_dir=output_dir,
                device="cpu",
                epochs=1,
                batch_size=2,
                validation_batch_size=2,
                num_workers=0,
                hidden_dimension=16,
                validation_every=1,
                retrieval_block_size=2,
                seed=11,
            )
            with patch(
                "scripts.muq_audio_training.load_frozen_muq",
                return_value=FakeMuQ(),
            ):
                result = run_training(
                    split_records(10, 4),
                    split_records(20, 3),
                    config,
                    test_records=split_records(30, 3),
                )

            self.assertTrue((output_dir / "last.pt").is_file())
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "test_metrics.json").is_file())
            self.assertEqual(result["tested_checkpoint"], str(output_dir / "best.pt"))
            self.assertIn("audio_to_chart_recall_at_10", result["test_metrics"])


if __name__ == "__main__":
    unittest.main()
