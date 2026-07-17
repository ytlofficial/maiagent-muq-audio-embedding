import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audio_embedding_dataset import (
    AudioEmbeddingRecord,
    audio_path_candidates,
    dataset_summary,
    normalize_teacher_vector,
    numeric_conditions,
)


class AudioEmbeddingDatasetTest(unittest.TestCase):
    def test_audio_path_candidates_remap_repo_absolute_path(self):
        root = Path("/portable-data")
        candidates = audio_path_candidates(
            "/legacy-host/workspace/outputs/audio_chunks/chart/1-4.mp3",
            root,
        )
        self.assertIn(
            root / "outputs/audio_chunks/chart/1-4.mp3",
            candidates,
        )
        self.assertIn(root / "audio_chunks/chart/1-4.mp3", candidates)

    def test_numeric_conditions_scale_level_and_six_scores(self):
        conditions = numeric_conditions(13.6, [100, 200, 0, 50, 25, 150])
        self.assertEqual(conditions.shape, (7,))
        self.assertAlmostEqual(float(conditions[0]), 13.6 / 15.0, places=6)
        np.testing.assert_allclose(
            conditions[1:],
            np.asarray([0.5, 1.0, 0.0, 0.25, 0.125, 0.75], dtype=np.float32),
        )

    def test_teacher_vector_is_l2_normalized(self):
        vector = np.ones(512, dtype=np.float32)
        normalized = normalize_teacher_vector(vector, "chunk")
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=6)

    def test_dataset_summary_counts_charts_songs_and_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = str(Path(directory) / "chunk.mp3")
            records = [
                AudioEmbeddingRecord(
                    key=f"chunk-{index}",
                    chart_id=f"chart-{index}",
                    song_id=index,
                    difficulty=5 + index,
                    difficulty_id=index,
                    level_value=13.0 + index,
                    segment_id=index,
                    segment_key=f"segment-{index}",
                    audio_file=audio,
                    teacher_vector=np.ones(512, dtype=np.float32),
                    numeric_conditions=np.ones(7, dtype=np.float32),
                )
                for index in range(2)
            ]
        summary = dataset_summary(records)
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["chart_count"], 2)
        self.assertEqual(summary["song_count"], 2)
        self.assertEqual(summary["difficulty_counts"], {5: 1, 6: 1})


if __name__ == "__main__":
    unittest.main()
