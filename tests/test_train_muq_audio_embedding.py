import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.train_muq_audio_embedding import (
    ensure_splits_song_disjoint,
    parse_arguments,
)

try:
    import yaml  # noqa: F401

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class TrainMuQAudioEmbeddingCliTest(unittest.TestCase):
    def write_config(self, directory: str, content: str) -> Path:
        path = Path(directory) / "training.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    @unittest.skipUnless(HAS_YAML, "PyYAML is installed in the CUDA training image")
    def test_yaml_values_are_typed_and_cli_can_override_them(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.write_config(
                directory,
                "batch_size: 8\n"
                "learning_rate: 0.0005\n"
                "output_dir: outputs/audio_embedding_runs/configured\n"
                "skip_audio_file_check: false\n",
            )
            args = parse_arguments(
                [
                    "--config",
                    str(config),
                    "--batch-size",
                    "4",
                ]
            )
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.learning_rate, 0.0005)
        self.assertEqual(
            args.output_dir,
            Path("outputs/audio_embedding_runs/configured"),
        )
        self.assertFalse(args.skip_audio_file_check)

    @unittest.skipUnless(HAS_YAML, "PyYAML is installed in the CUDA training image")
    def test_unknown_yaml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.write_config(directory, "batch_szie: 8\n")
            with self.assertRaisesRegex(ValueError, "unknown configuration keys"):
                parse_arguments(["--config", str(config)])

    def test_all_split_pairs_are_checked_for_song_leakage(self):
        with self.assertRaisesRegex(ValueError, "validation/test song leakage"):
            ensure_splits_song_disjoint(
                train=[SimpleNamespace(song_id=1)],
                validation=[SimpleNamespace(song_id=2)],
                test=[SimpleNamespace(song_id=2)],
            )


if __name__ == "__main__":
    unittest.main()
