import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_segment_chunk_audio_lancedb import (
    SkippedSourceAudioFormat,
    build_audio_rows_and_index_for_report,
    chunk_audio_path,
    chunk_time_range,
    relative_song_dir_from_created_file,
    resolve_existing_source_audio_file,
    resolve_source_audio_file,
)


class BuildSegmentChunkAudioLancedbTest(unittest.TestCase):
    def test_relative_song_dir_uses_chartdata_rebuilt_marker(self):
        self.assertEqual(
            relative_song_dir_from_created_file("chartdata-rebuilt/Foo Bar/maidata.txt"),
            Path("Foo Bar"),
        )
        self.assertEqual(
            relative_song_dir_from_created_file(
                "/tmp/work/chartdata-rebuilt/Nested/Foo Bar/maidata.txt"
            ),
            Path("Nested/Foo Bar"),
        )

    def test_resolve_source_audio_file_uses_chartdata_rebuilt_root(self):
        root = Path("/repo/chartdata-rebuilt")
        self.assertEqual(
            resolve_source_audio_file(
                "chartdata-rebuilt/Foo Bar/maidata.txt",
                chartdata_root=root,
                audio_filename="track.mp3",
            ),
            root / "Foo Bar" / "track.mp3",
        )

    def test_resolve_existing_source_audio_file_falls_back_to_ogg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chartdata-rebuilt"
            song_dir = root / "Foo Bar"
            song_dir.mkdir(parents=True)
            (song_dir / "track.ogg").write_bytes(b"fake ogg")

            self.assertEqual(
                resolve_existing_source_audio_file(
                    "chartdata-rebuilt/Foo Bar/maidata.txt",
                    chartdata_root=root,
                    audio_filename="track.mp3",
                    fallback_audio_filenames=["track.ogg"],
                ),
                song_dir / "track.ogg",
            )

    def test_chunk_time_range_uses_start_and_end_measure_boundaries(self):
        measures = {
            "measures": [
                {"index": 1, "start_seconds": 0.0, "end_seconds": 1.5},
                {"index": 2, "start_seconds": 1.5, "end_seconds": 3.0},
                {"index": 3, "start_seconds": 3.0, "end_seconds": 4.5},
                {"index": 4, "start_seconds": 4.5, "end_seconds": 6.0},
            ]
        }

        self.assertEqual(chunk_time_range(measures, 2, 4), (1.5, 6.0))

    def test_chunk_audio_path_is_stable(self):
        self.assertEqual(
            chunk_audio_path(
                Path("/tmp/audio"),
                "00001_ST_5_Master",
                3,
                7,
                10,
                suffix=".mp3",
            ),
            Path("/tmp/audio/00001_ST_5_Master/003_0007-0010.mp3"),
        )

    def test_build_rows_use_same_chunk_key_as_pattern_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranges_dir = root / "ranges"
            measures_dir = root / "measures"
            chartdata_root = root / "chartdata-rebuilt"
            audio_out_dir = root / "audio"
            ranges_dir.mkdir()
            measures_dir.mkdir()
            source_dir = chartdata_root / "Song"
            source_dir.mkdir(parents=True)
            (source_dir / "track.mp3").write_bytes(b"fake mp3")

            report_path = ranges_dir / "00001_ST_5_Master.json"
            report_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "song_id": 1,
                            "title": "Song",
                            "chart_kind": "ST",
                            "difficulty_index": 5,
                            "difficulty_name": "Master",
                            "measure_count": 4,
                            "created_from_file": "chartdata-rebuilt/Song/maidata.txt",
                        },
                        "segments": [{"start_measure": 1, "end_measure": 4}],
                    }
                ),
                encoding="utf-8",
            )
            (measures_dir / report_path.name).write_text(
                json.dumps(
                    {
                        "song": {"song_id": 1, "title": "Song"},
                        "chart": {
                            "chart_kind": "ST",
                            "difficulty_index": 5,
                            "difficulty_name": "Master",
                            "created_from_file": "chartdata-rebuilt/Song/maidata.txt",
                        },
                        "timeline": {"measure_count": 4},
                        "measures": [
                            {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0},
                            {"index": 2, "start_seconds": 1.0, "end_seconds": 2.0},
                            {"index": 3, "start_seconds": 2.0, "end_seconds": 3.0},
                            {"index": 4, "start_seconds": 3.0, "end_seconds": 4.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows, index_row = build_audio_rows_and_index_for_report(
                report_path,
                measures_dir=measures_dir,
                chartdata_root=chartdata_root,
                audio_filename="track.mp3",
                fallback_audio_filenames=["track.ogg"],
                audio_out_dir=audio_out_dir,
                audio_suffix=".mp3",
                converted_source_dir=root / "converted",
                extract_audio=False,
                ffmpeg=None,
                mp3_quality=2,
                reuse_existing_audio=False,
                store_audio_bytes=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["key"], "00001_ST_5_Master:1-4")
        self.assertEqual(index_row["chunk_keys_json"], '["00001_ST_5_Master:1-4"]')
        self.assertTrue(rows[0]["audio_file"].endswith("00001_ST_5_Master/001_0001-0004.mp3"))

    def test_only_source_audio_format_skips_mp3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranges_dir = root / "ranges"
            measures_dir = root / "measures"
            chartdata_root = root / "chartdata-rebuilt"
            ranges_dir.mkdir()
            measures_dir.mkdir()
            source_dir = chartdata_root / "Song"
            source_dir.mkdir(parents=True)
            (source_dir / "track.mp3").write_bytes(b"fake mp3")

            report_path = ranges_dir / "00001_ST_5_Master.json"
            report_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "song_id": 1,
                            "title": "Song",
                            "chart_kind": "ST",
                            "difficulty_index": 5,
                            "difficulty_name": "Master",
                            "measure_count": 4,
                            "created_from_file": "chartdata-rebuilt/Song/maidata.txt",
                        },
                        "segments": [{"start_measure": 1, "end_measure": 4}],
                    }
                ),
                encoding="utf-8",
            )
            (measures_dir / report_path.name).write_text(
                json.dumps(
                    {
                        "song": {"song_id": 1, "title": "Song"},
                        "chart": {
                            "chart_kind": "ST",
                            "difficulty_index": 5,
                            "difficulty_name": "Master",
                            "created_from_file": "chartdata-rebuilt/Song/maidata.txt",
                        },
                        "timeline": {"measure_count": 4},
                        "measures": [
                            {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0},
                            {"index": 2, "start_seconds": 1.0, "end_seconds": 2.0},
                            {"index": 3, "start_seconds": 2.0, "end_seconds": 3.0},
                            {"index": 4, "start_seconds": 3.0, "end_seconds": 4.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SkippedSourceAudioFormat):
                build_audio_rows_and_index_for_report(
                    report_path,
                    measures_dir=measures_dir,
                    chartdata_root=chartdata_root,
                    audio_filename="track.mp3",
                    fallback_audio_filenames=["track.ogg"],
                    audio_out_dir=root / "audio",
                    audio_suffix=".mp3",
                    converted_source_dir=root / "converted",
                    extract_audio=False,
                    ffmpeg=None,
                    mp3_quality=2,
                    reuse_existing_audio=False,
                    store_audio_bytes=False,
                    only_source_audio_formats={"ogg"},
                )

    def test_build_rows_prepare_ogg_as_converted_mp3_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranges_dir = root / "ranges"
            measures_dir = root / "measures"
            chartdata_root = root / "chartdata-rebuilt"
            audio_out_dir = root / "audio"
            converted_source_dir = root / "converted"
            ranges_dir.mkdir()
            measures_dir.mkdir()
            source_dir = chartdata_root / "Song"
            source_dir.mkdir(parents=True)
            (source_dir / "track.ogg").write_bytes(b"fake ogg")

            report_path = ranges_dir / "00001_ST_5_Master.json"
            report_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "song_id": 1,
                            "title": "Song",
                            "chart_kind": "ST",
                            "difficulty_index": 5,
                            "difficulty_name": "Master",
                            "measure_count": 4,
                            "created_from_file": "chartdata-rebuilt/Song/maidata.txt",
                        },
                        "segments": [{"start_measure": 1, "end_measure": 4}],
                    }
                ),
                encoding="utf-8",
            )
            (measures_dir / report_path.name).write_text(
                json.dumps(
                    {
                        "song": {"song_id": 1, "title": "Song"},
                        "chart": {
                            "chart_kind": "ST",
                            "difficulty_index": 5,
                            "difficulty_name": "Master",
                            "created_from_file": "chartdata-rebuilt/Song/maidata.txt",
                        },
                        "timeline": {"measure_count": 4},
                        "measures": [
                            {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0},
                            {"index": 2, "start_seconds": 1.0, "end_seconds": 2.0},
                            {"index": 3, "start_seconds": 2.0, "end_seconds": 3.0},
                            {"index": 4, "start_seconds": 3.0, "end_seconds": 4.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows, index_row = build_audio_rows_and_index_for_report(
                report_path,
                measures_dir=measures_dir,
                chartdata_root=chartdata_root,
                audio_filename="track.mp3",
                fallback_audio_filenames=["track.ogg"],
                audio_out_dir=audio_out_dir,
                audio_suffix=".mp3",
                converted_source_dir=converted_source_dir,
                extract_audio=False,
                ffmpeg=None,
                mp3_quality=2,
                reuse_existing_audio=False,
                store_audio_bytes=False,
            )

        self.assertEqual(rows[0]["original_source_audio_file"], str(source_dir / "track.ogg"))
        self.assertEqual(rows[0]["source_audio_file"], str(converted_source_dir / "Song" / "track.mp3"))
        self.assertTrue(rows[0]["source_audio_was_converted"])
        self.assertEqual(rows[0]["source_audio_original_format"], "ogg")
        self.assertEqual(index_row["source_audio_file"], str(converted_source_dir / "Song" / "track.mp3"))


if __name__ == "__main__":
    unittest.main()
