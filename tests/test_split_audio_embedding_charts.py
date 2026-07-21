import unittest

from scripts.split_audio_embedding_charts import (
    assign_song_disjoint_splits,
    select_charts,
    validate_assignment,
    version_number,
)


def chart(
    chart_id,
    song_id,
    version,
    level_value,
    *,
    difficulty=5,
    chart_kind="ST",
):
    level = f"{int(level_value)}+" if level_value % 1 else str(int(level_value))
    return {
        "chart_id": chart_id,
        "chart_name": chart_id.replace(":", "_"),
        "song_id": song_id,
        "title": chart_id,
        "artist": "artist",
        "chart_kind": chart_kind,
        "chart_version": version,
        "difficulty": difficulty,
        "difficulty_name": "Master" if difficulty == 5 else "Re:Master",
        "level": level,
        "level_value": level_value,
        "segment_count": 5,
        "chunk_count": 10,
        "first_chunk_key": f"{chart_id}:1-4",
        "last_chunk_key": f"{chart_id}:7-10",
    }


class SplitAudioEmbeddingChartsTest(unittest.TestCase):
    def test_version_number_reads_numeric_prefix(self):
        self.assertEqual(version_number("01. maimai"), 1)
        self.assertEqual(version_number("27. CiRCLE PLUS"), 27)

    def test_selection_excludes_earliest_low_level_charts(self):
        rows = [
            chart("1:ST:5", 1, "01. maimai", 11.0),
            chart("2:ST:5", 2, "01. maimai", 13.0),
            chart("3:ST:5", 3, "02. PLUS", 12.6),
            chart("4:ST:5", 4, "03. GreeN", 10.0),
            chart("5:ST:5", 5, "04. GreeN PLUS", 14.0),
        ]
        selected, excluded = select_charts(rows, selected_size=3, seed=7)

        self.assertEqual({row["chart_id"] for row in excluded}, {"1:ST:5", "3:ST:5"})
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(row["level_value"] < 13 for row in excluded))

    def test_optimizer_produces_exact_song_disjoint_splits(self):
        rows = [
            chart("1:ST:5", 1, "01. maimai", 13.0),
            chart("1:ST:6", 1, "01. maimai", 14.0, difficulty=6),
            chart("2:ST:5", 2, "01. maimai", 13.0),
            chart("3:ST:5", 3, "02. PLUS", 13.6),
            chart("4:ST:5", 4, "02. PLUS", 14.0),
            chart("5:DX:5", 5, "03. GreeN", 13.0, chart_kind="DX"),
            chart("6:DX:5", 6, "03. GreeN", 13.6, chart_kind="DX"),
            chart("7:DX:6", 7, "03. GreeN", 14.6, difficulty=6, chart_kind="DX"),
        ]
        assigned = assign_song_disjoint_splits(
            rows,
            split_sizes={"train": 4, "validation": 2, "test": 2},
            seed=7,
            time_limit_seconds=10,
        )
        validate_assignment(
            assigned,
            [],
            source_count=8,
            split_sizes={"train": 4, "validation": 2, "test": 2},
        )

        song_one_splits = {row["split"] for row in assigned if row["song_id"] == 1}
        self.assertEqual(len(song_one_splits), 1)


if __name__ == "__main__":
    unittest.main()
