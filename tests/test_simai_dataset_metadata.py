import unittest

from scripts.simai_dataset_metadata import (
    chart_id_for,
    level_value,
    segment_key_for,
    segment_rows_from_report,
)


class SimaiDatasetMetadataTest(unittest.TestCase):
    def test_chart_id_matches_existing_dataset_convention(self):
        self.assertEqual(chart_id_for(32, "st", 5), "32:ST:5")
        self.assertEqual(chart_id_for("1035", "DX", "6"), "1035:DX:6")

    def test_plus_level_uses_point_six(self):
        self.assertEqual(level_value("13+"), 13.6)
        self.assertEqual(level_value("14+"), 14.6)
        self.assertEqual(level_value("13"), 13.0)
        self.assertIsNone(level_value(""))

    def test_segment_rows_use_zero_based_ids_and_six_scores(self):
        segments = []
        for segment_id in range(5):
            segments.append(
                {
                    "start_measure": segment_id * 10 + 1,
                    "end_measure": segment_id * 10 + 10,
                    "label": "STEADY",
                    "scores": {
                        "note": segment_id + 0.1,
                        "peak": segment_id + 0.2,
                        "charge": segment_id + 0.3,
                        "slide": segment_id + 0.4,
                        "handtrip": segment_id + 0.5,
                        "tricky": segment_id + 0.6,
                    },
                }
            )
        rows = segment_rows_from_report(
            {
                "metadata": {
                    "song_id": 32,
                    "chart_kind": "ST",
                    "difficulty_index": 5,
                },
                "segments": segments,
            },
            chart_name="00032_ST_5_Master",
            report_file="report.json",
        )

        self.assertEqual([row["segment_id"] for row in rows], list(range(5)))
        self.assertEqual(rows[0]["chart_id"], "32:ST:5")
        self.assertEqual(rows[4]["key"], segment_key_for("32:ST:5", 4))
        self.assertEqual(len(rows[0]["score_vector"]), 6)
        self.assertEqual(rows[0]["slide"], 0.4)


if __name__ == "__main__":
    unittest.main()
