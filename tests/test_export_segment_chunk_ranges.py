import unittest

from scripts.export_segment_chunk_ranges import (
    body_ranges_for_segment,
    chunk_ranges_from_report,
    chunk_specs_from_report,
    expand_body_to_chunk,
)


def report(measure_count, segments):
    return {
        "metadata": {"measure_count": measure_count},
        "segments": [
            {
                "start_measure": start,
                "end_measure": end,
                "measure_count": end - start + 1,
            }
            for start, end in segments
        ],
    }


class ExportSegmentChunkRangesTest(unittest.TestCase):
    def test_normal_body_appends_one_overlap_measure_after_body(self):
        self.assertEqual(expand_body_to_chunk(1, 3, 20), [1, 4])
        spec = chunk_specs_from_report(report(20, [(1, 6)]))[0]
        self.assertEqual(spec.range, [1, 4])
        self.assertEqual(spec.body_range, [1, 3])
        self.assertEqual(spec.segment_id, 0)
        self.assertEqual(spec.overlap_measure_slots, [3])
        self.assertEqual(spec.overlap_measure_ranges, [4])

    def test_chart_end_moves_missing_overlap_before_body(self):
        self.assertEqual(expand_body_to_chunk(18, 20, 20), [17, 20])
        spec = chunk_specs_from_report(report(20, [(11, 20)]))[-1]
        self.assertEqual(spec.range, [17, 20])
        self.assertEqual(spec.body_range, [18, 20])
        self.assertEqual(spec.overlap_measure_slots, [0])
        self.assertEqual(spec.overlap_measure_ranges, [17])

    def test_segment_tail_remainder_gets_final_body_ending_at_segment_end(self):
        self.assertEqual(
            body_ranges_for_segment(1, 10),
            [(1, 3), (4, 6), (7, 9), (8, 10)],
        )

    def test_overlap_can_cross_segment_boundary(self):
        payload = report(20, [(1, 6), (7, 12), (13, 20)])
        chunks = chunk_ranges_from_report(payload)

        self.assertEqual(
            chunks,
            [
                [1, 4],
                [4, 7],
                [7, 10],
                [10, 13],
                [13, 16],
                [16, 19],
                [17, 20],
            ],
        )
        self.assertEqual(
            [spec.segment_id for spec in chunk_specs_from_report(payload)],
            [0, 0, 1, 1, 2, 2, 2],
        )

    def test_short_segment_still_exports_four_measure_chunk(self):
        chunks = chunk_ranges_from_report(report(20, [(1, 6), (7, 8), (9, 20)]))

        self.assertIn([7, 10], chunks)

    def test_chart_tail_drops_duplicate_expanded_chunk(self):
        chunks = chunk_ranges_from_report(report(20, [(1, 10), (11, 20)]))

        self.assertEqual(chunks[-3:], [[11, 14], [14, 17], [17, 20]])
        self.assertEqual(chunks.count([17, 20]), 1)


if __name__ == "__main__":
    unittest.main()
