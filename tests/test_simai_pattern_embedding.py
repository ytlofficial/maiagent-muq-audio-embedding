import math
import unittest

from scripts.simai_pattern_embedding import (
    OVERLAP_MEASURE_WEIGHT,
    SLIDE_PATH_LENGTH_FEATURE_WEIGHT,
    SLIDE_SHAPE_FEATURE_WEIGHT,
    SlideSegment,
    build_embedding_from_measures,
    chart_data_from_simai,
    cosine_similarity,
    select_measures,
    slide_segment_distance,
)


def four_measure_source(first_measure: str, fourth_measure: str = "{1},") -> str:
    return f"(120){{4}}{first_measure}" + "{1}," * 2 + fourth_measure + "E"


def embedding_for_first_measure(first_measure: str, *, sparse: bool = False) -> dict:
    chart_data = chart_data_from_simai(four_measure_source(first_measure))
    measures = select_measures(chart_data, (1, 4))
    return build_embedding_from_measures(measures, include_sparse_features=sparse)


class SimaiPatternEmbeddingTest(unittest.TestCase):
    def test_embedding_has_stable_dimension_and_unit_norm(self):
        payload = embedding_for_first_measure("1,2,3,4,")

        self.assertEqual(payload["dimension"], len(payload["feature_names"]))
        self.assertEqual(payload["dimension"], payload["block_slices"]["rhythm_fft"]["end"])
        norm = math.sqrt(sum(value * value for value in payload["embedding"]))
        self.assertAlmostEqual(norm, 1.0)
        self.assertIn("rhythm_fft", payload["block_weights"])
        self.assertIn("hand_balance", payload["block_weights"])
        self.assertGreater(payload["block_weights"]["topology"], payload["block_weights"]["slide"])
        self.assertGreater(payload["block_weights"]["rhythm_fft"], 0.08)
        self.assertEqual(payload["rhythm_tick_count"], 384)
        self.assertEqual(payload["measure_weights"], [1.0, 1.0, 1.0, OVERLAP_MEASURE_WEIGHT])
        self.assertAlmostEqual(payload["rhythm_bpm"], 120.0)

    def test_slide_geometry_keeps_near_outer_arc_closer_than_long_outer_arc(self):
        straight = SlideSegment("-", 2, 4)
        near_arc = SlideSegment(">", 2, 4)
        long_arc = SlideSegment("<", 2, 4)

        self.assertLess(
            slide_segment_distance(straight, near_arc),
            slide_segment_distance(straight, long_arc),
        )

    def test_embedding_slide_similarity_uses_soft_geometry(self):
        straight = embedding_for_first_measure("2-4[8:1],,,,")["embedding"]
        near_arc = embedding_for_first_measure("2>4[8:1],,,,")["embedding"]
        long_arc = embedding_for_first_measure("2<4[8:1],,,,")["embedding"]

        self.assertGreater(
            cosine_similarity(straight, near_arc),
            cosine_similarity(straight, long_arc),
        )

    def test_rotated_tap_pattern_is_closer_than_different_topology(self):
        base = embedding_for_first_measure("1,2,3,4,")["embedding"]
        rotated = embedding_for_first_measure("3,4,5,6,")["embedding"]
        different = embedding_for_first_measure("1,5,2,6,")["embedding"]

        self.assertGreater(
            cosine_similarity(base, rotated),
            cosine_similarity(base, different),
        )

    def test_sparse_features_preserve_slide_start_and_touch_details(self):
        payload = embedding_for_first_measure("1@-5[8:1]/B7f/Ch[4:3],,,,", sparse=True)
        slide_names = {item["name"] for item in payload["nonzero_features"]["slide"]}
        touch_names = {item["name"] for item in payload["nonzero_features"]["touch"]}

        self.assertIn("start_tap_normal", slide_names)
        self.assertIn("shape_start_end_-_1_5", slide_names)
        self.assertIn("sensor_B7_touch", touch_names)
        self.assertIn("sensor_C_touch_hold", touch_names)
        self.assertIn("touch_modifier_firework", touch_names)

    def test_slide_shape_and_path_features_are_attenuated(self):
        payload = embedding_for_first_measure("1@-5[8:1],,,,", sparse=True)
        slide = {
            item["name"]: item["value"]
            for item in payload["nonzero_features"]["slide"]
        }
        path_values = [
            value
            for name, value in slide.items()
            if name.startswith("path_length_bin_")
        ]

        self.assertAlmostEqual(slide["shape_start_end_-_1_5"], SLIDE_SHAPE_FEATURE_WEIGHT)
        self.assertAlmostEqual(slide["shape_-"], SLIDE_SHAPE_FEATURE_WEIGHT)
        self.assertEqual(len(path_values), 1)
        self.assertAlmostEqual(path_values[0], SLIDE_PATH_LENGTH_FEATURE_WEIGHT)

    def test_overlap_measure_slots_weight_event_features(self):
        chart_data = chart_data_from_simai(
            four_measure_source("1,2,3,4,", "{4}5,6,7,8,")
        )
        measures = select_measures(chart_data, (1, 4))

        default_tail = build_embedding_from_measures(
            measures,
            overlap_measure_slots=[3],
            include_sparse_features=True,
        )
        tail_time = {
            item["name"]: item["value"]
            for item in default_tail["nonzero_features"]["time"]
        }
        self.assertEqual(default_tail["measure_weights"], [1.0, 1.0, 1.0, 0.5])
        self.assertAlmostEqual(tail_time["measure_0_event_count"], 4.0)
        self.assertAlmostEqual(tail_time["measure_3_event_count"], 2.0)

        chart_tail = build_embedding_from_measures(
            measures,
            overlap_measure_slots=[0],
            include_sparse_features=True,
        )
        head_time = {
            item["name"]: item["value"]
            for item in chart_tail["nonzero_features"]["time"]
        }
        self.assertEqual(chart_tail["measure_weights"], [0.5, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(head_time["measure_0_event_count"], 2.0)
        self.assertAlmostEqual(head_time["measure_3_event_count"], 4.0)

    def test_hand_balance_forces_pair_each_to_use_both_hands(self):
        payload = embedding_for_first_measure("1/8,,,,", sparse=True)
        hand = {
            item["name"]: item["value"]
            for item in payload["nonzero_features"]["hand_balance"]
        }

        self.assertAlmostEqual(hand["left_load"], 1.0)
        self.assertAlmostEqual(hand["right_load"], 1.0)
        self.assertAlmostEqual(hand["forced_each_pair_count"], 1.0)
        self.assertAlmostEqual(hand["left_lane_8_load"], 1.0)
        self.assertAlmostEqual(hand["right_lane_1_load"], 1.0)
        self.assertAlmostEqual(hand["each_lr_pair_side_split"], 1.0)

    def test_hand_balance_counts_same_side_each_as_cross_hand_load(self):
        payload = embedding_for_first_measure("1/2,,,,", sparse=True)
        hand = {
            item["name"]: item["value"]
            for item in payload["nonzero_features"]["hand_balance"]
        }

        self.assertAlmostEqual(hand["left_load"], 1.0)
        self.assertAlmostEqual(hand["right_load"], 1.0)
        self.assertAlmostEqual(hand["cross_hand_load"], 1.0)
        self.assertAlmostEqual(hand["each_lr_pair_side_right"], 1.0)

    def test_hand_balance_soft_assigns_single_notes_by_lane_side(self):
        payload = embedding_for_first_measure("1,,,,", sparse=True)
        hand = {
            item["name"]: item["value"]
            for item in payload["nonzero_features"]["hand_balance"]
        }

        self.assertAlmostEqual(hand["left_load"], 0.25)
        self.assertAlmostEqual(hand["right_load"], 0.75)
        self.assertAlmostEqual(hand["soft_single_event_count"], 1.0)
        self.assertGreater(hand["right_lane_1_load"], hand["left_lane_1_load"])

    def test_star_slide_uses_guiding_each_hand_near_trace_start(self):
        payload = embedding_for_first_measure("1-5[4:1],1/2,,,", sparse=True)
        hand = {
            item["name"]: item["value"]
            for item in payload["nonzero_features"]["hand_balance"]
        }

        self.assertAlmostEqual(hand["star_each_guided_count"], 1.0)
        self.assertAlmostEqual(hand["left_lane_1_load"], 2.0)
        self.assertAlmostEqual(hand["right_lane_2_load"], 1.0)
        self.assertNotIn("star_slide_overlap_forced_count", hand)

    def test_star_slide_forces_other_hand_when_another_star_is_sliding(self):
        payload = embedding_for_first_measure("1-5[2:1],2-6[4:1],,,", sparse=True)
        hand = {
            item["name"]: item["value"]
            for item in payload["nonzero_features"]["hand_balance"]
        }

        self.assertAlmostEqual(hand["star_head_default_count"], 1.0)
        self.assertAlmostEqual(hand["star_slide_overlap_forced_count"], 1.0)
        self.assertAlmostEqual(hand["right_lane_1_load"], 1.0)
        self.assertAlmostEqual(hand["left_lane_2_load"], 1.0)


if __name__ == "__main__":
    unittest.main()
