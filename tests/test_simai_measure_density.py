import unittest

from scripts.simai_density_segmenter import (
    SegmentationConfig,
    annotate_segment_labels,
    attach_six_dimension_scores,
    burst_peak_score,
    chart_report_metadata,
    choose_segmentation,
    compute_handtrip_scores,
    compute_peak_note_scores,
    compute_tricky_scores,
    effective_min_segment_measures,
    four_measure_burst_summary,
    saturated_ratio_density_score,
    summarize_segment,
    segment_lengths,
    tricky_final_curve,
)
from scripts.simai_global_six_dimension_table import compute_global_baselines, score_handtrip, score_rows
from scripts.simai_measure_density import measure_density, summarize_density_range
from scripts.simai_segment_scorer import parse_measure_range, score_segment_summary


class SimaiMeasureDensityTest(unittest.TestCase):
    def test_measure_density_is_weighted_sum_over_measure_time(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "duration_seconds": 2.0,
                "bpm": 120.0,
                "beats": 4.0,
                "simai": "1,2,3,4,",
            }
        )

        self.assertEqual(density["weighted_sum"], 4)
        self.assertEqual(density["density"], 2)

    def test_measure_density_uses_bpm_for_exact_duration_weights(self):
        density = measure_density(
            {
                "index": 1,
                "duration_seconds": 2.0,
                "bpm": 120.0,
                "simai": "1h[#1.5],",
            }
        )

        self.assertEqual(density["weighted_sum"], 3)
        self.assertEqual(density["density"], 1.5)

    def test_non_touch_density_excludes_touch_notes_for_note_scoring(self):
        density = measure_density(
            {
                "index": 1,
                "duration_seconds": 2.0,
                "bpm": 120.0,
                "simai": "1/C/B1,",
            }
        )

        self.assertEqual(density["weighted_sum"], 3)
        self.assertEqual(density["non_touch_weighted_sum"], 1)
        self.assertEqual(density["density"], 1.5)
        self.assertEqual(density["non_touch_density"], 0.5)

    def test_tap_distance_treats_1_and_8_as_adjacent(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 60.0,
                "simai": "{4}1,8,",
            }
        )

        self.assertEqual(density["tap_distance"]["pair_count"], 1)
        self.assertEqual(density["tap_distance"]["total_distance"], 1)
        self.assertAlmostEqual(density["tap_distance"]["distance_per_second"], 0.5)

    def test_tap_distance_uses_nearest_lane_for_multi_to_single(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 60.0,
                "simai": "{4}1/6,4,",
            }
        )

        self.assertEqual(density["tap_distance"]["total_distance"], 2)

    def test_measure_density_strips_pipe_comments_for_all_note_analysis_paths(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 120.0,
                "simai": "{4}1,\n|| LET'S GO!\n 2 / 4,\n|| proofread by zoya {1}\n8-5[8:1],",
            }
        )

        self.assertEqual(density["note_counts"]["tap"], 4)
        self.assertEqual(density["note_counts"]["slide"], 1)
        self.assertGreater(density["tap_distance"]["moment_count"], 0)
        self.assertEqual(density["slide_movement"]["path_count"], 1)

    def test_slide_movement_accepts_break_markers_inside_tracks(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 120.0,
                "simai": "{4}3p1b>2-4b[4:5],5bv2b[8:1]*bv8b[8:1],",
            }
        )

        self.assertEqual(density["note_counts"]["slide"], 3)
        self.assertEqual(density["note_counts"]["break_slides"], 3)
        self.assertEqual(density["slide_movement"]["path_count"], 3)
        self.assertEqual(
            [path["lanes"] for path in density["slide_movement"]["paths"]],
            [[3, 1, 2, 4], [5, 2], [5, 8]],
        )

    def test_tap_distance_uses_average_centers_for_multi_to_multi(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 60.0,
                "simai": "{4}1/6,2/4,",
            }
        )

        self.assertEqual(density["tap_moments"][0]["center"], 7.5)
        self.assertEqual(density["tap_moments"][1]["center"], 3.0)
        self.assertEqual(density["tap_distance"]["total_distance"], 4.5)

    def test_tap_distance_treats_near_shifted_double_pairs_as_one(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 4.0,
                "bpm": 60.0,
                "simai": "{4}1/6,2/5,7/4,8/3,",
            }
        )

        self.assertEqual(density["tap_distance"]["pair_count"], 3)
        self.assertEqual(density["tap_distance"]["total_distance"], 4.0)

    def test_tap_distance_skips_intervals_at_one_sixteenth_or_less(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 0.5,
                "bpm": 60.0,
                "simai": "{16}1,8,",
            }
        )

        self.assertEqual(density["tap_distance"]["pair_count"], 0)
        self.assertEqual(density["tap_distance"]["skipped_short_interval_count"], 1)
        self.assertEqual(density["tap_distance"]["total_distance"], 0)

    def test_handtrip_uses_twelfth_boundary_when_bpm_is_over_200(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 1.0,
                "bpm": 240.0,
                "simai": "{12}1,4,8,",
            }
        )

        self.assertEqual(density["tap_distance"]["pair_count"], 2)
        self.assertEqual(density["tap_distance"]["total_distance"], 7)
        self.assertEqual(density["handtrip_tap_distance"]["pair_count"], 0)
        self.assertEqual(density["handtrip_tap_distance"]["skipped_short_interval_count"], 2)
        self.assertEqual(density["handtrip_tap_distance"]["high_bpm_boundary_pair_count"], 2)
        self.assertTrue(density["handtrip_tap_distance"]["uses_high_bpm_twelfth_boundary"])

    def test_handtrip_keeps_sixteenth_boundary_when_bpm_is_200(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 1.0,
                "bpm": 200.0,
                "simai": "{12}1,4,8,",
            }
        )

        self.assertEqual(density["handtrip_tap_distance"]["pair_count"], 2)
        self.assertEqual(density["handtrip_tap_distance"]["total_distance"], 7)
        self.assertEqual(density["handtrip_tap_distance"]["high_bpm_boundary_pair_count"], 0)

    def test_same_button_triple_tap_finds_shortest_non_adjacent_window(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 5.0,
                "bpm": 60.0,
                "simai": "{4}1,,1,8,1,",
            }
        )

        triple = density["same_button_triple_tap"]
        self.assertTrue(triple["found"])
        self.assertEqual(triple["lane"], 1)
        self.assertEqual(triple["shortest_time_seconds"], 4.0)
        self.assertEqual(triple["tap_occurrences_by_lane"]["1"], 3)

    def test_same_button_triple_tap_ignores_holds(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 3.0,
                "bpm": 60.0,
                "simai": "{4}1,1h[4:1],1,",
            }
        )

        self.assertFalse(density["same_button_triple_tap"]["found"])
        self.assertEqual(density["same_button_triple_tap"]["tap_occurrences_by_lane"]["1"], 2)

    def test_slide_heads_are_counted_as_tap_moments(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 3.0,
                "bpm": 60.0,
                "simai": "{4}1-4[4:1],1,1,",
            }
        )

        self.assertEqual(density["tap_only_moments"][0]["lanes"], [1])
        self.assertTrue(density["same_button_triple_tap"]["found"])
        self.assertEqual(density["same_button_triple_tap"]["lane"], 1)
        self.assertEqual(density["same_button_triple_tap"]["shortest_time_seconds"], 2.0)

    def test_slide_movement_sums_head_to_tail_and_chained_segments(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 120.0,
                "simai": "1-4[4:1],1-4q7[4:1],1-4[4:1]*-8[4:1],",
            }
        )

        self.assertEqual(
            [path["lanes"] for path in density["slide_movement"]["paths"]],
            [[1, 4], [1, 4, 7], [1, 4], [1, 8]],
        )
        self.assertEqual(density["slide_movement"]["total_distance"], 13)
        self.assertEqual(density["handtrip_movement"]["slide_total_distance"], 13)

    def test_slide_heads_are_excluded_from_handtrip_tap_distance(self):
        density = measure_density(
            {
                "index": 1,
                "start_seconds": 0.0,
                "duration_seconds": 2.0,
                "bpm": 60.0,
                "simai": "{4}1-4[4:1],8,",
            }
        )

        self.assertEqual(density["tap_distance"]["total_distance"], 1)
        self.assertEqual(density["handtrip_tap_distance"]["moment_count"], 1)
        self.assertEqual(density["handtrip_tap_distance"]["total_distance"], 0)
        self.assertEqual(density["slide_movement"]["total_distance"], 3)
        self.assertEqual(density["handtrip_movement"]["tap_total_distance"], 0)
        self.assertEqual(density["handtrip_movement"]["total_distance"], 3)

    def test_range_summary_sums_weights_and_time(self):
        summary = summarize_density_range(
            [
                {"index": 3, "weighted_sum": 4, "duration_seconds": 2.0},
                {"index": 4, "weighted_sum": 8, "duration_seconds": 2.0},
            ]
        )

        self.assertEqual(summary["start_measure"], 3)
        self.assertEqual(summary["end_measure"], 4)
        self.assertEqual(summary["weighted_sum"], 12)
        self.assertEqual(summary["density"], 3)

    def test_segmenter_avoids_first_and_last_eight_measures(self):
        densities = [
            {"index": index + 1, "density": value, "weighted_sum": value, "duration_seconds": 1}
            for index, value in enumerate([1] * 16 + [6] * 16 + [2] * 16)
        ]

        plan = choose_segmentation(
            densities,
            SegmentationConfig(fixed_segments=3, edge_exclusion=8, window=4),
        )

        self.assertEqual(plan.segment_count, 3)
        self.assertEqual(len(plan.boundaries), 2)
        self.assertTrue(all(boundary > 8 for boundary in plan.boundaries))
        self.assertTrue(all(boundary < len(densities) - 8 for boundary in plan.boundaries))

    def test_segmenter_splits_long_low_density_before_high_density(self):
        values = [8] * 12 + [2] * 20 + [9] * 20 + [5] * 12
        densities = [
            {"index": index + 1, "density": value, "weighted_sum": value, "duration_seconds": 1}
            for index, value in enumerate(values)
        ]

        plan = choose_segmentation(
            densities,
            SegmentationConfig(fixed_segments=4, edge_exclusion=8, window=4),
        )

        self.assertIn(32, plan.boundaries)

    def test_segmenter_defaults_to_five_segments(self):
        values = [2] * 12 + [8] * 12 + [3] * 12 + [9] * 12 + [4] * 12 + [10] * 12
        densities = [
            {"index": index + 1, "density": value, "weighted_sum": value, "duration_seconds": 1}
            for index, value in enumerate(values)
        ]

        plan = choose_segmentation(
            densities,
            SegmentationConfig(edge_exclusion=8, window=4),
        )

        self.assertEqual(plan.segment_count, 5)

    def test_default_min_segment_length_is_total_divided_by_ten_capped_at_ten(self):
        self.assertEqual(effective_min_segment_measures(39, SegmentationConfig()), 3)
        self.assertEqual(effective_min_segment_measures(131, SegmentationConfig()), 10)
        self.assertEqual(effective_min_segment_measures(240, SegmentationConfig()), 10)
        self.assertEqual(
            effective_min_segment_measures(
                131,
                SegmentationConfig(min_segment_measures=6),
            ),
            6,
        )

    def test_segmenter_respects_dynamic_min_segment_length(self):
        values = [2] * 20 + [8] * 20 + [3] * 20 + [9] * 20 + [4] * 20 + [10] * 31
        densities = [
            {"index": index + 1, "density": value, "weighted_sum": value, "duration_seconds": 1}
            for index, value in enumerate(values)
        ]

        plan = choose_segmentation(
            densities,
            SegmentationConfig(edge_exclusion=8, window=4),
        )

        self.assertTrue(all(length >= 10 for length in segment_lengths(len(densities), plan.boundaries)))

    def test_segment_summary_contains_note_mix_and_special_ratios(self):
        densities = [
            measure_density(
                {
                    "index": 1,
                    "duration_seconds": 2.0,
                    "bpm": 120.0,
                    "simai": "1x,2h[4:1],C,",
                }
            ),
            measure_density(
                {
                    "index": 2,
                    "duration_seconds": 2.0,
                    "bpm": 120.0,
                    "simai": "3-5[4:1],Ch[4:1],",
                }
            ),
        ]

        segment = summarize_segment(densities, 0, 2)

        self.assertEqual(segment["note_mix"]["tap"]["count"], 2)
        self.assertEqual(segment["note_mix"]["slide"]["count"], 1)
        self.assertEqual(segment["note_mix"]["hold"]["count"], 2)
        self.assertEqual(segment["note_mix"]["touch"]["count"], 1)
        self.assertEqual(segment["special_note_ratios"]["protected"]["count"], 1)
        self.assertEqual(segment["button_distribution"]["counts"]["1"], 1)
        self.assertEqual(segment["button_distribution"]["counts"]["2"], 1)
        self.assertEqual(segment["button_distribution"]["counts"]["3"], 1)
        self.assertEqual(segment["button_distribution"]["total"], 3)
        self.assertGreater(segment["button_distribution"]["variance"], 0)

    def test_density_profile_filters_very_short_measures(self):
        densities = [
            {
                "index": 1,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "duration_seconds": 1.0,
                "beats": 4.0,
                "weighted_sum": 4,
                "density": 4.0,
                "note_counts": {},
                "tap_moments": [],
                "handtrip_tap_moments": [],
                "tap_only_moments": [],
                "slide_movement": {"paths": []},
            },
            {
                "index": 2,
                "start_seconds": 1.0,
                "end_seconds": 1.01,
                "duration_seconds": 0.01,
                "beats": 0.04,
                "weighted_sum": 100,
                "density": 10000.0,
                "note_counts": {},
                "tap_moments": [],
                "handtrip_tap_moments": [],
                "tap_only_moments": [],
                "slide_movement": {"paths": []},
            },
            {
                "index": 3,
                "start_seconds": 1.01,
                "end_seconds": 2.01,
                "duration_seconds": 1.0,
                "beats": 4.0,
                "weighted_sum": 6,
                "density": 6.0,
                "note_counts": {},
                "tap_moments": [],
                "handtrip_tap_moments": [],
                "tap_only_moments": [],
                "slide_movement": {"paths": []},
            },
        ]

        segment = summarize_segment(densities, 0, 3)
        profile = segment["density_profile"]

        self.assertEqual(profile["filtered_short_measure_count"], 1)
        self.assertEqual(profile["window_count"], 2)
        self.assertAlmostEqual(profile["mean"], 5.0)
        self.assertLess(profile["q90"], 6.1)
        self.assertFalse(profile["includes_touch"])

    def test_density_profile_uses_non_touch_density(self):
        densities = [
            measure_density(
                {
                    "index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "duration_seconds": 2.0,
                    "bpm": 120.0,
                    "beats": 4.0,
                    "simai": "1/C/B1,",
                }
            )
        ]

        segment = summarize_segment(densities, 0, 1)

        self.assertEqual(segment["density"], 1.5)
        self.assertEqual(segment["density_profile"]["mean"], 0.5)

    def test_segment_summary_recomputes_tap_distance_across_measures(self):
        densities = [
            measure_density(
                {
                    "index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "duration_seconds": 2.0,
                    "bpm": 60.0,
                    "beats": 2.0,
                    "simai": "{4}1/6,2/4,",
                }
            ),
            measure_density(
                {
                    "index": 2,
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "duration_seconds": 2.0,
                    "bpm": 60.0,
                    "beats": 2.0,
                    "simai": "{4}8,4,",
                }
            ),
        ]

        segment = summarize_segment(densities, 0, 2)

        self.assertEqual(segment["tap_distance"]["moment_count"], 4)
        self.assertEqual(segment["tap_distance"]["pair_count"], 3)
        self.assertEqual(segment["tap_distance"]["total_distance"], 10.5)
        self.assertAlmostEqual(segment["tap_distance"]["distance_per_second"], 2.625)

    def test_segment_summary_finds_same_button_triple_tap_across_measures(self):
        densities = [
            measure_density(
                {
                    "index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "duration_seconds": 2.0,
                    "bpm": 60.0,
                    "beats": 2.0,
                    "simai": "{4}1,8,",
                }
            ),
            measure_density(
                {
                    "index": 2,
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "duration_seconds": 2.0,
                    "bpm": 60.0,
                    "beats": 2.0,
                    "simai": "{4}1,1,",
                }
            ),
        ]

        segment = summarize_segment(densities, 0, 2)
        triple = segment["same_button_triple_tap"]

        self.assertTrue(triple["found"])
        self.assertEqual(triple["lane"], 1)
        self.assertEqual(triple["shortest_time_seconds"], 3.0)

    def test_same_button_triple_tap_deduplicates_same_lane_same_beat(self):
        densities = [
            {
                "index": 1,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "duration_seconds": 1.0,
                "beats": 1.0,
                "weighted_sum": 0,
                "density": 0,
                "note_counts": {},
                "tap_only_moments": [
                    {"time_seconds": 0.0, "local_seconds": 0.0, "beat": 0.0, "local_beat": 0.0, "lanes": [4]},
                    {"time_seconds": 0.2, "local_seconds": 0.2, "beat": 0.5, "local_beat": 0.5, "lanes": [4]},
                ],
            },
            {
                "index": 2,
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "duration_seconds": 1.0,
                "beats": 1.0,
                "weighted_sum": 0,
                "density": 0,
                "note_counts": {},
                "tap_only_moments": [
                    {"time_seconds": 1.0, "local_seconds": 0.0, "beat": 1.0, "local_beat": 0.0, "lanes": [4]},
                    {"time_seconds": 1.0, "local_seconds": 0.0, "beat": 1.0, "local_beat": 0.0, "lanes": [4]},
                    {"time_seconds": 1.4, "local_seconds": 0.4, "beat": 1.5, "local_beat": 0.5, "lanes": [4]},
                ],
            },
        ]

        segment = summarize_segment(densities, 0, 2)
        triple = segment["same_button_triple_tap"]

        self.assertEqual(triple["deduplicated_occurrence_count"], 1)
        self.assertEqual(triple["tap_occurrences_by_lane"]["4"], 4)
        self.assertEqual(triple["shortest_time_seconds"], 1.0)

    def test_segment_handtrip_movement_includes_slide_distance(self):
        densities = [
            measure_density(
                {
                    "index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "duration_seconds": 2.0,
                    "bpm": 120.0,
                    "beats": 4.0,
                    "simai": "1-4q7[4:1],",
                }
            )
        ]

        segment = summarize_segment(densities, 0, 1)

        self.assertEqual(segment["tap_distance"]["total_distance"], 0)
        self.assertEqual(segment["slide_movement"]["total_distance"], 6)
        self.assertEqual(segment["handtrip_movement"]["total_distance"], 6)
        self.assertEqual(segment["handtrip_movement"]["distance_per_second"], 3)

    def test_four_measure_burst_counts_tap_hold_and_slide_heads(self):
        densities = [
            {"index": 1, "duration_seconds": 1.0, "note_counts": {"tap": 2, "hold": 1}},
            {"index": 2, "duration_seconds": 1.0, "note_counts": {"tap": 3, "hold": 0}},
            {"index": 3, "duration_seconds": 1.0, "note_counts": {"tap": 8, "hold": 2}},
            {"index": 4, "duration_seconds": 1.0, "note_counts": {"tap": 3, "hold": 0}},
            {"index": 5, "duration_seconds": 1.0, "note_counts": {"tap": 1, "hold": 0}},
        ]

        burst = four_measure_burst_summary(densities)

        self.assertEqual(burst["note_count"], 19)
        self.assertEqual(burst["start_measure"], 1)
        self.assertEqual(burst["end_measure"], 4)
        self.assertAlmostEqual(burst["density"], 4.75)

    def test_burst_peak_score_uses_p98_cap_and_steeper_low_penalty(self):
        self.assertAlmostEqual(burst_peak_score(0.0, 10.0)["score"], 10.0)
        self.assertGreater(burst_peak_score(2.0, 10.0)["score"], 10.0)
        self.assertLess(burst_peak_score(2.0, 10.0)["score"], 40.0)
        self.assertAlmostEqual(burst_peak_score(10.0, 10.0)["score"], 200.0)
        self.assertAlmostEqual(burst_peak_score(12.0, 10.0)["score"], 200.0)

    def test_peak_note_scores_use_density_and_note_volatility(self):
        segments = [
            {"density_profile": {"mean": 3.0, "q90": 4.0, "coefficient_of_variation": 0.2}},
            {"density_profile": {"mean": 6.0, "q90": 8.0, "coefficient_of_variation": 0.2}},
            {"density_profile": {"mean": 6.0, "q90": 8.0, "coefficient_of_variation": 0.8}},
            {"density_profile": {"mean": 9.0, "q90": 12.0, "coefficient_of_variation": 0.5}},
        ]

        compute_peak_note_scores(segments)

        low_density = segments[0]["peak_note_score"]
        mid_low_volatility = segments[1]["peak_note_score"]
        mid_high_volatility = segments[2]["peak_note_score"]
        high_density = segments[3]["peak_note_score"]

        self.assertGreater(low_density["peak"], 0)
        self.assertGreater(low_density["note"], 0)
        self.assertGreater(mid_low_volatility["peak"], low_density["peak"])
        self.assertGreater(mid_low_volatility["note"], low_density["note"])
        self.assertGreater(high_density["peak"], mid_low_volatility["peak"])
        self.assertGreater(mid_high_volatility["note"], mid_low_volatility["note"])

    def test_saturated_ratio_density_score_needs_ratio_and_density(self):
        balanced = saturated_ratio_density_score(
            ratio_value=0.20,
            density_value=1.5,
            ratio_98=0.20,
            density_98=1.5,
        )
        low_ratio = saturated_ratio_density_score(
            ratio_value=0.02,
            density_value=4.0,
            ratio_98=0.20,
            density_98=1.5,
        )
        low_density = saturated_ratio_density_score(
            ratio_value=0.60,
            density_value=0.1,
            ratio_98=0.20,
            density_98=1.5,
        )

        self.assertAlmostEqual(balanced["score"], 200.0, delta=0.4)
        self.assertLess(low_ratio["score"], balanced["score"])
        self.assertLess(low_density["score"], balanced["score"])

    def test_handtrip_score_uses_p98_density(self):
        segments = [
            {"handtrip_movement": {"distance_per_second": 0.0}},
            {"handtrip_movement": {"distance_per_second": 2.0}},
            {"handtrip_movement": {"distance_per_second": 4.0}},
            {"handtrip_movement": {"distance_per_second": 8.0}},
        ]

        compute_handtrip_scores(segments)

        self.assertEqual(segments[0]["handtrip_score"]["score"], 0)
        self.assertAlmostEqual(segments[1]["handtrip_score"]["density_98"], 7.76)
        self.assertLess(segments[2]["handtrip_score"]["score"], 200.0)
        self.assertEqual(segments[3]["handtrip_score"]["score"], 200.0)
        self.assertEqual(
            segments[1]["handtrip_score"]["normalization"],
            "current_segments_p98_power_to_200",
        )

    def test_global_handtrip_score_uses_power_mapping(self):
        half = score_handtrip(5.0, 10.0)
        full = score_handtrip(10.0, 10.0)

        self.assertLess(half["score"], 100.0)
        self.assertAlmostEqual(half["alpha"], 1.7)
        self.assertEqual(half["mapping"], "density_p98_power_to_200")
        self.assertEqual(full["score"], 200.0)

    def test_tricky_score_uses_shortest_triple_tap_window(self):
        segments = [
            {"same_button_triple_tap": {"shortest_time_seconds": None}},
            {"same_button_triple_tap": {"shortest_time_seconds": 0.30}},
            {"same_button_triple_tap": {"shortest_time_seconds": 0.60}},
            {"same_button_triple_tap": {"shortest_time_seconds": 1.20}},
        ]

        compute_tricky_scores(segments)

        self.assertAlmostEqual(segments[0]["tricky_score"]["base_score"], 0.2)
        self.assertEqual(segments[1]["tricky_score"]["score"], 200.0)
        self.assertLess(segments[2]["tricky_score"]["score"], segments[1]["tricky_score"]["score"])
        self.assertLess(segments[3]["tricky_score"]["score"], segments[2]["tricky_score"]["score"])
        self.assertEqual(
            segments[1]["tricky_score"]["normalization"],
            "current_segments_inverse_time_p98_to_200",
        )

    def test_tricky_final_curve_compresses_middle_and_keeps_high_end(self):
        self.assertAlmostEqual(tricky_final_curve(0), 0)
        self.assertAlmostEqual(tricky_final_curve(100), 84)
        self.assertAlmostEqual(tricky_final_curve(200), 200)
        self.assertLess(tricky_final_curve(150), 150)
        self.assertGreater(
            tricky_final_curve(190) - tricky_final_curve(170),
            tricky_final_curve(130) - tricky_final_curve(110),
        )

    def test_segment_labels_use_unified_steady_density_state(self):
        segments = [
            {"density": 0.03, "density_curve": [0.0, 0.0, 0.1, 0.0]},
            {"density": 1.05, "density_curve": [1.0, 1.1, 1.0, 1.1]},
            {"density": 3.05, "density_curve": [3.0, 3.1, 3.2, 3.0]},
            {"density": 8.05, "density_curve": [8.0, 8.1, 7.9, 8.2, 8.0]},
            {"density": 3.3, "density_curve": [1.0, 2.0, 3.0, 4.5, 6.0]},
            {"density": 4.0, "density_curve": [6.0, 5.0, 4.0, 3.0, 2.0]},
            {"density": 4.7, "density_curve": [2.0, 7.0, 2.0, 8.0, 2.0, 7.0]},
        ]

        annotate_segment_labels(segments)

        self.assertEqual(
            [segment["label"] for segment in segments],
            [
                "REST",
                "STEADY",
                "STEADY",
                "STEADY",
                "RISING",
                "FALLING",
                "VOLATILE",
            ],
        )
        self.assertTrue(
            all(segment["label_reason"]["classifier"] == "density_state_v1" for segment in segments)
        )

    def test_segment_labels_can_apply_level_steady_density_tiers(self):
        segments = [
            {"density": 5.5, "density_curve": [5.5, 5.5, 5.5, 5.5]},
            {"density": 7.0, "density_curve": [7.0, 7.0, 7.0, 7.0]},
            {"density": 8.5, "density_curve": [8.5, 8.5, 8.5, 8.5]},
        ]
        standards = {
            "standards": [],
            "by_level": {
                "13": {
                    "level": "13",
                    "steady_density_p33": 6.0,
                    "steady_density_p67": 8.0,
                    "source": "test",
                }
            },
        }

        annotate_segment_labels(segments, steady_standards=standards, chart_level="13")

        self.assertEqual(
            [segment["label"] for segment in segments],
            ["LOW_STEADY", "MID_STEADY", "HIGH_STEADY"],
        )
        self.assertEqual([segment["base_label"] for segment in segments], ["STEADY", "STEADY", "STEADY"])
        self.assertEqual([segment["steady_tier"] for segment in segments], ["LOW", "MID", "HIGH"])

    def test_burst_is_non_exclusive_event_label(self):
        segments = [
            {"density": 1.0, "density_curve": [1.0, 1.1, 1.0, 1.1]},
            {
                "density": 5.2,
                "measure_count": 10,
                "density_curve": [3.0, 3.0, 3.0, 3.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0],
                "four_measure_burst": {
                    "density": 10.0,
                    "note_count": 80,
                    "duration_seconds": 8.0,
                    "window_size": 4,
                    "start_measure": 5,
                    "end_measure": 8,
                },
            },
            {"density": 8.0, "density_curve": [8.0, 8.1, 8.0, 8.1]},
        ]

        annotate_segment_labels(segments)

        self.assertIn(segments[1]["label"], {"STEADY", "VOLATILE"})
        self.assertEqual(segments[1]["event_labels"], ["BURST"])
        self.assertTrue(segments[1]["has_burst"])
        self.assertNotEqual(segments[1]["label"], "BURST")

    def test_segment_labels_ignore_six_dimension_feature_metrics(self):
        segments = [
            {"density": 0.2, "density_curve": [0.2, 0.2, 0.2, 0.2]},
            {
                "density": 3.0,
                "density_curve": [3.0, 3.0, 3.0, 3.0],
                "note_mix": {
                    "slide": {"count": 1, "ratio": 0.02},
                    "hold": {"count": 30, "ratio": 0.55},
                },
                "same_button_triple_tap": {"shortest_time_seconds": 0.25},
                "handtrip_movement": {"distance_per_second": 12.0},
            },
            {"density": 8.0, "density_curve": [8.0, 8.0, 8.0, 8.0]},
        ]

        annotate_segment_labels(segments)

        self.assertEqual(segments[1]["label"], "STEADY")
        self.assertNotIn("peak_note_score", segments[1])
        self.assertNotIn("handtrip_score", segments[1])
        self.assertNotIn("tricky_score", segments[1])
        self.assertNotIn("slide_charge_score", segments[1])

    def test_segment_report_attaches_six_dimension_score_vector(self):
        segments = [
            {
                "duration_seconds": 8.0,
                "density_profile": {
                    "mean": 4.0,
                    "q90": 7.0,
                    "coefficient_of_variation": 0.4,
                },
                "four_measure_burst": {
                    "density": 10.0,
                    "note_count": 80,
                    "duration_seconds": 8.0,
                    "start_measure": 1,
                    "end_measure": 4,
                },
                "note_counts": {
                    "total": 100,
                    "tap": 70,
                    "hold": 10,
                    "slide": 20,
                    "touch": 0,
                    "touch_hold": 0,
                },
                "handtrip_movement": {"distance_per_second": 5.0},
                "same_button_triple_tap": {"shortest_time_seconds": 0.5},
            }
        ]
        baselines = {
            "density_note_mean_p95": 8.0,
            "density_cv_p05": 0.2,
            "density_cv_p95": 0.6,
            "burst_density_p98": 10.0,
            "slide_density_p98": 2.5,
            "slide_ratio_p98": 0.2,
            "charge_density_p98": 1.25,
            "charge_ratio_p98": 0.1,
            "handtrip_density_p98": 5.0,
            "tricky_intensity_p98": 2.0,
        }

        dimensions = attach_six_dimension_scores(segments, baselines)

        self.assertEqual(dimensions, ["note", "peak", "charge", "slide", "handtrip", "tricky"])
        self.assertEqual(len(segments[0]["score_vector"]), 6)
        self.assertEqual(segments[0]["score_vector"][dimensions.index("peak")], 200)
        self.assertEqual(segments[0]["scores"]["slide"], 200)
        self.assertIn(segments[0]["dominant_dimension"], dimensions)
        self.assertIn("slide_density", segments[0]["score_raw"])

    def test_segment_report_metadata_starts_with_song_and_chart_fields(self):
        chart_data = {
            "song": {
                "song_id": 1024,
                "title": "Sample Song",
                "artist": "Sample Artist",
                "bpm": "190",
                "genre": "niconico",
                "cabinet": "DX",
                "version": "18. UNiVERSE",
            },
            "chart": {
                "chart_kind": "DX",
                "chart_version": "18. UNiVERSE",
                "difficulty_index": 5,
                "difficulty_name": "Master",
                "level": "12+",
                "charter": "隅田川星人",
                "created_from_file": "chartdata-rebuilt/Sample/maidata.txt",
            },
            "timeline": {
                "duration_seconds": 144.0,
                "start_seconds": 0.0,
                "end_seconds": 144.0,
            },
        }

        metadata = chart_report_metadata("sample.json", chart_data, [{"index": 1}])

        self.assertEqual(metadata["title"], "Sample Song")
        self.assertEqual(metadata["difficulty_name"], "Master")
        self.assertEqual(metadata["charter"], "隅田川星人")
        self.assertEqual(metadata["chart_version"], "18. UNiVERSE")
        self.assertEqual(metadata["measure_count"], 1)

    def test_global_six_dimension_rows_are_scaled_to_zero_to_two_hundred(self):
        rows = [
            {
                "density_note_mean": 3.0,
                "density_peak_q90": 4.0,
                "density_cv": 0.2,
                "burst_density": 1.0,
                "slide_ratio": 0.1,
                "slide_density": 0.5,
                "charge_ratio": 0.1,
                "charge_density": 0.5,
                "handtrip_density": 1.0,
                "tricky_shortest_time": None,
            },
            {
                "density_note_mean": 8.0,
                "density_peak_q90": 12.0,
                "density_cv": 0.7,
                "burst_density": 4.0,
                "slide_ratio": 0.4,
                "slide_density": 2.0,
                "charge_ratio": 0.3,
                "charge_density": 1.5,
                "handtrip_density": 4.0,
                "tricky_shortest_time": 0.6,
            },
        ]

        baselines = compute_global_baselines(rows)
        scored = score_rows(rows, baselines)

        for row in scored:
            for dimension in ("note", "peak", "charge", "slide", "handtrip", "tricky"):
                self.assertGreaterEqual(row[f"{dimension}_score"], 0)
                self.assertLessEqual(row[f"{dimension}_score"], 200)
            self.assertIn(
                row["dominant_dimension"],
                ("note", "peak", "charge", "slide", "handtrip", "tricky"),
            )
            self.assertIn("burst", row["score_detail"]["peak_note"])
            self.assertIn(
                "burst_density",
                row["score_detail"]["peak_note"]["normalized"],
            )
            self.assertEqual(
                row["score_detail"]["peak_note"]["parameters"]["mapping"],
                "burst_p98_steeper_floor_to_200",
            )
            self.assertIn("burst_density_p98", baselines)

    def test_segment_scorer_parses_one_based_inclusive_ranges(self):
        self.assertEqual(parse_measure_range("1:16"), (1, 16))
        self.assertEqual(parse_measure_range("17-32"), (17, 32))
        self.assertEqual(parse_measure_range("9"), (9, 9))

    def test_segment_scorer_uses_global_baselines_for_six_dimensions(self):
        summary = {
            "duration_seconds": 8.0,
            "density_profile": {
                "mean": 4.0,
                "q90": 7.0,
                "coefficient_of_variation": 0.4,
            },
            "four_measure_burst": {
                "density": 10.0,
                "note_count": 80,
                "duration_seconds": 8.0,
                "start_measure": 1,
                "end_measure": 4,
            },
            "note_counts": {
                "total": 100,
                "tap": 70,
                "hold": 10,
                "slide": 20,
                "touch": 0,
                "touch_hold": 0,
            },
            "handtrip_movement": {"distance_per_second": 5.0},
            "same_button_triple_tap": {"shortest_time_seconds": 0.5},
        }
        baselines = {
            "density_note_mean_p95": 8.0,
            "density_cv_p05": 0.2,
            "density_cv_p95": 0.6,
            "burst_density_p98": 10.0,
            "slide_density_p98": 2.5,
            "slide_ratio_p98": 0.2,
            "charge_density_p98": 1.25,
            "charge_ratio_p98": 0.1,
            "handtrip_density_p98": 5.0,
            "tricky_intensity_p98": 2.0,
        }

        scored = score_segment_summary(summary, baselines)

        self.assertEqual(scored["scores"]["peak"], 200)
        self.assertEqual(scored["scores"]["slide"], 200)
        self.assertEqual(scored["scores"]["charge"], 200)
        self.assertEqual(scored["scores"]["handtrip"], 200)
        self.assertEqual(scored["raw"]["slide_density"], 2.5)
        self.assertEqual(scored["raw"]["charge_density"], 1.25)
        self.assertIn(scored["dominant_dimension"], scored["scores"])


if __name__ == "__main__":
    unittest.main()
