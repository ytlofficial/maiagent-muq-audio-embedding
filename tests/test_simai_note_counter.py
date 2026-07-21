import unittest

from scripts.simai_note_counter import SimaiNoteParseError, count_maidata_charts, count_simai_notes


class SimaiNoteCounterTest(unittest.TestCase):
    def test_counts_basic_note_types(self):
        result = count_simai_notes("(120){4}1,5h[2:1],B1,Ch[4:3],E")

        self.assertEqual(result.counts.tap, 1)
        self.assertEqual(result.counts.hold, 1)
        self.assertEqual(result.counts.touch, 1)
        self.assertEqual(result.counts.touch_hold, 1)
        self.assertEqual(result.counts.total, 4)

    def test_slide_counts_as_start_tap_plus_slide(self):
        result = count_simai_notes("1-4[8:3],E")

        self.assertEqual(result.counts.tap, 1)
        self.assertEqual(result.counts.slide_start_taps, 1)
        self.assertEqual(result.counts.slide, 1)
        self.assertEqual(result.counts.total, 2)

    def test_same_start_multi_slide_shares_one_start_tap(self):
        result = count_simai_notes("1-4[4:3]*-6[8:5],E")

        self.assertEqual(result.counts.tap, 1)
        self.assertEqual(result.counts.slide_start_taps, 1)
        self.assertEqual(result.counts.slide, 2)
        self.assertEqual(result.counts.total, 3)

    def test_chained_slide_is_one_slide(self):
        result = count_simai_notes("1-4q7-2[1:2],1-4[2:1]q7[2:1]-2[1:1],E")

        self.assertEqual(result.counts.tap, 2)
        self.assertEqual(result.counts.slide, 2)
        self.assertEqual(result.counts.total, 4)

    def test_starless_slides_do_not_add_start_taps(self):
        result = count_simai_notes("1?-5[2:1],1!-5[2:1],E")

        self.assertEqual(result.counts.tap, 0)
        self.assertEqual(result.counts.slide, 2)
        self.assertEqual(result.counts.starless_slides, 2)
        self.assertEqual(result.counts.total, 2)

    def test_compact_and_explicit_each(self):
        result = count_simai_notes("12,1/8h[2:1],E")

        self.assertEqual(result.counts.tap, 3)
        self.assertEqual(result.counts.hold, 1)
        self.assertEqual(result.counts.each_groups, 2)
        self.assertEqual(result.counts.total, 4)

    def test_compact_modified_button_taps_are_expanded(self):
        result = count_simai_notes("1b5b,2x6x,E")

        self.assertEqual(result.counts.tap, 4)
        self.assertEqual(result.counts.break_notes, 2)
        self.assertEqual(result.counts.break_taps, 2)
        self.assertEqual(result.counts.zetsuan_notes, 2)
        self.assertEqual(result.counts.ex_notes, 2)
        self.assertEqual(result.counts.protected_notes, 2)
        self.assertEqual(result.counts.each_groups, 2)
        self.assertEqual([note.raw for note in result.notes], ["1b", "5b", "2x", "6x"])

    def test_mixed_compact_modified_button_taps_are_left_for_separate_handling(self):
        with self.assertRaises(SimaiNoteParseError):
            count_simai_notes("33b,E")

    def test_pseudo_each_backticks(self):
        result = count_simai_notes("1`2`3/4,E")

        self.assertEqual(result.counts.tap, 4)
        self.assertEqual(result.counts.pseudo_each_groups, 3)
        self.assertEqual(result.counts.each_groups, 1)

    def test_modifiers_and_special_taps(self):
        result = count_simai_notes("1b,2x,3$,4$$,B7f/Chf[1:2],E")

        self.assertEqual(result.counts.tap, 4)
        self.assertEqual(result.counts.touch, 1)
        self.assertEqual(result.counts.touch_hold, 1)
        self.assertEqual(result.counts.break_notes, 1)
        self.assertEqual(result.counts.zetsuan_notes, 1)
        self.assertEqual(result.counts.break_taps, 1)
        self.assertEqual(result.counts.ex_notes, 1)
        self.assertEqual(result.counts.protected_notes, 1)
        self.assertEqual(result.counts.star_taps, 2)
        self.assertEqual(result.counts.firework_touches, 2)

    def test_count_dict_contains_note_mix_and_special_ratios(self):
        data = count_simai_notes("1,2h[4:1],C,C1xh[2:1],E5h[2:1]f,5x,E").to_dict()

        self.assertEqual(data["note_mix"]["tap"]["count"], 2)
        self.assertEqual(data["note_mix"]["hold"]["count"], 3)
        self.assertEqual(data["note_mix"]["touch"]["count"], 1)
        self.assertEqual(data["special_note_ratios"]["protected"]["count"], 2)

    def test_button_distribution_variance_is_zero_for_even_lanes(self):
        data = count_simai_notes("12345678,E").to_dict()

        self.assertEqual(data["button_distribution"]["counts"], {
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
            "7": 1,
            "8": 1,
        })
        self.assertEqual(data["button_distribution"]["total"], 8)
        self.assertEqual(data["button_distribution"]["variance"], 0)

    def test_button_distribution_variance_grows_for_concentrated_lanes(self):
        data = count_simai_notes("1/1/1/1,E").to_dict()

        self.assertEqual(data["button_distribution"]["counts"]["1"], 4)
        self.assertEqual(data["button_distribution"]["total"], 4)
        self.assertAlmostEqual(data["button_distribution"]["mean"], 0.5)
        self.assertAlmostEqual(data["button_distribution"]["variance"], 1.75)
        self.assertAlmostEqual(data["button_distribution"]["normalized_variance"], 7.0)

    def test_same_start_multi_slide_counts_one_button_head(self):
        data = count_simai_notes("1-4[4:1]*-6[4:1],E").to_dict()

        self.assertEqual(data["button_distribution"]["counts"]["1"], 1)
        self.assertEqual(data["button_distribution"]["total"], 1)

    def test_break_hold_and_break_slide_count_as_zetsuan(self):
        result = count_simai_notes("5bh[2:1],1-4[8:3]b,7x-5b[8:1],E")

        self.assertEqual(result.counts.break_notes, 3)
        self.assertEqual(result.counts.zetsuan_notes, 3)
        self.assertEqual(result.counts.break_holds, 1)
        self.assertEqual(result.counts.break_slides, 2)

    def test_slide_break_markers_inside_tracks_are_accepted(self):
        result = count_simai_notes(
            "3p1b>2-4b[4:5],"
            "5bv2b[8:1]*bv8b[8:1],"
            "6bx-3b-6[4:13]b,"
            "7b>3b>7b[8:4],E"
        )

        self.assertEqual(result.counts.tap, 4)
        self.assertEqual(result.counts.slide, 5)
        self.assertEqual(result.counts.break_notes, 8)
        self.assertEqual(result.counts.break_taps, 3)
        self.assertEqual(result.counts.break_slides, 5)

    def test_hold_density_weights_use_duration(self):
        result = count_simai_notes("1h[32:1],2h[16:1],3h[4:1],4h[4:3],5h[4:5],E")

        self.assertEqual(result.counts.hold, 5)
        self.assertEqual(result.counts.density_weight, 1 + 1 + 2 + 3 + 4)
        self.assertEqual(
            [note.duration_values for note in result.notes],
            [["1/32"], ["1/16"], ["1/4"], ["3/4"], ["5/4"]],
        )

    def test_slide_density_counts_start_tap_plus_slide_duration_weight(self):
        result = count_simai_notes("1-4[4:1],2-5[4:3],3-6[4:5],E")

        self.assertEqual(result.counts.tap, 3)
        self.assertEqual(result.counts.slide, 3)
        self.assertEqual(result.counts.density_weight, (1 + 2) + (1 + 3) + (1 + 4))

    def test_per_segment_slide_duration_is_combined_for_density(self):
        result = count_simai_notes("1-4[4:1]q7[4:1]-2[4:1],E")

        self.assertEqual(result.counts.slide, 1)
        self.assertEqual(result.notes[0].duration_values, ["3/4"])
        self.assertEqual(result.notes[0].density_weight, 1 + 3)

    def test_touch_density_is_capped_to_three_per_same_time_value(self):
        result = count_simai_notes("A1/A2/A3/A4/A5/A6/A7/A8,E")

        self.assertEqual(result.counts.touch, 8)
        self.assertEqual(result.counts.density_weight, 3)

    def test_touch_hold_density_uses_hold_weight_but_group_is_capped(self):
        result = count_simai_notes("A1h[4:3]/A2h[4:3]/A3h[4:3]/A4h[4:3],E")

        self.assertEqual(result.counts.touch_hold, 4)
        self.assertEqual(result.counts.density_weight, 3)

    def test_exact_seconds_duration_uses_current_bpm_when_available(self):
        result = count_simai_notes("(120)1h[#1.5],E")

        self.assertEqual(result.notes[0].duration_values, ["3/4"])
        self.assertEqual(result.counts.density_weight, 3)

    def test_touch_sensor_e_is_not_terminator(self):
        result = count_simai_notes("(120){4}E4,E5,E")

        self.assertEqual(result.counts.touch, 2)
        self.assertEqual(result.counts.slots, 2)

    def test_pipe_comments_are_removed_before_tokenization(self):
        result = count_simai_notes(
            "(120){4}1,\n"
            "||LET'S GO! 8\n"
            "2,\n"
            "||proofread by zoya {1}\n"
            "3,E"
        )

        self.assertEqual(result.counts.tap, 3)
        self.assertEqual(result.counts.slots, 3)

    def test_button_hold_accepts_post_duration_break_and_ex_modifiers(self):
        result = count_simai_notes("6hx[4:1]b,4h[8:2]x,1h[12:1]b,E")

        self.assertEqual(result.counts.hold, 3)
        self.assertEqual(result.counts.break_notes, 2)
        self.assertEqual(result.counts.break_holds, 2)
        self.assertEqual(result.counts.ex_notes, 2)
        self.assertEqual(result.counts.protected_notes, 2)
        self.assertEqual(
            [note.duration_values for note in result.notes],
            [["1/4"], ["1/4"], ["1/12"]],
        )

    def test_invalid_c_alias_is_rejected(self):
        with self.assertRaises(SimaiNoteParseError):
            count_simai_notes("C3,E")

    def test_counts_maidata_inote_fields(self):
        data = count_maidata_charts(
            "&title=demo\n"
            "&inote_4=(120){4}1,2,E\n"
            "&des_4=someone\n"
            "&inote_5=1-4[8:3],E\n"
        )

        self.assertEqual(data["inote_4"]["tap"], 2)
        self.assertEqual(data["inote_5"]["tap"], 1)
        self.assertEqual(data["inote_5"]["slide"], 1)


if __name__ == "__main__":
    unittest.main()
