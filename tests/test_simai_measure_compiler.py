import unittest

from scripts.simai_measure_compiler import compile_chart, compiled_to_dict


class SimaiMeasureCompilerTest(unittest.TestCase):
    def test_four_quarter_notes_make_one_measure(self):
        compiled = compile_chart("(120){4}1,2,3,4,E")

        self.assertEqual(len(compiled.measures), 1)
        measure = compiled.measures[0]
        self.assertAlmostEqual(measure.start_seconds, 0.0)
        self.assertAlmostEqual(measure.end_seconds, 2.0)
        self.assertAlmostEqual(measure.beats, 4.0)
        self.assertEqual(measure.ended_by, "measure_full")

    def test_bpm_change_ends_current_measure_early(self):
        compiled = compile_chart("(120){4}1,2,(150)3,4,E")

        self.assertEqual(len(compiled.measures), 2)
        self.assertAlmostEqual(compiled.measures[0].start_seconds, 0.0)
        self.assertAlmostEqual(compiled.measures[0].end_seconds, 1.0)
        self.assertAlmostEqual(compiled.measures[0].beats, 2.0)
        self.assertEqual(compiled.measures[0].ended_by, "bpm_change")
        self.assertAlmostEqual(compiled.measures[1].start_seconds, 1.0)
        self.assertAlmostEqual(compiled.measures[1].end_seconds, 1.8)

    def test_touch_sensor_e_is_not_terminator(self):
        compiled = compile_chart("(120){4}E4,E5,E")

        self.assertEqual(len(compiled.slots), 2)
        self.assertEqual(compiled.slots[0].note_text, "E4")
        self.assertEqual(compiled.slots[1].note_text, "E5")

    def test_json_contains_measure_simai_fragment(self):
        compiled = compile_chart("(120){4}1,2,3,4,E")
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertEqual(data["measures"][0]["simai"], "(120){4}1,2,3,4,")

    def test_missing_terminator_is_appended(self):
        compiled = compile_chart("(120){4}1,2,3,4,")
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("appended missing E terminator", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}1,2,3,4,")
        self.assertNotIn("normalized_simai", data)

    def test_missing_initial_divider_defaults_to_four(self):
        compiled = compile_chart("(120)1,2,3,4,E")
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("inserted default initial {4} divider", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}1,2,3,4,")

    def test_measure_start_gets_active_divider(self):
        compiled = compile_chart("(120){4}1,2,3,4,5,6,7,8,E")
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertEqual(data["measures"][1]["simai"], "{4}5,6,7,8,")

    def test_bpm_change_measure_start_gets_divider_after_bpm(self):
        compiled = compile_chart("(120){4}1,2,(150)3,4,E")
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertEqual(data["measures"][1]["simai"], "(150){4}3,4,")

    def test_large_divider_run_converts_to_32_when_lossless(self):
        chart = "(120){640}1,,,,,,,,,,,,,,,,,,,,2,,,,,,,,,,,,,,,,,,,,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertAlmostEqual(data["duration_seconds"], 0.125)
        self.assertEqual(data["measures"][0]["simai"], "(120){32}1,2,")

    def test_639_and_641_are_treated_as_640_before_conversion(self):
        chart = "(120){641}1,,,,,,,,,,,,,,,,,,,,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertIn(
            "treated 1 large divider run(s) with {639}/{641}->{640} before converting",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){32}1,")

    def test_641_21_slots_rounds_to_one_32nd_slot(self):
        chart = "(120){641}1" + "," * 21 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){32}1,")

    def test_639_19_slots_rounds_to_one_32nd_slot(self):
        chart = "(120){639}1" + "," * 19 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){32}1,")

    def test_641_multi_note_run_expands_to_nearest_32nd_slots(self):
        chart = "(120){641}6b" + "," * 20 + "5b,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){32}6b/5b,")

    def test_983_and_625_are_treated_as_triplet_approximations(self):
        chart = (
            "(120)"
            + "{983}1b/8b"
            + "," * 41
            + "{625}2b/7b"
            + "," * 26
            + "E"
        )
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 2 large divider run(s) to {24}", data["normalization_notes"])
        self.assertIn(
            "treated 1 large divider run(s) with {983}->{984} before converting",
            data["normalization_notes"],
        )
        self.assertIn(
            "treated 1 large divider run(s) with {625}->{624} before converting",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){24}1b/8b,{24}2b/7b,")

    def test_983_and_625_multi_note_runs_expand_to_nearest_24th_slots(self):
        chart = (
            "(120)"
            + "{983}1b/8b"
            + "," * 41
            + "3b/6b,"
            + "{625}2b/7b"
            + "," * 26
            + "4b/5b,E"
        )
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 2 large divider run(s) to {24}", data["normalization_notes"])
        self.assertEqual(
            data["measures"][0]["simai"],
            "(120){24}1b/8b/3b/6b,{24}2b/7b/4b/5b,",
        )

    def test_995_is_treated_as_one_12th_slot(self):
        chart = "(120){12}1h[4:3]," + "{995}2" + "," * 83 + "{12}3,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {12}", data["normalization_notes"])
        self.assertIn(
            "treated 1 large divider run(s) with {995}->{996} before converting",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){12}1h[4:3],{12}2,{12}3,")

    def test_995_multi_note_run_expands_to_nearest_12th_slots(self):
        chart = "(120){995}2" + "," * 83 + "3,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {12}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){12}2/3,")

    def test_1000_long_run_converts_to_quarter_grid(self):
        chart = "(120){1000}4h[4:1]" + "," * 249 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {4}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}4h[4:1],")

    def test_1000_41_slot_run_converts_to_24_grid(self):
        chart = "(120){1000}1b" + "," * 40 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {24}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){24}1b,")

    def test_1000_77_slot_run_converts_to_64_grid(self):
        chart = "(120){1000}1b" + "," * 76 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {64}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){64}1b,,,,,")

    def test_1000_128_slot_multi_note_run_converts_to_192_grid(self):
        chart = "(120){1000}1b" + "," * 127 + "2b,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {192}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){192}1b" + "," * 24 + "2b,")

    def test_1000_micro_run_is_kept(self):
        chart = "(120){1000}1b,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn(
            "kept 1 large divider run(s) because they do not align to normalized grids",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){1000}1b,")

    def test_1000_micro_run_merges_into_previous_each(self):
        chart = "(120){4}8,{1000}1b,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn(
            "merged 1 micro large-divider run(s) into previous EACH",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){4}8/1b,")

    def test_two_slot_micro_run_merges_multiple_notes_into_previous_each(self):
        chart = "(120){4}8,{1000}1b,2b,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn(
            "merged 1 micro large-divider run(s) into previous EACH",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){4}8/1b/2b,")

    def test_977_run_converts_to_24_grid(self):
        chart = "(120){977}3" + "," * 284 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {24}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){24}3,,,,,,,")

    def test_673_multi_note_run_converts_to_96_grid(self):
        chart = "(120){673}1" + "," * 7 + "2" + "," * 7 + "3,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {96}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){96}1,2/3,")

    def test_649_multi_note_run_converts_to_64_grid(self):
        chart = "(120){649}1" + "," * 71 + "2,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {64}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){64}1,,,,,,2,")

    def test_empty_833_declaration_is_removed(self):
        chart = "(120){4}1,{833},{4}2,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("removed 1 empty {833} declaration run(s)", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}1,{4}2,")

    def test_922_117_slot_run_converts_to_8_grid(self):
        chart = "(120){922}5" + "," * 116 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {8}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){8}5,")

    def test_empty_922_39_slot_run_converts_to_24_grid(self):
        chart = "(120){922}" + "," * 38 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {24}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){24},")

    def test_400_one_beat_run_converts_to_4_grid(self):
        chart = "(120){400}3" + "," * 49 + "4" + "," * 51 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {4}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}3/4,")

    def test_400_four_beat_run_converts_to_4_grid(self):
        chart = "(120){400}" + "," * 329 + "3^8[4:1]" + "," * 71 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {4}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4},,,3^8[4:1],")

    def test_657_empty_run_converts_to_128_grid(self):
        chart = "(120){657}" + "," * 77 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {128}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){128},,,,,,,,,,,,,,,")

    def test_417_half_beat_run_converts_to_8_grid(self):
        chart = "(120){417}1" + "," * 52 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {8}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){8}1,")

    def test_768_short_offset_run_converts_to_192_grid(self):
        chart = "(120){768}" + "," * 3 + "8" + "," * 2 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {192}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){192}8,")

    def test_999_half_beat_run_converts_to_8_grid(self):
        chart = "(120){999}6/4" + "," * 125 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {8}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){8}6/4,")

    def test_909_short_run_converts_to_128_grid(self):
        chart = "(120){909}5" + "," * 7 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {128}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){128}5,")

    def test_622_short_run_converts_to_48_grid(self):
        chart = "(120){622}5" + "," * 12 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {48}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){48}5,")

    def test_601_run_converts_to_96_grid(self):
        chart = "(120){601}4" + "," * 143 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {96}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){96}4,,,,,,,,,,,,,,,,,,,,,,,")

    def test_617_run_converts_to_32_grid(self):
        chart = "(120){617}4" + "," * 134 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){32}4,,,,,,,")

    def test_631_run_converts_to_32_grid(self):
        chart = "(120){631}3" + "," * 137 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){32}3,,,,,,,")

    def test_832_one_beat_run_converts_to_4_grid(self):
        chart = "(120){832}5" + "," * 206 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {4}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}5,")

    def test_921_empty_run_converts_to_64_grid(self):
        chart = "(120){921}" + "," * 13 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {64}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){64},")

    def test_989_long_run_converts_to_24_grid(self):
        chart = "(120){989}1h[1253:261]" + "," * 206 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {24}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){24}1h[1253:261],,,,,")

    def test_621_long_run_converts_to_32_grid(self):
        chart = "(120){621}5" + "," * 97 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {32}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){32}5,,,,,")

    def test_637_long_run_converts_to_32_grid(self):
        chart = "(120){637}7>4[8:3]" + "," * 418 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {192}", data["normalization_notes"])
        self.assertTrue(data["measures"][0]["simai"].startswith("(120){192}7>4[8:3],"))
        self.assertEqual(data["measures"][0]["simai"].count(","), 125)

    def test_993_half_beat_run_converts_to_8_grid(self):
        chart = "(120){993}5h[1251:469]" + "," * 124 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {8}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){8}5h[1251:469],")

    def test_828_short_run_converts_to_64_grid(self):
        chart = "(120){828}6b" + "," * 13 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {64}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){64}6b,")

    def test_833_one_beat_run_converts_to_4_grid(self):
        chart = "(120){833}7-2[4:1]" + "," * 208 + "E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn("converted 1 large divider run(s) to {4}", data["normalization_notes"])
        self.assertEqual(data["measures"][0]["simai"], "(120){4}7-2[4:1],")

    def test_other_large_divider_run_stays_when_not_lossless(self):
        chart = "(120){643}1,,,,,,,,,,,,,,,,,,,,E"
        compiled = compile_chart(chart)
        data = compiled_to_dict(compiled, include_slots=False, include_raw=True)

        self.assertIn(
            "kept 1 large divider run(s) because they do not align to normalized grids",
            data["normalization_notes"],
        )
        self.assertEqual(data["measures"][0]["simai"], "(120){643}1,,,,,,,,,,,,,,,,,,,,")


if __name__ == "__main__":
    unittest.main()
