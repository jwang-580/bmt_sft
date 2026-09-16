"""Offline regression tests: note/GT isolation, chronology, scoring and resumable runs."""
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_eval import research_extract as extraction
from research_eval.research_common import (RESEARCH_FIELDS, gold_label, load_encounters,
    load_ground_truth, load_questions, normalize_value, progress_date, record_id,
    unique_encounters, write_csv)
from research_eval.research_eval import evaluate_file, load_predictions, score_variant
from prompt_eval.extraction_prompts import (EXTRACTION_SYSTEM_PROMPT, build_sft_user_prompt,
    build_questions_block, build_user_prompt)


def fixture(directory):
    """Two admissions for the same synthetic identifier; one has no progress notes."""
    cohort = [dict(MRN="001", BMT_date=d, admit_date=a, crs_y_n="COHORT_LABEL_SENTINEL")
              for d, a in (("2020-01-10", "2020-01-01"), ("2021-01-10", "2021-01-01"))]
    write_csv(directory / "target_filtered.csv", cohort, list(cohort[0]))
    gold = []
    for c in cohort:
        row = dict(c, **{f: "" for f in RESEARCH_FIELDS})
        row.update(crs_y_n="N", hypotension_y_n="N", pressor_use_num="0", hypoxia_y_n="N",
                   high_flow_o2_y_n="N", bipap_or_intubation_y_n="N", neurotox_y_n="N",
                   toci_y_n="N", total_dose_toci="0")
        row["review_comment"] = "GT_COMMENT_SENTINEL"
        gold.append(row)
    write_csv(directory / "toci_ptcy_crs.csv", gold, list(gold[0]))
    progress = [dict(MRN="001", bmt_date="2020-01-10", note_date="0",
                     note="PROGRESS NOTE 1/9/2020. FIRST_ADMISSION_PROGRESS.", source_sheet="A")]
    write_csv(directory / "progress_notes.csv", progress*2, list(progress[0]))
    history = [dict(MRN="001", admit_date="2020-01-01", note="FIRST_BASELINE. Home oxygen."),
               dict(MRN="001", admit_date="2019-01-01", note="UNRELATED_ADMISSION_SENTINEL")]
    write_csv(directory / "h_p_notes.csv", history, list(history[0]))
    discharge = [dict(MRN="001", admit_date=c["admit_date"], discharge_date=c["BMT_date"][:4]+"-02-01",
                      note="FIRST_DISCHARGE" if i == 0 else "SECOND_DISCHARGE") for i, c in enumerate(cohort)]
    write_csv(directory / "dc_summary_notes.csv", discharge, list(discharge[0]))
    return cohort, gold


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cohort, self.gold = fixture(self.path)
        self.questions = load_questions()
        self.encounters = load_encounters(self.path)

    def test_all_fields_in_five_question_bundles(self):
        batches = extraction.bundles(list(self.questions))
        self.assertEqual([len(b) for b in batches], [5, 5, 4])
        self.assertEqual([f for b in batches for f in b], list(self.questions))

    def test_gt_selection_and_duplicate_encounter_rejection(self):
        self.assertEqual(len(load_ground_truth(self.path / "toci_ptcy_crs.csv")), 2)
        with self.assertRaises(ValueError):
            load_ground_truth(self.path / "target_filtered.csv")
        with self.assertRaises(ValueError):
            unique_encounters(self.cohort+self.cohort, "test")
        rows = [dict(c, **{f: "" for f in RESEARCH_FIELDS}) for c in self.cohort]
        write_csv(self.path / "blank.csv", rows, list(rows[0]))
        with self.assertRaisesRegex(ValueError, "All research labels are blank"):
            load_ground_truth(self.path / "blank.csv")

    def test_source_routing_no_gt_leakage_or_cross_admission_fallback(self):
        first = extraction.build_prompt(self.encounters[0], ["R10"], self.questions, "baseline")
        self.assertIn("FIRST_BASELINE", first)
        self.assertIn("FIRST_DISCHARGE", first)
        self.assertIn("FIRST_ADMISSION_PROGRESS", first)
        for secret in ("COHORT_LABEL_SENTINEL", "GT_COMMENT_SENTINEL", "UNRELATED_ADMISSION_SENTINEL", "SECOND_DISCHARGE"):
            self.assertNotIn(secret, first)
        toci = extraction.build_prompt(self.encounters[0], ["R11"], self.questions, "baseline")
        self.assertNotIn("FIRST_BASELINE", toci)
        second = extraction.build_prompt(self.encounters[1], ["R10"], self.questions, "baseline")
        self.assertIn("SECOND_DISCHARGE", second)
        self.assertNotIn("FIRST_BASELINE", second)
        self.assertNotIn("FIRST_DISCHARGE", second)

    def test_dates_preserve_conflict_and_deduplicate_notes(self):
        notes = [n for n in self.encounters[0]["notes"] if n["source"] == "progress"]
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0]["date_conflict"])
        self.assertEqual(notes[0]["date"], "2020-01-09")
        self.assertEqual(notes[0]["expected_date"], "2020-01-10")
        self.assertEqual(progress_date(dict(note_date="5", bmt_date="2020-01-10", note="No header")),
                         ("2020-01-15", "2020-01-15", False))

    def test_normalization_and_unknown_gt_are_not_negative(self):
        self.assertEqual(normalize_value("hypoxia_y_n", " n "), ("N", "valid"))
        self.assertEqual(normalize_value("fever_onset_date", "1/2/2020"), ("2020-01-02", "valid"))
        self.assertEqual(normalize_value("total_dose_toci", "2.0"), (2, "valid"))
        self.assertEqual(normalize_value("total_dose_toci", "2.5")[1], "invalid")
        self.assertEqual(normalize_value("bipap_or_intubation_y_n", "B")[1], "invalid")
        self.assertEqual(gold_label("max_temp", {"crs_y_n": "N"}), (None, "not_applicable"))
        self.assertEqual(gold_label("max_temp", {"crs_y_n": "Y"}), (None, "unlabeled"))
        self.assertEqual(gold_label("neurotox_y_n", {}), (None, "unlabeled"))

    def test_strict_response_ids_and_retries(self):
        fields = ["R01", "R05"]
        bad = '[{"field_id":"R01","answers":"N"}]'
        good = '[{"field_id":"R05","answers":"N"},{"field_id":"R01","answers":"Y"}]'
        calls = iter([bad, good])
        rows, error, attempts = extraction.request_bundle(lambda *_: next(calls), "", fields, self.questions, "baseline")
        self.assertFalse(error)
        self.assertEqual(len(attempts), 2)
        self.assertEqual([r["field_id"] for r in rows], fields)
        for raw in (bad, good.replace("R05", "R01"), good.replace('"N"', '"B"')):
            with self.assertRaises(ValueError):
                extraction.parse_response(raw, fields, self.questions, "baseline")
        with self.assertRaises(ValueError):
            extraction.parse_response(good, fields, self.questions, "baseline", require_evidence=True)

    def test_note_failure_does_not_become_successful_na(self):
        def fail(*_):
            raise RuntimeError("synthetic runtime failure")
        rows, attempts = extraction.extract_bundle(fail, self.encounters[0], ["R10"], self.questions,
                                                   "baseline", "per_note", retries=0)
        self.assertFalse(rows[0]["extraction_success"])
        self.assertTrue(rows[0]["parse_error"])
        self.assertEqual(attempts[0]["stage"], "note")

    def test_all_prompt_modes_and_non_json_parsing(self):
        for mode in extraction.DEFAULT_MODES:
            prompt = extraction.build_prompt(self.encounters[0], ["R11"], self.questions, mode)
            self.assertIn("FIRST_DISCHARGE", prompt)
            self.assertNotIn("COHORT_LABEL_SENTINEL", prompt)
            self.assertNotIn("GT_COMMENT_SENTINEL", prompt)
            expected_note = dict(record_id=self.encounters[0]["record_id"], record_id_type="MRN+BMT_date",
                                 date=self.encounters[0]["date"], note=extraction.clinical_note_text(self.encounters[0],
                                 "\n\n".join(extraction.render_note(n) for n in extraction.select_notes(
                                     self.encounters[0], ["R11"], self.questions))))
            self.assertEqual(prompt, build_user_prompt(expected_note, ["R11"], self.questions, mode))
        parsed = extraction.parse_response("Field R11: Y\nField R14: 2", ["R11", "R14"], self.questions, "non_json")
        self.assertEqual(parsed[1]["answers"], "2")

    def test_manifest_includes_completely_missing_variant(self):
        variant = dict(model="fake", mode="baseline", note_strategy="full", fields_per_call=5)
        (self.path / "predictions.json").write_text("[]", encoding="utf-8")
        (self.path / "run_manifest.json").write_text(json.dumps(dict(
            record_ids=[self.encounters[0]["record_id"]], variants=[variant])), encoding="utf-8")
        summary = evaluate_file(self.path / "predictions.json", self.path / "toci_ptcy_crs.csv",
                                self.path / "target_filtered.csv", self.path / "scores")[0]
        self.assertEqual(summary["expected_answers"], 14)
        self.assertEqual(summary["exact_accuracy"], 0)
        self.assertEqual(summary["prediction_coverage"], 0)

    def test_missing_explicit_manifest_rejected(self):
        (self.path / "predictions.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest does not exist"):
            evaluate_file(self.path / "predictions.json", self.path / "toci_ptcy_crs.csv",
                          self.path / "target_filtered.csv", self.path / "scores",
                          manifest_path=self.path / "absent.json")

    def test_final_prompt_preserves_verbatim_evidence_without_model_answers(self):
        prompts = []
        for note in self.encounters[0]["notes"]:
            note["text"] = note["text"].rstrip(".") + ". Tocilizumab given once."
        def fake(messages, mode):
            prompt = messages[-1]["content"]
            prompts.append(prompt)
            return json.dumps([dict(field_id="R14", answers="1", evidence_from_text="Tocilizumab given once",
                                    reasoning_trace="MODEL_RATIONALE_SENTINEL")])
        rows, log = extraction.extract_bundle(fake, self.encounters[0], ["R14"], self.questions,
                                              "baseline", "per_note", retries=0)
        self.assertTrue(rows[0]["extraction_success"])
        self.assertEqual(rows[0]["answers"], "1")
        self.assertIn("Tocilizumab given once", prompts[-1])
        self.assertIn("Source: discharge", prompts[-1])
        self.assertNotIn("FIRST_ADMISSION_PROGRESS", prompts[-1])
        self.assertNotIn("FIRST_DISCHARGE", prompts[-1])
        self.assertEqual(len(prompts), 3)  # One progress note, one discharge, one synthesis.
        self.assertNotIn('"answers":', prompts[-1])
        self.assertNotIn("MODEL_RATIONALE_SENTINEL", prompts[-1])
        self.assertEqual(log[-1]["stage"], "final")

    def test_missing_and_failed_predictions_penalized_no_temp_wildcard(self):
        cohort = unique_encounters(self.cohort, "test")
        gold = unique_encounters(self.gold, "test")
        rid = record_id(next(iter(cohort)))
        preds = [dict(record_id=rid, field_id="R04", answers=98.6),
                 dict(record_id=rid, field_id="R12", answers="N/A", parse_error="bad JSON")]
        summary, metrics, details = score_variant(preds, gold, cohort, self.questions)
        self.assertEqual(summary["expected_answers"], 28)
        self.assertEqual(summary["scorable_answers"], 28)
        self.assertEqual(summary["exact_accuracy"], 0)
        self.assertFalse(next(d for d in details if d["field_id"] == "R04")["exact"])
        self.assertFalse(next(d for d in details if d["field_id"] == "R12")["exact"])
        with self.assertRaises(ValueError):
            score_variant(preds+preds, gold, cohort, self.questions)

    def test_tolerance_does_not_replace_exact_and_excludes_bad_gt(self):
        gt = dict(self.gold[0], max_temp="102.0", fever_onset_date="2020-01-10", bipap_or_intubation_y_n="B")
        key = ("001", "2020-01-10")
        preds = [dict(record_id=record_id(key), field_id="R04", answers=102.1),
                 dict(record_id=record_id(key), field_id="R02", answers="2020-01-11")]
        summary, metrics, details = score_variant(preds, {key: gt}, {key: gt}, self.questions)
        self.assertEqual(summary["excluded_gt_answers"], 1)
        for fid in ("R04", "R02"):
            d = next(d for d in details if d["field_id"] == fid)
            self.assertFalse(d["exact"])
            self.assertTrue(d["tolerance_match"])

    def test_legacy_wide_requires_transplant_date(self):
        write_csv(self.path / "wide.csv", [self.gold[0]], list(self.gold[0]))
        self.assertEqual(len(load_predictions(self.path / "wide.csv", self.questions)), 14)
        row = {k: v for k, v in self.gold[0].items() if k != "BMT_date"}
        write_csv(self.path / "no_date.csv", [row], list(row))
        with self.assertRaises(ValueError):
            load_predictions(self.path / "no_date.csv", self.questions)

    def test_complete_cli_mock_inference_scoring_resume_and_changed_inputs(self):
        output = self.path / "out"
        calls = []
        def fake(messages, mode):
            fields = re.findall(r"^Field (R\d+)$", messages[-1]["content"], flags=re.M)
            calls.append(fields)
            return json.dumps([dict(field_id=f, answers=self.gold[0][self.questions[f]["target_column"]] or "N/A") for f in fields])
        argv = ["research_extract.py", "--data_dir", str(self.path), "--output_dir", str(output), "--limit", "1",
                "--note_strategy", "full"]
        with patch("sys.argv", argv), patch.object(extraction, "make_infer", return_value=(fake, lambda: None)), contextlib.redirect_stdout(io.StringIO()):
            extraction.main()
        self.assertEqual([len(c) for c in calls], [5, 5, 4])
        summary = json.loads((output / "research_evaluation_summary.json").read_text())[0]
        self.assertEqual(summary["expected_encounters"], 1)
        self.assertEqual(summary["exact_accuracy"], 1)
        with patch("sys.argv", argv+["--resume"]), patch.object(extraction, "make_infer") as loader, contextlib.redirect_stdout(io.StringIO()):
            extraction.main()
            loader.assert_not_called()
        with (self.path / "h_p_notes.csv").open("a") as h:
            h.write("\n")
        with patch("sys.argv", argv+["--resume"]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            extraction.main()

    def test_whole_notes_source_filtering_compact_evidence_and_resume(self):
        fields = ["R01", "R02", "R03", "R04", "R05"]
        cache = {}
        prompts = []
        def fake(messages, mode):
            prompt = messages[-1]["content"]
            prompts.append(prompt)
            ids = re.findall(r"^Field (R\d+)$", prompt, flags=re.M)
            return json.dumps([dict(field_id=f, answers="N/A", evidence_from_text="", reasoning_trace="") for f in ids])
        rows, log = extraction.extract_bundle(fake, self.encounters[0], fields, self.questions,
            "baseline", "per_note", evidence_cache=cache, cache_scope="model-a")
        self.assertTrue(all(r["extraction_success"] for r in rows))
        self.assertEqual(len(prompts), 4)
        hp_prompt = next(p for p in prompts if "FIRST_BASELINE" in p)
        self.assertEqual(re.findall(r"^Field (R\d+)$", hp_prompt, flags=re.M), ["R05"])
        for p in prompts[:-1]:
            self.assertEqual(sum(marker in p for marker in
                ["FIRST_BASELINE", "FIRST_DISCHARGE", "FIRST_ADMISSION_PROGRESS"]), 1)
        for note in self.encounters[0]["notes"]:
            self.assertTrue(any(note["text"] in p for p in prompts[:-1]))
        for marker in ["FIRST_BASELINE", "FIRST_DISCHARGE", "FIRST_ADMISSION_PROGRESS"]:
            self.assertNotIn(marker, prompts[-1])
        self.assertEqual(len(cache), 3)
        self.assertTrue(all(c["record_id"] == self.encounters[0]["record_id"] for c in cache.values()))
        prompts.clear()
        extraction.extract_bundle(fake, self.encounters[0], fields, self.questions,
            "baseline", "per_note", evidence_cache=cache, cache_scope="model-a")
        self.assertEqual(len(prompts), 1)  # Only final extraction is repeated.
        prompts.clear()
        extraction.extract_bundle(fake, self.encounters[0], fields, self.questions,
            "baseline", "per_note", evidence_cache=cache, cache_scope="model-b")
        self.assertEqual(len(prompts), 4)  # Never reuse another model's evidence.

    def test_unsupported_quotes_fail_validation(self):
        raw = json.dumps([dict(field_id="R14", answers=1, evidence_from_text="invented quote")])
        rows, error, attempts = extraction.request_bundle(lambda *_: raw, "prompt", ["R14"], self.questions,
            "baseline", retries=0, require_evidence=True, source_text="No medication information.")
        self.assertTrue(error)
        self.assertEqual(rows, [])

    def test_context_budget_reserves_output_and_does_not_retry(self):
        extraction.check_context_budget(20, 30, 10, 30)
        with self.assertRaises(extraction.ContextLimitError):
            extraction.check_context_budget(21, 30, 10, 30)
        calls = []
        def oversized(*_):
            calls.append(1)
            raise extraction.ContextLimitError("Too long")
        rows, error, attempts = extraction.request_bundle(oversized, "prompt", ["R01"], self.questions,
                                                         "baseline", retries=3)
        self.assertEqual(len(calls), 1)
        self.assertTrue(error)

    def test_per_note_default_cli_checkpoint_and_resume(self):
        output = self.path / "note_out"
        calls = []
        def fake(messages, mode):
            prompt = messages[-1]["content"]
            fields = re.findall(r"^Field (R\d+)$", prompt, flags=re.M)
            calls.append(fields)
            return json.dumps([dict(field_id=f, answers="N/A", evidence_from_text="", reasoning_trace="") for f in fields])
        argv = ["research_extract.py", "--data_dir", str(self.path), "--output_dir", str(output), "--limit", "1"]
        with patch("sys.argv", argv), patch.object(extraction, "make_infer", return_value=(fake, lambda: None)), contextlib.redirect_stdout(io.StringIO()):
            extraction.main()
        self.assertEqual([len(c) for c in calls], [1, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4])
        saved = json.loads((output / "note_evidence.json").read_text())
        self.assertEqual(len(saved), 8)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["variants"][0]["note_strategy"], "per_note")
        self.assertEqual(manifest["prompt_contract"], "shared_sft_baseline")
        self.assertTrue(all("events" not in field for v in saved.values() for field in v["fields"]))
        with patch("sys.argv", argv+["--resume"]), patch.object(extraction, "make_infer") as loader, contextlib.redirect_stdout(io.StringIO()):
            extraction.main()
            loader.assert_not_called()

    def test_baseline_note_and_final_prompts_equal_sft_template_exactly(self):
        fields = ["R11", "R12", "R13", "R14"]
        encounter = self.encounters[0]
        texts_and_dates = [("ONE COMPLETE ORIGINAL NOTE", "2020-01-09"),
                           ("[Source: progress; note date: 2020-01-09]\nAn original source excerpt.", encounter["date"])]
        self.assertEqual(extraction.SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT)
        for text, note_date in texts_and_dates:
            prompt = extraction.build_prompt(encounter, fields, self.questions, "baseline", note_text=text, note_date=note_date)
            expected = build_sft_user_prompt(record_id=encounter["record_id"], record_id_type=encounter["record_id_type"],
                date=note_date, note_text=extraction.clinical_note_text(encounter, text),
                fields_block=build_questions_block(fields, self.questions))
            self.assertEqual(prompt, expected)
            for obsolete in ("events array", "Synthesize final answers", "Available note counts", "4000 JSON characters"):
                self.assertNotIn(obsolete, prompt)

    def test_retry_keeps_system_and_user_prompts_identical(self):
        messages_seen = []
        def fake(messages, mode):
            messages_seen.append(messages)
            return "not valid JSON" if len(messages_seen) == 1 else '[{"field_id":"R01","answers":"N/A"}]'
        _, error, attempts = extraction.request_bundle(fake, "fixed prompt", ["R01"], self.questions, "baseline")
        self.assertFalse(error)
        self.assertEqual(messages_seen[0], messages_seen[1])
        self.assertEqual(messages_seen[0][0]["content"], EXTRACTION_SYSTEM_PROMPT)

    def test_excerpts_preserve_dates_negation_and_deduplicate_overlaps(self):
        note = dict(text="Unrelated history. On 1/2 there was no hypoxia and no hypotension. Another detail.")
        rows = [dict(evidence_from_text="hypoxia"), dict(evidence_from_text="no hypotension")]
        self.assertEqual(extraction.collect_excerpts(note, rows),
                         ["On 1/2 there was no hypoxia and no hypotension."])
        self.assertEqual(extraction.collect_excerpts(note, [dict(evidence_from_text="")]), [])
        self.assertEqual(extraction.quote_spans("No\n hypoxia", "No hypoxia"), [(0, 11)])

    def test_standard_response_needs_no_events_or_custom_fields(self):
        raw = json.dumps([dict(record_id="001|2020-01-10", record_id_type="MRN+BMT_date", date="2020-01-10",
                              field_id="R14", field_category="tocilizumab", field_name="Dose count",
                              field_instructions="Nonnegative integer", answers="1",
                              evidence_from_text="Tocilizumab given once", reasoning_trace="One administered dose.")])
        rows, error, _ = extraction.request_bundle(lambda *_: raw, "canonical prompt", ["R14"], self.questions,
            "baseline", retries=0, require_evidence=True, source_text="Tocilizumab given once.")
        self.assertFalse(error)
        self.assertEqual(rows[0]["answers"], "1")
        self.assertNotIn("events", rows[0])

    def test_modes_without_quotes_are_rejected_for_per_note(self):
        for mode in ("non_json", "no_evidence_field"):
            with self.assertRaisesRegex(ValueError, "evidence_from_text"):
                extraction.extract_bundle(lambda *_: "", self.encounters[0], ["R01"], self.questions, mode, "per_note")


if __name__ == "__main__":
    unittest.main()
