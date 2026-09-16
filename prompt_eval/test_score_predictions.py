import json
from pathlib import Path
import tempfile
import unittest

from prompt_eval.score_predictions import classify_answer_match, load_gold_answers


class AnswerEquivalenceTests(unittest.TestCase):
    def assert_match_type(
        self, expected, predicted, gold, field_id="1", field_metadata=None
    ):
        self.assertEqual(
            classify_answer_match(field_id, predicted, gold, field_metadata),
            expected,
        )

    def test_json_and_python_list_strings_match_real_lists(self):
        answer = ["Busulfan – Go to question 154", "Fludarabine – Go to question 154"]
        self.assert_match_type("equivalent_container", answer, str(answer))
        self.assert_match_type("equivalent_container", answer, '["Fludarabine", "Busulfan"]')

    def test_single_option_list_matches_scalar(self):
        self.assert_match_type(
            "normalized",
            ["Ursodiol – Go to question 162"],
            "Ursodiol",
        )

    def test_equivalent_numbers_ignore_representation(self):
        self.assert_match_type("equivalent_numeric", "74.0", "74")
        self.assert_match_type("normalized", 89.20, "89.2")

    def test_numeric_rounding_uses_gold_precision(self):
        self.assert_match_type("equivalent_numeric_rounding", "89.16", "89.2")
        self.assert_match_type("no_match", "89.14", "89.2")
        self.assert_match_type("no_match", "74.4", "74")

    def test_equivalent_dates_ignore_format(self):
        self.assert_match_type("equivalent_date", "2112 06 24", "2112-06-24")
        self.assert_match_type("equivalent_date", "06/24/2112", "2112-06-24")
        self.assert_match_type("equivalent_date", "06 24 2112", "2112-06-24")

    def test_narrow_text_artifacts_match(self):
        self.assert_match_type("equivalent_text_artifact", "Yes", "Y es")
        self.assert_match_type(
            "equivalent_text_artifact",
            "Hepatic, moderate / severe – Go to questio n 116",
            "Hepatic, moderate / severe – Go to question 116",
        )

    def test_date_equivalence_does_not_hide_century_errors(self):
        self.assert_match_type("no_match", "2012-06-24", "2112-06-24")

    def test_existing_field_31_equivalence_is_preserved(self):
        self.assert_match_type("normalized", "1", "N/A", field_id="31")

    def test_no_and_na_match_in_either_direction_for_yes_no_field(self):
        metadata = {
            "field_instructions": "['Yes – Go to question 12', 'No – Go to question 13']"
        }
        self.assert_match_type(
            "equivalent_negative_empty", "N/A", "No – Go to question 13", field_metadata=metadata
        )
        self.assert_match_type(
            "equivalent_negative_empty", "No – Go to question 13", "N/A", field_metadata=metadata
        )

    def test_negative_empty_rule_requires_yes_no_options_and_literal_na(self):
        yes_no_metadata = {"field_instructions": "['Yes', 'No']"}
        other_metadata = {"field_instructions": "['No', 'Not done']"}
        self.assert_match_type("no_match", "N/A", "No", field_metadata=other_metadata)
        self.assert_match_type("no_match", "Unknown", "No", field_metadata=yes_no_metadata)
        self.assert_match_type("no_match", "N/A", "Yes", field_metadata=yes_no_metadata)

    def test_q115_unknown_and_no_match_in_either_direction(self):
        self.assert_match_type(
            "equivalent_unknown_negative", "Unknown", "No", field_id="115"
        )
        self.assert_match_type(
            "equivalent_unknown_negative", "No", "Unknown", field_id="115"
        )
        self.assert_match_type("no_match", "Unknown", "No", field_id="114")
        self.assert_match_type("no_match", "Unknown", "Yes", field_id="115")

    def test_field_specific_empty_options_match_na(self):
        cases = [
            ("51", "Unknown – Go to question 53"),
            ("56", "Not done"),
            ("102", "Not done"),
            ("173", "None – Go to question 175"),
            ("120", "Unknown – Go to question 124"),
            ("124", "Unknown – Go to question 127"),
            ("56", "Not applicable (cord blood unit)"),
            ("178", "None of the above"),
        ]
        for field_id, option in cases:
            with self.subTest(field_id=field_id):
                self.assert_match_type(
                    "equivalent_empty_option", "N/A", option, field_id=field_id
                )
                self.assert_match_type(
                    "equivalent_empty_option", option, "N/A", field_id=field_id
                )

        self.assert_match_type("no_match", "Not done", "N/A", field_id="55")
        self.assert_match_type("no_match", "Reactive", "N/A", field_id="102")

    def test_q103_height_rounding_within_half_a_centimeter(self):
        self.assert_match_type(
            "equivalent_height_rounding", "175", "175.3", field_id="103"
        )
        self.assert_match_type(
            "equivalent_height_rounding", "173", "172.5", field_id="103"
        )
        self.assert_match_type("no_match", "173", "172.4", field_id="103")
        self.assert_match_type("no_match", "173", "172.5", field_id="104")

    def test_field_specific_embedded_start_dates_match(self):
        cases = [
            ("135", "395 02/17/2021", "2021-02-17"),
            ("155", "02/16/2024 — 02/20/2024", "2024-02-16"),
            ("155", "03/08/2018 — ___ ___ — ___ ___", "2018-03-08"),
            ("155", "01-13-2018 — 01-18-2018", ["2018-01-13"]),
            ("167", "2021 03 12 - 2021 03 - 13", "2021-03-12"),
        ]
        for field_id, predicted, gold in cases:
            with self.subTest(field_id=field_id, predicted=predicted):
                self.assert_match_type(
                    "equivalent_embedded_date", predicted, gold, field_id=field_id
                )

        self.assert_match_type(
            "no_match",
            "01-13-2018 — 01-18-2018",
            ["2018-01-13", "2018-01-18"],
            field_id="155",
        )
        self.assert_match_type(
            "no_match", "395 02/17/2021", "2021-02-17", field_id="136"
        )

    def test_more_specific_other_solid_tumor_text_matches(self):
        self.assert_match_type(
            "equivalent_specific_text",
            "Melanoma of right shoulder",
            "Melanoma",
            field_id="111",
        )
        self.assert_match_type("no_match", "Melanoma", "Sarcoma", field_id="111")

    def test_atg_name_and_abbreviation_match(self):
        self.assert_match_type(
            "equivalent_abbreviation",
            "Anti-thymocyte globulin (ATG)",
            "ATG",
            field_id="153",
        )

    def test_split_goto_option_is_rejoined(self):
        predicted = [
            "Genitourinary malignancy – Go to question 112",
        ]
        gold = ["Genitourinary malignancy –", "Go to question 112"]
        self.assert_match_type(
            "equivalent_split_option", predicted, gold, field_id="109"
        )

    def test_different_answers_do_not_match(self):
        self.assert_match_type("no_match", "No", "Yes")
        self.assert_match_type("no_match", "7", "130")

    def test_ground_truth_loader_preserves_utf8_option_punctuation(self):
        records = [
            {
                "hadm_id": "1",
                "answers": {"108": "Yes – Go to question 109"},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ground_truth.json"
            path.write_text(
                json.dumps(records, ensure_ascii=False), encoding="utf-8"
            )
            gold = load_gold_answers(str(path))

        self.assertEqual(gold[("1", "108")], "Yes – Go to question 109")
        self.assert_match_type(
            "exact",
            "Yes – Go to question 109",
            gold[("1", "108")],
            field_id="108",
        )


if __name__ == "__main__":
    unittest.main()
