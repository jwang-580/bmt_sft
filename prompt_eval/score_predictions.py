#!/usr/bin/env python3
"""Score saved extraction predictions against ground truth without loading a model.

Examples:
    python prompt_eval/score_predictions.py fine_tuning/outputs/gemma4_12b_lora/eval/held_out_predictions.json
    python prompt_eval/score_predictions.py predictions.json --output_dir scores
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .evaluate_gemma4_extraction import (
        answers_match,
        is_empty_answer,
        load_gold_answers,
        load_question_metadata,
        normalize_question_number,
    )
except ImportError:
    # Support direct execution: python prompt_eval/score_predictions.py ...
    from evaluate_gemma4_extraction import (
        answers_match,
        is_empty_answer,
        load_gold_answers,
        load_question_metadata,
        normalize_question_number,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = REPO_ROOT / "data" / "mimic_form_2400_answers.json"
DEFAULT_RECORD_IDS = REPO_ROOT / "fine_tuning" / "mimic_eval_record_ids.json"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "CIBMTR_Form_2400_R11_Items_simplified.csv"
GROUP_FIELDS = ("dataset", "model", "mode", "fields_per_call", "note_strategy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSON or JSONL prediction file to score.")
    parser.add_argument("--gold_json", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--record_ids",
        type=Path,
        default=DEFAULT_RECORD_IDS,
        help="JSON list defining the evaluation cohort (default: the fixed 20 held-out IDs).",
    )
    parser.add_argument(
        "--all_records",
        action="store_true",
        help="Score prediction record IDs instead of filtering to --record_ids.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--scope",
        choices=("all-gold-fields", "predicted-fields"),
        default="all-gold-fields",
        help=(
            "Strict-accuracy denominator. The default expects every ground-truth field; "
            "predicted-fields limits it to field IDs present in each prediction group."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: a scores directory beside the prediction file).",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_predictions(value: Any, inherited: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Accept flat prediction arrays and raw-run objects containing predictions arrays."""
    inherited = dict(inherited or {})
    if isinstance(value, list):
        flattened: list[dict[str, Any]] = []
        for item in value:
            flattened.extend(flatten_predictions(item, inherited))
        return flattened
    if not isinstance(value, dict):
        return []

    context = dict(inherited)
    context.update({name: value[name] for name in GROUP_FIELDS if name in value})

    if isinstance(value.get("predictions"), list):
        return flatten_predictions(value["predictions"], context)
    for wrapper in ("results", "records", "extractions"):
        if isinstance(value.get(wrapper), list):
            return flatten_predictions(value[wrapper], context)

    if value.get("record_id") is None or value.get("field_id") is None:
        return []
    prediction = dict(value)
    for name, metadata_value in context.items():
        prediction.setdefault(name, metadata_value)
    prediction["record_id"] = str(prediction["record_id"]).strip()
    prediction["field_id"] = normalize_question_number(prediction["field_id"])
    prediction.setdefault("answers", "N/A")
    return [prediction]


def read_record_ids(path: Path) -> set[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON list of record IDs in {path}")
    return {str(value).strip() for value in values if str(value).strip()}


def group_key(prediction: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(prediction.get(name, "") or "") for name in GROUP_FIELDS)


def group_metadata(key: tuple[str, ...]) -> dict[str, Any]:
    values: dict[str, Any] = dict(zip(GROUP_FIELDS, key))
    if values["fields_per_call"]:
        try:
            values["fields_per_call"] = int(values["fields_per_call"])
        except ValueError:
            pass
    return values


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def score_group(
    rows: list[dict[str, Any]],
    gold: dict[tuple[str, str], Any],
    cohort_ids: set[str] | None,
    scope: str,
    question_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key = (str(row["record_id"]), normalize_question_number(row["field_id"]))
        if cohort_ids is not None and key[0] not in cohort_ids:
            continue
        if key in prediction_by_key:
            duplicate_count += 1
        prediction_by_key[key] = row

    predicted_record_ids = {record_id for record_id, _ in prediction_by_key}
    record_scope = cohort_ids if cohort_ids is not None else predicted_record_ids
    predicted_field_ids = {field_id for _, field_id in prediction_by_key}
    expected_keys = {
        key
        for key in gold
        if key[0] in record_scope
        and (scope == "all-gold-fields" or key[1] in predicted_field_ids)
    }
    covered_keys = expected_keys & prediction_by_key.keys()
    missing_keys = expected_keys - prediction_by_key.keys()
    unscored_keys = prediction_by_key.keys() - gold.keys()

    details: list[dict[str, Any]] = []
    exact = 0
    empty_nonempty = 0
    parse_errors = 0
    for key in sorted(expected_keys, key=lambda item: (item[0], item[1])):
        prediction = prediction_by_key.get(key)
        gold_answer = gold[key]
        if prediction is None:
            details.append(
                {
                    "record_id": key[0],
                    "field_id": key[1],
                    "field_name": question_metadata.get(key[1], {}).get("field_name", ""),
                    "status": "missing_prediction",
                    "exact_match": False,
                    "empty_nonempty_match": False,
                    "predicted_answer": "",
                    "gold_answer": json_cell(gold_answer),
                    "parse_error": "",
                }
            )
            continue

        predicted_answer = prediction.get("answers", "N/A")
        exact_match = answers_match(key[1], predicted_answer, gold_answer)
        empty_match = exact_match or is_empty_answer(predicted_answer) == is_empty_answer(gold_answer)
        exact += int(exact_match)
        empty_nonempty += int(empty_match)
        parse_error = str(prediction.get("parse_error", "") or "")
        parse_errors += int(bool(parse_error))
        details.append(
            {
                "record_id": key[0],
                "field_id": key[1],
                "field_name": question_metadata.get(key[1], {}).get("field_name", ""),
                "status": "correct" if exact_match else "incorrect",
                "exact_match": exact_match,
                "empty_nonempty_match": empty_match,
                "predicted_answer": json_cell(predicted_answer),
                "gold_answer": json_cell(gold_answer),
                "parse_error": parse_error,
            }
        )

    metrics = {
        "prediction_rows": len(rows),
        "unique_predictions_in_scope": len(prediction_by_key),
        "duplicate_prediction_keys": duplicate_count,
        "expected_gold_cells": len(expected_keys),
        "gold_covered": len(covered_keys),
        "coverage": ratio(len(covered_keys), len(expected_keys)),
        "correct": exact,
        "exact_accuracy": ratio(exact, len(covered_keys)),
        "strict_exact_accuracy": ratio(exact, len(expected_keys)),
        "empty_nonempty_accuracy": ratio(empty_nonempty, len(covered_keys)),
        "missing_predictions": len(missing_keys),
        "predictions_without_ground_truth": len(unscored_keys),
        "parse_errors": parse_errors,
        "scored_record_ids": len({record_id for record_id, _ in expected_keys}),
        "predicted_record_ids": len(predicted_record_ids),
        "predicted_field_ids": len(predicted_field_ids),
        "scope": scope,
    }
    errors = [row for row in details if row["status"] != "correct"]
    return metrics, details, errors


def aggregate_details(
    details: Iterable[dict[str, Any]], group_columns: dict[str, Any], dimension: str
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        buckets[str(row[dimension])].append(row)

    output = []
    for value, rows in sorted(buckets.items()):
        covered = sum(row["status"] != "missing_prediction" for row in rows)
        correct = sum(bool(row["exact_match"]) for row in rows)
        output.append(
            {
                **group_columns,
                dimension: value,
                "expected": len(rows),
                "covered": covered,
                "missing": len(rows) - covered,
                "correct": correct,
                "exact_accuracy": ratio(correct, covered),
                "strict_exact_accuracy": ratio(correct, len(rows)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions.resolve()
    gold_path = args.gold_json.resolve()
    questions_path = args.questions.resolve()
    cohort_ids = None if args.all_records else read_record_ids(args.record_ids.resolve())

    predictions = flatten_predictions(read_json_or_jsonl(predictions_path))
    if not predictions:
        raise ValueError(
            "No parsed predictions found. Expected rows containing record_id, field_id, and answers, "
            "or raw-run rows containing a predictions array."
        )
    gold = load_gold_answers(str(gold_path))
    if not gold:
        raise ValueError(f"No ground-truth answers found in {gold_path}")
    question_metadata = load_question_metadata(str(questions_path))

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[group_key(prediction)].append(prediction)

    summaries = []
    error_rows = []
    per_field_rows = []
    per_record_rows = []
    for key, rows in sorted(grouped.items()):
        metadata = group_metadata(key)
        metrics, details, errors = score_group(
            rows=rows,
            gold=gold,
            cohort_ids=cohort_ids,
            scope=args.scope,
            question_metadata=question_metadata,
        )
        summaries.append({**metadata, **metrics})
        error_rows.extend({**metadata, **row} for row in errors)
        per_field_rows.extend(aggregate_details(details, metadata, "field_id"))
        per_record_rows.extend(aggregate_details(details, metadata, "record_id"))

    output_dir = (args.output_dir or predictions_path.parent / "scores").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = predictions_path.stem
    summary_path = output_dir / f"{prefix}_metrics.json"
    errors_path = output_dir / f"{prefix}_errors.csv"
    per_field_path = output_dir / f"{prefix}_per_field.csv"
    per_record_path = output_dir / f"{prefix}_per_record.csv"

    summary_payload = {
        "predictions_file": str(predictions_path),
        "ground_truth_file": str(gold_path),
        "record_ids_file": None if args.all_records else str(args.record_ids.resolve()),
        "groups": summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    detail_fields = list(GROUP_FIELDS) + [
        "record_id",
        "field_id",
        "field_name",
        "status",
        "exact_match",
        "empty_nonempty_match",
        "predicted_answer",
        "gold_answer",
        "parse_error",
    ]
    aggregate_fields = list(GROUP_FIELDS) + [
        "field_id",
        "expected",
        "covered",
        "missing",
        "correct",
        "exact_accuracy",
        "strict_exact_accuracy",
    ]
    record_aggregate_fields = aggregate_fields.copy()
    record_aggregate_fields[len(GROUP_FIELDS)] = "record_id"
    write_csv(errors_path, error_rows, detail_fields)
    write_csv(per_field_path, per_field_rows, aggregate_fields)
    write_csv(per_record_path, per_record_rows, record_aggregate_fields)

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {summary_path}")
    print(f"Saved errors: {errors_path}")
    print(f"Saved per-field scores: {per_field_path}")
    print(f"Saved per-record scores: {per_record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
