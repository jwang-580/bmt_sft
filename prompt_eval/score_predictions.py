#!/usr/bin/env python3
"""Score saved extraction predictions against ground truth without loading a model.

Examples:
    python prompt_eval/score_predictions.py fine_tuning/outputs/gemma4_12b_lora/eval/held_out_predictions.json
    python prompt_eval/score_predictions.py prompt_eval/results/mimic_google-gemma-4-12b-it_predictions.json
    python prompt_eval/score_predictions.py prompt_eval/results
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

try:
    from .evaluate_gemma4_extraction import (
        answers_match,
        is_empty_answer,
        load_gold_answers,
        load_question_metadata,
        normalize_answer,
        normalize_question_number,
    )
except ImportError:
    # Support direct execution: python prompt_eval/score_predictions.py ...
    from evaluate_gemma4_extraction import (
        answers_match,
        is_empty_answer,
        load_gold_answers,
        load_question_metadata,
        normalize_answer,
        normalize_question_number,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_BY_DATASET = {
    "mimic": REPO_ROOT / "data" / "mimic_form_2400_answers.json",
    "osu": REPO_ROOT / "data" / "osu_form_2400_all_extractions.json",
}
DEFAULT_MIMIC_RECORD_IDS = REPO_ROOT / "fine_tuning" / "mimic_eval_record_ids.json"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "CIBMTR_Form_2400_R11_Items_simplified.csv"
GROUP_FIELDS = ("dataset", "model", "mode", "fields_per_call", "note_strategy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions",
        type=Path,
        help="A prediction JSON/JSONL file, or a directory containing *_predictions.json files.",
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "mimic", "osu"),
        default="auto",
        help="Dataset type; auto detects it independently for each input file.",
    )
    parser.add_argument(
        "--gold_json",
        type=Path,
        default=None,
        help="Override GT path. By default, MIMIC and OSU use their corresponding GT files.",
    )
    parser.add_argument(
        "--record_ids",
        type=Path,
        default=None,
        help="Optional JSON list defining the evaluation cohort. MIMIC defaults to the fixed 20 IDs.",
    )
    parser.add_argument(
        "--all_records",
        action="store_true",
        help="Score record IDs found in each file instead of applying a cohort list.",
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

    record_id = value.get("record_id")
    if record_id is None:
        record_id = value.get("hadm_id") or value.get("MRN")
    if record_id is None or value.get("field_id") is None:
        return []
    prediction = dict(value)
    for name, metadata_value in context.items():
        prediction.setdefault(name, metadata_value)
    prediction["record_id"] = str(record_id).strip()
    prediction["field_id"] = normalize_question_number(prediction["field_id"])
    prediction.setdefault("answers", "N/A")
    return [prediction]


def read_record_ids(path: Path) -> set[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON list of record IDs in {path}")
    return {str(value).strip() for value in values if str(value).strip()}


def detect_dataset(path: Path, predictions: list[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    declared = {
        str(row.get("dataset", "")).strip().lower()
        for row in predictions
        if str(row.get("dataset", "")).strip()
    }
    if len(declared) == 1 and next(iter(declared)) in DEFAULT_GOLD_BY_DATASET:
        return next(iter(declared))
    filename = path.name.lower()
    if filename.startswith("mimic_"):
        return "mimic"
    if filename.startswith("osu_"):
        return "osu"
    record_id_types = {
        str(row.get("record_id_type", "")).strip().lower()
        for row in predictions
        if str(row.get("record_id_type", "")).strip()
    }
    if record_id_types == {"hadm_id"}:
        return "mimic"
    if record_id_types == {"mrn"}:
        return "osu"
    raise ValueError(
        f"Could not determine dataset for {path}. Pass --dataset mimic or --dataset osu."
    )


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Prediction path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Prediction path is neither a file nor directory: {path}")
    files = sorted(path.glob("*_predictions.json"))
    if not files:
        raise FileNotFoundError(f"No *_predictions.json files found in {path}")
    return files


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


def parse_serialized_container(value: Any) -> Any:
    """Decode JSON/Python list strings emitted by models or stored in labels."""
    if not isinstance(value, str) or not value.strip().startswith(("[", "{")):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, dict)):
            return parsed
    return value


def unwrap_singleton(value: Any) -> Any:
    value = parse_serialized_container(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def repair_text_artifacts(value: Any) -> Any:
    """Repair narrow token-splitting artifacts without changing answer meaning."""
    if isinstance(value, list):
        return [repair_text_artifacts(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_text_artifacts(item) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    value = re.sub(r"\bY\s+es\b", "Yes", value, flags=re.IGNORECASE)
    value = re.sub(r"\bN\s+o\b", "No", value, flags=re.IGNORECASE)
    return re.sub(r"\bquestio\s+n\b", "question", value, flags=re.IGNORECASE)


def join_split_goto_fragments(value: Any) -> Any:
    """Rejoin a list option split before its trailing ``Go to question`` text."""
    if not isinstance(value, list):
        return value
    joined: list[Any] = []
    for item in value:
        if (
            isinstance(item, str)
            and re.fullmatch(r"\s*go to (?:question|field)\s+[a-z]*\d+[a-z]*\s*\.?\s*", item, re.I)
            and joined
            and isinstance(joined[-1], str)
        ):
            joined[-1] = f"{joined[-1].rstrip()} {item.strip()}"
        else:
            joined.append(item)
    return joined


def is_literal_na(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() in {"N/A", "NA"}


def parse_decimal_answer(value: Any) -> Decimal | None:
    value = unwrap_singleton(value)
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        return None
    text = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def decimal_places(value: Any) -> int | None:
    value = unwrap_singleton(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"[+-]?\d+\.(\d+)", value.strip().replace(",", ""))
    return len(match.group(1)) if match else None


def parse_date_answer(value: Any) -> date | None:
    value = unwrap_singleton(value)
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value.strip())
    patterns = (
        (r"(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})", (1, 2, 3)),
        (r"(\d{4})\s+(\d{1,2})\s+(\d{1,2})", (1, 2, 3)),
        (r"(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{4})", (3, 1, 2)),
        (r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})", (3, 1, 2)),
    )
    for pattern, order in patterns:
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        parts = [int(match.group(index)) for index in order]
        try:
            return date(*parts)
        except ValueError:
            return None
    return None


def extract_date_answers(value: Any) -> list[date]:
    """Extract valid full dates from a combined answer or date range."""
    value = parse_serialized_container(value)
    if isinstance(value, list):
        return [
            parsed_date
            for item in value
            for parsed_date in extract_date_answers(item)
        ]
    if not isinstance(value, str):
        return []

    exact_date = parse_date_answer(value)
    if exact_date is not None:
        return [exact_date]

    text = re.sub(r"\s+", " ", value.strip())
    token_patterns = (
        r"(?<!\d)\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}(?!\d)",
        r"(?<!\d)\d{4}\s+\d{1,2}\s+\d{1,2}(?!\d)",
        r"(?<!\d)\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4}(?!\d)",
        r"(?<!\d)\d{1,2}\s+\d{1,2}\s+\d{4}(?!\d)",
    )
    extracted: list[date] = []
    for pattern in token_patterns:
        for match in re.finditer(pattern, text):
            parsed_date = parse_date_answer(match.group(0))
            if parsed_date is not None and parsed_date not in extracted:
                extracted.append(parsed_date)
    return extracted


def has_yes_no_options(field_metadata: dict[str, Any] | None) -> bool:
    if not field_metadata:
        return False
    options = parse_serialized_container(field_metadata.get("field_instructions", ""))
    if not isinstance(options, list):
        return False
    normalized_options = {
        normalize_answer(repair_text_artifacts(option)) for option in options
    }
    return {"yes", "no"}.issubset(normalized_options)


def classify_answer_match(
    field_id: str,
    predicted_answer: Any,
    gold_answer: Any,
    field_metadata: dict[str, Any] | None = None,
) -> str:
    """Describe why two answers receive credit, or return ``no_match``."""
    if predicted_answer == gold_answer:
        return "exact"
    if answers_match(field_id, predicted_answer, gold_answer):
        return "normalized"

    predicted_container = parse_serialized_container(predicted_answer)
    gold_container = parse_serialized_container(gold_answer)
    if answers_match(field_id, predicted_container, gold_container):
        return "equivalent_container"

    predicted_joined = join_split_goto_fragments(predicted_container)
    gold_joined = join_split_goto_fragments(gold_container)
    if answers_match(field_id, predicted_joined, gold_joined):
        return "equivalent_split_option"

    predicted_repaired = repair_text_artifacts(predicted_joined)
    gold_repaired = repair_text_artifacts(gold_joined)
    if answers_match(field_id, predicted_repaired, gold_repaired):
        return "equivalent_text_artifact"

    predicted_singleton = unwrap_singleton(predicted_repaired)
    gold_singleton = unwrap_singleton(gold_repaired)
    if answers_match(field_id, predicted_singleton, gold_singleton):
        return "equivalent_singleton"

    normalized_field_id = normalize_question_number(field_id)
    predicted_text = normalize_answer(predicted_singleton)
    gold_text = normalize_answer(gold_singleton)
    predicted_is_na = is_literal_na(predicted_singleton)
    gold_is_na = is_literal_na(gold_singleton)
    negative_empty_pair = (
        predicted_is_na and gold_text == "no"
    ) or (
        gold_is_na and predicted_text == "no"
    )
    if negative_empty_pair and has_yes_no_options(field_metadata):
        return "equivalent_negative_empty"

    if normalized_field_id == "115" and {predicted_text, gold_text} == {
        "unknown",
        "no",
    }:
        return "equivalent_unknown_negative"

    empty_equivalents = {
        "51": {"unknown"},
        "56": {"not applicable (cord blood unit)", "not done"},
        "102": {"not done"},
        "120": {"unknown"},
        "124": {"unknown"},
        "173": {"none"},
        "178": {"none of the above"},
    }.get(normalized_field_id, set())
    empty_option_pair = (
        predicted_is_na and gold_text in empty_equivalents
    ) or (
        gold_is_na and predicted_text in empty_equivalents
    )
    if empty_option_pair:
        return "equivalent_empty_option"

    if normalized_field_id == "111" and predicted_text and gold_text:
        shorter, longer = sorted((predicted_text, gold_text), key=len)
        if re.search(rf"(?<!\w){re.escape(shorter)}(?!\w)", longer):
            return "equivalent_specific_text"

    if normalized_field_id == "153":
        def normalize_atg(value: str) -> str:
            value = re.sub(r"\(\s*atg\s*\)", "", value, flags=re.I)
            value = re.sub(r"[^a-z]+", " ", value.lower()).strip()
            return "atg" if value in {"atg", "anti thymocyte globulin", "antithymocyte globulin"} else value

        if normalize_atg(predicted_text) == normalize_atg(gold_text):
            return "equivalent_abbreviation"

    predicted_number = parse_decimal_answer(predicted_singleton)
    gold_number = parse_decimal_answer(gold_singleton)
    if predicted_number is not None and predicted_number == gold_number:
        return "equivalent_numeric"
    if (
        normalized_field_id == "103"
        and predicted_number is not None
        and gold_number is not None
        and (
            predicted_number == predicted_number.to_integral_value()
            or gold_number == gold_number.to_integral_value()
        )
        and abs(predicted_number - gold_number) <= Decimal("0.5")
    ):
        return "equivalent_height_rounding"
    gold_precision = decimal_places(gold_singleton)
    if predicted_number is not None and gold_number is not None and gold_precision is not None:
        rounding_tolerance = Decimal(5).scaleb(-(gold_precision + 1))
        if abs(predicted_number - gold_number) <= rounding_tolerance:
            return "equivalent_numeric_rounding"

    predicted_date = parse_date_answer(predicted_singleton)
    gold_date = parse_date_answer(gold_singleton)
    if predicted_date is not None and predicted_date == gold_date:
        return "equivalent_date"
    if normalized_field_id in {"135", "155", "167"}:
        predicted_dates = extract_date_answers(predicted_repaired)
        gold_dates = extract_date_answers(gold_repaired)
        if len(gold_dates) == 1 and gold_dates[0] in predicted_dates:
            return "equivalent_embedded_date"

    return "no_match"


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
        match_type = classify_answer_match(
            key[1],
            predicted_answer,
            gold_answer,
            question_metadata.get(key[1]),
        )
        exact_match = match_type != "no_match"
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
                "match_type": match_type,
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
        "equivalent_matches": sum(
            row.get("match_type", "") not in {"", "exact", "no_match"} for row in details
        ),
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


def score_file(
    predictions_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    print_summary: bool = True,
) -> dict[str, Any]:
    predictions = flatten_predictions(read_json_or_jsonl(predictions_path))
    if not predictions:
        raise ValueError(
            f"No parsed predictions found in {predictions_path}. Expected rows containing "
            "record_id (or hadm_id/MRN), field_id, and answers, or raw-run rows "
            "containing a predictions array."
        )
    dataset = detect_dataset(predictions_path, predictions, args.dataset)
    for prediction in predictions:
        prediction.setdefault("dataset", dataset)

    gold_path = (args.gold_json or DEFAULT_GOLD_BY_DATASET[dataset]).resolve()
    gold = load_gold_answers(str(gold_path))
    if not gold:
        raise ValueError(f"No ground-truth answers found in {gold_path}")
    question_metadata = load_question_metadata(str(args.questions.resolve()))

    record_ids_path: Path | None = None
    if args.all_records:
        cohort_ids = None
    elif args.record_ids is not None:
        record_ids_path = args.record_ids.resolve()
        cohort_ids = read_record_ids(record_ids_path)
    elif dataset == "mimic":
        record_ids_path = DEFAULT_MIMIC_RECORD_IDS.resolve()
        cohort_ids = read_record_ids(record_ids_path)
    else:
        cohort_ids = None

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[group_key(prediction)].append(prediction)

    summaries = []
    error_rows = []
    equivalence_rows = []
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
        equivalence_rows.extend(
            {**metadata, **row}
            for row in details
            if row.get("match_type", "") not in {"", "exact", "no_match"}
        )
        per_field_rows.extend(aggregate_details(details, metadata, "field_id"))
        per_record_rows.extend(aggregate_details(details, metadata, "record_id"))

    prefix = predictions_path.stem
    summary_path = output_dir / f"{prefix}_metrics.json"
    errors_path = output_dir / f"{prefix}_errors.csv"
    equivalences_path = output_dir / f"{prefix}_equivalences.csv"
    per_field_path = output_dir / f"{prefix}_per_field.csv"
    per_record_path = output_dir / f"{prefix}_per_record.csv"

    summary_payload = {
        "predictions_file": str(predictions_path),
        "dataset": dataset,
        "ground_truth_file": str(gold_path),
        "record_ids_file": str(record_ids_path) if record_ids_path else None,
        "groups": summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    detail_fields = list(GROUP_FIELDS) + [
        "record_id",
        "field_id",
        "field_name",
        "status",
        "exact_match",
        "match_type",
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
    write_csv(equivalences_path, equivalence_rows, detail_fields)
    write_csv(per_field_path, per_field_rows, aggregate_fields)
    write_csv(per_record_path, per_record_rows, record_aggregate_fields)

    if print_summary:
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
        print(f"Saved metrics: {summary_path}")
        print(f"Saved errors: {errors_path}")
        print(f"Saved credited equivalences: {equivalences_path}")
        print(f"Saved per-field scores: {per_field_path}")
        print(f"Saved per-record scores: {per_record_path}")
    return summary_payload


def main() -> int:
    args = parse_args()
    input_path = args.predictions.resolve()
    files = prediction_files(input_path)
    output_dir = (
        args.output_dir
        or ((input_path if input_path.is_dir() else input_path.parent) / "scores")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    directory_mode = input_path.is_dir()
    scored = []
    for index, path in enumerate(files, start=1):
        if directory_mode:
            print(f"[{index}/{len(files)}] Scoring {path.name}")
        scored.append(score_file(path.resolve(), args, output_dir, print_summary=not directory_mode))

    if directory_mode:
        combined_groups = [
            {
                "predictions_file": payload["predictions_file"],
                "dataset": payload["dataset"],
                **group,
            }
            for payload in scored
            for group in payload["groups"]
        ]
        combined = {
            "predictions_directory": str(input_path),
            "files_scored": len(scored),
            "groups": combined_groups,
        }
        combined_json = output_dir / "all_predictions_metrics.json"
        combined_csv = output_dir / "all_predictions_metrics.csv"
        combined_json.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        fieldnames = list(combined_groups[0]) if combined_groups else []
        for row in combined_groups:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
        write_csv(combined_csv, combined_groups, fieldnames)
        print(f"Scored {len(scored)} concentrated prediction files.")
        print(f"Saved combined metrics: {combined_json}")
        print(f"Saved combined CSV: {combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
