#!/usr/bin/env python3
"""Score research predictions by MRN + transplant date, including missing predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

try:
    from .research_common import (DEFAULT_DATA, DEFAULT_QUESTIONS, YN_FIELDS, DATE_FIELDS,
        encounter_key, gold_label, load_ground_truth, load_questions, normalize_value,
        read_csv, record_id, unique_encounters, write_csv, write_json)
except ImportError:
    from research_common import (DEFAULT_DATA, DEFAULT_QUESTIONS, YN_FIELDS, DATE_FIELDS,
        encounter_key, gold_label, load_ground_truth, load_questions, normalize_value,
        read_csv, record_id, unique_encounters, write_csv, write_json)

VARIANT_FIELDS = ("model", "mode", "note_strategy", "fields_per_call")


def load_predictions(path, questions):
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8-sig")) if path.suffix == ".json" else read_csv(path)
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Predictions must be an array of records")
    result = []
    target_ids = {q["target_column"]: fid for fid, q in questions.items()}
    for row in rows:
        if "field_id" in row:
            row = dict(row)
            row["field_id"] = target_ids.get(str(row["field_id"]), str(row["field_id"]))
            if "MRN" in row and (row.get("bmt_date") or row.get("BMT_date")):
                expected_id = record_id(encounter_key(row))
                if row.get("record_id") and row["record_id"] != expected_id:
                    raise ValueError("Prediction record_id disagrees with MRN+BMT_date")
                row["record_id"] = expected_id
            if not row.get("record_id") or "|" not in row["record_id"]:
                raise ValueError("Research predictions require a composite MRN|BMT_date record_id")
            result.append(row)
        else:
            # MRN-only matching is unsafe for patients with repeated transplants.
            rid = record_id(encounter_key(row))
            for target, fid in target_ids.items():
                if target in row:
                    result.append(dict(record_id=rid, field_id=fid, answers=row[target],
                                       model=row.get("model_used", path.stem), mode="legacy",
                                       note_strategy="legacy", fields_per_call="legacy",
                                       extraction_success=row.get("extraction_success", True)))
    return result


def prediction_failed(row):
    return row.get("extraction_status") == "failed" or bool(row.get("parse_error") or row.get("error")) or str(
        row.get("extraction_success", True)).lower() in {"false", "0", "no"}


def score_variant(predictions, gold, cohort, questions, *, date_tolerance=1, temp_tolerance=0.1):
    index = {}
    for p in predictions:
        key = (str(p["record_id"]), str(p["field_id"]))
        if key in index:
            raise ValueError("Duplicate prediction for encounter+field within one variant")
        if key[1] not in questions:
            raise ValueError(f"Unknown research field id: {key[1]}")
        index[key] = p
    expected = {record_id(k) for k in cohort}
    unexpected = sum(rid not in expected for rid, fid in index)
    details = []
    for key in cohort:
        rid = record_id(key)
        gt = gold.get(key)
        for fid, q in questions.items():
            field = q["target_column"]
            target, gt_status = gold_label(field, gt) if gt is not None else (None, "missing_encounter")
            p = index.get((rid, fid))
            prediction, pred_status = normalize_value(field, p.get("answers")) if p else (None, "missing_prediction")
            if p is not None and prediction_failed(p):
                pred_status = "failed"
            scorable = gt_status in {"valid", "not_applicable"}
            usable = pred_status in {"valid", "empty"}
            exact = bool(scorable and usable and target == prediction)
            error = None
            if scorable and pred_status == "valid" and gt_status == "valid":
                if field in DATE_FIELDS:
                    error = abs((date.fromisoformat(prediction) - date.fromisoformat(target)).days)
                elif field not in YN_FIELDS:
                    error = abs(prediction - target)
            tolerance = date_tolerance if field in DATE_FIELDS else temp_tolerance if field == "max_temp" else 0
            tolerant = exact or (error is not None and error <= tolerance + 1e-9)
            details.append(dict(record_id=rid, MRN=key[0], bmt_date=key[1], field_id=fid,
                                target_column=field, gold_answer=gt.get(field) if gt else None,
                                gold_status=gt_status, predicted_answer=p.get("answers") if p else None,
                                prediction_status=pred_status, scorable=scorable, exact=exact,
                                tolerance_match=bool(scorable and tolerant), absolute_error=error,
                                normalized_gold=target, normalized_prediction=prediction))
    metrics = []
    for fid, q in questions.items():
        rows = [d for d in details if d["field_id"] == fid]
        scored = [d for d in rows if d["scorable"]]
        n = len(scored)
        errors = [d["absolute_error"] for d in scored if d["absolute_error"] is not None]
        row = dict(field_id=fid, target_column=q["target_column"], expected=len(rows), scorable=n,
                   excluded_gt=len(rows)-n,
                   predictions_present=sum(d["prediction_status"] != "missing_prediction" for d in rows),
                   missing_predictions=sum(d["prediction_status"] == "missing_prediction" for d in rows),
                   failed_predictions=sum(d["prediction_status"] == "failed" for d in rows),
                   invalid_predictions=sum(d["prediction_status"] == "invalid" for d in rows),
                   not_mentioned_predictions=sum(d["prediction_status"] == "not_mentioned" for d in rows),
                   correct=sum(d["exact"] for d in scored),
                   exact_accuracy=sum(d["exact"] for d in scored)/n if n else None,
                   tolerance_accuracy=sum(d["tolerance_match"] for d in scored)/n if n else None,
                   paired_valid_count=len(errors), mean_absolute_error=sum(errors)/len(errors) if errors else None)
        if q["target_column"] in YN_FIELDS:
            positive = [d for d in scored if d["normalized_gold"] == "Y"]
            negative = [d for d in scored if d["normalized_gold"] == "N"]
            tp = sum(d["exact"] for d in positive)
            fp = sum(d["prediction_status"] == "valid" and d["normalized_prediction"] == "Y" for d in negative)
            fn = len(positive)-tp
            tn = sum(d["exact"] for d in negative)
            row.update(tp=tp, fp=fp, fn=fn, tn=tn,
                       sensitivity=tp/len(positive) if positive else None,
                       specificity=tn/len(negative) if negative else None,
                       precision=tp/(tp+fp) if tp+fp else None,
                       f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
        metrics.append(row)
    n = sum(m["scorable"] for m in metrics)
    nonempty = [m["exact_accuracy"] for m in metrics if m["exact_accuracy"] is not None]
    summary = dict(expected_encounters=len(cohort), expected_answers=len(details), scorable_answers=n,
                   excluded_gt_answers=len(details)-n,
                   prediction_coverage=sum(m["predictions_present"] for m in metrics)/len(details) if details else 0,
                   valid_prediction_coverage=sum(d["prediction_status"] in {"valid", "empty"} for d in details)/len(details) if details else 0,
                   unexpected_predictions=unexpected,
                   failed_predictions=sum(m["failed_predictions"] for m in metrics),
                   not_mentioned_predictions=sum(m["not_mentioned_predictions"] for m in metrics),
                   exact_accuracy=sum(m["correct"] for m in metrics)/n if n else None,
                   macro_exact_accuracy=sum(nonempty)/len(nonempty) if nonempty else None,
                   date_tolerance_days=date_tolerance, temperature_tolerance_f=temp_tolerance)
    return summary, metrics, details


def evaluate_file(predictions_path, ground_truth_path, cohort_path, output_dir, questions_path=DEFAULT_QUESTIONS,
                  manifest_path=None, date_tolerance=1, temp_tolerance=0.1):
    questions = load_questions(questions_path)
    gold = load_ground_truth(ground_truth_path)
    cohort = unique_encounters(read_csv(cohort_path), cohort_path)
    predictions = load_predictions(predictions_path, questions)
    if manifest_path is not None and not Path(manifest_path).is_file():
        raise ValueError("The explicitly supplied run manifest does not exist")
    manifest_path = Path(manifest_path) if manifest_path else Path(predictions_path).parent / "run_manifest.json"
    expected_variants = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = set(manifest["record_ids"])
        cohort = {k: v for k, v in cohort.items() if record_id(k) in ids}
        if len(cohort) != len(ids):
            raise ValueError("Run manifest contains encounters absent from the selected cohort")
        expected_variants = [tuple(str(v[k]) for k in VARIANT_FIELDS) for v in manifest["variants"]]
    if not cohort:
        raise ValueError("No evaluation encounters selected")
    groups = defaultdict(list)
    for v in expected_variants:
        groups[v] = []
    for p in predictions:
        v = tuple(str(p.get(k, "unspecified")) for k in VARIANT_FIELDS)
        if expected_variants and v not in groups:
            raise ValueError("Prediction variant is not part of the run manifest")
        groups[v].append(p)
    if not groups:
        groups[("unspecified",)*4] = []
    summaries, field_metrics, details = [], [], []
    for variant, rows in groups.items():
        summary, metrics, scored = score_variant(rows, gold, cohort, questions,
                                                 date_tolerance=date_tolerance, temp_tolerance=temp_tolerance)
        metadata = dict(zip(VARIANT_FIELDS, variant))
        summaries.append({**metadata, **summary})
        field_metrics.extend({**metadata, **m} for m in metrics)
        details.extend({**metadata, **d} for d in scored)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "research_evaluation_summary.json", summaries)
    for name, rows in (("research_evaluation_overall_metrics.csv", summaries),
                       ("research_evaluation_field_metrics.csv", field_metrics),
                       ("research_evaluation_details.csv", details)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        write_csv(output_dir / name, rows, fields)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Long JSON/CSV or legacy wide CSV")
    parser.add_argument("--ground-truth", "--ground_truth", default=str(DEFAULT_DATA / "toci_ptcy_crs.csv"))
    parser.add_argument("--cohort", default=str(DEFAULT_DATA / "target_filtered.csv"))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--manifest", help="Subset manifest; auto-detected beside predictions")
    parser.add_argument("--output-dir", "--output_dir")
    parser.add_argument("--date-tolerance", type=int, default=1)
    parser.add_argument("--temperature-tolerance", type=float, default=0.1)
    args = parser.parse_args()
    if args.date_tolerance < 0 or args.temperature_tolerance < 0:
        parser.error("Tolerances must be nonnegative")
    summaries = evaluate_file(args.predictions, args.ground_truth, args.cohort,
                             args.output_dir or Path(args.predictions).parent,
                             args.questions, args.manifest, args.date_tolerance, args.temperature_tolerance)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
