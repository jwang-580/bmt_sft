"""Verifiable rewards for Form 2400 extraction rollouts.

The authoritative targets for this run are the fixed answers embedded in the
SFT training JSONL. Never pass targets in the model prompt.
"""

from __future__ import annotations

import json
from typing import Any

from prompt_eval.score_predictions import classify_answer_match
from prompt_eval.evaluate_gemma4_extraction import is_empty_answer
from prompt_eval.extraction_prompts import SFT_OUTPUT_FIELDS


def completion_text(completion: Any) -> str:
    """Accept either plain TRL completions or conversational completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    return ""


def score_completion(completion: Any, gold_json: str, note_text: str) -> dict[str, float]:
    """Score answer accuracy first, then output structure and grounded evidence.

    Max total = 5.6. Nonempty gold answers get twice the weight of N/A fields,
    which discourages the trivial all-N/A policy. Missing or duplicate fields
    receive no answer credit. Format rewards cannot outweigh one correct field.
    """
    result = {"answer": 0.0, "format": 0.0, "metadata": 0.0, "evidence": 0.0}
    text = completion_text(completion).strip()
    try:
        expected = json.loads(gold_json)
        predicted = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result
    if not isinstance(expected, list) or not expected or not isinstance(predicted, list):
        return result

    expected_ids = [str(item["field_id"]) for item in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate field_id in RLVR gold labels")
    counts: dict[str, int] = {}
    for item in predicted:
        if isinstance(item, dict) and "field_id" in item:
            field_id = str(item["field_id"])
            counts[field_id] = counts.get(field_id, 0) + 1
    by_id = {
        str(item["field_id"]): item
        for item in predicted
        if isinstance(item, dict)
        and "field_id" in item
        and counts[str(item["field_id"])] == 1
    }

    if (
        len(predicted) == len(expected)
        and all(isinstance(item, dict) and set(item) == set(SFT_OUTPUT_FIELDS) for item in predicted)
        and [str(item["field_id"]) for item in predicted] == expected_ids
        and all(isinstance(item["evidence_from_text"], str) and isinstance(item["reasoning_trace"], str) for item in predicted)
    ):
        result["format"] = 0.4

    total_weight = sum(1.0 if is_empty_answer(item["answers"]) else 2.0 for item in expected)
    metadata_count = 0
    evidence_count = 0
    for gold in expected:
        row = by_id.get(str(gold["field_id"]))
        if row is None:
            continue
        metadata_keys = SFT_OUTPUT_FIELDS[:7]
        if all(row.get(key) == gold[key] for key in metadata_keys):
            metadata_count += 1
        if "answers" not in row or not isinstance(row["answers"], (str, list)):
            continue
        field_weight = 1.0 if is_empty_answer(gold["answers"]) else 2.0
        matched = classify_answer_match(
            str(gold["field_id"]), row["answers"], gold["answers"], gold
        ) != "no_match"
        if matched:
            result["answer"] += 4.0 * field_weight / total_weight
        quote = row.get("evidence_from_text")
        if not matched or not isinstance(quote, str):
            continue
        if is_empty_answer(gold["answers"]):
            evidence_count += int(quote == "")
        else:
            evidence_count += int(bool(quote) and quote in note_text)

    result["metadata"] = 0.2 * metadata_count / len(expected)
    result["evidence"] = 1.0 * evidence_count / len(expected)
    return result


def make_reward_func(component: str):
    if component not in {"answer", "format", "metadata", "evidence"}:
        raise ValueError(f"Unknown reward component: {component}")

    def reward_func(completions: list[Any], gold_json: list[str], note_text: list[str], **_kwargs: Any) -> list[float]:
        if len(completions) != len(gold_json) or len(completions) != len(note_text):
            raise ValueError("TRL reward inputs have mismatched batch lengths")
        return [
            score_completion(completion, gold, note)[component]
            for completion, gold, note in zip(completions, gold_json, note_text)
        ]

    reward_func.__name__ = f"{component}_reward"
    return reward_func
