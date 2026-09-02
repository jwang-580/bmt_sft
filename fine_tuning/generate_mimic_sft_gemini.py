#!/usr/bin/env python3
"""Generate a MIMIC-only FPC-5 SFT dataset with Gemini-authored evidence and rationales.

The existing ground-truth answers are immutable. Gemini receives each answer as a
fixed value and supplies only ``evidence_from_text`` and a short visible
``reasoning_trace``. The final training examples omit the supplied answers from
the user message and place the fixed answers plus generated annotations in the
assistant message.

The script is resumable. Successful API responses are appended to a checkpoint
JSONL file before final datasets are assembled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_eval.extraction_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT,
    build_annotation_user_prompt,
    build_questions_block,
    build_sft_user_prompt,
)


DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_OUTPUT_DIR = "fine_tuning/outputs/mimic_gemini37_flash_sft_fpc5"
EMPTY_ANSWERS = {"", "N/A", "NA", "NONE", "NULL", "NOT FOUND", "UNKNOWN"}


@dataclass(frozen=True)
class Note:
    record_id: str
    date: str
    text: str


@dataclass(frozen=True)
class Field:
    field_id: str
    category: str
    name: str
    instructions: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic_path", default="data/mimic_iv_bmt_initial_progression.csv")
    parser.add_argument("--gold_json", default="data/mimic_form_2400_answers.json")
    parser.add_argument("--questions", default="data/CIBMTR_Form_2400_R11_Items_simplified.csv")
    parser.add_argument(
        "--exclude_record_ids_json",
        default="fine_tuning/mimic_eval_record_ids.json",
        help="JSON list of MIMIC record ids excluded from fine-tuning to prevent evaluation leakage.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fields_per_call", type=int, default=5)
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--env_file", default=".env", help="Dotenv file loaded at startup; existing process environment values take precedence.")
    parser.add_argument("--thinking_level", choices=["LOW", "MEDIUM", "HIGH"], default="LOW")
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--annotation_retries", type=int, default=2)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit_notes", type=int, default=None)
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--build_only", action="store_true")
    return parser.parse_args()


def read_notes(path: Path) -> list[Note]:
    notes: list[Note] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            record_id = str(row.get("hadm_id", "") or "").strip()
            if not record_id or record_id in seen:
                continue
            seen.add(record_id)
            note_text = str(row.get("progression_prompt", "") or "").strip()
            assess_plan = str(row.get("assess_plan", "") or "").strip()
            if assess_plan and assess_plan.lower() != "nan":
                note_text += f"\n\nASSESSMENT AND PLAN:\n{assess_plan}"
            notes.append(
                Note(
                    record_id=record_id,
                    date=str(row.get("current_time", "") or "").strip(),
                    text=note_text,
                )
            )
    return notes


def normalize_field_id(value: Any) -> str:
    text = str(value or "").strip()
    match = re.match(r"^(?:Question|Field)\s+(.+)$", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else text


def normalize_answers_container(value: Any, record_id: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return {normalize_field_id(key): answer for key, answer in value.items()}
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return {normalize_field_id(key): answer for key, answer in value[0].items()}
    raise ValueError(f"Ground-truth answers for {record_id} are not a dictionary or one-item dictionary list.")


def read_gold(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    gold: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("hadm_id") or record.get("record_id") or "").strip()
        if not record_id:
            continue
        gold[record_id] = normalize_answers_container(record.get("answers"), record_id)
    return gold


def read_fields(path: Path) -> list[Field]:
    fields: list[Field] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            field_id = str(row.get("Question Number", "") or "").strip()
            if not field_id:
                continue
            fields.append(
                Field(
                    field_id=field_id,
                    category=str(row.get("category", "") or "").strip(),
                    name=str(row.get("Question Text", "") or "").strip(),
                    instructions=str(row.get("Options / Instructions", "") or "").strip(),
                )
            )
    return fields


def read_excluded_record_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation exclusion file not found: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Evaluation exclusion file must contain a JSON list: {path}")
    excluded = {str(value).strip() for value in values if str(value).strip()}
    if len(excluded) != len(values):
        raise ValueError(f"Evaluation exclusion file contains blank or duplicate record ids: {path}")
    return excluded


def chunked(values: list[Field], size: int) -> Iterable[list[Field]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def annotation_key(record_id: str, fields: list[Field]) -> str:
    return f"{record_id}|{','.join(field.field_id for field in fields)}"


def is_empty_answer(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in EMPTY_ANSWERS
    if isinstance(value, list):
        return not value or all(is_empty_answer(item) for item in value)
    if isinstance(value, dict):
        return not value
    return False


def field_payload(field: Field, answer: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "field_id": field.field_id,
        "field_category": field.category,
        "field_name": field.name,
        "field_instructions": field.instructions,
    }
    if answer is not None:
        payload["fixed_ground_truth_answer"] = answer
    return payload


def build_annotation_prompt(note: Note, fields: list[Field], answers: dict[str, Any]) -> str:
    supplied = [field_payload(field, answers[field.field_id]) for field in fields]
    return build_annotation_user_prompt(
        record_id=note.record_id,
        date=note.date,
        note_text=note.text,
        supplied_fields=supplied,
    )


def build_training_user_prompt(note: Note, fields: list[Field]) -> str:
    field_ids = [field.field_id for field in fields]
    question_metadata = {
        field.field_id: {
            "field_category": field.category,
            "field_name": field.name,
            "field_instructions": field.instructions,
        }
        for field in fields
    }
    return build_sft_user_prompt(
        record_id=note.record_id,
        record_id_type="hadm_id",
        date=note.date,
        note_text=note.text,
        fields_block=build_questions_block(field_ids, question_metadata),
    )


def response_schema(expected_count: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": expected_count,
        "maxItems": expected_count,
        "items": {
            "type": "object",
            "properties": {
                "field_id": {"type": "string"},
                "evidence_from_text": {"type": "string"},
                "reasoning_trace": {"type": "string"},
            },
            "required": ["field_id", "evidence_from_text", "reasoning_trace"],
            "additionalProperties": False,
        },
    }


def interaction_request(prompt: str, expected_count: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "input": prompt,
        "system_instruction": ANNOTATION_SYSTEM_PROMPT,
        "generation_config": {
            "max_output_tokens": args.max_output_tokens,
            "thinking_level": args.thinking_level.lower(),
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": response_schema(expected_count),
        },
        "store": False,
    }


def sdk_value_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    return {"value": str(value)}


def call_gemini(client: Any, request_kwargs: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            interaction = client.interactions.create(**request_kwargs)
            output_text = interaction.output_text
            if not output_text:
                raise ValueError("Gemini interaction has an empty output_text value.")
            annotations = json.loads(output_text)
            if not isinstance(annotations, list):
                raise ValueError("Gemini structured output is not a JSON array.")
            return annotations, sdk_value_to_dict(getattr(interaction, "usage", None))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= args.max_retries:
                raise
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status_code in {408, 409, 429, 500, 502, 503, 504} or status_code is None
            if not retryable or attempt >= args.max_retries:
                raise
        delay = args.retry_base_seconds * (2**attempt) + random.random()
        time.sleep(min(delay, 60.0))
    raise RuntimeError(f"Gemini interaction failed: {last_error}")


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_annotations(note: Note, fields: list[Field], annotations: list[dict[str, Any]]) -> list[dict[str, str]]:
    expected_ids = [field.field_id for field in fields]
    if len(annotations) != len(fields):
        raise ValueError(f"Expected {len(fields)} annotations, received {len(annotations)}.")
    actual_ids = [str(item.get("field_id", "")).strip() for item in annotations]
    if actual_ids != expected_ids:
        raise ValueError(f"Expected field ids {expected_ids}, received {actual_ids}.")
    note_normalized = normalized_text(note.text)
    validated: list[dict[str, str]] = []
    for item in annotations:
        evidence = str(item.get("evidence_from_text", "") or "").strip()
        rationale = str(item.get("reasoning_trace", "") or "").strip()
        if not rationale:
            raise ValueError(f"Field {item.get('field_id')} has an empty reasoning_trace.")
        if evidence and normalized_text(evidence) not in note_normalized:
            raise ValueError(f"Field {item.get('field_id')} evidence is not an exact note substring: {evidence[:160]!r}")
        validated.append(
            {
                "field_id": str(item["field_id"]).strip(),
                "evidence_from_text": evidence,
                "reasoning_trace": rationale,
            }
        )
    return validated


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed[str(record["annotation_key"])] = record
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> int:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temp_path, path)
    return count


def build_datasets(
    output_dir: Path,
    notes: list[Note],
    fields: list[Field],
    gold: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    fields_per_call: int,
) -> dict[str, Any]:
    note_by_id = {note.record_id: note for note in notes}
    ordered_ids = sorted(note_by_id)
    flat_rows: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []

    for record_id in ordered_ids:
        note = note_by_id[record_id]
        answers = gold[record_id]
        for batch_index, field_batch in enumerate(chunked(fields, fields_per_call)):
            key = annotation_key(record_id, field_batch)
            checkpoint = completed.get(key)
            if checkpoint is None:
                raise ValueError(f"Missing annotation checkpoint for {key}.")
            annotation_by_id = {item["field_id"]: item for item in checkpoint["annotations"]}
            assistant_records: list[dict[str, Any]] = []
            for field in field_batch:
                annotation = annotation_by_id[field.field_id]
                row = {
                    "record_id": record_id,
                    "record_id_type": "hadm_id",
                    "date": note.date,
                    "field_id": field.field_id,
                    "field_category": field.category,
                    "field_name": field.name,
                    "field_instructions": field.instructions,
                    "answers": answers[field.field_id],
                    "evidence_from_text": annotation["evidence_from_text"],
                    "reasoning_trace": annotation["reasoning_trace"],
                }
                flat_rows.append(row)
                assistant_records.append(row)
            example = {
                "messages": [
                    {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                    {"role": "user", "content": build_training_user_prompt(note, field_batch)},
                    {"role": "assistant", "content": json.dumps(assistant_records, ensure_ascii=False, separators=(",", ":"))},
                ]
            }
            all_examples.append(example)

    counts = {
        "annotations": write_jsonl_atomic(output_dir / "mimic_annotations_gemini37_flash.jsonl", flat_rows),
        "all_examples": write_jsonl_atomic(output_dir / "mimic_sft_fpc5_all.jsonl", all_examples),
        "train_examples": write_jsonl_atomic(output_dir / "mimic_sft_fpc5_train.jsonl", all_examples),
        "train_notes": len(ordered_ids),
    }
    for stale_split in ("validation", "test"):
        stale_path = output_dir / f"mimic_sft_fpc5_{stale_split}.jsonl"
        if stale_path.exists():
            stale_path.unlink()
    return counts


def main() -> int:
    args = parse_args()
    if args.env_file:
        load_dotenv(dotenv_path=args.env_file, override=False)
    if args.fields_per_call <= 0:
        raise ValueError("--fields_per_call must be positive.")
    if args.limit_notes is not None and args.limit_notes <= 0:
        raise ValueError("--limit_notes must be positive.")
    if args.max_calls is not None and args.max_calls <= 0:
        raise ValueError("--max_calls must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")
    if args.annotation_retries < 0:
        raise ValueError("--annotation_retries must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "gemini_annotation_checkpoint.jsonl"
    manifest_path = output_dir / "generation_manifest.json"

    all_notes = read_notes(Path(args.mimic_path))
    gold = read_gold(Path(args.gold_json))
    fields = read_fields(Path(args.questions))
    excluded_eval_ids = read_excluded_record_ids(Path(args.exclude_record_ids_json))
    note_ids = {note.record_id for note in all_notes}
    gold_ids = set(gold)
    labeled_notes = [note for note in all_notes if note.record_id in gold]
    eligible_training_notes = [note for note in labeled_notes if note.record_id not in excluded_eval_ids]
    eligible_training_notes.sort(key=lambda note: note.record_id)
    usable_notes = list(eligible_training_notes)
    if args.limit_notes is not None:
        usable_notes = usable_notes[: args.limit_notes]

    field_ids = [field.field_id for field in fields]
    for note in usable_notes:
        missing = [field_id for field_id in field_ids if field_id not in gold[note.record_id]]
        if missing:
            raise ValueError(f"Ground truth for {note.record_id} is missing {len(missing)} fields: {missing[:10]}")

    planned: list[tuple[Note, int, list[Field]]] = []
    for note in usable_notes:
        for batch_index, field_batch in enumerate(chunked(fields, args.fields_per_call)):
            planned.append((note, batch_index, field_batch))
    if args.max_calls is not None:
        planned = planned[: args.max_calls]

    completed = read_checkpoint(checkpoint_path)
    manifest: dict[str, Any] = {
        "dataset": "mimic",
        "model": args.model,
        "fields_per_call": args.fields_per_call,
        "total_unique_mimic_notes": len(all_notes),
        "ground_truth_records": len(gold),
        "note_ground_truth_pairs_before_eval_exclusion": len(labeled_notes),
        "evaluation_record_ids_configured": len(excluded_eval_ids),
        "evaluation_record_ids": sorted(excluded_eval_ids),
        "evaluation_notes_present_in_mimic": len(excluded_eval_ids & note_ids),
        "evaluation_notes_with_ground_truth": len(excluded_eval_ids & note_ids & gold_ids),
        "usable_note_ground_truth_pairs_after_eval_exclusion": len(eligible_training_notes),
        "split_strategy": "The 20 configured evaluation record IDs are held out; every remaining labeled MIMIC note is assigned to training.",
        "notes_without_ground_truth": sorted(note_ids - gold_ids),
        "ground_truth_without_note": sorted(gold_ids - note_ids),
        "selected_notes": len(usable_notes),
        "fields_per_note": len(fields),
        "expected_annotations": len(usable_notes) * len(fields),
        "expected_fpc5_examples": len(usable_notes) * ((len(fields) + args.fields_per_call - 1) // args.fields_per_call),
        "planned_calls_this_run": len(planned),
        "completed_calls_before_run": len(completed),
        "checkpoint_file": str(checkpoint_path),
        "ground_truth_modified": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        if planned:
            sample_note, _, sample_fields = planned[0]
            sample = {
                "model": args.model,
                "interaction_create_kwargs": interaction_request(
                    build_annotation_prompt(sample_note, sample_fields, gold[sample_note.record_id]),
                    len(sample_fields),
                    args,
                ),
            }
            (output_dir / "sample_annotation_request.json").write_text(
                json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0

    if not args.build_only:
        api_key = os.getenv(args.api_key_env) or (os.getenv("GOOGLE_API_KEY") if args.api_key_env != "GOOGLE_API_KEY" else None)
        if not api_key:
            raise RuntimeError(
                f"No Gemini API key is configured. Set {args.api_key_env} (or GOOGLE_API_KEY) in the process environment or {args.env_file}; the checkpoint makes the job resumable."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                f"google-genai>=2.3.0 is required by the Gemini Interactions API runner but is unavailable to {sys.executable}. "
                "Install fine_tuning/requirements.txt with this same Python interpreter."
            ) from exc

        thread_local = threading.local()

        def get_client() -> Any:
            client = getattr(thread_local, "gemini_client", None)
            if client is None:
                client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": int(args.timeout * 1000)},
                )
                thread_local.gemini_client = client
            return client

        pending = [
            (note, batch_index, field_batch)
            for note, batch_index, field_batch in planned
            if annotation_key(note.record_id, field_batch) not in completed
        ]

        def annotate_batch(task: tuple[Note, int, list[Field]]) -> dict[str, Any]:
            note, batch_index, field_batch = task
            key = annotation_key(note.record_id, field_batch)
            prompt = build_annotation_prompt(note, field_batch, gold[note.record_id])
            request_kwargs = interaction_request(prompt, len(field_batch), args)
            last_error: Exception | None = None
            for annotation_attempt in range(args.annotation_retries + 1):
                try:
                    annotations, usage = call_gemini(get_client(), request_kwargs, args)
                    annotations = validate_annotations(note, field_batch, annotations)
                    if args.sleep:
                        time.sleep(args.sleep)
                    return {
                        "annotation_key": key,
                        "record_id": note.record_id,
                        "batch_index": batch_index,
                        "field_ids": [field.field_id for field in field_batch],
                        "model": args.model,
                        "annotations": annotations,
                        "usage_metadata": usage,
                    }
                except ValueError as exc:
                    last_error = exc
                    if annotation_attempt >= args.annotation_retries:
                        raise
                    time.sleep(args.retry_base_seconds * (annotation_attempt + 1))
            raise RuntimeError(f"Annotation failed for {key}: {last_error}")

        finished = 0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
        try:
            future_to_task = {executor.submit(annotate_batch, task): task for task in pending}
            for future in concurrent.futures.as_completed(future_to_task):
                checkpoint = future.result()
                append_jsonl(checkpoint_path, checkpoint)
                completed[checkpoint["annotation_key"]] = checkpoint
                finished += 1
                print(f"[{finished}/{len(pending)}] completed {checkpoint['annotation_key']}", flush=True)
        except Exception:
            for future in future_to_task:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    expected_keys = {
        annotation_key(note.record_id, field_batch)
        for note in usable_notes
        for field_batch in chunked(fields, args.fields_per_call)
    }
    completed_keys = set(completed)
    missing_keys = expected_keys - completed_keys
    if missing_keys:
        print(
            f"Checkpoint is incomplete: {len(completed_keys & expected_keys)}/{len(expected_keys)} expected batches completed. ",
            f"Rerun to resume. First missing key: {sorted(missing_keys)[0]}",
            file=sys.stderr,
        )
        return 2

    counts = build_datasets(output_dir, usable_notes, fields, gold, completed, args.fields_per_call)
    manifest.update(
        {
            "completed_calls_after_run": len(completed_keys & expected_keys),
            "integrity_checks": {
                "missing_batches": 0,
                "expected_annotations": len(usable_notes) * len(fields),
                "actual_annotations": counts["annotations"],
                "evaluation_ids_excluded_from_training": True,
                "all_remaining_notes_assigned_to_training": True,
            },
            "outputs": counts,
        }
    )
    if counts["annotations"] != manifest["expected_annotations"]:
        raise ValueError(
            f"Annotation count mismatch: expected {manifest['expected_annotations']}, got {counts['annotations']}."
        )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
