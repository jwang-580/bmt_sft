#!/usr/bin/env python3
"""
Evaluate GPT-OSS (20B, 120B) prompt variants for CIBMTR Form 2400 extraction.

Uses the HuggingFace Transformers backend with device_map="auto" sharding and a
memory-efficient attention implementation (SDPA by default, override with
--attn_implementation). GPT-OSS-20B is loaded at full precision and run in BF16;
GPT-OSS-120B is loaded in 4-bit (bitsandbytes) quantization to save VRAM. Both
behaviors are the per-model default and can be overridden with --load_in_4bit.

The default model repos are the BF16 weights (unsloth/gpt-oss-*-BF16), which load
cleanly under plain Transformers and allow bitsandbytes 4-bit quantization for
the 120B model.

GPT-OSS uses the harmony chat format with an analysis (reasoning) channel and a
final channel. The generation is decoded keeping special tokens, then only the
final channel is used as the answer. Reasoning depth is controlled with
--reasoning_effort (low/medium/high).

By default, --fields_per_call sweeps batch sizes for the baseline prompt only.
Non-JSON mode follows the same sweep as baseline. Other prompt modes run once
with --comparison_fields_per_call, which defaults to the largest baseline batch
size. For example, --fields_per_call 1,5,25 evaluates baseline and non_json at
1, 5, and 25 fields per call, while the remaining modes run at 25 fields per
call. The evaluator selects only the largest planned batch size worth of
questions unless --field_limit N or --all_fields is supplied.

By default, baseline also runs with --note_strategy chunked: it extracts from
overlapping text chunks and then runs a synthesis pass that produces one final
answer per field from all chunk-level extractions. Chunking is included only
for baseline unless the note strategy plan is overridden. Baseline/full is the
only note-strategy variant that sweeps --fields_per_call; baseline/chunked uses
--comparison_fields_per_call by default. Non-json/full also sweeps
--fields_per_call. Final result files include the dataset, model slug, and
completion timestamp; an unstamped checkpoint file is also maintained for
--resume.

Examples:
    python prompt_eval/evaluate_gptoss_extraction.py --dataset mimic --limit 5 --fields_per_call 1,5,25
    python prompt_eval/evaluate_gptoss_extraction.py --dataset osu --modes baseline,no_reasoning,non_json --limit 3
    python prompt_eval/evaluate_gptoss_extraction.py --models 20b,120b
    python prompt_eval/evaluate_gptoss_extraction.py --models 20b --load_in_4bit
    python prompt_eval/evaluate_gptoss_extraction.py --mode_fields_per_call baseline:25,non_json:5
    python prompt_eval/evaluate_gptoss_extraction.py --skip_runs baseline:full:100
    python prompt_eval/evaluate_gptoss_extraction.py --note_strategy full  # disable baseline chunking
    python prompt_eval/evaluate_gptoss_extraction.py --all_fields  # evaluate every form question
    python prompt_eval/evaluate_gptoss_extraction.py --save_raw_runs  # opt-in raw JSONL debugging
"""

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

if load_dotenv:
    load_dotenv()


DEFAULT_GPT_OSS_MODELS = {
    "20b": "unsloth/gpt-oss-20b-BF16",
    "120b": "unsloth/gpt-oss-120b-BF16",
}

# Per-model 4-bit defaults: 20B loads at full precision (BF16), 120B loads in 4-bit.
_MODEL_4BIT_DEFAULTS = {
    "unsloth/gpt-oss-20b-BF16": False,
    "unsloth/gpt-oss-120b-BF16": True,
}

DEFAULT_MODES = [
    "baseline",
    "non_json",
    "no_reasoning",
    "few_shot",
    "no_reasoning_trace_field",
    "no_evidence_field",
]

JSON_MODE_FIELDS = {
    "baseline": ["evidence_from_text", "reasoning_trace"],
    "few_shot": ["evidence_from_text", "reasoning_trace"],
    "no_reasoning": ["evidence_from_text"],
    "no_reasoning_trace_field": ["evidence_from_text"],
    "no_evidence_field": ["reasoning_trace"],
}

EMPTY_ANSWER_STRINGS = {"", "N/A", "NA", "NONE", "NULL", "NOT FOUND", "UNKNOWN"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_records(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_records(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json_list(path):
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def variant_stem(dataset, model_name, mode, fields_per_call, note_strategy="full"):
    return (
        f"{slugify(dataset)}_"
        f"{slugify(model_name)}_"
        f"{slugify(mode)}_"
        f"{slugify(note_strategy)}_"
        f"fpc{fields_per_call}"
    )


def variant_paths(output_dir, dataset, model_name, mode, fields_per_call, note_strategy="full"):
    stem = variant_stem(dataset, model_name, mode, fields_per_call, note_strategy)
    return {
        "runs": output_dir / f"{stem}_runs.jsonl",
    }


def model_slug(models):
    slugs = [slugify(model) for model in models]
    return slugs[0] if len(slugs) == 1 else "multi-model"


def concentrated_paths(output_dir, dataset, models=None, completed_at=None):
    parts = [slugify(dataset)]
    if models:
        parts.append(model_slug(models))
    if completed_at:
        parts.append(completed_at)
    stem = "_".join(parts)
    return {
        "predictions": output_dir / f"{stem}_predictions.json",
        "summary": output_dir / f"{stem}_aggregate_summary.csv",
    }


# ---------------------------------------------------------------------------
# Field / answer helpers
# ---------------------------------------------------------------------------

def normalize_question_number(value):
    normalized = str(value).strip()
    match = re.fullmatch(r"(?:Question|Field)\s+(.+)", normalized, flags=re.IGNORECASE)
    return match.group(1).strip() if match else normalized


def missing_field_identifier(value):
    if value is None:
        return True
    normalized = str(value).strip()
    return not normalized or normalized.upper() in EMPTY_ANSWER_STRINGS


def build_field_name_lookup(field_ids, question_metadata):
    by_name = defaultdict(list)
    for field_id in field_ids:
        name = str(question_metadata[field_id].get("field_name", "") or "").strip().lower()
        if name:
            by_name[name].append(field_id)
    return {name: ids[0] for name, ids in by_name.items() if len(ids) == 1}


def resolve_record_field_id(record, field_ids, field_name_lookup):
    allowed_field_ids = set(field_ids)

    for key in ("field_id", "question_id", "question_number", "question"):
        if key not in record or missing_field_identifier(record.get(key)):
            continue
        candidate = normalize_question_number(record.get(key))
        if candidate in allowed_field_ids:
            return candidate

    for key in ("field_name", "name", "question"):
        if key not in record or missing_field_identifier(record.get(key)):
            continue
        value = str(record.get(key)).strip()
        question_candidate = normalize_question_number(value)
        if question_candidate in allowed_field_ids:
            return question_candidate
        name_candidate = field_name_lookup.get(value.lower())
        if name_candidate:
            return name_candidate

    return ""


def normalize_answer(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(sorted(normalize_answer(item) for item in value if normalize_answer(item)))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(
        r"\s*(?:-\s*)?go to (?:question|field)\s+[a-z]*\d+[a-z]*\s*\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def is_empty_answer(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in EMPTY_ANSWER_STRINGS
    if isinstance(value, list):
        return not value or all(is_empty_answer(item) for item in value)
    if isinstance(value, dict):
        return not value
    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def question_number_column(rows):
    if not rows:
        raise ValueError("Question CSV is empty.")
    for column in rows[0]:
        if column.lstrip("﻿") == "Question Number":
            return column
    raise KeyError("Could not find Question Number column.")


def load_question_metadata(path):
    rows = read_csv_records(path)
    q_col = question_number_column(rows)
    metadata = {}
    for row in rows:
        field_id = normalize_question_number(row[q_col])
        metadata[field_id] = {
            "field_id": field_id,
            "field_category": str(row.get("category", "") or "").strip(),
            "field_name": str(row.get("Question Text", "") or "").strip(),
            "field_instructions": str(row.get("Options / Instructions", "") or "").strip(),
        }
    return metadata


def dedupe_by_key(rows, key):
    seen = set()
    deduped = []
    for row in rows:
        value = str(row.get(key, "") or "").strip()
        if value in seen:
            continue
        seen.add(value)
        deduped.append(row)
    return deduped


def load_notes(dataset, mimic_path, osu_path, limit):
    if dataset == "mimic":
        rows = dedupe_by_key(read_csv_records(mimic_path), "hadm_id")
        notes = []
        for row in rows:
            note = str(row.get("progression_prompt", "") or "").strip()
            assess_plan = str(row.get("assess_plan", "") or "").strip()
            if assess_plan and assess_plan.lower() != "nan":
                note += f"\n\nASSESSMENT AND PLAN:\n{assess_plan}"
            notes.append(
                {
                    "record_id": str(row.get("hadm_id", "") or "").strip(),
                    "record_id_type": "hadm_id",
                    "date": str(row.get("current_time", "") or "").strip(),
                    "note": note,
                }
            )
    elif dataset == "osu":
        rows = read_csv_records(osu_path)
        notes = [
            {
                "record_id": str(row.get("MRN", "") or "").strip(),
                "record_id_type": "MRN",
                "date": str(row.get("admit_date", "") or "").strip(),
                "note": str(row.get("note", "") or "").strip(),
            }
            for row in rows
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return notes[:limit] if limit else notes


def load_gold_answers(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        records = json.load(f)

    gold = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("hadm_id") or record.get("MRN") or record.get("record_id") or "")
        answers = record.get("answers")
        if isinstance(answers, dict):
            for field_id, answer in answers.items():
                gold[(record_id, normalize_question_number(field_id))] = answer
            continue
        field_id = record.get("field_id")
        if field_id:
            gold[(record_id, normalize_question_number(field_id))] = record.get("answers", "N/A")
    return gold


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_csv_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_list(value):
    values = [int(item) for item in parse_csv_list(value)]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if any(item <= 0 for item in values):
        raise ValueError("Integer values must be positive.")
    return values


def parse_mode_int_map(value):
    mode_values = {}
    for item in parse_csv_list(value):
        if ":" not in item:
            raise ValueError(f"Expected mode:value in --mode_fields_per_call, got {item!r}.")
        mode, raw_value = item.split(":", 1)
        mode = mode.strip()
        if mode not in DEFAULT_MODES:
            raise ValueError(f"Unknown mode {mode!r} in --mode_fields_per_call.")
        fields_per_call = int(raw_value.strip())
        if fields_per_call <= 0:
            raise ValueError("--mode_fields_per_call values must be positive.")
        mode_values[mode] = fields_per_call
    return mode_values


def parse_skip_runs(value):
    skip_runs = set()
    for item in parse_csv_list(value):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Expected mode:note_strategy:fields_per_call in --skip_runs, got {item!r}.")
        mode, note_strategy, raw_fields_per_call = parts
        if mode not in DEFAULT_MODES:
            raise ValueError(f"Unknown mode {mode!r} in --skip_runs.")
        if note_strategy not in {"full", "chunked"}:
            raise ValueError(f"Unknown note strategy {note_strategy!r} in --skip_runs.")
        fields_per_call = int(raw_fields_per_call)
        if fields_per_call <= 0:
            raise ValueError("--skip_runs fields_per_call values must be positive.")
        skip_runs.add((mode, note_strategy, fields_per_call))
    return skip_runs


def resolve_models(value):
    models = []
    for item in parse_csv_list(value):
        models.append(DEFAULT_GPT_OSS_MODELS.get(item.lower(), item))
    return models


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------

def build_note_strategy_plan(modes, strategy):
    if strategy == "auto":
        return {
            mode: ["full", "chunked"] if mode == "baseline" else ["full"]
            for mode in modes
        }
    if strategy == "chunked":
        return {
            mode: ["chunked"] if mode == "baseline" else ["full"]
            for mode in modes
        }
    return {mode: [strategy] for mode in modes}


def build_variant_fields_per_call_plan(modes, note_strategy_plan, baseline_values, comparison_value, mode_overrides):
    plan = {}
    for mode in modes:
        for note_strategy in note_strategy_plan[mode]:
            if mode in mode_overrides:
                plan[(mode, note_strategy)] = [mode_overrides[mode]]
            elif note_strategy == "full" and mode in {"baseline", "non_json"}:
                plan[(mode, note_strategy)] = baseline_values
            else:
                plan[(mode, note_strategy)] = [comparison_value]
    return plan


def apply_skip_runs(variant_plan, skip_runs):
    if not skip_runs:
        return variant_plan
    return {
        key: [
            fields_per_call
            for fields_per_call in fields_per_call_values
            if (key[0], key[1], fields_per_call) not in skip_runs
        ]
        for key, fields_per_call_values in variant_plan.items()
    }


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunked(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def chunk_text(text, chunk_chars, overlap_chars):
    text = text or ""
    if chunk_chars <= 0:
        raise ValueError("--chunk_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("--chunk_overlap_chars must be non-negative.")
    if overlap_chars >= chunk_chars:
        raise ValueError("--chunk_overlap_chars must be smaller than --chunk_chars.")
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            sentence_break = text.rfind(". ", start, end)
            whitespace_break = text.rfind(" ", start, end)
            split_at = max(paragraph_break, sentence_break, whitespace_break)
            min_split = start + max(chunk_chars // 2, 1)
            if split_at >= min_split:
                end = split_at + (2 if split_at == sentence_break else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def chunk_note(note, chunk, index, total):
    chunked_note = dict(note)
    chunked_note["note"] = f"[Text chunk {index + 1} of {total}]\n{chunk}"
    return chunked_note


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class NullProgress:
    def update(self, _value=1):
        pass

    def close(self):
        pass


def progress_bar(iterable=None, **kwargs):
    if tqdm is None:
        return iterable if iterable is not None else NullProgress()
    return tqdm(iterable, **kwargs)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_questions_block(field_ids, question_metadata):
    blocks = []
    for field_id in field_ids:
        field = question_metadata[field_id]
        block = [
            f"Field {field_id}",
            f"Category: {field['field_category']}",
            f"Field name: {field['field_name']}",
        ]
        if field["field_instructions"] and field["field_instructions"] != "[]":
            block.append(f"Options/Instructions: {field['field_instructions']}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def few_shot_block():
    return """Few-shot examples:

Example 1 clinical text:
Patient admitted for allogeneic matched sibling peripheral blood stem cell transplant. Donor is male. GVHD prophylaxis tacrolimus and methotrexate.

Example 1 answer:
[
  {
    "field_id": "32",
    "field_category": "donor_information",
    "field_name": "Specify donor",
    "field_instructions": "['Autologous', 'Allogeneic, related', 'Allogeneic, unrelated']",
    "answers": "Allogeneic, related",
    "evidence_from_text": "allogeneic matched sibling",
    "reasoning_trace": "Matched sibling indicates a related allogeneic donor."
  },
  {
    "field_id": "33",
    "field_category": "donor_information",
    "field_name": "Specify product type (check all that apply)",
    "field_instructions": "['Bone marrow - Go to question 35', 'PBSC - Go to question 35']",
    "answers": ["PBSC - Go to question 35"],
    "evidence_from_text": "peripheral blood stem cell transplant",
    "reasoning_trace": "Peripheral blood stem cell transplant maps to PBSC."
  }
]

Example 2 clinical text:
The note does not mention health insurance.

Example 2 answer:
[
  {
    "field_id": "17",
    "field_category": "recipient_information",
    "field_name": "Specify type of health insurance (check all that apply)",
    "field_instructions": "",
    "answers": "N/A",
    "evidence_from_text": "",
    "reasoning_trace": ""
  }
]"""


def mode_settings(mode):
    if mode not in DEFAULT_MODES:
        raise ValueError(f"Unknown mode {mode}. Valid modes: {', '.join(DEFAULT_MODES)}")
    return {
        "json": mode != "non_json",
        "few_shot": mode == "few_shot",
        "extra_fields": JSON_MODE_FIELDS.get(mode, []),
    }


def build_system_prompt():
    return (
        "You are a careful clinical information extraction system for BMT notes. "
        + "Extract only information supported by the supplied note. "
        + "Use standard CIBMTR Form 2400 option wording when an option list is supplied."
    )


def build_user_prompt(note, field_ids, question_metadata, mode):
    settings = mode_settings(mode)
    questions = build_questions_block(field_ids, question_metadata)
    common = f"""Record:
{note['record_id_type']}: {note['record_id']}
date: {note['date']}

Clinical note:
{note['note']}

Fields:

{questions}
"""

    if settings["few_shot"]:
        common = few_shot_block() + "\n\nNow extract from the target note.\n\n" + common

    if not settings["json"]:
        ordered_field_ids = ", ".join(field_ids)
        return (
            common
            + f"""
Return plain text only.

You must output exactly one line for every requested field id below, in this exact order:
{ordered_field_ids}

Use exactly this format for every line:
Field <field_id>: <answer>

Do not stop after the first answer. If an answer is not found, still include the line and use N/A."""
        )

    keys = [
        "record_id",
        "record_id_type",
        "date",
        "field_id",
        "field_category",
        "field_name",
        "field_instructions",
        "answers",
    ] + settings["extra_fields"]

    field_guidance = []
    if "evidence_from_text" in settings["extra_fields"]:
        field_guidance.append(
            '- evidence_from_text must be a concise quote or paraphrase from the note. Use "" for N/A answers.'
        )
    if "reasoning_trace" in settings["extra_fields"]:
        field_guidance.append(
            '- reasoning_trace must be a brief visible rationale. Use "" for N/A answers.'
        )

    return (
        common
        + f"""
Return a JSON array with exactly one object per field. Each object must have these keys:
{json.dumps(keys)}

Rules:
- Use the exact record_id, record_id_type, date, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Put the extracted answer in answers. Use "N/A" if the answer is not found.
- For check-all-that-apply fields, answers may be an array of selected options.
{chr(10).join(field_guidance)}
- Do not add any extra text outside the JSON array."""
    )


def build_synthesis_prompt(note, field_ids, question_metadata, mode, chunk_predictions):
    settings = mode_settings(mode)
    questions = build_questions_block(field_ids, question_metadata)
    compact_predictions = [
        {
            "chunk_index": prediction.get("chunk_index"),
            "field_id": prediction.get("field_id"),
            "field_name": prediction.get("field_name"),
            "answers": prediction.get("answers", "N/A"),
            "evidence_from_text": prediction.get("evidence_from_text", ""),
            "parse_error": prediction.get("parse_error", ""),
        }
        for prediction in chunk_predictions
    ]
    common = f"""Record:
{note['record_id_type']}: {note['record_id']}
date: {note['date']}

Fields:

{questions}

Chunk-level extractions:
{json.dumps(compact_predictions, indent=2, ensure_ascii=False)}
"""

    if not settings["json"]:
        ordered_field_ids = ", ".join(field_ids)
        return (
            common
            + f"""
Synthesize one final answer per field using the chunk-level extractions.
Prefer supported non-empty answers over N/A. Use N/A only when no chunk supports
an answer.

Return plain text only.

You must output exactly one line for every requested field id below, in this exact order:
{ordered_field_ids}

Use exactly this format for every line:
Field <field_id>: <answer>

Do not stop after the first answer. If an answer is not found, still include the line and use N/A."""
        )

    keys = [
        "record_id",
        "record_id_type",
        "date",
        "field_id",
        "field_category",
        "field_name",
        "field_instructions",
        "answers",
    ] + settings["extra_fields"]

    field_guidance = []
    if "evidence_from_text" in settings["extra_fields"]:
        field_guidance.append(
            '- evidence_from_text must summarize the best supporting chunk evidence. Use "" for N/A answers.'
        )
    if "reasoning_trace" in settings["extra_fields"]:
        field_guidance.append(
            '- reasoning_trace must be a brief visible rationale. Use "" for N/A answers.'
        )

    return (
        common
        + f"""
Synthesize one final answer per field using the chunk-level extractions.
Prefer supported non-empty answers over N/A. Use N/A only when no chunk supports
an answer. Resolve conflicts conservatively using the most specific supported
evidence.

Return a JSON array with exactly one object per field. Each object must have these keys:
{json.dumps(keys)}

Rules:
- Use the exact record_id, record_id_type, date, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Put the final synthesized answer in answers.
{chr(10).join(field_guidance)}
- Do not add any extra text outside the JSON array."""
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_json_generation(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        salvaged = salvage_json_records(cleaned)
        if salvaged:
            return salvaged
        raise
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("records", "answers", "extractions"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
    raise ValueError("JSON output was not an array of records.")


def parse_salvage_value(value):
    cleaned = value.strip().rstrip(",").strip()
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
    return json.loads(cleaned)


def salvage_json_records(text):
    records = []
    current = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if stripped == "{":
            current = {}
            continue
        if stripped in {"}", "},"}:
            if current and "field_id" in current:
                records.append(current)
            current = None
            continue
        if current is None:
            continue

        match = re.match(r'^"([^"]+)"\s*:\s*(.+?)\s*$', stripped)
        if not match:
            continue
        key, raw_value = match.groups()
        value_lines = [raw_value]
        try:
            value = parse_salvage_value("\n".join(value_lines))
        except (json.JSONDecodeError, ValueError):
            if not raw_value.lstrip().startswith(("[", "{")):
                continue
            while index < len(lines):
                value_lines.append(lines[index].strip())
                index += 1
                try:
                    value = parse_salvage_value("\n".join(value_lines))
                    break
                except (json.JSONDecodeError, ValueError):
                    continue
            else:
                continue

        try:
            if key in {"evidence_from_text", "reasoning_trace"} and value in {'"', "'"}:
                value = ""
            current[key] = value
        except (TypeError, ValueError):
            continue

    if current and "field_id" in current:
        records.append(current)
    return records


def parse_non_json_generation(text, field_ids):
    by_field = {}
    for line in text.splitlines():
        match = re.match(r"\s*(?:(?:Question|Field)\s*)?([A-Za-z]*\d+[A-Za-z]*)\s*[:=-]\s*(.+?)\s*$", line)
        if match:
            by_field[normalize_question_number(match.group(1))] = match.group(2)
    return [{"field_id": field_id, "answers": by_field.get(field_id, "N/A")} for field_id in field_ids]


def coerce_records(raw_records, note, field_ids, question_metadata, mode):
    settings = mode_settings(mode)
    by_field = {}
    field_name_lookup = build_field_name_lookup(field_ids, question_metadata)
    for index, record in enumerate(raw_records):
        if isinstance(record, dict):
            field_id = resolve_record_field_id(record, field_ids, field_name_lookup)
            if not field_id and index < len(field_ids):
                field_id = field_ids[index]
            if field_id:
                by_field[field_id] = record

    coerced = []
    for field_id in field_ids:
        metadata = question_metadata[field_id]
        raw = by_field.get(field_id, {})
        record = {
            "record_id": note["record_id"],
            "record_id_type": note["record_id_type"],
            "date": note["date"],
            "field_id": field_id,
            "field_category": raw.get("field_category") or metadata["field_category"],
            "field_name": raw.get("field_name") or metadata["field_name"],
            "field_instructions": raw.get("field_instructions") or metadata["field_instructions"],
            "answers": raw.get("answers", "N/A"),
        }
        if "evidence_from_text" in settings["extra_fields"]:
            record["evidence_from_text"] = raw.get("evidence_from_text", "")
        if "reasoning_trace" in settings["extra_fields"]:
            record["reasoning_trace"] = raw.get("reasoning_trace", "")
        coerced.append(record)
    return coerced


# ---------------------------------------------------------------------------
# HuggingFace model loading
# ---------------------------------------------------------------------------

def model_load_in_4bit(model_name, cli_override):
    """Return True if this model should be loaded in 4-bit quantization."""
    if cli_override is not None:
        return cli_override
    return _MODEL_4BIT_DEFAULTS.get(model_name, False)


def import_transformers_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install Transformers, Accelerate, and PyTorch before running inference."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, load_in_4bit, max_seq_length, attn_implementation):
    torch, AutoModelForCausalLM, AutoTokenizer = import_transformers_runtime()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_seq_length:
        tokenizer.model_max_length = max_seq_length

    print(
        f"Loading {model_name} "
        f"({'4-bit' if load_in_4bit else 'full precision BF16'}), "
        f"attn_implementation={attn_implementation or 'default'}"
    )

    # device_map="auto" shards across all visible GPUs (and CPU/disk if needed),
    # while a memory-efficient attention kernel (SDPA) avoids materializing the
    # full attention-score matrix that the eager path would.
    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit loading requires bitsandbytes support in Transformers.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    print_model_placement(model, torch)
    return torch, model, tokenizer


def print_model_placement(model, torch):
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        counts = defaultdict(int)
        for device in device_map.values():
            counts[str(device)] += 1
        print(f"Model device map: {dict(sorted(counts.items()))}")
        offloaded = {
            name: device
            for name, device in device_map.items()
            if str(device).lower() in {"cpu", "disk"} or str(device).startswith("cpu")
        }
        if offloaded:
            sample = dict(list(offloaded.items())[:8])
            print(f"Warning: model has CPU/disk offloaded modules; sample: {sample}")
        else:
            print("Model device map contains no CPU/disk offloaded modules.")
    else:
        device = getattr(model, "device", None)
        print(f"Model device: {device}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            allocated_gb = torch.cuda.memory_allocated(index) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(index) / (1024 ** 3)
            print(
                f"CUDA {index}: {torch.cuda.get_device_name(index)} "
                f"allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB"
            )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def resolve_model_device(model, torch):
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            mapped_str = str(mapped_device)
            if mapped_str.isdigit():
                return torch.device(f"cuda:{mapped_str}")
            if mapped_str not in {"cpu", "disk"} and not mapped_str.startswith("cpu"):
                return torch.device(mapped_str)
    return getattr(model, "device", torch.device("cpu"))


def split_harmony_response(response):
    """Split a decoded GPT-OSS harmony response into (final_answer, reasoning).

    GPT-OSS emits an analysis channel (chain-of-thought) followed by a final
    channel that holds the user-facing answer. Only the final channel should be
    parsed as the model's answer.
    """
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        response,
        flags=re.DOTALL,
    )
    analysis_match = re.search(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|<\|channel\|>)",
        response,
        flags=re.DOTALL,
    )
    reasoning = analysis_match.group(1).strip() if analysis_match else ""
    if final_match:
        return final_match.group(1).strip(), reasoning

    # Fallback for text decoded with skip_special_tokens=True, where channel
    # markers collapse into bare "analysis"/"assistantfinal" separators.
    if "assistantfinal" in response:
        head, answer = response.rsplit("assistantfinal", 1)
        if not reasoning and "analysis" in head:
            reasoning = head.split("analysis", 1)[1].strip()
        return answer.strip(), reasoning
    if "<|channel|>final" in response:
        return response.split("<|channel|>final", 1)[1].strip(), reasoning

    # No channel markers at all: strip any remaining special tokens.
    return re.sub(r"<\|[^>]*\|>", "", response).strip(), reasoning


def generate(model, tokenizer, torch, messages, max_new_tokens, temperature, top_p, top_k, reasoning_effort):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    ).to(resolve_model_device(model, torch))

    input_len = inputs["input_ids"].shape[-1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        # top_k=0 disables top-k filtering; only pass it when enabled.
        if top_k and top_k > 0:
            generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
    return split_harmony_response(response)


def generate_predictions(
    model,
    tokenizer,
    torch,
    note,
    field_batch,
    question_metadata,
    mode,
    prompt,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    reasoning_effort,
    parse_retries,
):
    settings = mode_settings(mode)
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    attempts = []
    max_attempts = max(parse_retries, 0) + 1
    generated = ""
    reasoning = ""
    parse_error = ""

    for attempt in range(1, max_attempts + 1):
        generated, reasoning = generate(
            model,
            tokenizer,
            torch,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            reasoning_effort=reasoning_effort,
        )

        try:
            if settings["json"]:
                raw_records = parse_json_generation(generated)
            else:
                raw_records = parse_non_json_generation(generated, field_batch)
            predictions = coerce_records(raw_records, note, field_batch, question_metadata, mode)
            attempts.append(
                {"attempt": attempt, "reasoning": reasoning, "raw_generation": generated, "parse_error": ""}
            )
            return generated, reasoning, predictions, "", attempts
        except Exception as exc:
            parse_error = str(exc)
            attempts.append(
                {"attempt": attempt, "reasoning": reasoning, "raw_generation": generated, "parse_error": parse_error}
            )

    predictions = coerce_records(
        [{"field_id": field_id, "answers": "N/A"} for field_id in field_batch],
        note,
        field_batch,
        question_metadata,
        mode,
    )
    for prediction in predictions:
        prediction["parse_error"] = parse_error

    return generated, reasoning, predictions, parse_error, attempts


def failed_generation_attempts(attempts):
    return [attempt for attempt in attempts if attempt.get("parse_error")]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(predictions, gold):
    if not gold:
        return {
            "gold_available": False,
            "n": len(predictions),
            "gold_covered": "",
            "exact": "",
            "empty_nonempty_accuracy": "",
        }

    exact = 0
    empty_match = 0
    covered = 0
    for pred in predictions:
        key = (str(pred["record_id"]), str(pred["field_id"]))
        if key not in gold:
            continue
        covered += 1
        gold_answer = gold[key]
        pred_answer = pred.get("answers", "N/A")
        if (
            is_empty_answer(pred_answer) and is_empty_answer(gold_answer)
        ) or normalize_answer(pred_answer) == normalize_answer(gold_answer):
            exact += 1
        if is_empty_answer(pred_answer) == is_empty_answer(gold_answer):
            empty_match += 1

    return {
        "gold_available": True,
        "n": len(predictions),
        "gold_covered": covered,
        "exact": exact / covered if covered else "",
        "empty_nonempty_accuracy": empty_match / covered if covered else "",
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def existing_completed(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for pred in record.get("predictions", []):
                done.add(
                    (
                        record["dataset"],
                        record["mode"],
                        record["model"],
                        int(record["fields_per_call"]),
                        record.get("note_strategy", "full"),
                        str(pred.get("record_id")),
                        str(pred.get("field_id")),
                    )
                )
    return done


def existing_completed_predictions(predictions):
    done = set()
    for pred in predictions:
        try:
            done.add(
                (
                    pred["dataset"],
                    pred["mode"],
                    pred["model"],
                    int(pred["fields_per_call"]),
                    pred.get("note_strategy", "full"),
                    str(pred.get("record_id")),
                    str(pred.get("field_id")),
                )
            )
        except KeyError:
            continue
    return done


def prediction_matches_variant(prediction, dataset, mode, model_name, fields_per_call, note_strategy):
    return (
        prediction.get("dataset") == dataset
        and prediction.get("mode") == mode
        and prediction.get("model") == model_name
        and int(prediction.get("fields_per_call", -1)) == fields_per_call
        and prediction.get("note_strategy", "full") == note_strategy
    )


def add_variant_metadata(predictions, dataset, mode, model_name, fields_per_call, note_strategy):
    for prediction in predictions:
        prediction.update(
            {
                "dataset": dataset,
                "mode": mode,
                "model": model_name,
                "fields_per_call": fields_per_call,
                "note_strategy": note_strategy,
            }
        )
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-OSS extraction prompt variants with HuggingFace Transformers."
    )
    parser.add_argument("--dataset", choices=["mimic", "osu"], default="mimic")
    parser.add_argument("--mimic_path", default="data/mimic_iv_bmt_initial_progression.csv")
    parser.add_argument("--osu_path", default="data/osu_h_p_notes.csv")
    parser.add_argument("--questions", default="data/CIBMTR_Form_2400_R11_Items_simplified.csv")
    parser.add_argument(
        "--gold_json",
        default=None,
        help=(
            "Optional gold answers JSON. Defaults to data/mimic_form_2400_answers.json "
            "for mimic and data/osu_form_2400_all_extractions.json for osu."
        ),
    )
    parser.add_argument("--output_dir", default="prompt_eval/results")
    parser.add_argument(
        "--models",
        default="20b",
        help=(
            "Comma-separated model aliases or HF names. "
            "Aliases: 20b=openai/gpt-oss-20b (full precision BF16), "
            "120b=openai/gpt-oss-120b (4-bit)."
        ),
    )
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated prompt modes.")
    parser.add_argument(
        "--fields_per_call",
        default="1,5,25",
        help="Comma-separated question batch sizes for baseline and non_json.",
    )
    parser.add_argument(
        "--comparison_fields_per_call",
        type=int,
        default=None,
        help="Fields per call for modes other than baseline and non_json. Defaults to max(--fields_per_call).",
    )
    parser.add_argument(
        "--mode_fields_per_call",
        default="",
        help="Comma-separated per-mode overrides, such as baseline:25,non_json:5.",
    )
    parser.add_argument(
        "--skip_runs",
        default="",
        help=(
            "Comma-separated variants to skip as mode:note_strategy:fields_per_call, "
            "such as baseline:full:100."
        ),
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of notes to evaluate.")
    parser.add_argument(
        "--field_limit",
        type=int,
        default=None,
        help="Limit number of form fields. Defaults to the largest planned fields-per-call value.",
    )
    parser.add_argument("--all_fields", action="store_true", help="Evaluate every form field unless --field_limit is set.")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help=(
            "Tokenizer model_max_length (context budget hint). Default 8192; "
            "GPT-OSS supports up to 131072."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help=(
            "Attention kernel for Transformers. sdpa (default) is memory-efficient; "
            "flash_attention_2 needs flash-attn installed; auto lets Transformers choose."
        ),
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="GPT-OSS harmony reasoning effort passed to the chat template.",
    )
    parser.add_argument(
        "--parse_retries",
        type=int,
        default=1,
        help="Number of extra generations to try when parsing fails.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=None,
        help=(
            "Force 4-bit quantization for all models. "
            "Default: 20b=False (full precision BF16), 120b=True (4-bit)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability. GPT-OSS recommended default 1.0.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k filtering; 0 disables it (GPT-OSS recommended). Try 100 for possibly better results.",
    )
    parser.add_argument(
        "--note_strategy",
        choices=["auto", "full", "chunked"],
        default="auto",
        help=(
            "auto runs baseline with full and chunked notes and other modes with full notes; "
            "full disables chunking; chunked runs baseline chunked and other modes full."
        ),
    )
    parser.add_argument("--chunk_chars", type=int, default=12000, help="Approximate characters per note chunk.")
    parser.add_argument("--chunk_overlap_chars", type=int, default=500, help="Overlapping characters between chunks.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save_raw_runs",
        action="store_true",
        help="Also save per-variant JSONL records with raw generations for debugging.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Build configs/prompts without loading GPT-OSS.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset

    baseline_fields_per_call_values = parse_int_csv_list(args.fields_per_call)
    if args.comparison_fields_per_call is not None and args.comparison_fields_per_call <= 0:
        raise ValueError("--comparison_fields_per_call must be positive.")
    if args.parse_retries < 0:
        raise ValueError("--parse_retries must be non-negative.")
    comparison_fields_per_call = args.comparison_fields_per_call or max(baseline_fields_per_call_values)
    models = resolve_models(args.models)
    checkpoint_paths = concentrated_paths(output_dir, dataset, models)
    modes = parse_csv_list(args.modes)
    mode_fields_per_call = parse_mode_int_map(args.mode_fields_per_call)
    skip_runs = parse_skip_runs(args.skip_runs)
    note_strategy_plan = build_note_strategy_plan(modes, args.note_strategy)
    variant_fields_per_call_plan = build_variant_fields_per_call_plan(
        modes,
        note_strategy_plan,
        baseline_fields_per_call_values,
        comparison_fields_per_call,
        mode_fields_per_call,
    )
    variant_fields_per_call_plan = apply_skip_runs(variant_fields_per_call_plan, skip_runs)
    planned_fields_per_call_values = [
        fields_per_call
        for mode_values in variant_fields_per_call_plan.values()
        for fields_per_call in mode_values
    ]
    if not planned_fields_per_call_values:
        raise ValueError("No runs remain after applying --skip_runs.")
    question_metadata = load_question_metadata(args.questions)
    total_fields = len(question_metadata)
    field_ids = list(question_metadata)
    if args.field_limit is not None:
        if args.field_limit <= 0:
            raise ValueError("--field_limit must be positive.")
        selected_field_count = args.field_limit
        field_selection = f"first {selected_field_count} fields from --field_limit"
    elif args.all_fields:
        selected_field_count = total_fields
        field_selection = "all fields"
    else:
        selected_field_count = max(planned_fields_per_call_values)
        field_selection = f"first {selected_field_count} fields from largest planned fields-per-call value"
    field_ids = field_ids[:selected_field_count]
    notes = load_notes(dataset, args.mimic_path, args.osu_path, args.limit)

    if args.gold_json is None:
        default_gold_paths = {
            "mimic": "data/mimic_form_2400_answers.json",
            "osu": "data/osu_form_2400_all_extractions.json",
        }
        gold_path = default_gold_paths.get(dataset, "")
    else:
        gold_path = args.gold_json
    gold = load_gold_answers(gold_path)

    # Determine per-model 4-bit settings
    cli_4bit_override = args.load_in_4bit if args.load_in_4bit else None
    model_4bit_map = {m: model_load_in_4bit(m, cli_4bit_override) for m in models}

    print(f"Dataset: {dataset}")
    print(f"Notes: {len(notes)}")
    print(f"Fields: {len(field_ids)} of {total_fields} ({field_selection})")
    print(f"Models: {models}")
    print(f"Model 4-bit settings: {model_4bit_map}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Attention implementation: {args.attn_implementation}")
    print(f"Modes: {modes}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Temperature: {args.temperature}  top_p: {args.top_p}  top_k: {args.top_k}")
    print(f"Note strategy plan: {note_strategy_plan}")
    print(f"Variant fields per call plan: {variant_fields_per_call_plan}")
    if skip_runs:
        print(f"Skipped runs: {sorted(skip_runs)}")
    if any("chunked" in strategies for strategies in note_strategy_plan.values()):
        print(f"Chunk chars: {args.chunk_chars}")
        print(f"Chunk overlap chars: {args.chunk_overlap_chars}")
    print(f"Gold answers: {len(gold)}")

    if args.dry_run:
        sample_mode = modes[0]
        sample_prompt = build_user_prompt(notes[0], field_ids[: min(3, len(field_ids))], question_metadata, sample_mode)
        print("\nDry run sample prompt:")
        print(sample_prompt[:3000])
        return

    summary_rows = []
    all_predictions = load_json_list(checkpoint_paths["predictions"]) if args.resume else []

    for model_name in models:
        try:
            torch, model, tokenizer = load_model(
                model_name,
                load_in_4bit=model_4bit_map[model_name],
                max_seq_length=args.max_seq_length,
                attn_implementation=None if args.attn_implementation == "auto" else args.attn_implementation,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            print("Install guidance: pip install -U transformers accelerate bitsandbytes torch")
            return

        for mode in modes:
            for note_strategy in note_strategy_plan[mode]:
                for fields_per_call in variant_fields_per_call_plan[(mode, note_strategy)]:
                    variant_run_paths = variant_paths(
                        output_dir,
                        dataset,
                        model_name,
                        mode,
                        fields_per_call,
                        note_strategy,
                    )
                    if args.save_raw_runs and not args.resume and variant_run_paths["runs"].exists():
                        variant_run_paths["runs"].unlink()
                    if args.resume:
                        completed = existing_completed_predictions(all_predictions)
                        if args.save_raw_runs:
                            completed.update(existing_completed(variant_run_paths["runs"]))
                        variant_predictions = [
                            prediction
                            for prediction in all_predictions
                            if prediction_matches_variant(
                                prediction,
                                dataset,
                                mode,
                                model_name,
                                fields_per_call,
                                note_strategy,
                            )
                        ]
                    else:
                        completed = set()
                        variant_predictions = []
                    parse_errors = 0
                    call_count = 0
                    started = time.time()
                    if note_strategy == "chunked":
                        note_chunks = {
                            note["record_id"]: chunk_text(
                                note["note"],
                                args.chunk_chars,
                                args.chunk_overlap_chars,
                            )
                            for note in notes
                        }
                        total_batches = sum(
                            len(note_chunks[note["record_id"]]) + 1
                            for note in notes
                            for _batch in chunked(field_ids, fields_per_call)
                        )
                    else:
                        note_chunks = {}
                        total_batches = sum(1 for _note in notes for _batch in chunked(field_ids, fields_per_call))
                    progress = progress_bar(
                        total=total_batches,
                        desc=f"{mode} {note_strategy} fpc={fields_per_call}",
                        unit="call",
                        leave=False,
                    )

                    try:
                        for note in notes:
                            for field_batch in chunked(field_ids, fields_per_call):
                                batch_done = [
                                    (
                                        dataset,
                                        mode,
                                        model_name,
                                        fields_per_call,
                                        note_strategy,
                                        note["record_id"],
                                        field_id,
                                    )
                                    in completed
                                    for field_id in field_batch
                                ]
                                if args.resume and all(batch_done):
                                    if note_strategy == "chunked":
                                        progress.update(len(note_chunks[note["record_id"]]) + 1)
                                    else:
                                        progress.update(1)
                                    continue

                                if note_strategy == "chunked":
                                    chunk_predictions = []
                                    chunk_generations = []
                                    chunks = note_chunks[note["record_id"]]
                                    for chunk_index, chunk in enumerate(chunks):
                                        chunked_record = chunk_note(note, chunk, chunk_index, len(chunks))
                                        generated, reasoning, predictions, parse_error, attempts = generate_predictions(
                                            model,
                                            tokenizer,
                                            torch,
                                            chunked_record,
                                            field_batch,
                                            question_metadata,
                                            mode,
                                            build_user_prompt(chunked_record, field_batch, question_metadata, mode),
                                            max_new_tokens=args.max_new_tokens,
                                            temperature=args.temperature,
                                            top_p=args.top_p,
                                            top_k=args.top_k,
                                            reasoning_effort=args.reasoning_effort,
                                            parse_retries=args.parse_retries,
                                        )
                                        call_count += len(attempts)
                                        if parse_error:
                                            parse_errors += 1
                                        for prediction in predictions:
                                            prediction["chunk_index"] = chunk_index + 1
                                            prediction["chunk_count"] = len(chunks)
                                        chunk_predictions.extend(predictions)
                                        chunk_generations.append(
                                            {
                                                "chunk_index": chunk_index + 1,
                                                "reasoning": reasoning,
                                                "raw_generation": generated,
                                                "parse_error": parse_error,
                                                "generation_attempts": failed_generation_attempts(attempts),
                                            }
                                        )
                                        progress.update(1)

                                    synthesis_prompt = build_synthesis_prompt(
                                        note,
                                        field_batch,
                                        question_metadata,
                                        mode,
                                        chunk_predictions,
                                    )
                                    generated, reasoning, predictions, parse_error, attempts = generate_predictions(
                                        model,
                                        tokenizer,
                                        torch,
                                        note,
                                        field_batch,
                                        question_metadata,
                                        mode,
                                        synthesis_prompt,
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature,
                                        top_p=args.top_p,
                                        top_k=args.top_k,
                                        reasoning_effort=args.reasoning_effort,
                                        parse_retries=args.parse_retries,
                                    )
                                    call_count += len(attempts)
                                    if parse_error:
                                        parse_errors += 1
                                    for prediction in predictions:
                                        prediction["chunk_count"] = len(chunks)
                                        prediction["synthesized_from_chunks"] = True
                                    raw_run_generation = {
                                        "chunk_generations": chunk_generations,
                                        "synthesis_generation": generated,
                                        "synthesis_reasoning": reasoning,
                                        "synthesis_parse_error": parse_error,
                                        "synthesis_generation_attempts": failed_generation_attempts(attempts),
                                    }
                                    raw_run_attempts = {
                                        "chunk_attempts": [
                                            {
                                                "chunk_index": generation["chunk_index"],
                                                "generation_attempts": generation["generation_attempts"],
                                            }
                                            for generation in chunk_generations
                                            if generation["generation_attempts"]
                                        ],
                                        "synthesis_generation_attempts": failed_generation_attempts(attempts),
                                    }
                                    raw_run_reasoning = {
                                        "chunk_reasonings": [
                                            {
                                                "chunk_index": generation["chunk_index"],
                                                "reasoning": generation["reasoning"],
                                            }
                                            for generation in chunk_generations
                                        ],
                                        "synthesis_reasoning": reasoning,
                                    }
                                    progress.update(1)
                                else:
                                    generated, reasoning, predictions, parse_error, attempts = generate_predictions(
                                        model,
                                        tokenizer,
                                        torch,
                                        note,
                                        field_batch,
                                        question_metadata,
                                        mode,
                                        build_user_prompt(note, field_batch, question_metadata, mode),
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature,
                                        top_p=args.top_p,
                                        top_k=args.top_k,
                                        reasoning_effort=args.reasoning_effort,
                                        parse_retries=args.parse_retries,
                                    )
                                    call_count += len(attempts)
                                    if parse_error:
                                        parse_errors += 1
                                    raw_run_generation = generated
                                    raw_run_reasoning = reasoning
                                    raw_run_attempts = failed_generation_attempts(attempts)
                                    progress.update(1)

                                predictions = add_variant_metadata(
                                    predictions,
                                    dataset,
                                    mode,
                                    model_name,
                                    fields_per_call,
                                    note_strategy,
                                )
                                variant_predictions.extend(predictions)
                                all_predictions.extend(predictions)
                                if args.save_raw_runs:
                                    append_jsonl(
                                        variant_run_paths["runs"],
                                        {
                                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                            "dataset": dataset,
                                            "mode": mode,
                                            "model": model_name,
                                            "fields_per_call": fields_per_call,
                                            "note_strategy": note_strategy,
                                            "record_id": note["record_id"],
                                            "field_ids": field_batch,
                                            "reasoning": raw_run_reasoning,
                                            "raw_generation": raw_run_generation,
                                            "generation_attempts": raw_run_attempts,
                                            "predictions": predictions,
                                        },
                                    )
                                for prediction in predictions:
                                    completed.add(
                                        (
                                            dataset,
                                            mode,
                                            model_name,
                                            fields_per_call,
                                            note_strategy,
                                            prediction["record_id"],
                                            prediction["field_id"],
                                        )
                                    )
                                if args.sleep:
                                    time.sleep(args.sleep)
                    finally:
                        if hasattr(progress, "close"):
                            progress.close()

                    metrics = evaluate_predictions(variant_predictions, gold)
                    metrics.update(
                        {
                            "dataset": dataset,
                            "mode": mode,
                            "model": model_name,
                            "fields_per_call": fields_per_call,
                            "note_strategy": note_strategy,
                            "chunk_chars": args.chunk_chars if note_strategy == "chunked" else "",
                            "chunk_overlap_chars": args.chunk_overlap_chars if note_strategy == "chunked" else "",
                            "calls": call_count,
                            "parse_errors": parse_errors,
                            "parse_error_rate": parse_errors / call_count if call_count else 0,
                            "seconds": round(time.time() - started, 2),
                        }
                    )
                    summary_rows.append(metrics)
                    write_json(checkpoint_paths["predictions"], all_predictions)
                    print(json.dumps(metrics, indent=2))

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_fields_list = summary_fields()
    completed_at = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_paths = concentrated_paths(output_dir, dataset, models, completed_at)
    write_json(final_paths["predictions"], all_predictions)
    write_csv_records(final_paths["summary"], summary_rows, summary_fields_list)
    write_csv_records(checkpoint_paths["summary"], summary_rows, summary_fields_list)
    print(f"\nSaved concentrated predictions to {final_paths['predictions']}")
    print(f"Saved aggregate summary to {final_paths['summary']}")
    print(f"Saved resume checkpoint to {checkpoint_paths['predictions']}")
    if args.save_raw_runs:
        print(f"Saved per-variant raw runs under {output_dir}")


def summary_fields():
    return [
        "dataset",
        "mode",
        "model",
        "fields_per_call",
        "note_strategy",
        "chunk_chars",
        "chunk_overlap_chars",
        "calls",
        "n",
        "gold_available",
        "gold_covered",
        "exact",
        "empty_nonempty_accuracy",
        "parse_errors",
        "parse_error_rate",
        "seconds",
    ]


if __name__ == "__main__":
    main()
