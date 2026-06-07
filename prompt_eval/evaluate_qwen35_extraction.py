#!/usr/bin/env python3
"""
Evaluate Qwen3.5 prompt variants for CIBMTR Form 2400 extraction.

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

Model placement defaults to --device auto, which lets Transformers shard across
available devices. Use --device cuda:0, --device cuda:1, or the shorthand
--device 0 to force a model onto a single GPU when it fits.

Qwen thinking defaults to --qwen_thinking auto, which enables Qwen's internal
thinking mode for prompt modes that ask for rationale fields. Use
--qwen_thinking off to keep the same extraction modes but avoid long hidden
thinking generations.

Examples:
    python prompt_eval/evaluate_qwen35_extraction.py --dataset mimic --limit 5 --fields_per_call 1,5,25
    python prompt_eval/evaluate_qwen35_extraction.py --dataset osu --modes baseline,no_reasoning,non_json --limit 3
    python prompt_eval/evaluate_qwen35_extraction.py --models 2b,9b,27b,35b-a3b
    python prompt_eval/evaluate_qwen35_extraction.py --models 9b --device cuda:0
    python prompt_eval/evaluate_qwen35_extraction.py --modes baseline --qwen_thinking off
    python prompt_eval/evaluate_qwen35_extraction.py --mode_fields_per_call baseline:25,non_json:5
    python prompt_eval/evaluate_qwen35_extraction.py --sampling_preset instruct_general
    python prompt_eval/evaluate_qwen35_extraction.py --note_strategy full  # disable baseline chunking
    python prompt_eval/evaluate_qwen35_extraction.py --all_fields  # evaluate every form question
    python prompt_eval/evaluate_qwen35_extraction.py --save_raw_runs  # opt-in raw JSONL debugging
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


DEFAULT_QWEN35_MODELS = {
    "2b": "Qwen/Qwen3.5-2B",
    "9b": "Qwen/Qwen3.5-9B",
    "27b": "Qwen/Qwen3.5-27B",
    "35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    "all": "Qwen/Qwen3.5-2B,Qwen/Qwen3.5-9B,Qwen/Qwen3.5-27B,Qwen/Qwen3.5-35B-A3B",
}

SAMPLING_PRESETS = {
    "thinking_general": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "thinking_precise_coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "instruct_general": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "instruct_reasoning": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}

SAMPLING_PRESET_CHOICES = ["auto"] + sorted(SAMPLING_PRESETS)

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


def question_number_column(rows):
    if not rows:
        raise ValueError("Question CSV is empty.")
    for column in rows[0]:
        if column.lstrip("\ufeff") == "Question Number":
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


def resolve_models(value):
    models = []
    for item in parse_csv_list(value):
        resolved = DEFAULT_QWEN35_MODELS.get(item.lower(), item)
        models.extend(parse_csv_list(resolved))
    return models


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


class NullProgress:
    def update(self, _value=1):
        pass

    def close(self):
        pass


def progress_bar(iterable=None, **kwargs):
    if tqdm is None:
        return iterable if iterable is not None else NullProgress()
    return tqdm(iterable, **kwargs)


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
        "thinking": mode in {"baseline", "non_json", "few_shot", "no_reasoning_trace_field", "no_evidence_field"},
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
            '- reasoning_trace must be a brief visible rationale, not hidden chain-of-thought. Use "" for N/A answers.'
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
            '- reasoning_trace must be a brief visible rationale, not hidden chain-of-thought. Use "" for N/A answers.'
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
    records = []
    by_field = {}
    for line in text.splitlines():
        match = re.match(r"\s*(?:(?:Question|Field)\s*)?([A-Za-z]*\d+[A-Za-z]*)\s*[:=-]\s*(.+?)\s*$", line)
        if match:
            by_field[normalize_question_number(match.group(1))] = match.group(2)
    for field_id in field_ids:
        records.append({"field_id": field_id, "answers": by_field.get(field_id, "N/A")})
    return records


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


def import_transformers_runtime():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install Transformers, Accelerate, and PyTorch before running inference."
        ) from exc
    return torch, AutoModelForImageTextToText, AutoProcessor


def normalize_device_arg(device):
    normalized = str(device).strip()
    if not normalized or normalized.lower() == "auto":
        return "auto"
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def load_model(model_name, load_in_4bit, device):
    torch, AutoModelForImageTextToText, AutoProcessor = import_transformers_runtime()
    processor = AutoProcessor.from_pretrained(model_name)

    model_kwargs = {
        "dtype": "auto",
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit loading requires bitsandbytes support in Transformers.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        if device != "auto":
            model_kwargs["device_map"] = {"": device}

    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    if device != "auto" and not load_in_4bit:
        model.to(torch.device(device))
    model._qwen35_input_device = resolve_input_device(model, torch, device)
    model.eval()
    print_model_placement(model, torch)
    return torch, model, processor


def resolve_input_device(model, torch, device):
    if device != "auto":
        return torch.device(device)
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for mapped_device in hf_device_map.values():
            mapped_device = str(mapped_device)
            if mapped_device not in {"cpu", "disk"} and not mapped_device.startswith("cpu"):
                return torch.device(mapped_device)
    return getattr(model, "device", torch.device("cpu"))


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


def processor_text(response):
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("answer", "content", "text", "response"):
            if key in response:
                return processor_text(response[key])
        return json.dumps(response, ensure_ascii=False)
    if isinstance(response, (list, tuple)):
        return "\n".join(processor_text(item) for item in response if processor_text(item))
    return str(response)


def extract_thought(response):
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, flags=re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    match = re.search(r"<\|channel>thought\s*", response)
    if not match:
        return ""
    thought = response[match.end() :]
    end_match = re.search(r"<channel\|>|<\|end\|>", thought)
    if end_match:
        thought = thought[: end_match.start()]
    return thought.strip()


def strip_thought(response):
    return re.sub(r"\s*<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()


def supported_generation_kwargs(model, kwargs):
    generation_config = getattr(model, "generation_config", None)
    supported = {}
    skipped = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key == "presence_penalty":
            skipped[key] = value
            continue
        if generation_config is not None and not hasattr(generation_config, key):
            skipped[key] = value
            continue
        supported[key] = value
    if skipped:
        skipped_keys = tuple(sorted(skipped))
        if getattr(model, "_qwen35_reported_unsupported_generation_kwargs", None) != skipped_keys:
            print(
                "Skipping unsupported generation parameters for this Transformers runtime: "
                + ", ".join(f"{key}={value}" for key, value in skipped.items())
            )
            setattr(model, "_qwen35_reported_unsupported_generation_kwargs", skipped_keys)
    return supported


def resolve_qwen_thinking(args, mode):
    if args.qwen_thinking == "auto":
        return mode_settings(mode)["thinking"]
    return args.qwen_thinking == "on"


def resolve_sampling(args, mode, enable_thinking=None):
    if enable_thinking is None:
        enable_thinking = resolve_qwen_thinking(args, mode)
    if args.sampling_preset == "auto":
        preset_name = "thinking_general" if enable_thinking else "instruct_general"
    else:
        preset_name = args.sampling_preset
    sampling = dict(SAMPLING_PRESETS[preset_name])
    for name in sampling:
        override = getattr(args, name)
        if override is not None:
            sampling[name] = override
    sampling["preset"] = preset_name
    return sampling


def generate(
    model,
    processor,
    torch,
    messages,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    presence_penalty,
    repetition_penalty,
    enable_thinking,
):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    input_device = getattr(model, "_qwen35_input_device", getattr(model, "device", None))
    if input_device is not None:
        inputs = inputs.to(input_device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    generate_kwargs = supported_generation_kwargs(model, generate_kwargs)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generate_kwargs,
        )
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    thought = extract_thought(response)
    return strip_thought(processor.decode(output_ids[0][input_len:], skip_special_tokens=True)), thought


def generate_predictions(
    model,
    processor,
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
    min_p,
    presence_penalty,
    repetition_penalty,
    enable_thinking,
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
    thought = ""
    parse_error = ""

    for attempt in range(1, max_attempts + 1):
        generated, thought = generate(
            model,
            processor,
            torch,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
        )

        try:
            if settings["json"]:
                raw_records = parse_json_generation(generated)
            else:
                raw_records = parse_non_json_generation(generated, field_batch)
            predictions = coerce_records(
                raw_records,
                note,
                field_batch,
                question_metadata,
                mode,
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "thought": thought,
                    "raw_generation": generated,
                    "parse_error": "",
                }
            )
            return generated, thought, predictions, "", attempts
        except Exception as exc:
            parse_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "thought": thought,
                    "raw_generation": generated,
                    "parse_error": parse_error,
                }
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

    return generated, thought, predictions, parse_error, attempts


def failed_generation_attempts(attempts):
    return [attempt for attempt in attempts if attempt.get("parse_error")]


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
            is_empty_answer(pred_answer)
            and is_empty_answer(gold_answer)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 extraction prompt variants with Transformers.")
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
    parser.add_argument("--models", default="2b", help="Comma-separated model aliases or HF names. Use all for every default Qwen3.5 model.")
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
    parser.add_argument("--limit", type=int, default=5, help="Number of notes to evaluate.")
    parser.add_argument(
        "--field_limit",
        type=int,
        default=None,
        help=(
            "Limit number of form fields. Defaults to the largest planned fields-per-call value."
        ),
    )
    parser.add_argument("--all_fields", action="store_true", help="Evaluate every form field unless --field_limit is set.")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument(
        "--parse_retries",
        type=int,
        default=1,
        help="Number of extra generations to try when parsing fails.",
    )
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--device",
        default="auto",
        help='Device for model placement. Use "auto" for Transformers sharding, or e.g. cuda:0, cuda:1, 0, 1, cpu.',
    )
    parser.add_argument(
        "--qwen_thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Controls Qwen chat-template internal thinking. "
            "auto follows the prompt mode, on always enables it, off always disables it."
        ),
    )
    parser.add_argument(
        "--sampling_preset",
        choices=SAMPLING_PRESET_CHOICES,
        default="auto",
        help="Qwen-recommended sampling preset. Explicit sampling flags override preset values.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
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
    parser.add_argument("--dry_run", action="store_true", help="Build configs/prompts without loading Qwen3.5.")
    args = parser.parse_args()
    args.device = normalize_device_arg(args.device)

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
    note_strategy_plan = build_note_strategy_plan(modes, args.note_strategy)
    variant_fields_per_call_plan = build_variant_fields_per_call_plan(
        modes,
        note_strategy_plan,
        baseline_fields_per_call_values,
        comparison_fields_per_call,
        mode_fields_per_call,
    )
    planned_fields_per_call_values = [
        fields_per_call
        for mode_values in variant_fields_per_call_plan.values()
        for fields_per_call in mode_values
    ]
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

    print(f"Dataset: {dataset}")
    print(f"Notes: {len(notes)}")
    print(f"Fields: {len(field_ids)} of {total_fields} ({field_selection})")
    print(f"Models: {models}")
    print(f"Device: {args.device}")
    print(f"Qwen thinking: {args.qwen_thinking}")
    print(f"Modes: {modes}")
    print(
        "Sampling preset: "
        f"{args.sampling_preset}"
        + (
            " with explicit overrides "
            + json.dumps(
                {
                    key: getattr(args, key)
                    for key in SAMPLING_PRESETS["thinking_general"]
                    if getattr(args, key) is not None
                },
                sort_keys=True,
            )
            if any(getattr(args, key) is not None for key in SAMPLING_PRESETS["thinking_general"])
            else ""
        )
    )
    print(f"Note strategy plan: {note_strategy_plan}")
    print(f"Variant fields per call plan: {variant_fields_per_call_plan}")
    print(
        "Resolved sampling by mode: "
        + json.dumps(
            {
                mode: resolve_sampling(args, mode)
                for mode in modes
            },
            sort_keys=True,
        )
    )
    print(
        "Resolved Qwen thinking by mode: "
        + json.dumps(
            {
                mode: resolve_qwen_thinking(args, mode)
                for mode in modes
            },
            sort_keys=True,
        )
    )
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
            torch, model, processor = load_model(
                model_name,
                load_in_4bit=args.load_in_4bit,
                device=args.device,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            print("Install guidance: pip install -U transformers torch accelerate")
            return

        for mode in modes:
            enable_thinking = resolve_qwen_thinking(args, mode)
            sampling = resolve_sampling(args, mode, enable_thinking=enable_thinking)
            print(
                f"Mode {mode}: qwen_thinking={enable_thinking}, sampling preset={sampling['preset']}, "
                f"temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
                f"top_k={sampling['top_k']}, min_p={sampling['min_p']}, "
                f"presence_penalty={sampling['presence_penalty']}, "
                f"repetition_penalty={sampling['repetition_penalty']}"
            )
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
                                        generated, thought, predictions, parse_error, attempts = generate_predictions(
                                            model,
                                            processor,
                                            torch,
                                            chunked_record,
                                            field_batch,
                                            question_metadata,
                                            mode,
                                            build_user_prompt(chunked_record, field_batch, question_metadata, mode),
                                            max_new_tokens=args.max_new_tokens,
                                            temperature=sampling["temperature"],
                                            top_p=sampling["top_p"],
                                            top_k=sampling["top_k"],
                                            min_p=sampling["min_p"],
                                            presence_penalty=sampling["presence_penalty"],
                                            repetition_penalty=sampling["repetition_penalty"],
                                            enable_thinking=enable_thinking,
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
                                                "thought": thought,
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
                                    generated, thought, predictions, parse_error, attempts = generate_predictions(
                                        model,
                                        processor,
                                        torch,
                                        note,
                                        field_batch,
                                        question_metadata,
                                        mode,
                                        synthesis_prompt,
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=sampling["temperature"],
                                        top_p=sampling["top_p"],
                                        top_k=sampling["top_k"],
                                        min_p=sampling["min_p"],
                                        presence_penalty=sampling["presence_penalty"],
                                        repetition_penalty=sampling["repetition_penalty"],
                                        enable_thinking=enable_thinking,
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
                                        "synthesis_parse_error": parse_error,
                                        "synthesis_generation_attempts": failed_generation_attempts(attempts),
                                    }
                                    raw_run_thought = {
                                        "chunk_thoughts": [
                                            {
                                                "chunk_index": generation["chunk_index"],
                                                "thought": generation["thought"],
                                            }
                                            for generation in chunk_generations
                                        ],
                                        "synthesis_thought": thought,
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
                                    progress.update(1)
                                else:
                                    generated, thought, predictions, parse_error, attempts = generate_predictions(
                                        model,
                                        processor,
                                        torch,
                                        note,
                                        field_batch,
                                        question_metadata,
                                        mode,
                                        build_user_prompt(note, field_batch, question_metadata, mode),
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=sampling["temperature"],
                                        top_p=sampling["top_p"],
                                        top_k=sampling["top_k"],
                                        min_p=sampling["min_p"],
                                        presence_penalty=sampling["presence_penalty"],
                                        repetition_penalty=sampling["repetition_penalty"],
                                        enable_thinking=enable_thinking,
                                        parse_retries=args.parse_retries,
                                    )
                                    call_count += len(attempts)
                                    if parse_error:
                                        parse_errors += 1
                                    raw_run_generation = generated
                                    raw_run_thought = thought
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
                                            "thought": raw_run_thought,
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
        del processor
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
