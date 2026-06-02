#!/usr/bin/env python3
"""
Evaluate Gemma 4 prompt variants for CIBMTR Form 2400 extraction.

Examples:
    python prompt_eval/evaluate_gemma4_extraction.py --dataset mimic --limit 5 --fields_per_call 1,5,25
    python prompt_eval/evaluate_gemma4_extraction.py --dataset osu --modes baseline,no_reasoning,non_json --limit 3
    python prompt_eval/evaluate_gemma4_extraction.py --models google/gemma-4-E2B-it,google/gemma-4-E4B-it
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


DEFAULT_GEMMA4_MODELS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
    "26b-a4b": "google/gemma-4-26B-A4B-it",
    "31b": "google/gemma-4-31B-it",
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


def variant_stem(dataset, model_name, mode, fields_per_call):
    return (
        f"{slugify(dataset)}_"
        f"{slugify(model_name)}_"
        f"{slugify(mode)}_"
        f"fpc{fields_per_call}"
    )


def variant_paths(output_dir, dataset, model_name, mode, fields_per_call):
    stem = variant_stem(dataset, model_name, mode, fields_per_call)
    return {
        "runs": output_dir / f"{stem}_runs.jsonl",
        "predictions": output_dir / f"{stem}_predictions.json",
        "summary": output_dir / f"{stem}_summary.csv",
    }


def normalize_question_number(value):
    normalized = str(value).strip()
    match = re.fullmatch(r"Question\s+(.+)", normalized, flags=re.IGNORECASE)
    return match.group(1).strip() if match else normalized


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


def resolve_models(value):
    models = []
    for item in parse_csv_list(value):
        models.append(DEFAULT_GEMMA4_MODELS.get(item.lower(), item))
    return models


def chunked(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_questions_block(field_ids, question_metadata):
    blocks = []
    for field_id in field_ids:
        field = question_metadata[field_id]
        block = [
            f"Question {field_id}",
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
        "thinking": mode in {"baseline", "few_shot", "no_reasoning_trace_field", "no_evidence_field"},
        "few_shot": mode == "few_shot",
        "extra_fields": JSON_MODE_FIELDS.get(mode, []),
    }


def build_system_prompt(mode):
    settings = mode_settings(mode)
    prefix = "<|think|>\n" if settings["thinking"] else ""
    return (
        prefix
        + "You are a careful clinical information extraction system for BMT notes. "
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

Questions:

{questions}
"""

    if settings["few_shot"]:
        common = few_shot_block() + "\n\nNow extract from the target note.\n\n" + common

    if not settings["json"]:
        return (
            common
            + """
Return plain text only. For each question, write one line in this exact format:
Question <field_id>: <answer>
Use N/A when the answer is not found."""
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
Return a JSON array with exactly one object per question. Each object must have these keys:
{json.dumps(keys)}

Rules:
- Use the exact record_id, record_id_type, date, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Put the extracted answer in answers. Use "N/A" if the answer is not found.
- For check-all-that-apply fields, answers may be an array of selected options.
{chr(10).join(field_guidance)}
- Do not add any extra text outside the JSON array."""
    )


def parse_json_generation(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("records", "answers", "extractions"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
    raise ValueError("JSON output was not an array of records.")


def parse_non_json_generation(text, field_ids):
    records = []
    by_field = {}
    for line in text.splitlines():
        match = re.match(r"\s*(?:Question\s*)?([A-Za-z]*\d+[A-Za-z]*)\s*[:=-]\s*(.+?)\s*$", line)
        if match:
            by_field[normalize_question_number(match.group(1))] = match.group(2)
    for field_id in field_ids:
        records.append({"field_id": field_id, "answers": by_field.get(field_id, "N/A")})
    return records


def coerce_records(raw_records, note, field_ids, question_metadata, mode):
    settings = mode_settings(mode)
    by_field = {}
    for record in raw_records:
        if isinstance(record, dict):
            field_id = normalize_question_number(record.get("field_id", ""))
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
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install Transformers, Accelerate, and PyTorch before running inference."
        ) from exc
    return torch, AutoModelForCausalLM, AutoProcessor


def load_model(model_name, max_seq_length, load_in_4bit):
    torch, AutoModelForCausalLM, AutoProcessor = import_transformers_runtime()
    processor = AutoProcessor.from_pretrained(model_name)

    model_kwargs = {
        "device_map": "auto",
        "dtype": "auto",
    }
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit loading requires bitsandbytes support in Transformers.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return torch, model, processor


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


def generate(model, processor, torch, messages, max_new_tokens, temperature, top_p, top_k, enable_thinking):
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = processor(text=text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    if eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generate_kwargs,
        )
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    if hasattr(processor, "parse_response"):
        try:
            return processor_text(processor.parse_response(response)).strip()
        except Exception:
            pass
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


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
        if normalize_answer(pred_answer) == normalize_answer(gold_answer):
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
                        str(pred.get("record_id")),
                        str(pred.get("field_id")),
                    )
                )
    return done


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma 4 extraction prompt variants with Transformers.")
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
    parser.add_argument("--models", default="e2b", help="Comma-separated model aliases or HF names.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated prompt modes.")
    parser.add_argument("--fields_per_call", default="1,5,25", help="Comma-separated question batch sizes.")
    parser.add_argument("--limit", type=int, default=5, help="Number of notes to evaluate.")
    parser.add_argument("--field_limit", type=int, default=None, help="Limit number of form fields for smoke tests.")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build configs/prompts without loading Gemma 4.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset
    summary_path = output_dir / f"{slugify(dataset)}_aggregate_summary.csv"

    question_metadata = load_question_metadata(args.questions)
    field_ids = list(question_metadata)
    if args.field_limit:
        field_ids = field_ids[: args.field_limit]
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
    models = resolve_models(args.models)
    modes = parse_csv_list(args.modes)
    fields_per_call_values = [int(value) for value in parse_csv_list(args.fields_per_call)]

    print(f"Dataset: {dataset}")
    print(f"Notes: {len(notes)}")
    print(f"Fields: {len(field_ids)}")
    print(f"Models: {models}")
    print(f"Modes: {modes}")
    print(f"Fields per call: {fields_per_call_values}")
    print(f"Gold answers: {len(gold)}")

    if args.dry_run:
        sample_mode = modes[0]
        sample_prompt = build_user_prompt(notes[0], field_ids[: min(3, len(field_ids))], question_metadata, sample_mode)
        print("\nDry run sample prompt:")
        print(sample_prompt[:3000])
        return

    summary_rows = []

    for model_name in models:
        try:
            torch, model, processor = load_model(
                model_name,
                max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            print("Install guidance: pip install -U transformers torch accelerate")
            return

        for mode in modes:
            settings = mode_settings(mode)
            for fields_per_call in fields_per_call_values:
                paths = variant_paths(output_dir, dataset, model_name, mode, fields_per_call)
                if not args.resume:
                    for path in paths.values():
                        if path.exists():
                            path.unlink()
                completed = existing_completed(paths["runs"]) if args.resume else set()
                variant_predictions = load_json_list(paths["predictions"]) if args.resume else []
                parse_errors = 0
                call_count = 0
                started = time.time()

                for note in notes:
                    for field_batch in chunked(field_ids, fields_per_call):
                        batch_done = [
                            (
                                dataset,
                                mode,
                                model_name,
                                fields_per_call,
                                note["record_id"],
                                field_id,
                            )
                            in completed
                            for field_id in field_batch
                        ]
                        if args.resume and all(batch_done):
                            continue

                        messages = [
                            {"role": "system", "content": build_system_prompt(mode)},
                            {
                                "role": "user",
                                "content": build_user_prompt(note, field_batch, question_metadata, mode),
                            },
                        ]
                        call_count += 1
                        generated = generate(
                            model,
                            processor,
                            torch,
                            messages,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            enable_thinking=settings["thinking"],
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
                        except Exception as exc:
                            parse_errors += 1
                            predictions = coerce_records(
                                [{"field_id": field_id, "answers": "N/A"} for field_id in field_batch],
                                note,
                                field_batch,
                                question_metadata,
                                mode,
                            )
                            for prediction in predictions:
                                prediction["parse_error"] = str(exc)

                        variant_predictions.extend(predictions)
                        append_jsonl(
                            paths["runs"],
                            {
                                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "dataset": dataset,
                                "mode": mode,
                                "model": model_name,
                                "fields_per_call": fields_per_call,
                                "record_id": note["record_id"],
                                "field_ids": field_batch,
                                "raw_generation": generated,
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
                                    prediction["record_id"],
                                    prediction["field_id"],
                                )
                            )
                        if args.sleep:
                            time.sleep(args.sleep)

                metrics = evaluate_predictions(variant_predictions, gold)
                metrics.update(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "model": model_name,
                        "fields_per_call": fields_per_call,
                        "calls": call_count,
                        "parse_errors": parse_errors,
                        "parse_error_rate": parse_errors / call_count if call_count else 0,
                        "seconds": round(time.time() - started, 2),
                    }
                )
                summary_rows.append(metrics)
                write_json(paths["predictions"], variant_predictions)
                write_csv_records(paths["summary"], [metrics], summary_fields())
                print(json.dumps(metrics, indent=2))

        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_fields_list = summary_fields()
    write_csv_records(summary_path, summary_rows, summary_fields_list)
    print(f"\nSaved per-variant raw runs, predictions, and summaries under {output_dir}")
    print(f"Saved aggregate summary to {summary_path}")


def summary_fields():
    return [
        "dataset",
        "mode",
        "model",
        "fields_per_call",
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
