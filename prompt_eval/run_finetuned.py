#!/usr/bin/env python3
"""Run BMT-GRAFT-12B on synthetic examples or local notes without reference labels.

Only weight/config downloads use Hugging Face. Notes are processed by the local
model, never passed to a hosted inference endpoint. --dry_run imports no GPU stack.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from prompt_eval import evaluate_gemma4_extraction as common
from prompt_eval.extraction_prompts import build_system_prompt, build_user_prompt

MODEL = "JWang580/BMT-GRAFT-12B"
DEFAULT_FIELDS = ["32", "33", "53", "100", "162"]
DEMO_NOTE = (
    "This is an invented example for testing software, not a patient record. "
    "Patient admitted for an allogeneic matched sibling peripheral blood stem cell transplant. "
    "The selected donor is male. Recipient blood type is A positive. "
    "GVHD prophylaxis consists of tacrolimus and methotrexate."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use one invented note; no clinical file is needed")
    source.add_argument("--note_file", type=Path, help="UTF-8 plain-text note")
    source.add_argument("--notes_csv", type=Path, help="CSV containing a note column; labels are unnecessary")
    parser.add_argument("--note_column", default="note")
    parser.add_argument("--id_column", default="record_id")
    parser.add_argument("--date_column", default="date")
    parser.add_argument("--record_id", default="local_record_0001", help="Identifier for a single note")
    parser.add_argument("--date", default="", help="Note date for a single note, when known")
    parser.add_argument("--model", default=MODEL, help="Hugging Face repository or local checkpoint directory")
    parser.add_argument("--revision", help="Adapter or merged-model revision, preferably a commit hash")
    parser.add_argument("--base_revision", help="Base model revision for an adapter checkpoint")
    parser.add_argument("--base_model_path", help="Explicit local base-model directory for an adapter, useful offline")
    parser.add_argument("--processor_path", help="Explicit saved tokenizer/processor directory or repository")
    parser.add_argument("--local_files_only", action="store_true", help="Use cached/local weights without Hub requests")
    parser.add_argument("--load_in_4bit", action="store_true", help="Optional bitsandbytes model quantization")
    parser.add_argument("--questions", type=Path, default=ROOT/"resources/CIBMTR_Form_2400_R11_Items_simplified.csv")
    fields = parser.add_mutually_exclusive_group()
    fields.add_argument("--fields", help="Comma-separated Form 2400 field IDs; default: 32,33,53,100,162")
    fields.add_argument("--all_fields", action="store_true", help="Extract all 129 questions")
    parser.add_argument("--fields_per_call", type=int, default=5)
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 uses greedy decoding")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_input_tokens", type=int, default=24000, help="Reject oversized prompts; do not truncate notes")
    parser.add_argument("--parse_retries", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT/"prompt_eval/results/finetuned_predictions.json")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and show a plan without loading weights or writing outputs")
    args = parser.parse_args(argv)
    if min(args.fields_per_call, args.max_new_tokens, args.max_input_tokens) < 1:
        parser.error("Question and token limits must be positive")
    if args.parse_retries < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Retry count and temperature must be nonnegative")
    return args


def load_notes(args):
    if args.demo:
        return [{"record_id":"synthetic_demo_0001", "record_id_type":"synthetic", "date":"", "note":DEMO_NOTE}]
    if args.note_file:
        text = args.note_file.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("The note file is empty")
        if not args.record_id.strip():
            raise ValueError("Record identifier must not be blank")
        return [{"record_id":args.record_id, "record_id_type":"local", "date":args.date, "note":text}]
    with args.notes_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or args.note_column not in headers:
            raise ValueError("CSV requires unique headers and the configured note column")
        rows = list(reader)
    notes = []
    identities = set()
    for i, row in enumerate(rows, 1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Malformed CSV row {i}")
        text = row[args.note_column].strip()
        rid = row.get(args.id_column, "").strip() or f"local_record_{i:06d}"
        if not text or rid in identities:
            raise ValueError(f"Empty note or duplicate record identifier at CSV row {i}")
        identities.add(rid)
        notes.append({"record_id":rid, "record_id_type":"local", "date":row.get(args.date_column, "").strip(), "note":text})
    if not notes:
        raise ValueError("CSV contains no note records")
    return notes


def selected_fields(args, metadata):
    ids = list(metadata) if args.all_fields else (common.parse_csv_list(args.fields) if args.fields else DEFAULT_FIELDS[:])
    if not ids or len(ids) != len(set(ids)) or any(fid not in metadata for fid in ids):
        raise ValueError("Field IDs must be nonempty, unique, and present in the question catalog")
    return ids


def checkpoint_layout(model_name, revision, local_files_only):
    """Read only checkpoint metadata; authorization failures never imply merged weights."""
    local = Path(model_name)
    if local.is_dir():
        path = local/"adapter_config.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    try:
        path = hf_hub_download(model_name, "adapter_config.json", revision=revision, local_files_only=local_files_only)
    except LocalEntryNotFoundError:
        # An offline merged checkpoint must have its own cached model config.
        hf_hub_download(model_name, "config.json", revision=revision, local_files_only=True)
        return None
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(args):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
    adapter = checkpoint_layout(args.model, args.revision, args.local_files_only)
    base = (args.base_model_path or adapter.get("base_model_name_or_path")) if adapter is not None else args.model
    if not isinstance(base, str) or not base.strip():
        raise ValueError("Adapter configuration does not identify its trained base model")
    base_revision = (args.base_revision or adapter.get("revision")) if adapter is not None else args.revision
    if adapter is None and (args.base_revision or args.base_model_path):
        raise ValueError("Base-model overrides apply only to an adapter checkpoint")
    kwargs = {"device_map":"auto", "dtype":torch.bfloat16 if torch.cuda.is_available() else torch.float32,
              "local_files_only":args.local_files_only}
    if base_revision:
        kwargs["revision"] = base_revision
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForMultimodalLM.from_pretrained(base, **kwargs)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model, revision=args.revision,
                                          local_files_only=args.local_files_only, is_trainable=False)
    processor_source = args.processor_path or args.model
    processor_kwargs = {"local_files_only":args.local_files_only}
    if args.revision and not args.processor_path:
        processor_kwargs["revision"] = args.revision
    try:
        processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    except (OSError, ValueError):
        try:
            processor = AutoTokenizer.from_pretrained(processor_source, **processor_kwargs)
        except (OSError, ValueError):
            if args.processor_path or adapter is None:
                raise
            warnings.warn("Adapter has no usable processor/tokenizer; loading its base processor. Prefer the saved training tokenizer.")
            base_kwargs = {"local_files_only":args.local_files_only}
            if base_revision:
                base_kwargs["revision"] = base_revision
            processor = AutoProcessor.from_pretrained(base, **base_kwargs)
    if not getattr(processor, "chat_template", None):
        raise ValueError("Tokenizer/processor has no chat template; supply --processor_path from the saved checkpoint")
    model.eval()
    return torch, model, processor, {"checkpoint_type":"adapter" if adapter is not None else "merged",
                                     "base_model":base, "base_revision":base_revision,
                                     "processor_source":processor_source}


def extract_note(note, field_ids, metadata, fields_per_call, parse_retries, generate_text):
    """Validated extraction with a testable generator boundary."""
    predictions = []
    for batch in common.chunked(field_ids, fields_per_call):
        messages = [{"role":"system", "content":build_system_prompt()},
                    {"role":"user", "content":build_user_prompt(note, batch, metadata, "baseline")}]
        error = ""
        rows = []
        for _ in range(parse_retries+1):
            # Runtime failures propagate; parsing failures use the configured retry.
            text = generate_text(messages)
            try:
                rows = common.coerce_records(common.parse_json_generation(text), note, batch, metadata, "baseline")
                common.require_complete_predictions(rows)
                break
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                error = str(exc)
        else:
            rows = common.coerce_records([], note, batch, metadata, "baseline")
            for row in rows:
                row["parse_error"] = error
        predictions.extend(rows)
    return predictions


def main(argv=None):
    args = parse_args(argv)
    metadata = common.load_question_metadata(str(args.questions))
    fields = selected_fields(args, metadata)
    notes = load_notes(args)
    calls = len(notes)*((len(fields)+args.fields_per_call-1)//args.fields_per_call)
    print(f"Records: {len(notes)}; fields per record: {len(fields)}; initial calls: {calls}; model: {args.model}")
    if args.dry_run:
        print("Dry run complete. No weights loaded, no clinical text printed, no output files written.")
        return 0
    output = args.output.resolve()
    manifest = output.with_suffix(".manifest.json")
    inputs = [p.resolve() for p in [args.note_file, args.notes_csv, args.questions] if p is not None]
    if output in inputs or manifest in inputs or output.exists() or manifest.exists():
        raise ValueError("Choose a new output path; inputs and existing outputs are never overwritten")
    torch, model, processor, layout = load_model(args)
    def generate_text(messages):
        formatted = common.format_messages_for_processor(messages, processor)
        rendered = processor.apply_chat_template(formatted, tokenize=True, return_dict=True,
                                                 return_tensors="pt", add_generation_prompt=True,
                                                 enable_thinking=args.thinking=="on")
        if rendered["input_ids"].shape[-1] > args.max_input_tokens:
            raise ValueError("Prompt exceeds --max_input_tokens; shorten the note or deliberately raise the budget")
        # The shared generator preserves model-specific final-channel handling.
        text, _ = common.generate(model, processor, torch, messages, args.max_new_tokens,
                                  args.temperature, .95, 64, args.thinking=="on")
        return text
    predictions = []
    for i, note in enumerate(notes, 1):
        rows = extract_note(note, fields, metadata, args.fields_per_call, args.parse_retries, generate_text)
        for row in rows:
            row.update(dataset="local", model=args.model, mode="baseline", fields_per_call=args.fields_per_call, note_strategy="full")
        predictions.extend(rows)
        print(f"Processed record {i}/{len(notes)}")
    failed = sum(not row["extraction_success"] for row in predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump({"model":args.model, "revision":args.revision, **layout, "records":len(notes), "fields":fields,
                   "fields_per_call":args.fields_per_call, "thinking":args.thinking, "temperature":args.temperature,
                   "max_new_tokens":args.max_new_tokens, "max_input_tokens":args.max_input_tokens,
                   "parse_retries":args.parse_retries, "load_in_4bit":args.load_in_4bit, "failed_fields":failed,
                   "questions_sha256":hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                   "prediction_sha256":hashlib.sha256(output.read_bytes()).hexdigest()},handle,indent=2)
        handle.write("\n")
    print(f"Saved {len(predictions)} field answers ({failed} failed) to {output}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
