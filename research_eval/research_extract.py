#!/usr/bin/env python3
"""Extract all research questions in bundles of five with question-specific notes.

Uses the same Gemma loader/generator and record format as evaluate_gemma4_extraction.py.
python research_eval/research_extract.py --dry_run --limit 0
python research_eval/research_extract.py --models e2b --limit 5
python research_eval/research_extract.py --backend ollama --models gemma4 --limit 5
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompt_eval.evaluate_gemma4_extraction import generate, load_model, resolve_models, format_messages_for_processor
from prompt_eval.extraction_prompts import (DEFAULT_MODES, EXTRACTION_SYSTEM_PROMPT,
    build_user_prompt, mode_settings)
from research_eval.research_common import (DEFAULT_DATA, DEFAULT_QUESTIONS, ROOT,
    data_audit, file_hash, load_encounters, load_questions, normalize_value,
    render_note, select_notes, write_json)
from research_eval.research_eval import evaluate_file, prediction_failed

# Imported verbatim from the same module used to construct the training examples.
SYSTEM_PROMPT = EXTRACTION_SYSTEM_PROMPT


class ContextLimitError(ValueError):
    """The complete input cannot fit; never truncate or retry the same oversized input."""


def check_context_budget(input_tokens, max_input_tokens, max_new_tokens, context_window=None, *, estimated=False):
    budget = max_input_tokens
    if context_window:
        budget = min(budget, context_window - max_new_tokens)
    if input_tokens > budget:
        measurement = "conservative token estimate" if estimated else "token count"
        raise ContextLimitError(f"Complete input {measurement} is {input_tokens}; input budget is {budget}. "
                                "Use a larger supported context/model. No note text was truncated.")


def bundles(fields, size=5):
    if size <= 0:
        raise ValueError("fields_per_call must be positive")
    return [fields[i:i+size] for i in range(0, len(fields), size)]


def clinical_note_text(encounter, text):
    """Factual source metadata, not additional extraction instructions."""
    return f"Transplant date: {encounter['bmt_date']}; admission: {encounter['admit_date']}\n\n{text}"


def build_prompt(encounter, fields, questions, mode, *, note_text=None, note_date=None):
    """Use the canonical evaluation/SFT builder with no stage-specific instructions."""
    if note_text is None:
        note_text = "\n\n".join(map(render_note, select_notes(encounter, fields, questions)))
    note = dict(record_id=encounter["record_id"], record_id_type=encounter["record_id_type"],
                date=note_date or encounter["date"], note=clinical_note_text(encounter, note_text))
    return build_user_prompt(note, fields, questions, mode)


def quote_spans(source_text, quote):
    """Find exact quotes, tolerating only whitespace differences."""
    if not quote.strip():
        return []
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    return [(match.start(), match.end()) for match in re.finditer(pattern, source_text)]


def collect_excerpts(note, rows):
    """Select whole sentences/paragraphs containing verified quotes; retain negation/dates.

    The model reads the entire original note. This is evidence selection afterward,
    not a text-splitting strategy. If a source has no sentence boundaries, its full
    enclosing passage is retained and remains subject to the input-token guard.
    """
    text = note["text"]
    boundaries = sorted({0, len(text), *(m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text))})
    spans = []
    for row in rows:
        for start, end in quote_spans(text, row.get("evidence_from_text", "")):
            left = boundaries[max(0, bisect_right(boundaries, start)-1)]
            right = boundaries[min(len(boundaries)-1, bisect_left(boundaries, end))]
            spans.append((left, right))
    merged = []
    for start, end in sorted(set(spans)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [text[start:end].strip() for start, end in merged]


def render_evidence_notes(findings):
    """Final input consists of verbatim source excerpts, never inferred answer values."""
    sections = []
    for finding in findings:
        note = dict(finding["note"], text="\n[...]\n".join(finding["excerpts"]))
        if note["text"]:
            sections.append(render_note(note))
    return "\n\n".join(sections)


def parse_response(text, fields, questions, mode, *, require_evidence=False):
    """Require every requested ID exactly once; never assign malformed output by position."""
    if mode == "non_json":
        records = [{"field_id": m[1], "answers": m[2].strip()} for m in
                   re.finditer(r"^Field\s+(\S+)\s*:\s*(.*?)\s*$", text, flags=re.M)]
    else:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        records = json.loads(clean)
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Expected an array of field records")
    ids = [str(r.get("field_id", "")) for r in records]
    if Counter(ids) != Counter(fields):
        raise ValueError("Response must contain every requested field_id exactly once, without extra IDs")
    by_id = dict(zip(ids, records))
    for fid in fields:
        row = by_id[fid]
        if "answers" not in row:
            raise ValueError(f"Missing answers for {fid}")
        if normalize_value(questions[fid]["target_column"], row["answers"])[1] == "invalid":
            raise ValueError(f"Invalid answer type/value for {fid}")
        if require_evidence and not isinstance(row.get("evidence_from_text"), str):
            raise ValueError(f"Expected the standard evidence_from_text string for {fid}")
    return [by_id[fid] for fid in fields]


def request_bundle(infer, prompt, fields, questions, mode, retries=1, *, require_evidence=False, source_text=None):
    attempts = []
    for attempt in range(retries+1):
        raw = ""
        try:
            raw = infer([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], mode)
            rows = parse_response(raw, fields, questions, mode, require_evidence=require_evidence)
            if require_evidence and source_text is not None:
                if any(row["evidence_from_text"].strip() and not quote_spans(source_text, row["evidence_from_text"])
                       for row in rows):
                    raise ValueError("evidence_from_text must be an exact quote from the supplied note")
            attempts.append(dict(attempt=attempt+1, raw_generation=raw, error=""))
            return rows, "", attempts
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(dict(attempt=attempt+1, raw_generation=raw, error=error))
            if isinstance(exc, ContextLimitError):
                break
            # Keep retries distribution-aligned too: do not append repair instructions.
    return [], attempts[-1]["error"], attempts


def note_jobs(encounter, fields, questions):
    for note_index, note in enumerate(select_notes(encounter, fields, questions)):
        relevant = [fid for fid in fields if note["source"] in questions[fid]["note_sources"]]
        yield note_index, note, relevant


def extract_bundle(infer, encounter, fields, questions, mode, strategy, *, retries=1,
                   evidence_cache=None, cache_scope="", save_cache=None):
    if strategy not in {"per_note", "full"}:
        raise ValueError("Supported note strategies are per_note and full; notes are never split")
    if strategy == "per_note" and "evidence_from_text" not in mode_settings(mode)["extra_fields"]:
        raise ValueError("Per-note review requires a prompt mode with evidence_from_text; use baseline for SFT alignment")
    notes = select_notes(encounter, fields, questions)
    attempts = []
    rows, error = [], "No matching notes for this bundle" if not notes else ""
    if notes and strategy == "per_note":
        findings = []
        for note_index, note, relevant in note_jobs(encounter, fields, questions):
            prompt = build_prompt(encounter, relevant, questions, mode, note_text=render_note(note), note_date=note["date"])
            key = hashlib.sha256((cache_scope + "\n" + mode + "\n" + SYSTEM_PROMPT + "\n" + prompt).encode()).hexdigest()
            if evidence_cache is not None and key in evidence_cache:
                note_rows = evidence_cache[key]["fields"]
                log = [dict(attempt=0, cached=True, error="")]
            else:
                note_rows, error, log = request_bundle(infer, prompt, relevant, questions, mode, retries,
                                                     require_evidence=True, source_text=note["text"])
                if not error:
                    # Cache standard response fields. No model-inferred answers or
                    # rationales are inserted into the final clinical-note content.
                    note_rows = [{"field_id": r["field_id"], "answers": r["answers"],
                                  "evidence_from_text": r["evidence_from_text"]}
                                 for r in note_rows]
                    if evidence_cache is not None:
                        evidence_cache[key] = dict(record_id=encounter["record_id"], model=cache_scope, mode=mode,
                                                  source=note["source"], note_date=note["date"],
                                                  note_index=note_index, fields=note_rows)
                        if save_cache:
                            save_cache(evidence_cache)
            attempts.extend(dict(stage="note", note_index=note_index, source=note["source"], **a) for a in log)
            if error:
                break
            excerpts = collect_excerpts(note, note_rows)
            findings.append(dict(note={k: v for k, v in note.items() if k != "text"}, excerpts=excerpts))
        if not error:
            evidence_text = render_evidence_notes(findings)
            prompt = build_prompt(encounter, fields, questions, mode, note_text=evidence_text)
            rows, error, log = request_bundle(infer, prompt, fields, questions, mode, retries,
                                             require_evidence=True, source_text=evidence_text)
            attempts.extend(dict(stage="final", **a) for a in log)
    elif notes:
        prompt = build_prompt(encounter, fields, questions, mode)
        rows, error, log = request_bundle(infer, prompt, fields, questions, mode, retries)
        attempts.extend(dict(stage="full", **a) for a in log)
    by_id = {r["field_id"]: r for r in rows}
    result = []
    for fid in fields:
        q = questions[fid]
        raw = by_id.get(fid, {})
        prediction = {k: encounter[k] for k in ("record_id", "record_id_type", "date", "MRN", "bmt_date")}
        prediction.update({k: q[k] for k in ("field_id", "field_category", "field_name", "field_instructions", "target_column")})
        prediction.update(answers=raw.get("answers", "N/A"), extraction_success=not bool(error),
                          parse_error=error, note_sources=q["note_sources"],
                          available_note_counts=dict(Counter(n["source"] for n in notes)),
                          missing_note_sources=[s for s in q["note_sources"] if not any(n["source"] == s for n in notes)])
        for extra in mode_settings(mode)["extra_fields"]:
            prediction[extra] = raw.get(extra, "")
        result.append(prediction)
    return result, attempts


def make_infer(args, model_name):
    if args.backend == "ollama":
        try:
            from ollama import chat
        except ImportError as exc:
            raise RuntimeError("Ollama backend requires the ollama Python package") from exc
        def infer(messages, mode):
            # Ollama exposes no tokenizer here. Use a conservative byte-based upper
            # estimate, with space for its chat template, rather than risk truncation.
            estimate = sum(len(m["content"].encode("utf-8")) for m in messages) + 4096
            check_context_budget(estimate, args.max_input_tokens, args.max_new_tokens, args.ollama_num_ctx, estimated=True)
            response = chat(model=model_name, messages=messages,
                            options=dict(temperature=args.temperature, num_predict=args.max_new_tokens,
                                         num_ctx=args.ollama_num_ctx))
            return response.message.content
        return infer, lambda: None
    torch, model, processor = load_model(model_name, args.load_in_4bit)
    def infer(messages, mode):
        encoded = processor.apply_chat_template(format_messages_for_processor(messages, processor),
            tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=mode_settings(mode)["thinking"])
        text_config = getattr(model.config, "text_config", model.config)
        context_window = getattr(text_config, "max_position_embeddings", None)
        check_context_budget(encoded["input_ids"].shape[-1], args.max_input_tokens,
                             args.max_new_tokens, context_window)
        del encoded
        response, _ = generate(model, processor, torch, messages, args.max_new_tokens,
                               args.temperature, args.top_p, args.top_k, mode_settings(mode)["thinking"])
        return response
    def cleanup():
        nonlocal model, processor
        model, processor = None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return infer, cleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", default=str(DEFAULT_DATA))
    parser.add_argument("--cohort")
    parser.add_argument("--ground-truth", "--ground_truth")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output-dir", "--output_dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--backend", choices=["transformers", "ollama"], default="transformers")
    parser.add_argument("--models", default="e2b")
    parser.add_argument("--modes", default="baseline")
    parser.add_argument("--fields-per-call", "--fields_per_call", type=int, default=5)
    parser.add_argument("--limit", type=int, default=5, help="Encounter limit; 0 means all cohort encounters")
    parser.add_argument("--note-strategy", "--note_strategy", choices=["per_note", "full"], default="per_note")
    parser.add_argument("--max-input-tokens", "--max_input_tokens", type=int, default=24000,
                        help="Input budget before inference; exact tokenizer count for Transformers, conservative byte estimate for Ollama")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=8192)
    parser.add_argument("--parse-retries", "--parse_retries", type=int, default=1)
    parser.add_argument("--load-in-4bit", "--load_in_4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.95)
    parser.add_argument("--top-k", "--top_k", type=int, default=64)
    parser.add_argument("--ollama-num-ctx", type=int, default=65536)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-raw-runs", "--save_raw_runs", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--save-prompt-preview", "--save_prompt_preview", action="store_true",
                        help="During dry run, also save first encounter's actual note prompts")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or len(modes) != len(set(modes)) or any(m not in DEFAULT_MODES for m in modes):
        parser.error("--modes must contain unique supported prompt modes: " + ", ".join(DEFAULT_MODES))
    if args.note_strategy == "per_note" and any("evidence_from_text" not in mode_settings(m)["extra_fields"] for m in modes):
        parser.error("per_note requires evidence_from_text; use baseline for SFT alignment or explicitly select full for modes without evidence")
    if args.limit < 0 or args.fields_per_call <= 0 or args.parse_retries < 0 or args.max_new_tokens <= 0:
        parser.error("Invalid limit, bundle size, retry count, or generation length")
    if args.max_input_tokens <= 0 or args.ollama_num_ctx <= args.max_new_tokens:
        parser.error("Require positive max_input_tokens and ollama_num_ctx > max_new_tokens")
    data_dir = Path(args.data_dir)
    cohort_path = Path(args.cohort or data_dir / "target_filtered.csv")
    gt_path = Path(args.ground_truth or data_dir / "toci_ptcy_crs.csv")
    questions = load_questions(args.questions)
    all_encounters = load_encounters(data_dir, cohort_path)
    audit = data_audit(data_dir, cohort_path, gt_path, all_encounters, questions)
    encounters = all_encounters[:args.limit] if args.limit else all_encounters
    if not encounters:
        parser.error("No cohort encounters selected")
    models = resolve_models(args.models) if args.backend == "transformers" else [m.strip() for m in args.models.split(",") if m.strip()]
    if not models or len(models) != len(set(models)):
        parser.error("--models must contain unique model names")
    batches = bundles(list(questions), args.fields_per_call)
    variants = [dict(model=model, mode=mode, note_strategy=args.note_strategy, fields_per_call=args.fields_per_call)
                for model in models for mode in modes]
    inputs = [cohort_path, gt_path, Path(args.questions)] + [data_dir / n for n in
              ("progress_notes.csv", "h_p_notes.csv", "dc_summary_notes.csv")]
    code = [Path(__file__), Path(__file__).with_name("research_common.py"),
            Path(__file__).with_name("research_eval.py"), ROOT / "prompt_eval/extraction_prompts.py",
            ROOT / "prompt_eval/evaluate_gemma4_extraction.py"]
    config = {k: v for k, v in vars(args).items() if k not in {"resume", "dry_run", "output_dir", "save_raw_runs", "save_prompt_preview"}}
    signature = dict(config=config, input_hashes={str(p.resolve()): file_hash(p) for p in inputs},
                     code_hashes={str(p.resolve()): file_hash(p) for p in code})
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    manifest = dict(fingerprint=fingerprint, **signature, record_ids=[e["record_id"] for e in encounters],
                    variants=variants, bundles=batches, dry_run=args.dry_run,
                    prompt_contract="shared_sft_baseline" if modes == ["baseline"] else "shared_evaluation_variants")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, pred_path = output / "run_manifest.json", output / "research_predictions.json"
    evidence_path = output / "note_evidence.json"
    predictions = []
    evidence_cache = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (pred_path.exists() or evidence_path.exists()) and not args.resume:
            parser.error("Existing predictions/evidence found: use --resume or a new output directory")
        if args.resume and old["fingerprint"] != fingerprint:
            parser.error("Resume rejected: inputs, questions, code, or run configuration changed")
        if args.resume and pred_path.exists():
            predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        if args.resume and evidence_path.exists():
            evidence_cache = json.loads(evidence_path.read_text(encoding="utf-8"))
    elif pred_path.exists() or evidence_path.exists():
        parser.error("Existing predictions/evidence have no run manifest; choose a new output directory")
    write_json(output / "data_audit.json", audit)
    write_json(manifest_path, manifest)
    print(f"Cohort: {len(all_encounters)}; selected: {len(encounters)}; GT matches: {audit['matched_ground_truth']}")
    print(f"Note coverage: {audit['note_coverage']}; date conflicts: {audit['note_date_conflicts']}")
    print(f"All {len(questions)} research fields; bundle sizes: {[len(b) for b in batches]}")
    if args.dry_run:
        if args.save_prompt_preview:
            previews = []
            for mode in modes:
                for batch in batches:
                    if args.note_strategy == "per_note":
                        for _, note, relevant in note_jobs(encounters[0], batch, questions):
                            previews.append(dict(stage="note", source=note["source"], mode=mode, fields=relevant,
                                system=SYSTEM_PROMPT, prompt=build_prompt(encounters[0], relevant, questions, mode,
                                                                        note_text=render_note(note), note_date=note["date"])))
                    else:
                        previews.append(dict(stage="full", mode=mode, fields=batch, system=SYSTEM_PROMPT,
                            prompt=build_prompt(encounters[0], batch, questions, mode)))
            write_json(output / "prompt_preview.json", previews)
        note_calls = sum(1 for e in encounters for batch in batches for _ in note_jobs(e, batch, questions))
        print(f"Strategy: {args.note_strategy}; planned per-note calls: {note_calls * len(variants) if args.note_strategy == 'per_note' else 0}; final bundle calls: {len(encounters)*len(batches)*len(variants)}")
        print(f"Dry run complete: {output}; no model loaded and no predictions scored")
        return
    def pkey(p):
        return tuple(str(p[k]) for k in ("model", "mode", "note_strategy", "fields_per_call", "record_id", "field_id"))
    if len({pkey(p) for p in predictions}) != len(predictions):
        raise ValueError("Duplicate checkpoint predictions")
    for model_name in models:
        infer = cleanup = None
        try:
            for variant in (v for v in variants if v["model"] == model_name):
                for encounter_index, encounter in enumerate(encounters):
                    for batch_index, batch in enumerate(batches):
                        wanted = {pkey({**variant, "record_id": encounter["record_id"], "field_id": fid}) for fid in batch}
                        old = [p for p in predictions if pkey(p) in wanted]
                        if len(old) == len(batch) and all(not prediction_failed(p) for p in old):
                            continue
                        if infer is None:
                            infer, cleanup = make_infer(args, model_name)
                        print(f"{model_name} / {variant['mode']}: encounter {encounter_index+1}/{len(encounters)}, bundle {batch_index+1}/{len(batches)}", flush=True)
                        rows, attempts = extract_bundle(infer, encounter, batch, questions, variant["mode"], args.note_strategy,
                                                        retries=args.parse_retries, evidence_cache=evidence_cache,
                                                        cache_scope=model_name,
                                                        save_cache=lambda cache: write_json(evidence_path, cache))
                        predictions = [p for p in predictions if pkey(p) not in wanted]
                        predictions.extend({**r, **variant, "bundle_index": batch_index+1} for r in rows)
                        write_json(pred_path, predictions)
                        if args.save_raw_runs:
                            with (output / "research_runs.jsonl").open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(dict(record_id=encounter["record_id"], **variant,
                                                             fields=batch, attempts=attempts), ensure_ascii=False) + "\n")
        finally:
            if cleanup:
                cleanup()
    completed = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(output / f"research_{completed}_predictions.json", predictions)
    summaries = evaluate_file(pred_path, gt_path, cohort_path, output, args.questions, manifest_path)
    print(json.dumps(summaries, indent=2))
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
