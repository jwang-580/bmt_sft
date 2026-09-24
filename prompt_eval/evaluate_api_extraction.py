#!/usr/bin/env python3
"""Evaluate Form 2400 extraction prompts with OpenAI, Anthropic, and Gemini APIs.

Uses the same prompts, parsing, score calculation, and prediction format as
evaluate_gemma4_extraction.py. API keys are loaded from .env or the environment.
The default evaluation runs baseline on all notes and all form fields, using
full notes and five fields per API call.

Examples:
    python prompt_eval/evaluate_api_extraction.py --dataset mimic --limit 5
    python prompt_eval/evaluate_api_extraction.py --dataset mimic --models gpt-6-sol
    python prompt_eval/evaluate_api_extraction.py --models claude-opus-5-5 --resume
    python prompt_eval/evaluate_api_extraction.py --models gemini-3.8-flash

Install: pip install -U -r prompt_eval/requirements-api.txt

API references:
    https://developers.openai.com/api/docs/models/gpt-6-sol
    https://developers.openai.com/api/docs/guides/migrate-to-responses
    https://platform.claude.com/docs/en/models/opus-5-5/overview
    https://platform.claude.com/docs/en/api/python/messages/create
    https://ai.google.dev/gemini-api/docs/latest-model
    https://ai.google.dev/gemini-api/docs/text-generation
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

try:
    from . import evaluate_gemma4_extraction as common
    from .extraction_prompts import DEFAULT_MODES, build_synthesis_prompt, build_system_prompt, build_user_prompt, mode_settings
except ImportError:
    import evaluate_gemma4_extraction as common
    from extraction_prompts import DEFAULT_MODES, build_synthesis_prompt, build_system_prompt, build_user_prompt, mode_settings


MODELS = {
    "gpt-6-sol": "openai",
    "claude-opus-5-5": "anthropic",
    "gemini-3.8-flash": "google",
}


def make_client(model):
    provider = MODELS[model]
    if provider == "google":
        # Match Google's documented key precedence when both variables exist.
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required. Put it in .env or the environment.")
        from google import genai

        return genai.Client(api_key=api_key)
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        raise ValueError(f"{key_name} is required for {model}. Put it in .env or the environment.")
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        if not hasattr(client, "responses"):
            raise RuntimeError("The installed openai package lacks the Responses API. Run: pip install -U openai")
        return client
    from anthropic import Anthropic

    return Anthropic()


def generate(client, model, system_prompt, user_prompt, mode, max_output_tokens):
    """Return visible answer text only; hidden reasoning is never used as a label."""
    provider = MODELS[model]
    if provider == "openai":
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": "none" if mode == "no_reasoning" else "medium"},
            store=False,
        )
        if response.status != "completed":
            raise RuntimeError(f"OpenAI response status: {response.status}; details: {response.incomplete_details}")
        return response.output_text or ""

    if provider == "google":
        # Gemini 3.8 Flash supports low/medium/high thinking; low is the
        # closest available setting for the existing no_reasoning variant.
        interaction = client.interactions.create(
            model=model,
            system_instruction=system_prompt,
            input=user_prompt,
            generation_config={
                "thinking_level": "low" if mode == "no_reasoning" else "medium",
                "max_output_tokens": max_output_tokens,
            },
            store=False,
        )
        if interaction.status != "completed":
            raise RuntimeError(f"Gemini interaction status: {interaction.status}")
        return interaction.output_text or ""

    # Opus 5.5 always uses adaptive thinking. Low effort is the closest
    # supported setting for the existing no_reasoning prompt variant.
    message = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_output_tokens,
        output_config={"effort": "low" if mode == "no_reasoning" else "medium"},
    )
    if message.stop_reason != "end_turn":
        raise RuntimeError(f"Anthropic message stopped with {message.stop_reason!r}")
    return "\n".join(block.text for block in message.content if block.type == "text")


def generate_predictions(client, model, note, field_batch, metadata, mode, prompt, max_output_tokens, parse_retries):
    settings = mode_settings(mode)
    attempts = []
    parse_error = ""
    generated = ""
    for attempt in range(1, parse_retries + 2):
        generated = generate(client, model, build_system_prompt(), prompt, mode, max_output_tokens)
        try:
            records = (
                common.parse_json_generation(generated)
                if settings["json"]
                else common.parse_non_json_generation(generated, field_batch)
            )
            predictions = common.coerce_records(records, note, field_batch, metadata, mode)
            attempts.append({"attempt": attempt, "raw_generation": generated, "parse_error": ""})
            return generated, predictions, "", attempts
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
            attempts.append({"attempt": attempt, "raw_generation": generated, "parse_error": parse_error})

    predictions = common.coerce_records([], note, field_batch, metadata, mode)
    for prediction in predictions:
        prediction["parse_error"] = parse_error
    return generated, predictions, parse_error, attempts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mimic", "osu"], default="mimic")
    parser.add_argument("--mimic_path", default="data/mimic_iv_bmt_initial_progression.csv")
    parser.add_argument("--osu_path", default="data/osu_h_p_notes.csv")
    parser.add_argument("--questions", default="data/CIBMTR_Form_2400_R11_Items_simplified.csv")
    parser.add_argument("--gold_json", default=None)
    parser.add_argument("--output_dir", default="prompt_eval/results")
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated API model IDs.")
    parser.add_argument("--modes", default="baseline")
    parser.add_argument("--fields_per_call", default="5")
    parser.add_argument("--comparison_fields_per_call", type=int, default=None)
    parser.add_argument("--mode_fields_per_call", default="")
    parser.add_argument("--skip_runs", default="")
    parser.add_argument("--limit", type=int, default=0, help="Maximum notes; 0 (default) means all notes.")
    parser.add_argument("--field_limit", type=int, default=None, help="Maximum form fields; default is all fields.")
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--parse_retries", type=int, default=1)
    parser.add_argument("--note_strategy", choices=["auto", "full", "chunked"], default="full")
    parser.add_argument("--chunk_chars", type=int, default=12000)
    parser.add_argument("--chunk_overlap_chars", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_raw_runs", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Show the plan without using API keys or sending notes.")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.parse_retries < 0 or args.sleep < 0:
        parser.error("--limit, --parse_retries, and --sleep must be non-negative")
    if args.max_output_tokens <= 0:
        parser.error("--max_output_tokens must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    load_dotenv()
    models = common.parse_csv_list(args.models)
    if not models or any(model not in MODELS for model in models):
        raise ValueError(f"--models must contain one or more of: {', '.join(MODELS)}")
    modes = common.parse_csv_list(args.modes)
    if not modes or any(mode not in DEFAULT_MODES for mode in modes):
        raise ValueError(f"--modes must contain one or more of: {', '.join(DEFAULT_MODES)}")
    baseline_sizes = common.parse_int_csv_list(args.fields_per_call)
    comparison_size = args.comparison_fields_per_call or max(baseline_sizes)
    if comparison_size <= 0:
        raise ValueError("--comparison_fields_per_call must be positive")
    strategies = common.build_note_strategy_plan(modes, args.note_strategy)
    plan = common.apply_skip_runs(
        common.build_variant_fields_per_call_plan(
            modes, strategies, baseline_sizes, comparison_size,
            common.parse_mode_int_map(args.mode_fields_per_call),
        ),
        common.parse_skip_runs(args.skip_runs),
    )
    planned_sizes = [size for sizes in plan.values() for size in sizes]
    if not planned_sizes:
        raise ValueError("No runs remain after --skip_runs")

    metadata = common.load_question_metadata(args.questions)
    if args.field_limit is not None and args.field_limit <= 0:
        raise ValueError("--field_limit must be positive")
    field_count = args.field_limit or len(metadata)
    field_ids = list(metadata)[:field_count]
    notes = common.load_notes(args.dataset, args.mimic_path, args.osu_path, args.limit)
    if not notes or not field_ids:
        raise ValueError("No notes or fields selected")
    gold_path = args.gold_json or {
        "mimic": "data/mimic_form_2400_answers.json",
        "osu": "data/osu_form_2400_all_extractions.json",
    }[args.dataset]
    gold = common.load_gold_answers(gold_path)
    print(f"Dataset: {args.dataset}; notes: {len(notes)}; fields: {len(field_ids)}; gold answers: {len(gold)}")
    print(f"Models: {models}; variants: {plan}")
    if args.dry_run:
        print("\nSample prompt:\n" + build_user_prompt(notes[0], field_ids[:3], metadata, modes[0])[:3000])
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = common.concentrated_paths(output_dir, args.dataset, models)
    all_predictions = common.load_json_list(checkpoint["predictions"]) if args.resume else []
    completed = common.existing_completed_predictions(all_predictions) if args.resume else set()
    summary_rows = []

    for model in models:
        client = make_client(model)
        for mode in modes:
            for strategy in strategies[mode]:
                for size in plan[(mode, strategy)]:
                    run_path = common.variant_paths(output_dir, args.dataset, model, mode, size, strategy)["runs"]
                    if args.save_raw_runs and not args.resume and run_path.exists():
                        run_path.unlink()
                    variant_predictions = [
                        pred for pred in all_predictions
                        if common.prediction_matches_variant(pred, args.dataset, mode, model, size, strategy)
                    ] if args.resume else []
                    calls = 0
                    parse_errors = 0
                    started = time.time()
                    chunks_by_note = {
                        note["record_id"]: common.chunk_text(note["note"], args.chunk_chars, args.chunk_overlap_chars)
                        for note in notes
                    } if strategy == "chunked" else {}
                    progress = common.progress_bar(
                        total=sum((len(chunks_by_note[note["record_id"]]) + 1) if strategy == "chunked" else 1
                                  for note in notes for _ in common.chunked(field_ids, size)),
                        desc=f"{model} {mode} {strategy} fpc={size}", unit="call", leave=False,
                    )
                    try:
                        for note in notes:
                            for batch in common.chunked(field_ids, size):
                                keys = [
                                    (args.dataset, mode, model, size, strategy, note["record_id"], field_id)
                                    for field_id in batch
                                ]
                                if args.resume and all(key in completed for key in keys):
                                    progress.update(len(chunks_by_note[note["record_id"]]) + 1 if strategy == "chunked" else 1)
                                    continue
                                raw = []
                                if strategy == "chunked":
                                    chunk_predictions = []
                                    chunks = chunks_by_note[note["record_id"]]
                                    for index, chunk in enumerate(chunks):
                                        chunk_note = common.chunk_note(note, chunk, index, len(chunks))
                                        text, preds, error, attempts = generate_predictions(
                                            client, model, chunk_note, batch, metadata, mode,
                                            build_user_prompt(chunk_note, batch, metadata, mode),
                                            args.max_output_tokens, args.parse_retries,
                                        )
                                        calls += len(attempts)
                                        parse_errors += bool(error)
                                        chunk_predictions.extend(preds)
                                        raw.append({"chunk_index": index + 1, "raw_generation": text,
                                                    "parse_error": error, "generation_attempts": attempts})
                                        progress.update(1)
                                    prompt = build_synthesis_prompt(note, batch, metadata, mode, chunk_predictions)
                                else:
                                    prompt = build_user_prompt(note, batch, metadata, mode)
                                text, preds, error, attempts = generate_predictions(
                                    client, model, note, batch, metadata, mode, prompt,
                                    args.max_output_tokens, args.parse_retries,
                                )
                                calls += len(attempts)
                                parse_errors += bool(error)
                                progress.update(1)
                                if strategy == "chunked":
                                    for pred in preds:
                                        pred["chunk_count"] = len(chunks)
                                        pred["synthesized_from_chunks"] = True
                                preds = common.add_variant_metadata(preds, args.dataset, mode, model, size, strategy)
                                variant_predictions.extend(preds)
                                all_predictions.extend(preds)
                                completed.update(keys)
                                common.write_json(checkpoint["predictions"], all_predictions)
                                if args.save_raw_runs:
                                    common.append_jsonl(run_path, {
                                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                                        "dataset": args.dataset, "mode": mode, "model": model,
                                        "fields_per_call": size, "note_strategy": strategy,
                                        "record_id": note["record_id"], "field_ids": batch,
                                        "chunk_generations": raw, "raw_generation": text,
                                        "parse_error": error, "generation_attempts": attempts,
                                        "predictions": preds,
                                    })
                                if args.sleep:
                                    time.sleep(args.sleep)
                    finally:
                        progress.close()
                    metrics = common.evaluate_predictions(variant_predictions, gold)
                    metrics.update({
                        "dataset": args.dataset, "mode": mode, "model": model,
                        "fields_per_call": size, "note_strategy": strategy,
                        "chunk_chars": args.chunk_chars if strategy == "chunked" else "",
                        "chunk_overlap_chars": args.chunk_overlap_chars if strategy == "chunked" else "",
                        "calls": calls, "parse_errors": parse_errors,
                        "parse_error_rate": parse_errors / calls if calls else 0,
                        "seconds": round(time.time() - started, 2),
                    })
                    summary_rows.append(metrics)
                    print(json.dumps(metrics, indent=2))

    completed_at = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_paths = common.concentrated_paths(output_dir, args.dataset, models, completed_at)
    common.write_json(final_paths["predictions"], all_predictions)
    common.write_csv_records(final_paths["summary"], summary_rows, common.summary_fields())
    common.write_csv_records(checkpoint["summary"], summary_rows, common.summary_fields())
    print(f"Saved predictions to {final_paths['predictions']}")
    print(f"Saved summary to {final_paths['summary']}")


if __name__ == "__main__":
    main()
