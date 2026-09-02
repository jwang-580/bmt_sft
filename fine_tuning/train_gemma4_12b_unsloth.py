#!/usr/bin/env python3
"""Fine-tune Gemma 4 12B on the generated MIMIC SFT conversations.

The training set contains all 90 labeled notes outside the fixed 20-record
evaluation cohort. No subset of the training set is held out for trainer-side
evaluation. After training, the script validates the in-memory LoRA model on
the 20 external records; it does not start or reload a separate evaluator.

The data formatting follows Unsloth's Gemma 4 text notebook:
  * install the ``gemma-4`` chat template;
  * render each conversation with ``add_generation_prompt=False``;
  * remove the leading ``<bos>`` because the processor adds it during training;
  * mask loss to assistant responses with ``train_on_responses_only``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = REPO_ROOT / "fine_tuning" / "outputs" / "mimic_sft_fpc5_train.jsonl"
DEFAULT_EVAL_IDS = REPO_ROOT / "fine_tuning" / "mimic_eval_record_ids.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fine_tuning" / "outputs" / "gemma4_12b_lora"
DEFAULT_MODEL = "unsloth/gemma-4-12b-it"
DEFAULT_MIMIC_FILE = REPO_ROOT / "data" / "mimic_iv_bmt_initial_progression.csv"
DEFAULT_QUESTIONS_FILE = REPO_ROOT / "data" / "CIBMTR_Form_2400_R11_Items_simplified.csv"
DEFAULT_GOLD_FILE = REPO_ROOT / "data" / "mimic_form_2400_answers.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune unsloth/gemma-4-12b-it and validate on the fixed MIMIC cohort."
    )
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval_record_ids", type=Path, default=DEFAULT_EVAL_IDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Opt into 4-bit model loading. The 12B model loads at full precision by default.",
    )
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override epoch-based training when positive; -1 trains for --epochs.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataset_num_proc", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wandb_project", default="bmt-sft")
    parser.add_argument("--wandb_run_name", default="gemma4-12b-lora-r16")
    parser.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=None)
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Save the adapter without validating on the 20 held-out records.",
    )
    parser.add_argument("--eval_fields_per_call", type=int, default=5)
    parser.add_argument("--eval_max_new_tokens", type=int, default=8192)
    parser.add_argument("--eval_parse_retries", type=int, default=1)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths, schema, split integrity, and settings without importing Unsloth.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_inputs(train_file: Path, eval_ids_path: Path) -> dict[str, Any]:
    if not train_file.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {train_file}")
    if not eval_ids_path.is_file():
        raise FileNotFoundError(f"Evaluation record-id file not found: {eval_ids_path}")

    eval_ids_raw = read_json(eval_ids_path)
    if not isinstance(eval_ids_raw, list) or not all(isinstance(value, (str, int)) for value in eval_ids_raw):
        raise ValueError(f"Expected a JSON list of record IDs in {eval_ids_path}")
    eval_ids = {str(value) for value in eval_ids_raw}
    if len(eval_ids) != 20:
        raise ValueError(f"Expected exactly 20 held-out record IDs, found {len(eval_ids)}")

    examples = 0
    record_ids: set[str] = set()
    max_chars = 0
    with train_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {train_file}:{line_number}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or [message.get("role") for message in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError(
                    f"Expected system/user/assistant messages at {train_file}:{line_number}"
                )
            if not all(isinstance(message.get("content"), str) and message["content"] for message in messages):
                raise ValueError(f"Every message must contain non-empty text at {train_file}:{line_number}")
            try:
                assistant_rows = json.loads(messages[-1]["content"])
                record_id = str(assistant_rows[0]["record_id"])
            except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Assistant content must be a non-empty JSON list with record_id at {train_file}:{line_number}"
                ) from exc
            record_ids.add(record_id)
            examples += 1
            max_chars = max(max_chars, sum(len(message["content"]) for message in messages))

    leaked_ids = sorted(record_ids & eval_ids)
    if leaked_ids:
        raise ValueError(f"Held-out evaluation IDs leaked into training: {leaked_ids}")
    if len(record_ids) != 90:
        raise ValueError(f"Expected examples from 90 training notes, found {len(record_ids)}")
    if examples != 2340:
        raise ValueError(f"Expected 2,340 training examples, found {examples}")

    return {
        "train_file": str(train_file),
        "examples": examples,
        "training_records": len(record_ids),
        "held_out_records": len(eval_ids),
        "max_message_chars": max_chars,
        "evaluation_leakage": False,
    }


def validate_positive_args(args: argparse.Namespace) -> None:
    positive_values = {
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "dataset_num_proc": args.dataset_num_proc,
        "eval_fields_per_call": args.eval_fields_per_call,
        "eval_max_new_tokens": args.eval_max_new_tokens,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max_steps must be -1 or a positive integer")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be non-negative")
    if args.eval_parse_retries < 0:
        raise ValueError("--eval_parse_retries must be non-negative")


def format_dataset(dataset: Any, tokenizer: Any, num_proc: int) -> Any:
    def formatting_prompts_func(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for messages in examples["messages"]
        ]
        return {"text": texts}

    formatted = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=num_proc,
        desc="Applying the Gemma 4 chat template",
    )
    sample = formatted[0]["text"]
    if sample.startswith("<bos>"):
        raise RuntimeError("Formatted training text contains a duplicate leading <bos> token")
    if "<|turn>model" not in sample or not sample.endswith("<turn|>\n"):
        raise RuntimeError("The installed chat template did not produce Gemma 4 conversation markers")
    return formatted


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_held_out_validation_data(eval_record_ids: Path) -> tuple[Any, list[Any], dict[str, Any], dict[Any, Any]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from prompt_eval import evaluate_gemma4_extraction as evaluation

    eval_ids = [str(value) for value in read_json(eval_record_ids)]
    all_notes = evaluation.load_notes("mimic", str(DEFAULT_MIMIC_FILE), "", limit=0)
    note_by_id = {note["record_id"]: note for note in all_notes}
    missing_notes = [record_id for record_id in eval_ids if record_id not in note_by_id]
    if missing_notes:
        raise ValueError(f"Held-out records missing from the MIMIC source: {missing_notes}")
    notes = [note_by_id[record_id] for record_id in eval_ids]
    question_metadata = evaluation.load_question_metadata(str(DEFAULT_QUESTIONS_FILE))
    gold = evaluation.load_gold_answers(str(DEFAULT_GOLD_FILE))
    return evaluation, notes, question_metadata, gold


def held_out_plan(eval_record_ids: Path, fields_per_call: int) -> dict[str, Any]:
    _, notes, question_metadata, gold = load_held_out_validation_data(eval_record_ids)
    eval_ids = {note["record_id"] for note in notes}
    gold_record_ids = {record_id for record_id, _ in gold if record_id in eval_ids}
    fields = len(question_metadata)
    calls_per_record = (fields + fields_per_call - 1) // fields_per_call
    return {
        "held_out_source_records": len(notes),
        "held_out_gold_records": len(gold_record_ids),
        "held_out_fields": fields,
        "held_out_planned_calls": len(notes) * calls_per_record,
    }


def run_held_out_validation(
    model: Any,
    tokenizer: Any,
    torch: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate and score all fields for the exact 20 held-out records."""
    eval_ids = [str(value) for value in read_json(args.eval_record_ids)]
    evaluation, notes, question_metadata, gold = load_held_out_validation_data(args.eval_record_ids)
    field_ids = list(question_metadata)
    predictions: list[dict[str, Any]] = []
    parse_errors = 0
    calls = 0
    eval_dir = args.output_dir / "eval"
    checkpoint_path = eval_dir / "held_out_predictions.json"

    model.eval()
    print(
        f"Validating the in-memory model on {len(notes)} held-out records, "
        f"{len(field_ids)} fields, {args.eval_fields_per_call} fields per call."
    )
    for note_index, note in enumerate(notes, start=1):
        for field_batch in evaluation.chunked(field_ids, args.eval_fields_per_call):
            prompt = evaluation.build_user_prompt(
                note,
                field_batch,
                question_metadata,
                mode="baseline",
            )
            _, _, batch_predictions, parse_error, attempts = evaluation.generate_predictions(
                model=model,
                processor=tokenizer,
                torch=torch,
                note=note,
                field_batch=field_batch,
                question_metadata=question_metadata,
                mode="baseline",
                prompt=prompt,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                parse_retries=args.eval_parse_retries,
            )
            calls += len(attempts)
            parse_errors += int(bool(parse_error))
            predictions.extend(batch_predictions)
        write_json(checkpoint_path, predictions)
        print(f"Validated held-out record {note_index}/{len(notes)}: {note['record_id']}")

    metrics = evaluation.evaluate_predictions(predictions, gold)
    metrics.update(
        {
            "records": len(notes),
            "fields": len(field_ids),
            "fields_per_call": args.eval_fields_per_call,
            "calls": calls,
            "parse_errors": parse_errors,
            "parse_error_rate": parse_errors / calls if calls else 0,
            "record_ids": eval_ids,
        }
    )
    write_json(eval_dir / "held_out_metrics.json", metrics)
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    f"held_out/{name}": value
                    for name, value in metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
        else:
            print("Warning: the W&B run was not active when held-out metrics were produced.")
    except ImportError:
        print("Warning: wandb is unavailable; held-out metrics were saved locally only.")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    args = parse_args()
    args.train_file = args.train_file.resolve()
    args.eval_record_ids = args.eval_record_ids.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_positive_args(args)
    dataset_summary = validate_inputs(args.train_file, args.eval_record_ids)
    validation_plan = held_out_plan(args.eval_record_ids, args.eval_fields_per_call)

    run_summary = {
        **dataset_summary,
        **validation_plan,
        "model": args.model_name,
        "load_in_4bit": args.load_in_4bit,
        "output_dir": str(args.output_dir),
        "max_seq_length": args.max_seq_length,
        "lora_rank": 16,
        "lora_alpha": 16,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "report_to": "wandb",
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "trainer_eval_dataset": None,
        "held_out_validation": not args.skip_eval,
        "held_out_validation_mode": "baseline",
        "held_out_validation_fields_per_call": args.eval_fields_per_call,
    }
    print(json.dumps(run_summary, indent=2))
    if args.dry_run:
        print("Dry run complete; Unsloth was not imported and no files were written.")
        return 0

    try:
        import torch
        # Unsloth must be imported before TRL/Transformers trainer classes so
        # its patches are applied before SFTConfig and SFTTrainer are bound.
        from unsloth import FastModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from datasets import load_dataset
        import transformers
        import trl
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing fine-tuning dependencies. Install fine_tuning/requirements.txt in the CUDA environment."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    torch._dynamo.config.recompile_limit = 64
    print(
        f"Runtime versions: transformers={transformers.__version__}, "
        f"trl={trl.__version__}"
    )
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    training_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(training_tokenizer, "chat_template", None) is None:
        training_tokenizer.chat_template = tokenizer.chat_template

    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    dataset = format_dataset(dataset, tokenizer, args.dataset_num_proc)

    trainer = SFTTrainer(
        model=model,
        processing_class=training_tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            dataset_text_field="text",
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to="wandb",
            run_name=args.wandb_run_name,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            dataset_num_proc=args.dataset_num_proc,
            packing=False,
        ),
    )
    trainer = train_on_responses_only(trainer)

    first_labels = trainer.train_dataset[0].get("labels", [])
    supervised_tokens = sum(label != -100 for label in first_labels)
    if not supervised_tokens:
        raise RuntimeError("Response-only masking left the first training example with no supervised tokens")
    print(f"Response-only masking verified: {supervised_tokens} supervised tokens in the first example.")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = args.output_dir / "final"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter and Gemma 4 chat template to {adapter_dir}")

    if not args.skip_eval:
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        FastModel.for_inference(model)
        run_held_out_validation(model, tokenizer, torch, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
