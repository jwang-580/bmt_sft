#!/usr/bin/env python3
"""Train vanilla Gemma 4 12B with GRPO and verifiable extraction rewards.

Each prompt is the original system/user pair. Fixed answers and note text travel
in dataset columns for reward functions only, never in the model prompt. The
20-record MIMIC evaluation cohort remains outside the training JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.rlvr_rewards import make_reward_func
from fine_tuning.train_gemma4_12b_unsloth import (
    DEFAULT_EVAL_IDS,
    DEFAULT_TRAIN_FILE,
    validate_inputs,
)
from prompt_eval.extraction_prompts import SFT_OUTPUT_FIELDS


DEFAULT_MODEL = "google/gemma-4-12B-it"
DEFAULT_OUTPUT = REPO_ROOT / "fine_tuning" / "outputs" / "gemma4_12b_rlvr"
NOTE_PREFIX = "\n\nClinical note:\n"
NOTE_SUFFIX = "\n\nFields:\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--held_out_record_ids", type=Path, default=DEFAULT_EVAL_IDS)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--load_in_4bit", action=argparse.BooleanOptionalAction, default=False,
        help="Use 4-bit base weights for GRPO (default: disabled).",
    )
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--max_prompt_length", type=int, default=12288)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument(
        "--steps_per_generation", type=int, default=4,
        help="Generate one rollout batch every N microsteps; independent of gradient accumulation.",
    )
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report_to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb_project", default="bmt-rlvr")
    parser.add_argument("--wandb_run_name", default="gemma4-12b-vanilla-grpo")
    parser.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Validate labels and split without loading a model.")
    return parser.parse_args()


def validate_settings(args: argparse.Namespace) -> None:
    positive = (
        "max_seq_length", "max_prompt_length", "max_completion_length", "max_steps",
        "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
        "num_generations", "steps_per_generation", "save_steps", "save_total_limit", "logging_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch % args.num_generations:
        raise ValueError("Batch size times gradient accumulation must be divisible by num_generations")
    if args.gradient_accumulation_steps % args.steps_per_generation:
        raise ValueError("gradient_accumulation_steps must be divisible by steps_per_generation")
    if args.per_device_train_batch_size * args.steps_per_generation % args.num_generations:
        raise ValueError("Rollout batch size must be divisible by num_generations")
    if args.max_prompt_length + args.max_completion_length + 1 > args.max_seq_length:
        raise ValueError("Prompt and completion limits exceed max_seq_length")
    if not args.model_name.strip():
        raise ValueError("--model_name must identify a base model")
    if (Path(args.model_name) / "adapter_config.json").is_file():
        raise ValueError("--model_name points to a LoRA adapter; RLVR must start from a base model")


def make_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"Unexpected conversation roles at line {line_number}")
            user = messages[1]["content"]
            if NOTE_PREFIX not in user or NOTE_SUFFIX not in user:
                raise ValueError(f"Cannot find clinical note boundaries at line {line_number}")
            note = user.split(NOTE_PREFIX, 1)[1].split(NOTE_SUFFIX, 1)[0]
            gold = json.loads(messages[2]["content"])
            if not isinstance(gold, list) or not 1 <= len(gold) <= 5:
                raise ValueError(f"Expected one to five gold fields at line {line_number}")
            ids = [str(item["field_id"]) for item in gold]
            if len(ids) != len(set(ids)) or any(list(item) != SFT_OUTPUT_FIELDS for item in gold):
                raise ValueError(f"Invalid gold field schema at line {line_number}")
            examples.append({
                "prompt": messages[:2],
                "gold_json": json.dumps(gold, ensure_ascii=False, separators=(",", ":")),
                "note_text": note,
            })
    return examples


def filter_prompt_lengths(examples: list[dict[str, Any]], tokenizer: Any, maximum: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept = []
    longest = 0
    skipped = 0
    for example in examples:
        ids = tokenizer.apply_chat_template(
            example["prompt"], tokenize=True, add_generation_prompt=True
        )
        length = len(ids)
        longest = max(longest, length)
        if length > maximum:
            skipped += 1
        else:
            kept.append(example)
    if not kept:
        raise ValueError("No training prompts fit max_prompt_length")
    return kept, {"kept": len(kept), "skipped_long_prompts": skipped, "longest_prompt_tokens": longest}


def main() -> int:
    args = parse_args()
    for name in ("train_file", "held_out_record_ids", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    validate_settings(args)
    split_summary = validate_inputs(args.train_file, args.held_out_record_ids)
    examples = make_examples(args.train_file)
    if len(examples) != split_summary["examples"]:
        raise ValueError("RLVR and SFT example counts disagree")
    print(json.dumps({
        **split_summary,
        "base_model": args.model_name,
        "output_dir": str(args.output_dir),
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "load_in_4bit": args.load_in_4bit,
        "effective_batch_size_per_process": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "rollout_batch_size_per_process": args.per_device_train_batch_size * args.steps_per_generation,
        "steps_per_generation": args.steps_per_generation,
        "prompts_per_step_single_process": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps // args.num_generations
        ),
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
    }, indent=2))
    if args.dry_run:
        print("Dry run complete; no model imported or training outputs written.")
        return 0

    try:
        from unsloth import FastModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install fine_tuning/requirements-rlvr.txt in a CUDA environment") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
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
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("Fresh LoRA adapter has no trainable parameters")

    examples, token_summary = filter_prompt_lengths(examples, training_tokenizer, args.max_prompt_length)
    print(json.dumps({"trainable_parameters": trainable, **token_summary}, indent=2))
    dataset = Dataset.from_list(examples)
    rewards = [make_reward_func(name) for name in ("answer", "format", "metadata", "evidence")]
    trainer = GRPOTrainer(
        model=model,
        processing_class=training_tokenizer,
        reward_funcs=rewards,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(args.output_dir),
            max_completion_length=args.max_completion_length,
            # TRL >=0.28 leaves prompt lengths to the caller; filtered above.
            remove_unused_columns=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            steps_per_generation=args.steps_per_generation,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_steps=max(1, int(args.max_steps * 0.03)),
            lr_scheduler_type="linear",
            optim="adamw_torch_fused",
            weight_decay=0.001,
            temperature=1.0,
            loss_type="bnpo",
            mask_truncated_completions=True,
            use_vllm=False,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to=args.report_to,
            run_name=args.wandb_run_name,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = args.output_dir / "final"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved RLVR adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
