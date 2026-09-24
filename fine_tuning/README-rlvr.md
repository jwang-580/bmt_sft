# Gemma 4 12B extraction RLVR

`train_gemma4_12b_rlvr_unsloth.py` starts from the vanilla
`google/gemma-4-12B-it` model and creates a fresh LoRA adapter using Unsloth
and TRL's GRPO trainer. It uses the same 2,340 five-field prompts from 90
training notes, but does not load the SFT adapter or its weights. The fixed
answers remain in reward columns and are never included in model prompts. The
20 held-out record IDs are checked before training to prevent data leakage.

Use a CUDA environment with the newer RL stack in
`fine_tuning/requirements-rlvr.txt`. The original SFT requirements remain
separate because they pin an older TRL release.

```bash
pip install -r fine_tuning/requirements-rlvr.txt
python fine_tuning/train_gemma4_12b_rlvr_unsloth.py --dry_run
python fine_tuning/train_gemma4_12b_rlvr_unsloth.py
```

The output adapter is saved under `fine_tuning/outputs/gemma4_12b_rlvr/final`.
To resume, add `--resume_from_checkpoint`. To change GPU memory use, start with
`--max_prompt_length`, `--max_completion_length`, and
`--num_generations`. Prompts that exceed the configured token limit are
counted and excluded, with the count printed before training. Four-bit loading
is off by default; use `--load_in_4bit` when GPU memory is limited.

With one GPU, the current defaults use a per-device batch of 2, 8 gradient
accumulation steps, 4 generations per prompt, and 4 steps per generation batch.
TRL generates 8 completions at a time (2 prompts), twice per optimizer step.
That is 4 distinct prompts per optimizer step. About 585 steps cover the 2,340
prompts once, before any long prompts are excluded. The configured 600 steps
are approximately one pass; two passes would be about 1,170 steps.

TRL otherwise defaults `steps_per_generation` to `gradient_accumulation_steps`.
With accumulation set to 8, that would generate 16 completions at once on one
GPU. Keeping `--steps_per_generation 4` separates rollout memory from the
optimizer's effective batch. Raising accumulation reduces optimizer updates,
but does not reduce the total generated completions or proportionally speed up
training. Raising the per-device batch can improve throughput but uses more
peak memory. Multi-GPU runs cover more distinct prompts per optimizer step.

## Reward design

Each generated JSON array earns up to 5.6 points:

| Component | Maximum | Verifier |
| --- | ---: | --- |
| Answers | 4.0 | Uses the project's field-aware answer equivalence rules. Correct nonempty answers have twice the weight of correct N/A answers. Missing or duplicate fields earn zero. |
| Format | 0.4 | JSON array with exactly the requested fields and keys, in field order, with string evidence/rationale fields. |
| Metadata | 0.2 | Exact record and field metadata from the prompt. |
| Evidence | 1.0 | Correct nonempty answers need an exact quote from the note; correct N/A answers need empty evidence. |

The answer reward dominates so the model cannot gain more by producing neat
JSON with wrong extractions. Reasoning text is required for schema compatibility
but is not rewarded for wording or length. Scores are computed by
`rlvr_rewards.py` without another model or judge.

`--max_completion_length` should be large enough for all five objects. The
trainer masks completions cut off at that limit. Monitor the separate reward
components during training. This script does not run held-out evaluation.

References: [Unsloth Gemma 4 Sudoku GRPO example](https://github.com/unslothai/notebooks/blob/main/nb/Gemma4_%28E2B%29_Reinforcement_Learning_Sudoku_Game.ipynb), [Unsloth RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), and [TRL GRPO trainer](https://huggingface.co/docs/trl/v0.28.0/en/grpo_trainer).
