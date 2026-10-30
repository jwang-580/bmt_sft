# Registry extraction and evaluation

## Run BMT-GRAFT-12B

For extraction without ground-truth labels, use the dedicated runner:

```bash
python prompt_eval/run_finetuned.py --demo --dry_run
python prompt_eval/run_finetuned.py --demo --output prompt_eval/results/demo_predictions.json
python prompt_eval/run_finetuned.py --note_file data/my_note.txt --all_fields --output prompt_eval/results/my_note_predictions.json
python prompt_eval/run_finetuned.py --notes_csv data/my_notes.csv --all_fields --output prompt_eval/results/batch_predictions.json
```

The default demonstration extracts fields 32, 33, 53, 100, and 162 in one five-field call. `--all_fields` extracts all 129 catalog questions. For CSV input, only a `note` column is required; `record_id` and `date` are optional. The runner detects adapter versus merged checkpoints, loads any required base weights, and saves validated JSON and a companion run manifest. It rejects an existing output path and does not silently truncate long notes.

See [running the model](../docs/running_the_model.md) for the full installation and deployment instructions. The evaluation commands below serve the separate task of reproducing the manuscript experiments.

## Reproduce registry evaluation

Use the shared `extraction_prompts.py` prompts and the 129-question catalog under `resources/`. Local evaluation supports Gemma, Qwen, and GPT-OSS; `evaluate_api_extraction.py` provides optional hosted comparators. Hosted services receive the supplied clinical text and require separate data authorization.

Install `requirements-local.txt` for local inference. Model weights are available at [JWang580/BMT-GRAFT-12B](https://huggingface.co/JWang580/BMT-GRAFT-12B); select that repository or a compatible downloaded checkpoint through `--models`. Training data will be released separately through PhysioNet.

Run from the repository root. With authorized input files installed, inspect a configuration without loading a model:

```bash
python prompt_eval/evaluate_gemma4_extraction.py --dataset mimic \
  --models JWang580/BMT-GRAFT-12B --modes baseline --fields_per_call 5 \
  --all_fields --note_strategy full --limit 20 --dry_run
```

The example's `--limit 20` selects the first 20 loaded notes; it is **not** the held-out admission split. For the fixed MIMIC validation set use the training script's held-out validation or prepare an input restricted by the controlled `mimic_eval_record_ids.json`. Never substitute the first 20 source rows for that manifest. Dry runs print prompt previews and can expose note text in local logs; use synthetic inputs for shared demonstrations.

Request sizes are `1,5,10,25,50,100`. Modes are `baseline`, `non_json`, `no_reasoning`, `few_shot`, `no_reasoning_trace_field`, and `no_evidence_field`. The last two distinguish omitting a visible response field from disabling model thinking. Use `--all_fields` for all 129 questions or an explicit `--field_limit` for a documented subset; the default field subset can depend on the largest requested bundle. Use `--note_strategy full` to avoid also running chunked baseline variants. Inspect each model script's `--help` for sampling, device, quantization, and token-budget options.

New runs use `registry_response_v2`: omitted answers are failures, ambiguous bindings are rejected, and incomplete bundles use the parse retry. See [scoring policy](../docs/scoring.md). Inference output, evidence, raw-run logs, and explanations remain private even when answers are suitable for release.

Score saved outputs without a GPU:

```bash
python prompt_eval/score_predictions.py prompt_eval/results/private_predictions.json \
  --dataset mimic --record_ids fine_tuning/mimic_eval_record_ids.json \
  --scoring_policy valid-response-v2
```

For institutional evaluation pass the intended cohort using `--record_ids`; `--all_records` scores only records appearing in the output. Supply `--gold_json` and `--questions` for nondefault inputs. Summaries record hashes and scoring policy. Keep per-record/error tables private until processed by `scripts/export_public_answers.py`; those tables can include source identifiers and clinical answer text.
