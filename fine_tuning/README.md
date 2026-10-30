# Supervised fine-tuning

BMT-GRAFT-12B adapts Gemma 4 12B using LoRA supervised fine-tuning. The released model is available at [JWang580/BMT-GRAFT-12B](https://huggingface.co/JWang580/BMT-GRAFT-12B). Training conversations and the admission split will be distributed separately through PhysioNet.

To run the released model, follow the [inference guide](../docs/running_the_model.md). The instructions below reproduce training.

Place the authorized training JSONL at `fine_tuning/outputs/mimic_sft_fpc5_train.jsonl` and the 20 reserved admission identifiers at `fine_tuning/mimic_eval_record_ids.json`, or supply explicit paths. Both are ignored by Git and excluded from the source archive. Each example has a `messages` array in system/user/assistant order. The user message contains the clinical note and questions; the assistant contains a JSON array of fixed reference answers, evidence quotations, and brief explanations. These conversations contain clinical text even if the source data are deidentified.

```bash
python fine_tuning/train_gemma4_12b_unsloth.py --dry_run
python fine_tuning/train_gemma4_12b_unsloth.py --report_to none
```

The dry run validates the 90-record training set, 2,340 examples, 20-record reserved list, and admission-level disjointness without importing Unsloth. Actual training requires the CUDA/Unsloth stack. The script uses assistant-response-only loss, LoRA rank/alpha 16, no dropout, one epoch, 16,384-token sequences, learning rate 2e-5, and gradient accumulation four with per-device batch one. The saved final checkpoint recorded 585 optimization steps. See [the reproducibility guide](../docs/reproducibility.md) for recorded software versions and limitations of the historical requirements.

Validation uses the reserved records after training. `--skip_eval` omits that step. The reserved records are not passed as an SFT trainer evaluation dataset. Training outputs and validation logs stay under ignored `fine_tuning/outputs/`; external telemetry is off by default and requires explicit `--report_to wandb`.

The [RLVR experiment](README-rlvr.md) starts from the unadapted backbone. Use a separate environment and output directory for that workflow.
