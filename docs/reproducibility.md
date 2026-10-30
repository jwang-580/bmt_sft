# Reproducibility record

Use a fixed code release and dataset version for each experiment. Record the analysis population, endpoint, reference-label version, and scoring policy alongside the results.

Record the following with each run:

- Code commit or source-archive manifest; dataset version, hashes, and admission split.
- Model and base-model repositories and revisions; adapter identity and tokenizer.
- Prompt version, question-catalog hash, fields per call, note strategy, and input limits.
- Sampling, reasoning, retry, and output-token settings.
- Software versions, GPU model/count/memory, precision, quantization, and timing boundaries.

Keep prompts, raw responses, and note-bearing logs in private result directories.

The SFT configuration uses 90 notes, 129 questions in bundles of five, 2,340 conversations, one epoch, rank/alpha 16, 16,384-token sequences, learning rate 2e-5, per-device batch one, and four accumulation steps. Twenty admissions are reserved for evaluation. The split is checked at admission level; verifying patient-level separation requires subject linkage across admissions.

Archived training artifacts record PyTorch 2.11.0, Transformers 5.14.1, TRL 1.0.0, Datasets 5.0.0, Tokenizers 0.22.2, and PEFT 0.18.1. The requirements files also cover development environments and do not provide a complete lockfile for the original run. Unsloth, driver, hardware, and checkpoint details are incomplete in the archived environment record.

For registry evaluation, specify the cohort and field set so missing predictions remain in the denominator. Record whether scoring used `valid-response-v2` or `legacy-answer-only`; estimates from different policies are not interchangeable. For research, record whether final synthesis was automated, assisted, or manually integrated from cached findings, and retain both original and revised label versions.

The manuscript specifies 10,000 percentile bootstrap resamples clustered by institutional patient or MIMIC admission, with paired comparisons using the same sampled units. The analysis uses seed 20261014. Include positive-event denominators for sensitivity and report date/temperature tolerance separately from exact agreement. Comparisons selected using the same evaluation cohorts are exploratory.

Share reviewed answer-only data and aggregate results. The source manifest records hashes of public files; private notes and identifier mappings are excluded. Clinical training data and split information are distributed separately under the dataset's access conditions.
