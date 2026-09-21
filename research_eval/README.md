# Research extraction and evaluation

This workflow replaces the obsolete daily Ollama extraction and wide-table evaluator.
It extracts the same **14 research targets**, using a form-style question catalog and
the Gemma evaluation record format. It does not extract Form 2400 fields.

## Inputs and ground truth

Defaults resolve relative to the repository, regardless of the working directory.
The original source CSV files are never rewritten.

| File under `data/research` | Purpose |
| --- | --- |
| `target_filtered.csv` | Evaluation cohort: 191 unique MRN + BMT date encounters, with admission dates. All 14 target columns are blank, so this is **not labeled GT**. |
| `toci_ptcy_crs.csv` | Labeled GT: 346 unique MRN + BMT date encounters (336 MRNs). All 191 cohort encounters match. |
| `progress_notes.csv` | 1,025 source rows covering 174 cohort encounters; `note_date` is a day slot 0–5, not a calendar date. |
| `dc_summary_notes.csv` | Exact admission matches for all 191 cohort encounters; supplements the daily course and supplies the full neurotoxicity window. |
| `h_p_notes.csv` | Exact admission matches for 175 cohort encounters; supplies baseline context. |

GT and progress notes join on **MRN + BMT date**. H&P and discharge notes join on
**MRN + admission date** from the cohort. There is no MRN-only fallback to another
admission. Encounters lacking progress notes remain in the cohort and use available
supplements. Missing source types are recorded on every prediction.

The source has 14 progress rows with note-header/day-slot date conflicts. Exact
duplicate text with the same header date is included once; seven conflicts remain
among the retained notes. Prompts include the embedded header date, slot-derived
date, original text, and a factual conflict flag. Dates are not silently shifted.

## Reconstructed questions

`research_questions.csv` has the same first four columns as the simplified 2400
catalog: `Question Number`, `category`, `Question Text`, `Options / Instructions`.
It adds `target_column` and `note_sources`. IDs `R01`–`R14` are research-specific.

The default always covers **all 14 fields**, in bundles of **5, 5, and 4**.
There is no invented fifteenth question or duplicated padding question.

| Bundle | Questions | Source selection |
| --- | --- | --- |
| 1 | CRS, first/last fever dates, maximum temperature, hypotension | Progress and discharge |
| 2 | Vasopressor count, hypoxia, high-flow oxygen, BiPAP/intubation, neurotoxicity | Progress, discharge, H&P baseline |
| 3 | Tocilizumab use, first/last administration dates, total dose count | Progress and discharge |

The default `per_note` workflow reviews each complete source note independently.
It never combines multiple original notes in one extraction call or splits a note.
Within each bundle, a source receives only its relevant questions: for example,
H&P receives R06–R10 from bundle 2 and none from bundles 1 or 3. R05 is assessed
from progress notes and discharge summaries, not the admission H&P.

Each extraction returns the standard SFT answer records, including `answers`,
`evidence_from_text`, and `reasoning_trace`. Each question's **Options / Instructions**
specifies what supports its answer, which misleading observations to exclude, and
how to handle missing information. There is one extraction call per note/bundle
and one final extraction per bundle, with no judge or additional validation call.

| Information in the source | Requested answer | Requested quote |
| --- | --- | --- |
| Documented value or event | Value / `Y` | Exact quote supporting this specific field |
| Explicit absence/non-use | `N` / zero | Exact statement of absence/non-use |
| Explicit non-applicability of a date/temperature | `N/A`, e.g. fever date when no fever occurred | Statement establishing non-applicability |
| Requested fact is not documented | `not mentioned` | Empty string |

Missing dates/counts can be `not mentioned` even when the underlying event is
described. For example, R05 explicitly says that 'no fevers', transplant day, and
'no transplant reactions' do not establish CRS-associated hypotension or its absence.
The extraction model makes this semantic decision using the question instructions.
Code checks exact quote membership and requires quotes for extracted values.
For an explicit `not mentioned` answer, any stray quote is cleared without
guessing a different answer; the original response remains in the attempt log.
Legacy empty `N/A` responses are
normalized to `not mentioned`. Failed format/quote checks retry the extraction;
unresolved failures use `answers: null` and `extraction_success: false`.

Every new prediction also includes an explicit `extraction_status`:

| Result | `answers` | `extraction_status` | `extraction_success` |
| --- | --- | --- | --- |
| Documented answer | Extracted value, including explicit `N`, `0`, or applicable `N/A` | `extracted` | `true` |
| No supporting information found | `not mentioned` | `not_mentioned` | `true` |
| Parsing, generation, or quote-validation failure | `null` | `failed` | `false` |

Successful results have an empty `parse_error`. Failed results retain the error
and raw attempt in `research_failures.jsonl`. A malformed or null model answer
cannot be relabeled as `not mentioned`. Resume skips successful `not_mentioned`
results and retries failures. Cached note fields carry the same successful statuses.

These checks do not independently prove that an exact quote supports an answer;
clinical relevance remains the extraction model's responsibility and needs accuracy
evaluation. Source-matched quotes contribute their enclosing sentences/paragraphs
to the final pass, preserving negation and dates. Overlapping passages are merged.

The final call receives these **verbatim source excerpts**, labeled with their
source and note dates, as its `Clinical note`. Intermediate model answers and
rationales are not inserted as clinical facts. The final model answers the same
5/5/4 question bundles using the standard extraction prompt and the research field
instructions. H&P supplies baseline context for bundle 2; discharge supplies the wider course.

`not mentioned` contributes no excerpts and is never converted into a negative
clinical finding. Final-stage `not mentioned` is scoped to the selected excerpts,
not a claim about unreviewed chart content. Successful note extractions are saved
immediately and reused on resume; failed extractions fail the bundle. The workflow keeps
each original note intact in the first pass. A long enclosing passage remains
subject to the input limit instead of being truncated. Events not represented by
first-pass quotes can be missed in the final pass; this needs accuracy evaluation.

Individual progress notes have a median of 7,522 characters;
the largest discharge summary is 62,493 characters. These are text lengths, not
token counts. Some individual notes can still be too large for a small context.
No clinical label columns or per-patient research annotation comments are included
in prompts. Only cohort identifiers/admission dates and the actual notes are used.

### Fine-tuned model prompt compatibility

Use `--modes baseline` (the default) for the fine-tuned model. Both note review and
final extraction call the existing shared `build_user_prompt(..., mode="baseline")`,
which calls `build_sft_user_prompt`. The saved conversations in
`fine_tuning/outputs/mimic_sft_fpc5_train.jsonl` were generated with that same builder.

- The **system message is identical**: the imported `EXTRACTION_SYSTEM_PROMPT`.
  Its existing Form 2400 wording is retained to match training; research questions
  and answer options still come from `research_questions.csv`.
- The **question layout and ten output keys** retain the SFT structure. Research
  question definitions supply the field-specific evidence rules. In research prompts
  only, the generic missing-answer line uses `not mentioned`, reserving `N/A` for
  field-defined non-applicability. The shared training template and training data
  are unchanged. The record identifier remains MRN+BMT date.
- The first pass supplies one complete note. Code prepares selected source excerpts
  for the final pass. Neither pass adds special review, summarization, synthesis,
  or event-output instructions to the extraction prompt. Retries reuse the same
  extraction messages without a repair suffix.
- Study rules live in each question's existing `Options / Instructions` slot.
- Other prompt modes are explicit experiments. Their research missing-answer and
  quote wording follows the same rules; they are not silently activated.

Prompt/template alignment does not establish equal clinical performance: research
fields and excerpt-based inputs differ from the training data. No fine-tuned model
inference has been performed as part of this prompt update.

### Study definitions and reconstruction choices

The long annotation-column header in `toci_ptcy_crs.csv` contains both open questions
and affirmative study instructions. The catalog uses the affirmative instructions:

- Record the first fever episode; exclude another episode beginning more than
  three days after the preceding fever. Do not record an afebrile maximum.
- Flag CRS after day +5; do not automatically exclude later CRS.
- Exclude intubation unrelated to CRS and CPAP for obstructive sleep apnea.
- Neurotoxicity includes confusion during this admission at any point before discharge.
- Count only tocilizumab for CRS, excluding GVHD-directed treatment. Deduplicate
  repeated mentions, orders, and cumulative dose totals.

The old script's `>100.3 F` fever threshold is retained. For ambiguities not resolved
by the source header, the catalog explicitly uses the first documented fever episode,
keeps infectious attribution distinct, and requires support for CRS rather than
equating every fever with CRS. Maximum pressors means the maximum concurrent count.
These are visible reconstruction choices, not a newly validated study protocol.
`max_crp` and demographic/outcome columns are outside the original 14-field scope.

## Run

Offline validation and prompt inspection (no model downloads or inference):

```powershell
python research_eval/research_extract.py --dry_run --limit 0 --output_dir research_eval/results/audit
```

Gemma extraction and automatic scoring, starting with five encounters:

```powershell
python research_eval/research_extract.py --models e2b --limit 5 --output_dir research_eval/results/gemma_e2b
```

Full cohort, reviewing complete notes individually, followed by extraction from selected evidence:

```powershell
python research_eval/research_extract.py --models e2b --limit 0 --output_dir research_eval/results/gemma_e2b_per_note
```

Inference uses the **Transformers** loader and generation functions from
`prompt_eval/evaluate_gemma4_extraction.py`. Install PyTorch, Transformers, and
Accelerate in the inference environment; 4-bit loading also needs bitsandbytes,
and saved LoRA adapters need PEFT. Unsloth is not required for research inference.
Text-only inference uses the saved tokenizer and chat template, keeping system
and user messages as plain strings as in SFT. Baseline extraction instructions
remain identical to the SFT template.
Gemma aliases, Hugging Face model names, and saved fine-tuned adapter directories
are accepted. `--load_in_4bit` is optional. Data preparation and scoring use only
the Python standard library.

After training, run the saved adapter directory containing its processor/tokenizer
files (adjust the path to its actual location):

```powershell
python research_eval/research_extract.py --models fine_tuning/outputs/gemma4_12b_lora/final --load_in_4bit --limit 5 --output_dir research_eval/results/gemma_finetuned
```

Useful options:

- `--limit 0`: all 191 cohort encounters. The default is 5 encounters.
- `--fields_per_call 5`: five questions per call; the last bundle has four.
- `--modes baseline,no_reasoning`: run multiple prompt variants.
  Keep `baseline` alone for the SFT layout with research question instructions.
  Other modes, including few-shot examples, use the shared Gemma evaluation variants
  with the research missing-answer and exact-quote wording.
- `--thinking auto`: default; disables internal thinking for local adapter folders
  containing `adapter_config.json`, matching this project's training format.
  Base models use the shared evaluator's mode defaults. `--thinking on` and
  `--thinking off` explicitly override this without changing question instructions.
- `--note_strategy per_note`: default; independent complete-note reviews with standard
  evidence quotes, then a final extraction call for each question bundle. Intermediate
  reviews use at most five relevant questions. Arbitrary text splitting is removed.
- `--note_strategy full`: explicitly opt into the older combined-notes comparison.
  The same question instructions and quote checks apply. `non_json` and
  `no_evidence_field` are rejected because research extraction requires quotes.
- `--parse_retries 1`: retry malformed, incomplete, invalid-valued, or failed responses.
- `--max_input_tokens 24000`: check every note and final evidence input before generation.
  Uses the actual chat-template token count and reserves output space within both
  the configured sequence cap and the model's declared context. Oversized inputs fail visibly without
  truncation or repeat retries. Use a larger supported context/model for such notes.
- `--max_seq_length 32768`: safety cap on total input and output tokens.
  This does not extend the model's supported context length.
- `--max_new_tokens 8192`: generation allowance. The input budget applies to final evidence as well as original notes;
  an unusually large evidence collection fails rather than being silently trimmed.
- `--resume`: rerun the same command with this flag and the original output directory.
  Retry failed and unfinished bundles, reusing valid cached note reviews; skip
  complete successful bundles. The startup summary reports loaded results.
  A missing manifest is an error rather than silently starting from encounter 1.
  Starting at encounter 1 is expected if its first bundle failed.
  Changed inputs or generation settings reject resume. After a code fix, a run
  containing only failed predictions can resume with its old note cache cleared;
  the previous configuration is retained in the manifest's resume history. Runs
  containing successful predictions require a new output directory after a code
  change, so results from different implementations are not mixed.
- `--max_consecutive_failures 3`: stop after three consecutive failed bundles,
  retaining checkpoints and diagnostics. Set to `0` to disable the guard.
- `--save_raw_runs`: save all attempts, including successes. Failed attempts are
  always saved in `research_failures.jsonl`; both logs can contain clinical text.
- `--save_prompt_preview`: with `--dry_run`, save the first encounter's actual
  per-note prompts. By default a dry run writes only the audit and run manifest.
- `--data_dir`, `--cohort`, `--ground_truth`, `--questions`: override inputs explicitly.

## Outputs and scoring

- `data_audit.json`: input choice, note coverage, date conflicts, label counts, and
  the exact encounter/field identifiers of invalid or missing GT labels.
- `run_manifest.json`: selected encounters, planned variants/bundles, configuration,
  prompt-contract label, and hashes of input files and relevant code.
- `prompt_preview.json`: optional first-encounter prompts, with one intact note per
  evidence call under `per_note`; never a fabricated inference result.
- `note_evidence.json`: standard field IDs, answers, and evidence quotes saved after
  each successful extraction, with encounter/source/date/model/mode metadata. Reusable
  during resume; only original source passages selected by the quotes enter the
  final clinical-note input, not the cached answer values.
- `research_predictions.json`: atomic checkpoint, one record per encounter/field/variant;
  carries standard answer/evidence/rationale metadata plus research source coverage.
- `research_failures.jsonl`: failed attempts with stage, source, raw answer text,
  original decoded output, input/output token counts, thinking setting, and token-limit
  status when generation completed. This is written even without `--save_raw_runs`.
- `research_<UTC timestamp>_predictions.json`: final prediction snapshot.
- `research_evaluation_summary.json`, `research_evaluation_overall_metrics.csv`,
  `research_evaluation_field_metrics.csv`, `research_evaluation_details.csv`: overall,
  per-field, and per-answer scores and coverage.

Score a saved result without loading a model:

```powershell
python research_eval/research_eval.py --predictions research_eval/results/gemma_e2b/research_predictions.json
```

### Recovering the all-N/A parsing-failure run

In older results, `answers: "N/A"` together with `extraction_success: false` is a
failure placeholder, not a clinical finding. New failures use `answers: null`.
The updated question instructions change the run contract: start a **new output
directory** so results from previous prompts are not reused:

```powershell
python research_eval/research_extract.py --models fine_tuning/outputs/gemma4_12b_lora/final --limit 0 --output_dir research_eval/results/gemma_finetuned_questions_v2
```

Subsequent restarts of this new run use the same command with `--resume`.

The parser accepts complete JSON arrays, standard response wrappers, and separate
complete field objects (including JSONL or individually fenced records). It combines
records only when every requested field ID occurs exactly once. Bracketed prose
such as `[End of response]` is not itself a second JSON answer. Truncated containers,
duplicate/missing field IDs, null/invalid answers, and quotes absent from the source
are still rejected; the parser never salvages inner records from a broken array.
Empty final output is reported explicitly; unfinished thought text is never used
as a clinical answer. If failures persist, inspect the last failed attempt in
`research_failures.jsonl`; the prediction file alone cannot show what the model generated.

The evaluator detects the adjacent run manifest and evaluates **every expected
encounter and field in that run**, including missing/failed predictions and entire
missing variants. Without a manifest, it evaluates the entire supplied cohort.
Keep the manifest with exported subsets, or supply `--manifest` explicitly. A
standalone run with custom inputs must also supply `--cohort` and `--ground-truth`.
Legacy wide CSV predictions are accepted if they contain MRN and BMT date; the old
MRN-only merge is intentionally unsupported. Duplicate encounter/field predictions
within a variant are rejected. Extra-cohort predictions are counted separately.

Scoring rules:

- Normalize Y/N casing/whitespace, dates, and numeric representations.
- Blank fever date/temperature GT is a valid N/A target only when CRS is explicitly
  N. Blank tocilizumab date GT is valid N/A only when tocilizumab is N. Nonblank
  labels are preserved even if they conflict with a parent field.
- Other blank or invalid GT labels are excluded and reported, never converted to N.
  In the supplied cohort this excludes one `B` in the BiPAP Y/N field and one blank
  neurotoxicity label: **2,672 of 2,674 answers are scorable**.
- Missing/failed/invalid predictions count as wrong for scorable GT. A successfully
  returned N/A only matches an applicable empty target. Blank temperature GT is
  **not** a wildcard accepting any predicted temperature.
- `not mentioned` is a separate abstention, reported as `not_mentioned_predictions`.
  It counts as incorrect for scorable GT and does not match a negative, zero, or
  not-applicable GT value. It is not counted as a parsing failure. Summary and
  per-field metrics separately report `failed_predictions`.
- Report exact accuracy separately from tolerance accuracy (default ±1 day and
  ±0.1 F). Counts require exact matches. MAE uses paired valid values and reports
  the paired denominator. Missing values still lower overall accuracy.
- Report binary precision, sensitivity, specificity, and F1 for Y/N fields, plus
  prediction presence and valid-response coverage. Failed responses cannot earn
  true-negative credit. Undefined metrics are null.

## Verification

```powershell
python -m unittest research_eval.test_research -v
```

Tests use synthetic notes and mocked inference to check encounter isolation,
source routing, no GT leakage, date conflicts, duplicate handling, strict output
validation, quote checking, input limits, retry prompt stability, SFT layout with
research-specific missing-answer wording in both passes, single-call extraction,
source-excerpt selection, missing-answer penalties, tolerances,
legacy CSV input, full extraction-to-scoring execution, and resume protection.
The provided data was audited with a full-cohort dry run. Actual model inference
and clinical extraction accuracy must be measured in the intended model runtime.
