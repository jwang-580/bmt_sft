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
| 1 | CRS, first/last fever dates, maximum temperature, hypotension | Progress and discharge; H&P added for hypotension baseline |
| 2 | Vasopressor count, hypoxia, high-flow oxygen, BiPAP/intubation, neurotoxicity | Progress, discharge, H&P baseline |
| 3 | Tocilizumab use, first/last administration dates, total dose count | Progress and discharge |

The default `per_note` workflow reviews each complete source note independently.
It never combines multiple original notes in one extraction call or splits a note.
Within each bundle, a source receives only its relevant questions: for example,
H&P receives R05 from bundle 1, R06–R10 from bundle 2, and none from bundle 3.

Each review returns the standard SFT answer records, including `answers`,
`evidence_from_text`, and `reasoning_trace`. There is no custom `events` schema.
Code verifies each nonempty quote against the original note and retrieves its
containing source sentence/paragraph, preserving negation and dates in that
passage. Duplicate or overlapping excerpts within a note are merged. If the same
quote occurs in multiple passages, all matching passages remain available.

The final call receives these **verbatim source excerpts**, labeled with their
source and note dates, as its `Clinical note`. Intermediate model answers and
rationales are not inserted as clinical facts. The final model answers the same
5/5/4 question bundles using the standard extraction prompt and the research field
instructions. H&P supplies baseline context; discharge supplies the wider course.

Empty evidence strings are valid under the training format and contribute no
excerpts. A missing quote is not converted into a negative clinical finding.
Successful note reviews are saved immediately and reused on resume; failed reviews
fail the bundle. This uses more calls than concatenating the admission, but keeps
each original note intact in the first pass. A long enclosing passage remains
subject to the input limit instead of being truncated. Events not represented by
first-pass quotes can be missed in the final pass; this needs accuracy evaluation.

The supplied data has a median combined bundle-1 note length of 76,651 characters
and a maximum of 311,777. Individual progress notes have a median of 7,522 characters;
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
- The **user-message template, question layout, fixed rules, and ten output keys
  are identical** to SFT. Only variable record metadata, clinical text, and research
  question definitions change. The record identifier correctly remains MRN+BMT date.
- The first pass supplies one complete note. Code prepares selected source excerpts
  for the final pass. Neither pass adds special review, summarization, synthesis,
  or event-output instructions. Retries reuse the same messages without a repair suffix.
- Study rules live in each question's existing `Options / Instructions` slot.
- Other prompt modes are explicit experiments, not the training prompt. The
  shared evaluation variants are reused as-is; they are not silently activated.

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

The Transformers backend imports the existing Gemma loader/generator from
`prompt_eval/evaluate_gemma4_extraction.py`, so it uses that script's installed
PyTorch/Transformers/Accelerate environment. Gemma aliases and Hugging Face model
names are accepted. `--load_in_4bit` is optional. Data preparation and scoring use
only the Python standard library; no pandas or sklearn installation is required.

Optional Ollama backend (requires its Python package and a running local server):

```powershell
python research_eval/research_extract.py --backend ollama --models YOUR_INSTALLED_MODEL --limit 5 --output_dir research_eval/results/ollama
```

Useful options:

- `--limit 0`: all 191 cohort encounters. The default is 5 encounters.
- `--fields_per_call 5`: five questions per call; the last bundle has four.
- `--modes baseline,no_reasoning`: run multiple prompt variants.
  Keep `baseline` alone for the exact SFT prompt. Other modes, including few-shot
  examples, use the shared Gemma evaluation variants without research-specific rewrites.
  Gemma's Transformers backend also switches internal thinking with the prompt mode;
  Ollama's internal thinking behavior remains controlled by its model/server defaults.
- `--note_strategy per_note`: default; independent complete-note reviews with standard
  evidence quotes, then a final extraction call for each question bundle. Intermediate
  reviews use at most five relevant questions. Arbitrary text splitting is removed.
- `--note_strategy full`: explicitly opt into the older combined-notes comparison.
  `non_json` and `no_evidence_field` require `full`; per-note review needs the
  standard `evidence_from_text` output field to select source excerpts.
- `--parse_retries 1`: retry malformed, incomplete, invalid-valued, or failed responses.
- `--max_input_tokens 24000`: check every note and final evidence input before generation.
  Transformers uses the actual chat-template token count and also reserves output
  space within the model's declared context. Oversized inputs fail visibly without
  truncation or repeat retries. Use a larger supported context/model for such notes.
  Ollama lacks a tokenizer endpoint in this wrapper, so it uses a conservative UTF-8
  byte estimate plus template allowance; this can reject notes that might fit.
- `--max_new_tokens 8192`: generation allowance. For Ollama, `--ollama-num-ctx`
  defaults to 65536. The input budget applies to final evidence as well as original notes;
  an unusually large evidence collection fails rather than being silently trimmed.
- `--resume`: rerun the same command with this flag. Complete successful bundles are
  skipped; failed bundles reuse successful note reviews and retry failed work. Changed inputs, question definitions, runtime
  wrapper code, or generation settings reject resume. Use a new output directory
  for a changed experiment. A dry run can be followed by a real run in its directory.
- `--save_raw_runs`: opt-in raw model responses. These contain clinical text.
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
  each successful review with encounter/source/date/model/mode metadata. Reusable
  during resume; only original source passages selected by the quotes enter the
  final clinical-note input, not the cached answer values.
- `research_predictions.json`: atomic checkpoint, one record per encounter/field/variant;
  carries standard answer/evidence/rationale metadata plus research source coverage.
- `research_<UTC timestamp>_predictions.json`: final prediction snapshot.
- `research_evaluation_summary.json`, `research_evaluation_overall_metrics.csv`,
  `research_evaluation_field_metrics.csv`, `research_evaluation_details.csv`: overall,
  per-field, and per-answer scores and coverage.

Score a saved result without loading a model:

```powershell
python research_eval/research_eval.py --predictions research_eval/results/gemma_e2b/research_predictions.json
```

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
validation, quote checking, input limits, retry prompt stability, exact SFT template
equality in both passes, source-excerpt selection, missing-answer penalties, tolerances,
legacy CSV input, full extraction-to-scoring execution, and resume protection.
The provided data was audited with a full-cohort dry run. Actual model inference
and clinical extraction accuracy must be measured in the intended model runtime.
