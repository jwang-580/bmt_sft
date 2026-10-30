# Research extraction and evaluation

This workflow extracts 14 longitudinal transplant research targets from clinical
notes and evaluates encounter-level answers. The released model is
[JWang580/BMT-GRAFT-12B](https://huggingface.co/JWang580/BMT-GRAFT-12B).
Clinical inputs, cached findings, evidence quotations, and full results remain
private. Reviewed answers can be shared using the
[answer-only exporter](../docs/publication.md#export-answers).

## Reference labels and archived results

The archived `gemma_finetuned` results use manual final synthesis of cached daily
findings, completed October 5, 2026, without rerunning inference. The canonical
`research_predictions.json` and `run_manifest.json` record this synthesis.
Earlier aggregate scores are retained under `model_final_pass_before_manual_synthesis`.

An error audit reviewed all 459 exact mismatches against daily findings, source
notes, question scope, and annotation comments. The audit distinguished
first-pass errors, missing reviews, synthesis errors, and unresolved source/label
disagreements. Review ledgers are retained with the private analysis records.

The `gemma_finetuned` metrics use the versioned
`data/research/toci_ptcy_crs.csv`: 59 source-supported values corrected,
135 unverifiable/conflicting fields excluded, and three already-tolerated rounding
differences retained. The run manifest records this label file, so the evaluator
CLI selects it automatically unless `--ground-truth` is supplied. Exact
accuracy is 88.1% (2,002/2,273), and tolerance accuracy is 88.8% (2,018/2,273).
These are extraction-informed label revisions, not an independent model benchmark.
Original-label scores (80.9% / 82.4%) are preserved in
`gemma_finetuned/labels_before_adjudication`, with field-level revisions recorded
in the private label-review ledger.

## Scoring policy

Scoring treats a successful `not mentioned` prediction as the field's negative
value: `N` for yes/no questions, zero for counts, and `N/A` for dates/temperature.
Raw answers and extraction statuses remain unchanged; evaluation details include
the effective normalized prediction, scoring status, and conversion flag. Failed,
invalid, and missing predictions remain incorrect. `valid_prediction_coverage`
measures actual extracted values; `scoring_prediction_coverage` also includes
successful `not mentioned` answers converted by this policy.

Blank fever onset, last-fever, and maximum-temperature labels are **unlabeled**,
regardless of CRS status, and are excluded. An explicit `N/A` remains scorable as
not applicable. Negative CRS does not rule out infectious or other fever. Blank
tocilizumab dates remain not applicable when `toci_y_n=N`.

Each evaluation writes a separate `research_evaluation_manifest.json` recording
the policy and hashes of the scoring inputs/code. For `gemma_finetuned`,
`run_manifest.json` describes the canonical manual synthesis;
`source_generation_manifest.json` preserves the original model-run metadata.

Inspect note-to-final aggregation reversions in an earlier excerpt-based run:

```powershell
python research_eval/analyze_aggregation_losses.py --run-dir path/to/historical_run --output-dir outputs/aggregation_losses
```

This audits runs using the excerpt-based final pass. It emits
CSV/JSON cases and reconstructed prompts using exact note-cache fingerprints,
checking quote retention, lost CRS context, label agreement and final rationales.
New structured-integration runs have a different final-input contract. The
`gemma_finetuned` folder contains manual final answers; analysis of the earlier
model final pass requires the corresponding archived run.

## Inputs and ground truth

Defaults resolve relative to the repository. The active reference-label table is
`toci_ptcy_crs.csv`; archived scores may use earlier label versions.

| File under `data/research` | Purpose |
| --- | --- |
| `target_filtered.csv` | Evaluation cohort: 191 unique MRN + BMT date encounters, with admission dates. All 14 target columns are blank, so this is **not labeled GT**. |
| `toci_ptcy_crs.csv` | Canonical corrected label table: 346 unique MRN + BMT date encounters (336 MRNs), including all 191 cohort encounters. Used by default and by the saved `gemma_finetuned` run manifest. Unverified values are blank/unscorable. |
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

The default covers all 14 fields in bundles of 5, 5, and 4.

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
and one final integration call per bundle, with no judge or additional model call.
The v3 question catalog uses concise instructions with the shared SFT system
prompt, response layout, and five-question bundles. Repetition was observed with
earlier prompts; the revised prompts require model evaluation before concluding
that this failure mode has been resolved.

| Information in the source | Requested answer | Requested quote |
| --- | --- | --- |
| Documented value or event | Value / `Y` | Exact quote supporting this specific field |
| Explicit absence/non-use | `N` / zero | Exact statement of absence/non-use |
| Explicit non-applicability of a date/temperature | `N/A`, e.g. no fever throughout the admission | Statement establishing non-applicability for that period |
| Requested fact is not documented | `not mentioned` | Empty string |

Missing dates/counts can be `not mentioned` even when the underlying event is
described. For example, R05 explicitly says that 'no fevers', transplant day, and
'no transplant reactions' do not establish CRS-associated hypotension or its absence.
Daily no-fever statements and a normal current temperature do not establish an
afebrile admission. R04 requests the highest value across the first fever episode,
not its first measurement; episodes beginning more than three days later are excluded.
The extraction model makes these semantic decisions using the question instructions.
Code requires source-matched quotes for extracted values. Matching tolerates
whitespace, typographic quotes/dashes, and a standalone dash separator between
words. Accepted quotes are restored to the exact source wording. It does not use
fuzzy matching or tolerate changed words, negation, numbers, signs or numeric ranges.
Explicit Fahrenheit suffixes such as `100.4F` are normalized to `100.4`; arbitrary
prose, ranges and Celsius suffixes are not silently treated as Fahrenheit.
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

First-pass quote checks establish source matching; clinical attribution is assessed
by that first extraction. The final call receives **structured daily findings JSON**:
field answers, extraction statuses, rationales, note IDs, source types, note dates,
and date-conflict metadata. Original notes, selected passages, and evidence quotes
are not sent again. Excerpt selection is not used in the current per-note path.

The final model integrates the same 5/5/4 bundles with a dedicated integration
prompt/system message. It preserves first-pass clinical attribution, reconciles
chronology and conflicts, retains a positive event despite later recovery or
non-mention, selects fever-episode dates/maximum temperature, takes maximum
concurrent pressors, and deduplicates tocilizumab doses across repeated findings.
H&P establishes baseline; discharge findings can reconcile the wider course.
Unresolvable totals or dates remain `not mentioned`; the scorer separately applies
the requested negative-scoring policy.

Integration outputs include `supporting_note_ids`. Code checks that each cited
note contains an extracted finding for that field, then restores/deduplicates its
already validated quotes into the saved `evidence_from_text`. The model does not
generate new quotes in the final pass. Successful note extractions, including
rationales, are saved immediately and reused on resume; failures fail the bundle.
No original note is split or truncated. Integration still needs model evaluation;
it cannot recover a fact missed by all first-pass findings.

Individual progress notes have a median of 7,522 characters;
the largest discharge summary is 62,493 characters. These are text lengths, not
token counts. Some individual notes can still be too large for a small context.
No clinical label columns or per-patient research annotation comments are included
in prompts. Only cohort identifiers/admission dates and the actual notes are used.

### Fine-tuned model prompt compatibility

Use `--modes baseline` (the default) for the fine-tuned model. Note review calls
the existing shared `build_user_prompt(..., mode="baseline")`,
which calls `build_sft_user_prompt`. The saved conversations in
`fine_tuning/outputs/mimic_sft_fpc5_train.jsonl` were generated with that same builder.

- The **note-review system message is identical**: the imported `EXTRACTION_SYSTEM_PROMPT`.
  Its existing Form 2400 wording is retained to match training; research questions
  and answer options still come from `research_questions.csv`.
- The **question layout and ten output keys** retain the SFT structure. Research
  question definitions supply the field-specific evidence rules. In research prompts
  only, the generic missing-answer line uses `not mentioned`, reserving `N/A` for
  field-defined non-applicability. The shared training template and training data
  are unchanged. The record identifier remains MRN+BMT date.
- The first pass supplies one complete note. The final pass uses the dedicated
  `structured_findings_integration_v1` contract and system message, with JSON
  findings and integration rules. Its purpose and input differ from note extraction.
  Retries reuse the same stage-specific messages without a repair suffix.
- Study rules live in each question's existing `Options / Instructions` slot.
- Other prompt modes are explicit experiments. Their research missing-answer and
  quote wording follows the same rules; they are not silently activated.

Research fields and structured-integration inputs differ from the training data.
The revised prompt and integration procedure have not been validated by a new
fine-tuned model run.

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
These rules operationalize the available annotation instructions; they have not
been independently validated as a study protocol.
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

Full cohort, with individual note extraction followed by structured integration:

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
  evidence quotes, then a final JSON integration call for each question bundle. Intermediate
  reviews use at most five relevant questions. Arbitrary text splitting is removed.
- `--note_strategy full`: explicitly opt into the older combined-notes comparison.
  The same question instructions and quote checks apply. `non_json` and
  `no_evidence_field` are rejected because research extraction requires quotes.
- `--parse_retries 1`: retry malformed, incomplete, invalid-valued, or failed responses.
- `--max_input_tokens 24000`: check every note and final findings input before generation.
  Uses the actual chat-template token count and reserves output space within both
  the configured sequence cap and the model's declared context. Oversized inputs fail visibly without
  truncation or repeat retries. Use a larger supported context/model for such notes.
- `--max_seq_length 32768`: safety cap on total input and output tokens.
  This does not extend the model's supported context length.
- `--max_new_tokens 8192`: generation allowance. The input budget applies to final JSON findings as well as original notes;
  an unusually large findings collection fails rather than being silently trimmed.
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
  `encounter_note_coverage` adds per-encounter source counts, missing sources and
  first/last progress-note dates. `labeled_value_candidate_gaps` flags nonnegative
  labeled facts (positive events, nonzero counts, dates and temperatures) without
  candidate terms in the question's available sources. This is keyword screening,
  not evidence verification or a GT correction; missing terms can reflect missing
  notes or different wording. Neither labels nor audit flags enter model prompts.
- `run_manifest.json`: selected encounters, planned variants/bundles, configuration,
  prompt-contract label, and hashes of input files and relevant code.
- `prompt_preview.json`: optional first-encounter prompts, with one intact note per
  evidence call under `per_note`.
- `note_evidence.json`: standard field IDs, answers, rationales, and evidence quotes saved after
  each successful extraction, with encounter/source/date/model/mode metadata. Reusable
  during resume. Answers/statuses/rationales and source dates enter the final
  findings JSON. Quotes remain local provenance and are not sent to integration.
- `research_predictions.json`: atomic checkpoint, one record per encounter/field/variant;
  carries standard answer/evidence/rationale metadata plus research source coverage,
  final-pass contract, and contributing note IDs for per-note integration.
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

### Failed responses and reruns

In older results, `answers: "N/A"` together with `extraction_success: false` is a
failure placeholder, not a clinical finding. New failures use `answers: null`.
Use a new output directory after changing question instructions so cached
results from earlier prompts are not reused:

```powershell
python research_eval/research_extract.py --models fine_tuning/outputs/gemma4_12b_lora/final --limit 0 --output_dir research_eval/results/gemma_finetuned_questions_v3
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
- Blank fever date/temperature GT is unlabeled regardless of CRS status. Explicit
  N/A date/temperature labels are not applicable. Blank tocilizumab date GT is
  not applicable when tocilizumab is N. Nonblank values are preserved even if
  they conflict with a parent field.
- Other blank or invalid GT labels are excluded and reported, never converted to N.
  In the supplied cohort this excludes one `B` in the BiPAP Y/N field and one blank
  neurotoxicity label, plus 264 blank fever labels: **2,408 of 2,674 answers are scorable**.
- Missing/failed/invalid predictions count as wrong for scorable GT. A successfully
  returned N/A only matches an applicable empty target. Blank temperature GT is
  **not** a wildcard accepting any predicted temperature.
- Successful `not mentioned` is retained in `not_mentioned_predictions`, but scores
  as N for Y/N, zero for counts, and N/A for dates/temperature. It matches negative,
  zero, or not-applicable GT and is incorrect for a positive value. Summary and
  per-field metrics report `not_mentioned_as_no_correct` and `failed_predictions`.
- Report exact accuracy separately from tolerance accuracy (default ±1 day and
  ±0.1 F). Counts require exact matches. MAE uses paired scoring values and reports
  the paired denominator, including not-mentioned counts scored as zero. Missing
  predictions still lower overall accuracy.
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
structured integration and provenance, historical excerpt audits, missing-answer penalties, tolerances,
legacy CSV input, full extraction-to-scoring execution, and resume protection.
The provided data was audited with a full-cohort dry run. Actual model inference
and clinical extraction accuracy must be measured in the intended model runtime.
