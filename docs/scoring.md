# Registry response and scoring policies

The manuscript defines registry valid-answer agreement as correct, successfully processed answers divided by all targeted reference answers. Failed and missing responses count as incorrect.

## New inference runs

`registry_response_v2` is shared by the Gemma, Qwen, GPT-OSS, Unsloth GPT-OSS, and hosted-comparator backends. Fields bind by explicit identifier or an unambiguous field name; list position is not used to guess a field. Unknown/conflicting identifiers and duplicates are rejected. An omitted answer remains `null` with `extraction_success=false`. Incomplete bundles use the configured retry; unresolved bundles return failed records, never successful default `N/A` values. Metadata is copied from the requested catalog.

An explicit, successfully returned `N/A` is a valid response. A missing `answers` key, null/blank answer, empty option list, or unsupported answer structure is not. Failed bundles are distinct from clinical absence. A concise rationale is a saved explanation, not a verified internal reasoning trace.

Resume accepts only outputs carrying the same response contract. Use a new output directory for runs from older parser versions. This prevents mixing historical default-filled answers with newly validated responses. Duplicate record/field predictions in a scored variant are rejected by the default policy rather than silently replacing an earlier answer.

## Saved-output scoring

`score_predictions.py` defaults to `--scoring_policy valid-response-v2`. It denies correctness credit to parse failures, explicit failed statuses, missing answer keys, and null/invalid answers. `valid_answer_accuracy` and `strict_exact_accuracy` use all targeted labels as their denominator; the compatibility field `exact_accuracy` uses covered labels. Use an explicit `--record_ids` cohort for both datasets when absent entire records must be penalized. `--all_records` instead evaluates records present in the prediction file. `--scope predicted-fields` does not penalize omitted entire fields and should be used only for a documented target subset.

`--scoring_policy legacy-answer-only` applies the answer matcher without requiring successful response validation. Use it for comparisons with historical scores. Summaries identify the selected policy; both policies use the same normalization and field-specific equivalence rules.

Old parsed outputs sometimes already contain `N/A` inserted by the historical parser. If neither raw-response completeness nor a failure flag was saved, the new scorer cannot distinguish those values from genuinely returned `N/A`. Use the corresponding raw generation audit or a new inference run; do not claim that rescoring alone reconstructs missing provenance. New inference outputs explicitly record the response contract.

Each scoring summary records the policy and hashes of predictions, reference answers, and scorer code. Scoring writes new files; historical result artifacts are not automatically regenerated. Correcting response handling can change performance estimates and requires an explicitly versioned analysis.

## Research task

The 14-target research evaluator has separate definitions and missing-data rules. Successful `not mentioned` answers score as negative, zero, or not applicable according to field type; failed answers receive no credit. Blank fever labels remain unlabeled. Report date/temperature tolerance and positive-event sensitivity separately from exact agreement. The [research documentation](../research_eval/README.md) describes synthesis and label review.
