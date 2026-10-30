# Local inputs and public metadata

The table lists local input paths and public question metadata. Obtain clinical inputs separately through the appropriate data-access process. Preserve identifiers as strings, including leading zeros.

| Path | Required content | Access |
| --- | --- | --- |
| `resources/CIBMTR_Form_2400_R11_Items_simplified.csv` | 129 entries with Question Number, category, Question Text, and Options / Instructions | Public question metadata; source Form 2400 R11 attribution retained |
| `data/mimic_iv_bmt_initial_progression.csv` | `hadm_id`, `current_time`, `progression_prompt`, `assess_plan`; other reconstruction fields may be retained | Controlled clinical text |
| `data/osu_h_p_notes.csv` | `MRN`, `admit_date`, `note` | Institutional clinical text |
| `data/mimic_form_2400_answers.json` | Record/field identifiers and reference `answers` | Controlled source; answer-only export may be reviewed for publication |
| `data/osu_form_2400_all_extractions.json` | Institutional registry reference records | Controlled source; may contain quoted evidence and explanations |
| `fine_tuning/outputs/mimic_sft_fpc5_train.jsonl` | One `messages` array per conversation in system/user/assistant order | Planned PhysioNet dataset; includes clinical text |
| `fine_tuning/mimic_eval_record_ids.json` | JSON list of 20 reserved MIMIC admission identifiers | Obtain with controlled dataset/split manifest; not included in source archive |
| `data/research/` | Cohort, labels, progress notes, admission notes, and discharge notes as described in the research README | Institutional source data; never included in public source archive |
| `research_eval/research_questions.csv` | R01–R14, target columns, options, and source routing | Public question metadata |

MIMIC registry loading retains the first stored row per admission, then applies the reserved-ID selection. Institutional registry inference uses admission H&P records. Longitudinal research links progress notes by patient and transplant date and supplements by patient and admission date; an MRN-only fallback is not equivalent. The research `note_date` can be a day slot rather than a calendar date.

Annotation and prediction JSON can include evidence quotations and rationales alongside answers. Use the [answer-only exporter](publication.md#export-answers) before reviewing these records for release.
