# Dataset construction and annotation

These scripts construct MIMIC-IV-derived admission and daily progress notes and prepare Form 2400 reference annotations.

Run commands from the repository root. Input and output paths are relative to the working directory. Obtain the MIMIC source files separately through credentialed access.

| Script | Purpose |
| --- | --- |
| `mimic_iv_dataset_creation.py` | Shared processing of discharge summaries, admissions, pharmacy orders, medication administration details, radiology, microbiology, and laboratories into admission and progression summaries. Also provides a general dataset-construction CLI. |
| `mimic_iv_transplant_dataset_creation.py` | Transplant keyword screening and summary construction using the shared processing module. The optional `--optimize` step applies medication-target filtering and model-based admission screening. |
| `mimic_iv_transplant_dataset_only_hct_admission.py` | Additional medication-target filtering and GPT-4o screening for initial transplant admissions. |
| `mimic_iv_transplant_reasoning_dataset_creation.py` | Optional generation of medication-specific reasoning examples from the reconstructed notes and their assessment and plan. This is separate from CIBMTR registry-answer annotation. |
| `process_bmt_notes.py` | Gemini-assisted Form 2400 annotation of reconstructed MIMIC notes, with question sampling based on previous answers. |
| `extract_hp_notes.py` | Gemini-assisted Form 2400 annotation of institutional H&P notes, supplemented by reference-table values. |

The transplant builder imports `mimic_iv_dataset_creation.py` from this directory. Keep these files together and invoke the scripts directly as shown below.

## Inputs and dependencies

The default raw-data directory is `data/MIMIC-IV`. The shared processor expects `discharge.csv`, `admissions.csv`, `d_labitems.csv`, and `labevents.csv` (or an existing `labs_with_labels.csv`); it also reads `pharmacy.csv`, `emar_detail.csv`, `radiology.csv`, and `microbiologyevents.csv` when present. Input filenames and schemas follow the original scripts.

Python dependencies are `pandas`, `numpy`, `openai`, and `python-dotenv`. The optional reasoning script also imports `google-genai`, `matplotlib`, and `seaborn`. The shared general builder only requires `pandas` beyond the Python standard library. For API-based screening, the original scripts read `OAI_API_KEY`; Gemini reasoning reads `GEMINI_API_KEY`. The reasoning script initializes its provider client when run.

## Example workflow

Construct transplant summaries from locally available source files:

```powershell
python dataset_creation/mimic_iv_transplant_dataset_creation.py --data_dir data/MIMIC-IV --range all --output_dir results/mimic_iv_transplant_dataset
```

Filter the resulting dataset to initial transplant admissions:

```powershell
python dataset_creation/mimic_iv_transplant_dataset_only_hct_admission.py --data_dir results/mimic_iv_transplant_dataset --output_dir results/mimic_iv_initial_transplant_dataset
```

The second command uses the screening API. Reproducing the study cohort also requires the source-data release and final encounter-selection manifest. The annotation scripts below prepare reference answers; `prompt_eval/` contains model inference and evaluation.

The reconstructed notes reuse discharge-derived context. They are retrospective summaries and should be distinguished from contemporaneous admission or progress notes.

## Reference annotation

Run annotation commands from the repository root so the default `data/` and `resources/` paths resolve correctly:

```bash
python dataset_creation/process_bmt_notes.py --help
python dataset_creation/extract_hp_notes.py --help
```

Both scripts require `google-genai` and `GEMINI_API_KEY` for annotation. They send the supplied notes and reference information to Gemini; use only inputs authorized for that service. Help commands do not make API requests.

`process_bmt_notes.py` reads `data/bmt_initial_progression_notes_all.csv`, the public question catalog, and previous answers from `data/bmt_form_2400_answers.json` by default. It samples ten questions per note, targeting eight previously nonempty answers. Use `--previous_answers`, `--questions_per_note`, and `--non_empty_per_note` to change the sampling inputs and settings.

`extract_hp_notes.py` reads `data/h_p_notes.csv`, `data/final_merged.csv`, and the public question catalog by default. Use `--notes`, `--reference`, and `--questions` for alternative paths. Its reference matching selects the closest transplant date among rows with the same MRN.

Outputs include evidence and model explanations and belong in private storage. Both scripts retain their original command-line options and default input/output paths.
