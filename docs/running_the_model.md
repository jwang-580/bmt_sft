# Run BMT-GRAFT-12B

Use [JWang580/BMT-GRAFT-12B](https://huggingface.co/JWang580/BMT-GRAFT-12B) to extract registry fields from your own local notes. You do not need the PhysioNet training dataset, reference answers, Unsloth, or an API key for a hosted language model to run inference. The dedicated `prompt_eval/run_finetuned.py` runner accepts a text file or a CSV and uses the same registry prompt builder and response validation as the repository evaluation code.

## 1 Set up a local environment

Run commands from the repository root. Use Python 3.12 and a compatible PyTorch/CUDA installation. A CUDA-capable GPU with enough memory is the practical starting point; CPU loading uses float32 and requires considerably more host memory and time.

Create an environment on Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Or in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch build appropriate for your GPU and operating system using the [official installation selector](https://pytorch.org/get-started/locally/), then install the local inference dependencies:

```bash
python -m pip install -r prompt_eval/requirements-local.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The local dependency file requires Transformers 5.14.1 or newer and PEFT 0.18.1 or newer within their listed major-version bounds. Older Transformers installations may not recognize the model architecture. The manuscript's recorded training environment is described separately in the reproducibility guide; installing inference dependencies does not reproduce training.

Full BF16 storage for approximately 12 billion parameters is roughly 24 GB before adapter, cache, activations, and runtime overhead. The manuscript describes a 32-GB GPU deployment example; memory needed for a particular note depends on sequence length, output allowance, and device placement. Optional four-bit loading reduces the base-weight footprint, but does not guarantee that every context will fit.

## 2 Obtain access to the model and base weights

The runner defaults to `JWang580/BMT-GRAFT-12B`. Login if the release or its base model requires authorization:

```bash
hf auth login
```

If the `hf` command is unavailable, install the Hugging Face Hub package in the active environment:

```bash
python -m pip install --upgrade huggingface_hub
```

Accept any required model-access terms using your Hugging Face account. The runner reads the cached Hugging Face token or `HF_TOKEN`; keep tokens out of commands, source code, and output files.

Checkpoint handling is automatic:

- **LoRA adapter:** if `adapter_config.json` is present, the runner reads `base_model_name_or_path`, loads that base model, and attaches the BMT-GRAFT-12B adapter with PEFT. Running only `google/gemma-4-12B-it` would run the unadapted backbone.
- **Merged model:** if no adapter configuration exists, the runner loads the specified checkpoint directly.
- **Saved local adapter:** pass `--model fine_tuning/outputs/gemma4_12b_lora/final` after reproducing SFT, or pass another complete local checkpoint directory.

The local SFT adapter identifies `google/gemma-4-12B-it` as its base. The runner inspects checkpoint metadata before loading. For adapters, access to both the adapter repository and its base model is required.

## 3 Check the installation with a synthetic note

Validate the input and planned request without downloading weights or loading the GPU runtime:

```bash
python prompt_eval/run_finetuned.py --demo --dry_run
```

This reports one record, five requested fields, and one initial model call. It does not print clinical text or write a result file. Then run actual inference:

```bash
python prompt_eval/run_finetuned.py --demo --output prompt_eval/results/demo_predictions.json
```

The synthetic note describes an allogeneic matched sibling peripheral blood transplant, a male donor, recipient A-positive blood type, and tacrolimus/methotrexate prophylaxis. The default fields are:

| Field | Question |
| --- | --- |
| 32 | Specify donor |
| 33 | Specify product type |
| 53 | Donor sex |
| 100 | Recipient blood type |
| 162 | GVHD prophylaxis drugs |

Compare the generated answers with the example note. The first inference run may take longer while weights download and load.

## 4 Extract fields from a text file

Save one note as UTF-8 text in a private local directory. For five selected fields:

```bash
python prompt_eval/run_finetuned.py --note_file data/my_note.txt --record_id local_001 --date 2026-10-29 --fields 32,33,53,100,162 --output prompt_eval/results/local_001_predictions.json
```

Replace the example date with the note's actual date, or omit `--date` if unknown. `record_id` is a local linking identifier; it need not be a patient identifier. Use only field IDs present in `resources/CIBMTR_Form_2400_R11_Items_simplified.csv`.

For the full catalog:

```bash
python prompt_eval/run_finetuned.py --note_file data/my_note.txt --all_fields --output prompt_eval/results/full_form_predictions.json
```

The runner sends five questions at a time by default. All 129 questions require 26 initial calls, with four questions in the final call. Retry attempts can add calls. Every call contains the complete note; the runner does not silently select only the first five questions when `--all_fields` is supplied.

## 5 Process a CSV of notes

The CSV requires a `note` column. `record_id` and `date` are optional; missing identifiers become local sequential identifiers. Explicit identifiers must be unique per note. Each row is processed independently, with one final answer per selected question for that row.

Example CSV using synthetic notes:

```csv
record_id,date,note
example_001,2000-01-01,"Invented example: allogeneic matched sibling peripheral blood transplant."
example_002,2000-01-02,"Invented example: donor is female; recipient blood type is O positive."
```

Use a CSV writer or spreadsheet export so notes containing commas, quotes, or newlines are correctly quoted. With your authorized local file:

```bash
python prompt_eval/run_finetuned.py --notes_csv data/my_notes.csv --all_fields --output prompt_eval/results/batch_predictions.json
```

For different column names:

```bash
python prompt_eval/run_finetuned.py --notes_csv data/my_notes.csv --note_column clinical_text --id_column encounter_key --date_column note_date --fields 32,33,53,100,162 --output prompt_eval/results/selected_predictions.json
```

This runner does not integrate several notes into one longitudinal encounter answer. For the separate 14-question research workflow, use `research_eval/research_extract.py` and its documented encounter-linkage inputs.

## 6 Configure decoding and memory use

| Option | Default | Use |
| --- | --- | --- |
| `--model` | `JWang580/BMT-GRAFT-12B` | Published checkpoint or a local model/adapter directory |
| `--fields_per_call` | `5` | Maximum questions in each bundle |
| `--thinking` | `on` | Model reasoning mode; `off` retains the requested visible JSON rationale |
| `--temperature` | `0.0` | Greedy generation; positive values enable sampling |
| `--max_new_tokens` | `8192` | Completion allowance per call, including any generated thinking |
| `--max_input_tokens` | `24000` | Reject a rendered prompt exceeding this input budget |
| `--parse_retries` | `1` | Additional attempt after invalid or incomplete JSON |
| `--load_in_4bit` | Off | Quantize base weights with bitsandbytes |
| `--revision` | Repository default | Pin the adapter or merged-model version |
| `--base_revision` | Adapter configuration/default | Pin the base version for adapter loading |
| `--base_model_path` | Adapter configuration | Explicit local base-model directory for an adapter |
| `--processor_path` | Checkpoint | Explicit saved training tokenizer/processor |
| `--local_files_only` | Off | Require complete local/cached weights and configurations |

The starter commands use greedy decoding for convenience. Registry experiments in the manuscript used their recorded generation settings; the repository Gemma baseline defaults include temperature 1.0, top-p 0.95, and top-k 64. Set `--temperature 1.0` when deliberately matching that sampling configuration. A new run still requires the original cohort, prompt/version, and model revision to reproduce an archived estimate.

Example with a smaller weight footprint and model thinking disabled:

```bash
python prompt_eval/run_finetuned.py --note_file data/my_note.txt --all_fields --load_in_4bit --thinking off --output prompt_eval/results/quantized_predictions.json
```

Four-bit loading requires a compatible bitsandbytes environment. It is a deployment option, not a claim that the manuscript's BF16 model was evaluated at four-bit precision. Lowering request size or output allowance can reduce memory use but changes the configuration; an output allowance that is too small can cause failed bundles. The runner does not truncate long input notes. Raise the input budget only after checking the model context and available memory.

## 7 Read the outputs

A successful invocation writes the requested JSON and a companion file with the same stem ending in `.manifest.json`. Existing outputs are never overwritten; choose a new path for another run. The manifest records the model, checkpoint type, supplied revisions, selected fields, generation settings, question-catalog hash, prediction hash, and number of failed fields.

Each answer row carries record and field metadata, `answers`, `evidence_from_text`, `reasoning_trace`, `extraction_success`, `extraction_status`, `parse_error`, and the response-contract version. A successful explicit `N/A` means the requested answer was unavailable in the note. A null answer with `extraction_success=false` means extraction failed and needs review. Omitted/ambiguous answers cannot silently become successful `N/A` responses.

The process returns code `0` when all requested fields were processed successfully, `2` when it saved results with failed fields, and a nonzero error on a runtime/input failure. Successful processing is not proof that an answer is clinically correct. Verify selected values, source attribution, form branching, and dates against the input.

Complete outputs contain quoted clinical text and must remain private. To produce an answer-only file for review before publication:

```bash
python scripts/export_public_answers.py prompt_eval/results/local_001_predictions.json --output public_results/local_001_answers.json
```

No clinical text is sent to a hosted inference service by this runner. Weight/config downloads can contact Hugging Face unless offline mode is selected. Avoid redirecting private notes or full prediction contents into shared logs, issues, notebooks, or public repositories.

## 8 Use complete local checkpoints offline

Download the model before entering an offline environment, using an authorized networked machine. For an adapter release, both the adapter and its base weights are required. One practical layout is:

```bash
hf download JWang580/BMT-GRAFT-12B --local-dir /private/models/BMT-GRAFT-12B
hf download google/gemma-4-12B-it --local-dir /private/models/gemma-4-12B-it
```

If the release is already merged, a separate base download is unnecessary. Model folders are not code-release files.

For an adapter snapshot whose `adapter_config.json` names a Hub base repository, `--local_files_only` resolves the base from the standard Hugging Face cache. To use an explicit transferred base folder instead, pass `--base_model_path`. Keep the adapter's saved tokenizer and chat template with it.

```bash
python prompt_eval/run_finetuned.py --model /private/models/BMT-GRAFT-12B --base_model_path /private/models/gemma-4-12B-it --local_files_only --note_file data/my_note.txt --all_fields --output prompt_eval/results/offline_predictions.json
```

Omit `--base_model_path` for a merged checkpoint. Use local Windows paths when running on Windows. Pin `--revision` and `--base_revision` to release commit hashes when loading from the Hub or cache. The dry run skips model loading and does not check credentials or checkpoint availability.

## Troubleshooting

| Problem | Check |
| --- | --- |
| `401`, `403`, repository not found, or gated model | Confirm the model ID, login, release visibility, and authorization for both adapter and base; the runner does not substitute a different model |
| Unknown Gemma architecture or missing auto-model class | Activate the intended environment and install the listed local dependencies; inspect the checkpoint's architecture and recorded runtime |
| Adapter appears to load only a small file | A LoRA release still needs its base weights; verify that both are available |
| No chat template, processor errors, or empty decoded response | Use the saved training tokenizer with `--processor_path`; keep its chat-template files. The shared generator preserves final-channel decoding |
| CUDA unavailable | Install a matching GPU PyTorch build and check the driver; package installation alone does not verify CUDA |
| Out of GPU memory | Try supported quantization, shorter notes, fewer requested fields per call, or a deliberate smaller output allowance; confirm device offloading and available memory |
| Input budget exceeded | Shorten the source or raise the budget within supported context/memory limits; the note was not silently truncated |
| Null answers or exit code `2` | Review `parse_error`; check output allowance, prompt format, and saved tokenizer before retrying to a new output file |
| Output already exists | Choose a new path; this starter runner does not resume or overwrite prior results |

Reference implementation APIs: [Transformers Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4), [PEFT adapter loading](https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config), [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), and [bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
