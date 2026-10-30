# Manuscript figures

Figure 1 is provided in [PNG](figs/fig1_study_design.png) and [PDF](figs/fig1_study_design.pdf) format. R scripts in `Figs_scripts/` generate the remaining figures and their source-data tables under `figs/`.

## Inputs

| Input | Figures |
| --- | --- |
| `bmt_analysis_results.xlsx` | Figures 2–4 and S1; sheets `Settings`, `Research`, `Research fields`, and `Runtime` |
| `data/form_paired_cells.json` | Figure S2; paired base/fine-tuned answer-match indicators for 100 questions and 20 records in each cohort |

The workbook is not included in this checkout. Place it directly under `data_analysis/` before running the workbook-based scripts. The Figure S2 input contains record labels, question metadata, and answer-match flags; it contains no note text, evidence quotations, or model rationales.

## Run

Install R and the required packages:

```r
install.packages(c("readxl", "jsonlite", "dplyr", "tidyr", "ggplot2", "ggsci", "patchwork", "ggrepel"))
```

Run scripts from the repository root:

```bash
Rscript data_analysis/Figs_scripts/fig2ab_model_size.R
Rscript data_analysis/Figs_scripts/fig2cd_extraction_settings.R
Rscript data_analysis/Figs_scripts/fig3_finetuned_model.R
Rscript data_analysis/Figs_scripts/fig4_research_extraction.R
Rscript data_analysis/Figs_scripts/figS1_runtime.R
Rscript data_analysis/Figs_scripts/figS2_question_accuracy.R
```

Each script documents its comparison set, scoring rules, and plotting settings. Keep those definitions with the exported figure when comparing results.
