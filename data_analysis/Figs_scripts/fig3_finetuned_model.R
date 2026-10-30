#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Figure 3: fine-tuned BMT-GRAFT-12B (Gemma-4-12B fine-tuned on MIMIC) vs its
#           base model, the best open-weight model and proprietary models.
#   Bar chart of answer accuracy with 95% CI at 5 fields per call, baseline
#   prompt; MIMIC (held-out validation) and OSU (external test).
#
# Input : ../bmt_analysis_results.xlsx, sheet "Settings"
# Output: ../figs/fig3_finetuned_model.{pdf,png}
#         ../figs/fig3_finetuned_model_source_data.csv
#
# Notes
#   Each run is shown on its own target set: BMT-GRAFT-12B and the
#     proprietary models on the 129-field set (2,580 cells); Gemma-4-12B and
#     Gemma-4-31B on the 100-field set (2,000 cells, their only baseline runs
#     at 5 fields per call). Matched-field differences are reported in the
#     text from the workbook "Paired" sheet.
#   Proprietary models were run on MIMIC only.
#   All runs shown are complete (coverage = 1); the script stops if not.
#
# Style: Nature figure guidelines - Helvetica, 5-7 pt text, 0.5 pt rules,
#   bars from zero, 150 mm width, vector PDF. Colours from ggsci "npg".
#
# Run from anywhere:  Rscript figs_scripts/fig3_finetuned_model.R
# Packages: readxl, dplyr, ggplot2 (>= 3.4), ggsci
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  needed <- c("readxl", "dplyr", "ggplot2", "ggsci")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Install missing packages first: install.packages(c(",
         paste0('"', missing, '"', collapse = ", "), "))", call. = FALSE)
  }
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(ggsci)
})

# ---- Paths ------------------------------------------------------------------
script_dir <- local({
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg)) {
    dirname(normalizePath(sub("^--file=", "", arg[1])))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable()) {
    dirname(rstudioapi::getSourceEditorContext()$path)
  } else {
    getwd()
  }
})
proj_dir  <- normalizePath(file.path(script_dir, ".."))
xlsx_path <- file.path(proj_dir, "bmt_analysis_results.xlsx")
out_dir   <- file.path(proj_dir, "figs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(xlsx_path))

# ---- Read the Settings sheet (header row located, not hard-coded) -----------
raw <- read_excel(xlsx_path, sheet = "Settings", col_names = FALSE,
                  .name_repair = "minimal")
hdr <- which(raw[[1]] == "Cohort" & raw[[2]] == "Model")[1]
if (is.na(hdr)) stop("Header row not found in sheet 'Settings'.", call. = FALSE)
settings <- raw[-seq_len(hdr), ]
names(settings) <- make.unique(as.character(unlist(raw[hdr, ])))
settings <- settings[!is.na(settings[[1]]), ] |>
  transmute(cohort = Cohort, model = Model, target_set = `Target set`,
            mode = Mode, fpc = as.numeric(`Fields per call`), notes = Notes,
            correct = as.numeric(Correct),
            labelled = as.numeric(`Labeled targets`),
            accuracy = as.numeric(`Answer accuracy`),
            ci_lo = as.numeric(`95% CI lower`),
            ci_hi = as.numeric(`95% CI upper`),
            coverage = as.numeric(Coverage),
            parse_fail = as.numeric(`Parse failure`),
            run_id = `Run ID`)

# ---- Models shown -----------------------------------------------------------
models <- tibble::tribble(
  ~model,                ~label,
  "gemma-4-12B-it",      "Gemma-4-12B (base)",
  "gemma-4-31B-it",      "Gemma-4-31B",
  "BMT-GRAFT-12B (SFT)", "BMT-GRAFT-12B (fine-tuned)",
  "gpt-6-sol",           "GPT-6 Sol",
  "gemini-3.8-flash",    "Gemini 3.8 Flash"
)
cohort_lab <- c(MIMIC = "MIMIC", OSU = "OSU")

acc <- settings |>
  filter(mode == "baseline", fpc == 5, notes == "full",
         model %in% models$model) |>
  left_join(models, by = "model")
stopifnot(!anyDuplicated(acc[c("cohort", "model")]),
          all(acc$coverage >= 1 - 1e-9))

acc_plot <- acc |>
  mutate(label  = factor(label, levels = models$label),
         cohort = factor(cohort_lab[cohort], levels = cohort_lab))

acc |>
  transmute(cohort, model, target_set, fields_per_call = fpc, run_id,
            correct, n_scored = labelled, coverage,
            parse_failure_rate = parse_fail, accuracy,
            ci_lower = ci_lo, ci_upper = ci_hi,
            ci_method = "encounter-cluster bootstrap (workbook)") |>
  arrange(cohort, factor(model, levels = models$model)) |>
  write.csv(file.path(out_dir, "fig3_finetuned_model_source_data.csv"),
            row.names = FALSE)

# ---- Style ------------------------------------------------------------------
lw  <- 0.5 / .pt
npg <- pal_npg("nrc")(10)
model_col <- setNames(c("grey60", npg[4], npg[1], npg[3], npg[2]),
                      models$label)   # base grey, 31B blue, SFT red,
                                      # GPT green, Gemini cyan

fig3 <- ggplot(acc_plot, aes(label, accuracy * 100, fill = label)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = ci_lo * 100, ymax = ci_hi * 100),
                width = 0.25, linewidth = lw) +
  geom_text(aes(y = ci_hi * 100 + 1.5, label = sprintf("%.1f", accuracy * 100)),
            size = 5.5 / .pt, family = "Helvetica", vjust = 0) +
  # proprietary models were not run on OSU: those slots are dropped
  facet_grid(. ~ cohort, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = model_col, guide = "none") +
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 20),
                     expand = c(0, 0)) +
  labs(x = NULL, y = "Answer accuracy (%)") +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(axis.line        = element_line(linewidth = lw, colour = "black"),
        axis.ticks       = element_line(linewidth = lw, colour = "black"),
        axis.ticks.x     = element_blank(),
        axis.ticks.length = unit(1, "mm"),
        axis.text        = element_text(size = 6, colour = "black"),
        axis.text.x      = element_text(size = 6, angle = 45, hjust = 1, vjust = 1),
        axis.title       = element_text(size = 7, colour = "black"),
        strip.background = element_blank(),
        strip.text       = element_text(size = 7, colour = "black",
                                        margin = margin(b = 2)),
        panel.spacing.x  = unit(4, "mm"),
        plot.margin      = margin(3, 3, 3, 3))

ggsave(file.path(out_dir, "fig3_finetuned_model.pdf"), fig3,
       width = 150, height = 85, units = "mm", device = "pdf",
       useDingbats = FALSE)
ggsave(file.path(out_dir, "fig3_finetuned_model.png"), fig3,
       width = 150, height = 85, units = "mm", dpi = 600, bg = "white")

acc |>
  transmute(cohort, model, target_set,
            acc = sprintf("%.1f (%.1f-%.1f)", accuracy * 100, ci_lo * 100,
                          ci_hi * 100)) |>
  as.data.frame() |> print(row.names = FALSE)
message("Figure written to ", out_dir)
