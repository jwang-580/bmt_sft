#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Figure 4: longitudinal research abstraction (CRS and related events after
#           HCT; 191 encounters, 14 fields, 2,273 scorable labels)
#   a - overall exact accuracy (95% CI) for the automated Gemma 4 workflows
#       and the BMT-GRAFT-12B workflow (fine-tuned note-level extraction
#       followed by assisted synthesis)
#   b - exact accuracy by field (95% Wilson CI), Gemma-4-31B vs the
#       BMT-GRAFT-12B workflow
#
# Input : ../bmt_analysis_results.xlsx, sheets "Research", "Research fields"
# Output: ../figs/fig4_research_extraction.{pdf,png}
#         ../figs/fig4_research_extraction_source_data.csv
#
# Notes
#   Full expected cohort; failed or missing outputs count as incorrect.
#   Reference labels were revised after review of fine-tuned extraction
#   errors, and the BMT-GRAFT-12B workflow used assisted synthesis whereas
#   the Gemma workflows were fully automated (see Methods).
#
# Style: Nature figure guidelines - Helvetica, 5-7 pt text, 0.5 pt rules,
#   bars from zero, 183 mm double-column width, vector PDF. ggsci "npg".
#
# Run from anywhere:  Rscript figs_scripts/fig4_research_extraction.R
# Packages: readxl, dplyr, ggplot2 (>= 3.4), ggsci, patchwork
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  needed <- c("readxl", "dplyr", "ggplot2", "ggsci", "patchwork")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Install missing packages first: install.packages(c(",
         paste0('"', missing, '"', collapse = ", "), "))", call. = FALSE)
  }
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(ggsci)
  library(patchwork)
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

read_block <- function(sheet) {
  raw <- read_excel(xlsx_path, sheet = sheet, col_names = FALSE,
                    .name_repair = "minimal")
  hdr <- which(raw[[1]] == "Workflow")[1]
  if (is.na(hdr)) stop("Header row not found in sheet '", sheet, "'.",
                       call. = FALSE)
  out <- raw[-seq_len(hdr), ]
  names(out) <- make.unique(as.character(unlist(raw[hdr, ])))
  out[!is.na(out[[1]]), ]
}

# ---- Labels -----------------------------------------------------------------
workflows <- tibble::tribble(
  ~workflow,                  ~label,
  "Gemma 12b",                "Gemma-4\n12B",
  "Gemma 26b-a4b",            "Gemma-4\n26B-A4B",
  "Gemma 31b",                "Gemma-4\n31B",
  "SFT plus human synthesis", "BMT-GRAFT-12B"
)
fields <- tibble::tribble(
  ~target,                   ~field,
  "crs_y_n",                 "CRS",
  "fever_onset_date",        "Fever onset date",
  "last_fever_date",         "Last fever date",
  "max_temp",                "Maximum temperature",
  "hypotension_y_n",         "Hypotension",
  "pressor_use_num",         "Vasopressor use",
  "hypoxia_y_n",             "Hypoxia",
  "high_flow_o2_y_n",        "High-flow oxygen",
  "bipap_or_intubation_y_n", "BiPAP or intubation",
  "neurotox_y_n",            "Neurotoxicity",
  "toci_y_n",                "Tocilizumab use",
  "toci_start_date",         "Tocilizumab start date",
  "toci_stop_date",          "Tocilizumab stop date",
  "total_dose_toci",         "Tocilizumab total dose"
)

# ---- Data -------------------------------------------------------------------
overall <- read_block("Research") |>
  transmute(workflow = Workflow, scope = `Cohort scope`, metric = Metric,
            num = as.numeric(Numerator), den = as.numeric(Denominator),
            est = as.numeric(Estimate), lo = as.numeric(`95% CI lower`),
            hi = as.numeric(`95% CI upper`)) |>
  filter(scope == "full expected cohort", metric == "exact_accuracy") |>
  inner_join(workflows, by = "workflow") |>
  mutate(label = factor(label, levels = workflows$label))
stopifnot(nrow(overall) == nrow(workflows))

by_field <- read_block("Research fields") |>
  transmute(workflow = Workflow, scope = `Cohort scope`, target = Target,
            metric = Metric, num = as.numeric(Numerator),
            den = as.numeric(Denominator), est = as.numeric(Estimate),
            lo = as.numeric(`95% CI lower`), hi = as.numeric(`95% CI upper`),
            ci_method = `CI method`) |>
  filter(scope == "full expected cohort", metric == "exact_accuracy",
         workflow %in% c("Gemma 31b", "SFT plus human synthesis")) |>
  inner_join(workflows, by = "workflow") |>
  inner_join(fields, by = "target") |>
  mutate(label = factor(if_else(workflow == "Gemma 31b", "Gemma-4-31B",
                                "BMT-GRAFT-12B"),
                        levels = c("Gemma-4-31B", "BMT-GRAFT-12B")),
         field = factor(field, levels = rev(fields$field)))
stopifnot(nrow(by_field) == 2 * nrow(fields))

bind_rows(
  overall |> transmute(panel = "a", workflow, field = "all fields",
                       correct = num, scorable = den, accuracy = est,
                       ci_lower = lo, ci_upper = hi,
                       ci_method = "encounter-cluster bootstrap (workbook)"),
  by_field |> transmute(panel = "b", workflow, field = as.character(field),
                        correct = num, scorable = den, accuracy = est,
                        ci_lower = lo, ci_upper = hi, ci_method)
) |>
  write.csv(file.path(out_dir, "fig4_research_extraction_source_data.csv"),
            row.names = FALSE)

# ---- Style ------------------------------------------------------------------
lw  <- 0.5 / .pt
npg <- pal_npg("nrc")(10)
wf_col <- c("Gemma-4\n12B"           = "grey70",
            "Gemma-4\n26B-A4B"       = "grey45",
            "Gemma-4\n31B"           = npg[4],    # #3C5488
            "BMT-GRAFT-12B" = npg[1])   # #E64B35
wf_col_b <- c("Gemma-4-31B" = npg[4], "BMT-GRAFT-12B" = npg[1])

theme_nature <- function(base_size = 7, base_family = "Helvetica") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(axis.line         = element_line(linewidth = lw, colour = "black"),
          axis.ticks        = element_line(linewidth = lw, colour = "black"),
          axis.ticks.length = unit(1, "mm"),
          axis.text         = element_text(size = 6, colour = "black"),
          axis.title        = element_text(size = 7, colour = "black"),
          plot.tag          = element_text(size = 8, face = "bold"),
          legend.text       = element_text(size = 6.5),
          legend.title      = element_blank(),
          legend.key.size   = unit(3, "mm"),
          plot.margin       = margin(3, 4, 3, 3))
}

# ---- Panel a: overall accuracy ----------------------------------------------
p_a <- ggplot(overall, aes(label, est * 100, fill = label)) +
  geom_col(width = 0.68) +
  geom_errorbar(aes(ymin = lo * 100, ymax = hi * 100), width = 0.25,
                linewidth = lw) +
  geom_text(aes(y = hi * 100 + 2, label = sprintf("%.1f", est * 100)),
            size = 5.5 / .pt, family = "Helvetica", vjust = 0) +
  scale_fill_manual(values = wf_col, guide = "none") +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20),
                     expand = c(0, 0)) +
  labs(x = NULL, y = "Exact accuracy (%)") +
  coord_cartesian(clip = "off") +
  theme_nature() +
  theme(axis.ticks.x = element_blank(),
        axis.text.x  = element_text(size = 6, lineheight = 0.9))

# ---- Panel b: accuracy by field ---------------------------------------------
dodge <- position_dodge(width = 0.6)
p_b <- ggplot(by_field, aes(est * 100, field, colour = label)) +
  geom_vline(xintercept = seq(20, 80, 20), colour = "grey92",
             linewidth = 0.35 / .pt) +
  geom_errorbarh(aes(xmin = lo * 100, xmax = hi * 100), height = 0,
                 linewidth = 0.7 / .pt, position = dodge) +
  geom_point(size = 1.5, position = dodge) +
  scale_colour_manual(values = wf_col_b,
                      breaks = c("BMT-GRAFT-12B", "Gemma-4-31B")) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20),
                     expand = c(0, 0)) +
  labs(x = "Exact accuracy (%)", y = NULL) +
  coord_cartesian(clip = "off") +
  theme_nature() +
  theme(axis.ticks.y = element_blank(),
        legend.position = "bottom",
        legend.margin = margin(0, 0, 0, 0))

fig4 <- (p_a | p_b) +
  plot_layout(widths = c(0.9, 1.2)) +
  plot_annotation(tag_levels = "a")

ggsave(file.path(out_dir, "fig4_research_extraction.pdf"), fig4,
       width = 183, height = 80, units = "mm", device = "pdf",
       useDingbats = FALSE)
ggsave(file.path(out_dir, "fig4_research_extraction.png"), fig4,
       width = 183, height = 80, units = "mm", dpi = 600, bg = "white")

# ---- Console summary --------------------------------------------------------
overall |>
  transmute(workflow, acc = sprintf("%.1f (%.1f-%.1f)", est * 100, lo * 100,
                                    hi * 100), n = paste0(num, "/", den)) |>
  as.data.frame() |> print(row.names = FALSE)
message("Figure written to ", out_dir)
