#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Figure 2a-b: Form 2400 extraction accuracy by model size
#   Fig. 2a = MIMIC cohort, Fig. 2b = OSU cohort
#   baseline prompt, 1 field per call, legacy 100-field target set
#   Fig. 2a also shows the proprietary models (gemini-3.8-flash, gpt-6-sol).
#     These were only run at MIMIC with 5 fields per call on the 129-field
#     target set (2,580 cells), so they are not a matched comparison; their
#     labels carry "*" and the legend must say so.
#   Fig. 2b omits gpt-oss (and the proprietary models, which were not run).
#
# Input : ../bmt_analysis_results.xlsx, sheet "Settings"
# Output: ../figs/fig2a_model_size_mimic.{pdf,png}
#         ../figs/fig2b_model_size_osu.{pdf,png}
#         ../figs/fig2ab_model_size_source_data.csv
#
# Scoring
#   Complete runs (coverage = 1): strict answer accuracy = correct / 2,000
#     labelled cells (20 encounters x 100 fields). Error bars are the
#     workbook's 95% encounter-cluster bootstrap CI (10,000 resamples,
#     seed 20261014).
#   Incomplete runs (coverage < 1): accuracy on completed cells only,
#     correct / (coverage x labelled cells), drawn identically to complete
#     runs (no visual distinction; state the scoring in the legend). The workbook
#     has no cell-level outcomes, so a cluster bootstrap CI cannot be computed
#     here. Set `ci_incomplete` below:
#       "wilson" - Wilson score 95% CI on completed cells (default). Treats
#                  cells as independent, so it is narrower than an
#                  encounter-cluster interval; say so in the figure legend.
#       "none"   - no error bar for incomplete runs.
#
# Style: Nature figure guidelines - Helvetica, 5-7 pt text, 0.5 pt rules,
#   bars from zero, 89 mm single-column width, vector PDF.
#   Colours: ggsci "npg" (Nature Publishing Group) palette, one per family.
#
# Run from anywhere:  Rscript figs_scripts/fig2ab_model_size.R
# Packages: readxl, dplyr, ggplot2 (>= 3.4), ggsci
# -----------------------------------------------------------------------------

ci_incomplete <- "wilson"   # "wilson" or "none"

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

figures <- tibble::tribble(
  ~cohort, ~stem,
  "MIMIC", "fig2a_model_size_mimic",
  "OSU",   "fig2b_model_size_osu"
)

# ---- Read the Settings sheet (header row located, not hard-coded) -----------
raw <- read_excel(xlsx_path, sheet = "Settings", col_names = FALSE,
                  .name_repair = "minimal")
hdr <- which(raw[[1]] == "Cohort" & raw[[2]] == "Model")[1]
if (is.na(hdr)) stop("Header row not found in sheet 'Settings'.", call. = FALSE)
settings <- raw[-seq_len(hdr), ]
names(settings) <- make.unique(as.character(unlist(raw[hdr, ])))

settings <- settings |>
  transmute(
    cohort     = Cohort,
    model      = Model,
    target_set = `Target set`,
    mode       = Mode,
    fpc        = as.numeric(`Fields per call`),
    notes      = Notes,
    correct    = as.numeric(Correct),
    labelled   = as.numeric(`Labeled targets`),
    ci_lo      = as.numeric(`95% CI lower`),
    ci_hi      = as.numeric(`95% CI upper`),
    coverage   = as.numeric(Coverage),
    parse_fail = as.numeric(`Parse failure`),
    run_id     = `Run ID`
  ) |>
  filter(!is.na(cohort), !is.na(model))

# ---- Model metadata (display order = family, then size) ---------------------
model_meta <- tibble::tribble(
  ~model,               ~family,   ~label,     ~size_b,
  "Qwen3.5-2B",         "Qwen3.5", "2B",          2,
  "Qwen3.5-9B",         "Qwen3.5", "9B",          9,
  "Qwen3.5-27B",        "Qwen3.5", "27B",        27,
  "Qwen3.5-35B-A3B",    "Qwen3.5", "35B-A3B",    35,
  "gemma-4-E4B-it",     "Gemma 4", "E4B",         4,
  "gemma-4-12B-it",     "Gemma 4", "12B",        12,
  "gemma-4-26B-A4B-it", "Gemma 4", "26B-A4B",    26,
  "gemma-4-31B-it",     "Gemma 4", "31B",        31,
  "gpt-oss-20b",        "gpt-oss", "20B",        20,
  "gpt-oss-120b",       "gpt-oss", "120B",      120,
  "gemini-3.8-flash",   "Proprietary", "Gemini 3.8 Flash*", NA,
  "gpt-6-sol",          "Proprietary", "GPT-6 Sol*",        NA
)
family_levels <- c("Qwen3.5", "Gemma 4", "gpt-oss", "Proprietary")
proprietary   <- c("gemini-3.8-flash", "gpt-6-sol")

# Models shown in each figure (display order follows model_meta)
fig_models <- list(
  MIMIC = model_meta$model,
  OSU   = model_meta$model[model_meta$family %in% c("Qwen3.5", "Gemma 4")]
)

wilson <- function(x, n, z = qnorm(0.975)) {
  p <- x / n
  centre <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  half   <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  cbind(centre - half, centre + half)
}

# ---- Select runs and score --------------------------------------------------
# Open-weight models: baseline, 1 field per call, legacy 100-field set.
# Proprietary models: their only baseline run (5 fields/call, 129-field set).
dat <- settings |>
  filter(mode == "baseline", notes == "full",
         (fpc == 1 & target_set == "legacy 100-field" &
            !model %in% proprietary) |
           model %in% proprietary) |>
  left_join(model_meta, by = "model")

# keep only models listed for that cohort's figure (drops OSU gpt-oss)
dat <- dat[mapply(function(m, c) m %in% fig_models[[c]], dat$model, dat$cohort), ]
stopifnot(!anyDuplicated(dat[c("cohort", "model")]))

if (anyNA(dat$family)) {
  stop("No metadata for: ", paste(unique(dat$model[is.na(dat$family)]),
                                  collapse = ", "), call. = FALSE)
}

dat <- dat |>
  mutate(
    complete  = coverage >= 1 - 1e-9,
    n_scored  = if_else(complete, labelled, round(coverage * labelled)),
    accuracy  = correct / n_scored,
    # Source data always carries a 95% CI. Complete runs: workbook cluster
    # bootstrap. Incomplete runs: Wilson score on completed cells, which
    # treats cells as independent (no cell-level data to cluster), so it is
    # narrower than an encounter-level interval would be.
    ci_method = if_else(complete,
                        "encounter-cluster bootstrap (workbook)",
                        "Wilson score on completed cells (unclustered)"),
    ci_lo = if_else(complete, ci_lo, wilson(correct, n_scored)[, 1]),
    ci_hi = if_else(complete, ci_hi, wilson(correct, n_scored)[, 2]),
    # What the figures draw (controlled by `ci_incomplete`)
    plot_lo = if_else(complete | ci_incomplete == "wilson", ci_lo, NA_real_),
    plot_hi = if_else(complete | ci_incomplete == "wilson", ci_hi, NA_real_)
  )

# Source data for both figures (Nature requires source data)
dat |>
  transmute(figure = if_else(cohort == "MIMIC", "Fig. 2a", "Fig. 2b"),
            cohort, model, family, target_set, fields_per_call = fpc,
            run_id, correct, n_scored, labelled,
            coverage, parse_failure_rate = parse_fail,
            scoring = if_else(complete, "all labelled cells",
                              "completed cells only"),
            accuracy, ci_lower = ci_lo, ci_upper = ci_hi, ci_method) |>
  arrange(figure, factor(family, levels = family_levels), model) |>
  write.csv(file.path(out_dir, "fig2ab_model_size_source_data.csv"),
            row.names = FALSE)

# ---- Nature-style theme and NPG colours -------------------------------------
lw <- 0.5 / .pt                        # 0.5 pt rule, in ggplot linewidth (mm)
npg <- pal_npg("nrc")(10)
fam_col <- c("Qwen3.5"     = npg[1],   # #E64B35 red
             "Gemma 4"     = npg[4],   # #3C5488 dark blue
             "gpt-oss"     = npg[3],   # #00A087 green
             "Proprietary" = npg[9])   # #7E6148 brown

theme_nature <- function(base_size = 7, base_family = "Helvetica") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line          = element_line(linewidth = lw, colour = "black"),
      axis.ticks         = element_line(linewidth = lw, colour = "black"),
      axis.ticks.length  = unit(1.2, "mm"),
      axis.ticks.x       = element_blank(),
      axis.text          = element_text(size = 6, colour = "black"),
      axis.title         = element_text(size = 7, colour = "black"),
      strip.background   = element_blank(),
      strip.placement    = "outside",
      strip.text         = element_text(size = 7, colour = "black",
                                        margin = margin(t = 2, b = 2)),
      panel.spacing.x    = unit(2, "mm"),
      legend.text        = element_text(size = 6),
      legend.title       = element_blank(),
      legend.key.size    = unit(2.6, "mm"),
      legend.spacing.x   = unit(1, "mm"),
      legend.margin      = margin(0, 0, 0, 0),
      legend.box.spacing = unit(1, "mm"),
      legend.position    = "top",
      plot.margin        = margin(3, 3, 6, 3)
    )
}

# ---- Figure builder ---------------------------------------------------------
make_fig <- function(cohort_name) {
  d <- model_meta |>
    filter(model %in% fig_models[[cohort_name]]) |>
    left_join(filter(dat, cohort == cohort_name),
              by = c("model", "family", "label", "size_b")) |>
    mutate(
      family = factor(family, levels = intersect(family_levels, family)),
      label  = factor(label, levels = model_meta$label)
    )

  ggplot(d, aes(label, accuracy * 100)) +
    geom_col(aes(fill = family), width = 0.72, na.rm = TRUE,
             show.legend = FALSE) +
    geom_errorbar(aes(ymin = plot_lo * 100, ymax = plot_hi * 100),
                  width = 0.3, linewidth = lw, na.rm = TRUE) +
    geom_text(aes(y = 1.5, label = if_else(is.na(accuracy), "not run",
                                            NA_character_)),
              angle = 90, hjust = 0, size = 5 / .pt, family = "Helvetica",
              colour = "grey40", na.rm = TRUE) +
    facet_grid(. ~ family, scales = "free_x", space = "free_x",
               switch = "x") +
    scale_fill_manual(values = fam_col, guide = "none") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20),
                       expand = c(0, 0)) +
    labs(x = NULL, y = "Answer accuracy (%)") +
    coord_cartesian(clip = "off") +
    theme_nature() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
}

# ---- Save -------------------------------------------------------------------
w_mm <- 89; h_mm <- 65
for (i in seq_len(nrow(figures))) {
  fig <- make_fig(figures$cohort[i])
  ggsave(file.path(out_dir, paste0(figures$stem[i], ".pdf")), fig,
         width = w_mm, height = h_mm, units = "mm", device = "pdf",
         useDingbats = FALSE)
  ggsave(file.path(out_dir, paste0(figures$stem[i], ".png")), fig,
         width = w_mm, height = h_mm, units = "mm", dpi = 600, bg = "white")
}

# ---- Console summary --------------------------------------------------------
dat |>
  arrange(cohort, factor(family, levels = family_levels), size_b) |>
  transmute(cohort, model, n_scored,
            acc_pct = sprintf("%.1f", accuracy * 100),
            ci_95   = if_else(is.na(ci_lo), "-",
                              sprintf("%.1f-%.1f", ci_lo * 100, ci_hi * 100)),
            ci_method) |>
  as.data.frame() |>
  print(row.names = FALSE)
message("Figures written to ", out_dir)
