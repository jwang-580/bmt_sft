#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Figure 2c-d: extraction settings and extraction performance
#   Fig. 2c - accuracy by number of fields per call (baseline JSON prompt),
#            one small-multiple panel per open-weight model, MIMIC and OSU
#            overlaid, 95% CI error bars.
#            Legacy 100-field target set (20 encounters x 100 fields).
#   Fig. 2d - extraction settings at 5 fields per call: baseline (JSON with
#            reasoning, supporting citation and brief explanation) vs
#            free-text output, few-shot examples, no reasoning, no supporting
#            citation (evidence field) and no brief explanation (reasoning-
#            trace field). One panel per model (rows: MIMIC, OSU); each
#            setting is a point with 95% CI, dotted line = model baseline. Models: Qwen3.5-27B, Qwen3.5-35B-A3B,
#            Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B.
#            Baseline uses the 100-field set (2,000 cells); the variants use
#            the 129-field set (2,580 cells) except free text where only a
#            100-field run exists - see target_set in the source data.
#
# Input : ../bmt_analysis_results.xlsx, sheet "Settings"
# Output: ../figs/fig2c_fields_per_call.{pdf,png}
#         ../figs/fig2d_extraction_settings.{pdf,png}
#         ../figs/fig2c_fields_per_call_source_data.csv
#         ../figs/fig2d_extraction_settings_source_data.csv
#
# Scoring (same rules as Fig. 2a-b)
#   Complete runs (coverage = 1): correct / 2,000 labelled cells, with the
#     workbook's 95% encounter-cluster bootstrap CI (10,000 resamples).
#   Incomplete runs (coverage < 1): correct / completed cells, with a Wilson
#     score 95% CI on completed cells (treats cells as independent, so it is
#     narrower than an encounter-level interval). Not visually distinguished;
#     the source data records the scoring and CI method for every value.
#
# Style: Nature figure guidelines - Helvetica, 5-7 pt text, 0.5 pt rules,
#   183 mm double-column width, vector PDF. Colours from ggsci "npg".
#
# Run from anywhere:  Rscript figs_scripts/fig2cd_extraction_settings.R
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

# ---- Model metadata (display order) -----------------------------------------
model_meta <- tibble::tribble(
  ~model,               ~family,   ~short,
  "Qwen3.5-2B",         "Qwen3.5", "Qwen3.5-2B",
  "Qwen3.5-9B",         "Qwen3.5", "Qwen3.5-9B",
  "Qwen3.5-27B",        "Qwen3.5", "Qwen3.5-27B",
  "Qwen3.5-35B-A3B",    "Qwen3.5", "Qwen3.5-35B-A3B",
  "gpt-oss-20b",        "gpt-oss", "gpt-oss-20b",
  "gemma-4-E4B-it",     "Gemma 4", "Gemma-4-E4B",
  "gemma-4-12B-it",     "Gemma 4", "Gemma-4-12B",
  "gemma-4-26B-A4B-it", "Gemma 4", "Gemma-4-26B-A4B",
  "gemma-4-31B-it",     "Gemma 4", "Gemma-4-31B",
  "gpt-oss-120b",       "gpt-oss", "gpt-oss-120b"
)
fpc_levels <- c(1, 5, 10, 25, 50, 100)

wilson <- function(x, n, z = qnorm(0.975)) {
  p <- x / n
  centre <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  half   <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  cbind(centre - half, centre + half)
}

# ---- Select and score: baseline vs non-JSON, legacy set, full prompts -------
dat <- settings |>
  filter(target_set == "legacy 100-field", notes == "full",
         mode %in% c("baseline", "non_json"),
         fpc %in% fpc_levels,
         model %in% model_meta$model) |>
  left_join(model_meta, by = "model") |>
  mutate(
    format    = if_else(mode == "baseline", "JSON", "Free text"),
    complete  = coverage >= 1 - 1e-9,
    n_scored  = if_else(complete, labelled, round(coverage * labelled)),
    accuracy  = correct / n_scored,
    ci_method = if_else(complete,
                        "encounter-cluster bootstrap (workbook)",
                        "Wilson score on completed cells (unclustered)"),
    ci_lo = if_else(complete, ci_lo, wilson(correct, n_scored)[, 1]),
    ci_hi = if_else(complete, ci_hi, wilson(correct, n_scored)[, 2])
  )
stopifnot(!anyDuplicated(dat[c("cohort", "model", "format", "fpc")]))

# Source data
dat |>
  transmute(figure = if_else(format == "JSON", "Fig. 2c",
                             "not plotted (free-text runs, all fields per call)"),
            cohort, model, output_format = format, fields_per_call = fpc,
            target_set, run_id, correct, n_scored, labelled, coverage,
            parse_failure_rate = parse_fail,
            scoring = if_else(complete, "all labelled cells",
                              "completed cells only"),
            accuracy, ci_lower = ci_lo, ci_upper = ci_hi, ci_method) |>
  arrange(cohort, factor(model, levels = model_meta$model), output_format,
          fields_per_call) |>
  write.csv(file.path(out_dir, "fig2c_fields_per_call_source_data.csv"),
            row.names = FALSE)

# ---- Nature-style theme and NPG colours -------------------------------------
lw  <- 0.5 / .pt
npg <- pal_npg("nrc")(10)
cohort_col <- c(MIMIC = npg[4], OSU = npg[1])   # #3C5488 blue, #E64B35 red

theme_nature <- function(base_size = 7, base_family = "Helvetica") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line          = element_line(linewidth = lw, colour = "black"),
      axis.ticks         = element_line(linewidth = lw, colour = "black"),
      axis.ticks.length  = unit(1, "mm"),
      axis.text          = element_text(size = 6, colour = "black"),
      axis.title         = element_text(size = 7, colour = "black"),
      strip.background   = element_blank(),
      strip.text         = element_text(size = 6.5, colour = "black",
                                        margin = margin(b = 2)),
      panel.spacing      = unit(2.5, "mm"),
      plot.tag           = element_text(size = 8, face = "bold"),
      legend.text        = element_text(size = 6),
      legend.title       = element_text(size = 6),
      legend.key.size    = unit(2.6, "mm"),
      legend.margin      = margin(0, 0, 0, 0),
      legend.box.spacing = unit(1, "mm"),
      legend.position    = "top",
      plot.margin        = margin(3, 3, 3, 3)
    )
}

# ---- Fig. 2c: fields per call ------------------------------------------------
f3 <- dat |>
  filter(format == "JSON") |>
  mutate(short  = factor(short, levels = model_meta$short),
         cohort = factor(cohort, levels = c("MIMIC", "OSU")),
         x      = factor(fpc, levels = fpc_levels))

# Style: "family" - colour = model family (same NPG colours as Fig. 2a-b),
#                    MIMIC solid line + filled circle, OSU dashed + open circle
#        "strips"  - colour = cohort (navy/red); panel headers tinted by family
fig3_style <- "family"

fam_col3  <- c("Qwen3.5" = npg[1], "Gemma 4" = npg[4], "gpt-oss" = npg[3])
fam_tint3 <- c("Qwen3.5" = "#FBE3DE", "Gemma 4" = "#DFE3EE",
               "gpt-oss" = "#D6EFEA")
f3 <- f3 |> mutate(family = factor(family, levels = names(fam_col3)))

dodge <- position_dodge(width = 0.3)
base3 <- function(p) {
  p +
    geom_hline(yintercept = seq(60, 90, 10), colour = "grey90",
               linewidth = 0.35 / .pt) +
    scale_y_continuous(limits = c(50, 100), breaks = seq(50, 100, 10),
                       expand = c(0, 0)) +
    labs(x = "Fields per call", y = "Answer accuracy (%)") +
    coord_cartesian(clip = "off") +
    theme_nature() +
    theme(strip.text = element_text(size = 6.5, colour = "black",
                                    margin = margin(t = 1.5, b = 1.5)),
          legend.text = element_text(size = 6.5),
          legend.key.width = unit(6, "mm"))
}

if (fig3_style == "family") {
  fig2c <- base3(
    ggplot(f3, aes(x, accuracy * 100, colour = family,
                   group = cohort, linetype = cohort, shape = cohort))
  ) +
    geom_line(linewidth = 0.8 / .pt, position = dodge) +
    geom_errorbar(aes(ymin = ci_lo * 100, ymax = ci_hi * 100),
                  width = 0, linewidth = 0.6 / .pt, linetype = "solid",
                  position = dodge) +
    geom_point(size = 1.3, stroke = 0.6, fill = "white", position = dodge) +
    facet_wrap(~ short, nrow = 2) +
    scale_colour_manual(values = fam_col3, guide = "none") +
    scale_linetype_manual(values = c(MIMIC = "solid", OSU = "22"),
                          name = NULL) +
    scale_shape_manual(values = c(MIMIC = 16, OSU = 21), name = NULL) +
    # linetype and shape share one merged legend; one override suffices
    guides(linetype = guide_legend(override.aes = list(colour = "grey25")),
           shape    = guide_legend())
} else {
  strip_fill <- fam_tint3[as.character(
    distinct(f3, short, family) |> arrange(short) |> pull(family))]
  fig2c <- base3(
    ggplot(f3, aes(x, accuracy * 100, colour = cohort, fill = cohort,
                   group = cohort))
  ) +
    geom_line(linewidth = 0.8 / .pt, position = dodge) +
    geom_errorbar(aes(ymin = ci_lo * 100, ymax = ci_hi * 100),
                  width = 0, linewidth = 0.6 / .pt, position = dodge) +
    geom_point(shape = 21, size = 1.4, stroke = 0.35, colour = "white",
               position = dodge) +
    facet_wrap(~ short, nrow = 2) +
    scale_colour_manual(values = cohort_col, name = NULL) +
    scale_fill_manual(values = cohort_col, name = NULL) +
    theme(strip.background = element_rect(fill = "grey95", colour = NA))
  # tint each panel header by family
  g <- ggplotGrob(fig2c)
  strips <- grep("^strip-t", g$layout$name)
  ord <- order(g$layout$t[strips], g$layout$l[strips])
  for (k in seq_along(strips)) {
    st <- g$grobs[[strips[ord][k]]]
    if (inherits(st, "zeroGrob") || k > length(strip_fill)) next
    bg <- grep("background", st$grobs[[1]]$childrenOrder)
    st$grobs[[1]]$children[[bg]]$gp$fill <- strip_fill[k]
    g$grobs[[strips[ord][k]]] <- st
  }
  fig2c <- patchwork::wrap_elements(full = g)
}

ggsave(file.path(out_dir, "fig2c_fields_per_call.pdf"), fig2c,
       width = 183, height = 92, units = "mm", device = "pdf",
       useDingbats = FALSE)
ggsave(file.path(out_dir, "fig2c_fields_per_call.png"), fig2c,
       width = 183, height = 92, units = "mm", dpi = 600, bg = "white")

# ---- Fig. 2d: extraction settings at 5 fields per call ---------------------------
setting_map <- c(
  baseline                 = "Baseline",
  non_json                 = "Free-text output",
  few_shot                 = "Few-shot examples",
  no_reasoning             = "No reasoning",
  no_evidence_field        = "No supporting citation",
  no_reasoning_trace_field = "No brief explanation"
)
f4_models <- c("Qwen3.5-27B", "Qwen3.5-35B-A3B",
               "gemma-4-12B-it", "gemma-4-26B-A4B-it", "gemma-4-31B-it")

f4 <- settings |>
  filter(fpc == 5, notes == "full", mode %in% names(setting_map),
         model %in% f4_models,
         # baseline at 5 fields/call exists only on the 100-field set
         !(mode == "baseline" & target_set != "legacy 100-field")) |>
  # one run per model x cohort x setting: prefer the 129-field set for
  # variants (where the ablations were run), else the 100-field run
  group_by(cohort, model, mode) |>
  arrange(desc(target_set == "129-field"), .by_group = TRUE) |>
  slice(1) |>
  ungroup() |>
  left_join(model_meta, by = "model") |>
  mutate(
    complete  = coverage >= 1 - 1e-9,
    n_scored  = if_else(complete, labelled, round(coverage * labelled)),
    accuracy  = correct / n_scored,
    ci_method = if_else(complete,
                        "encounter-cluster bootstrap (workbook)",
                        "Wilson score on completed cells (unclustered)"),
    ci_lo = if_else(complete, ci_lo, wilson(correct, n_scored)[, 1]),
    ci_hi = if_else(complete, ci_hi, wilson(correct, n_scored)[, 2]),
    setting = factor(setting_map[mode], levels = setting_map),
    short   = factor(short, levels = model_meta$short[
                       model_meta$model %in% f4_models]),
    cohort  = factor(cohort, levels = c("MIMIC", "OSU"))
  )

f4 |>
  transmute(cohort, model, setting, mode, fields_per_call = fpc, target_set,
            run_id, correct, n_scored, labelled, coverage,
            parse_failure_rate = parse_fail,
            scoring = if_else(complete, "all labelled cells",
                              "completed cells only"),
            accuracy, ci_lower = ci_lo, ci_upper = ci_hi, ci_method) |>
  arrange(cohort, factor(model, levels = f4_models), setting) |>
  write.csv(file.path(out_dir, "fig2d_extraction_settings_source_data.csv"),
            row.names = FALSE)

# Baseline in dark grey; variants in NPG colours
set_col <- c("Baseline"               = "grey30",
             "Free-text output"       = npg[2],   # #4DBBD5
             "Few-shot examples"      = npg[3],   # #00A087
             "No reasoning"           = npg[1],   # #E64B35
             "No supporting citation" = npg[4],   # #3C5488
             "No brief explanation"   = npg[5])   # #F39B7F

base_ref <- f4 |> filter(setting == "Baseline") |>
  select(cohort, short, base = accuracy)

fig2d <- ggplot(f4, aes(setting, accuracy * 100, colour = setting)) +
  geom_hline(yintercept = seq(85, 95, 5), colour = "grey90",
             linewidth = 0.35 / .pt) +
  geom_hline(data = base_ref, aes(yintercept = base * 100),
             colour = "grey30", linetype = "22", linewidth = 0.5 / .pt) +
  geom_errorbar(aes(ymin = ci_lo * 100, ymax = ci_hi * 100),
                width = 0, linewidth = 0.7 / .pt) +
  geom_point(size = 1.6) +
  facet_grid(cohort ~ short) +
  scale_colour_manual(values = set_col, name = NULL, drop = FALSE) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(limits = c(85, 100), breaks = seq(85, 100, 5),
                     expand = c(0, 0)) +
  labs(x = NULL, y = "Answer accuracy (%)") +
  coord_cartesian(clip = "off") +
  guides(colour = guide_legend(nrow = 1, override.aes = list(size = 2))) +
  theme_nature() +
  theme(axis.text.x      = element_blank(),
        axis.ticks.x     = element_blank(),
        strip.text.x     = element_text(size = 6.5, margin = margin(b = 2)),
        strip.text.y     = element_text(size = 7, angle = -90,
                                        margin = margin(l = 2)),
        panel.spacing.x  = unit(2, "mm"),
        panel.spacing.y  = unit(3, "mm"),
        legend.position  = "bottom",
        legend.text      = element_text(size = 6.5),
        legend.key.width = unit(2.5, "mm"))

ggsave(file.path(out_dir, "fig2d_extraction_settings.pdf"), fig2d,
       width = 183, height = 85, units = "mm", device = "pdf",
       useDingbats = FALSE)
ggsave(file.path(out_dir, "fig2d_extraction_settings.png"), fig2d,
       width = 183, height = 85, units = "mm", dpi = 600, bg = "white")

# ---- Console summary --------------------------------------------------------
best <- f3 |>
  group_by(cohort, model) |>
  slice_max(accuracy, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(cohort, model, best_fpc = fpc,
            best_acc = sprintf("%.1f (%.1f-%.1f)", accuracy * 100,
                               ci_lo * 100, ci_hi * 100))
print(as.data.frame(best), row.names = FALSE)
message("Peak within 5-25 fields per call: ",
        sum(best$best_fpc %in% c(5, 10, 25)), " of ", nrow(best))
message("Figures written to ", out_dir)
