#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Supplementary Figure S1: inference running time by fields per call (MIMIC)
#   Running time per record (minutes, log scale) for each open-weight model
#   at 1, 5, 10, 25, 50 and 100 fields per call, baseline prompt (JSON output
#   with model reasoning). One panel per model family.
#
# Input : ../bmt_analysis_results.xlsx, sheet "Runtime" (minutes per record
#         from the original run records)
# Output: ../figs/figS1_runtime.{pdf,png}
#         ../figs/figS1_runtime_source_data.csv
#
# Notes
#   Single runs from the original run records; hardware was not
#   standardized, so no CI is shown.
#   gpt-oss-120b at 100 fields per call was not available.
#
# Style: Nature figure guidelines - Helvetica, 5-7 pt text, 0.5 pt rules,
#   183 mm double-column width, vector PDF. Colours from ggsci "npg".
#
# Run from anywhere:  Rscript figs_scripts/figS1_runtime.R
# Packages: readxl, dplyr, ggplot2 (>= 3.4), ggsci, ggrepel
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  needed <- c("readxl", "dplyr", "ggplot2", "ggsci", "ggrepel")
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

# ---- Read the Runtime sheet (header row located, not hard-coded) ------------
raw <- read_excel(xlsx_path, sheet = "Runtime", col_names = FALSE,
                  .name_repair = "minimal")
hdr <- which(raw[[1]] == "Cohort" & raw[[2]] == "Model")[1]
if (is.na(hdr)) stop("Header row not found in sheet 'Runtime'.", call. = FALSE)
runtime <- raw[-seq_len(hdr), ]
names(runtime) <- make.unique(as.character(unlist(raw[hdr, ])))
runtime <- runtime[!is.na(runtime[[1]]), ] |>
  transmute(cohort = Cohort, model = Model, mode = Mode,
            fpc = as.numeric(`Fields per call`), target_set = `Target set`,
            minutes = as.numeric(`Minutes per record`))

# ---- Models -----------------------------------------------------------------
models <- tibble::tribble(
  ~model,               ~short,       ~family,
  "Qwen3.5-2B",         "2B",         "Qwen3.5",
  "Qwen3.5-9B",         "9B",         "Qwen3.5",
  "Qwen3.5-27B",        "27B",        "Qwen3.5",
  "Qwen3.5-35B-A3B",    "35B-A3B",    "Qwen3.5",
  "gemma-4-E4B-it",     "E4B",        "Gemma 4",
  "gemma-4-12B-it",     "12B",        "Gemma 4",
  "gemma-4-26B-A4B-it", "26B-A4B",    "Gemma 4",
  "gemma-4-31B-it",     "31B",        "Gemma 4",
  "gpt-oss-20b",        "20B",        "gpt-oss",
  "gpt-oss-120b",       "120B",       "gpt-oss"
)
fpc_levels <- c(1, 5, 10, 25, 50, 100)

dat <- runtime |>
  filter(cohort == "MIMIC", mode == "baseline", fpc %in% fpc_levels) |>
  inner_join(models, by = "model") |>
  filter(!is.na(minutes)) |>
  mutate(family = factor(family, levels = c("Qwen3.5", "Gemma 4", "gpt-oss")),
         x      = match(fpc, fpc_levels)) |>
  # colour slot = size rank within family (smallest = 1)
  group_by(family) |>
  mutate(slot = factor(match(model, models$model[models$family ==
                                                   first(family)]))) |>
  ungroup()
stopifnot(nrow(dat) == 59)   # 10 models x 6 settings, minus gpt-oss-120b @100

dat |>
  transmute(cohort, model, mode, fields_per_call = fpc, target_set,
            minutes_per_record = minutes) |>
  arrange(factor(model, levels = models$model), fields_per_call) |>
  write.csv(file.path(out_dir, "figS1_runtime_source_data.csv"),
            row.names = FALSE)

# End-of-line labels at each model's last available setting
lab <- dat |> group_by(model) |> slice_max(x, n = 1) |> ungroup()

# ---- Style ------------------------------------------------------------------
lw  <- 0.5 / .pt
npg <- pal_npg("nrc")(10)
slot_col <- c("1" = npg[2], "2" = npg[3], "3" = npg[4], "4" = npg[1])

fig <- ggplot(dat, aes(x, minutes, colour = slot, group = model)) +
  geom_hline(yintercept = c(5, 10, 20, 50, 100, 200, 400), colour = "grey92",
             linewidth = 0.35 / .pt) +
  geom_line(linewidth = 0.9 / .pt) +
  geom_point(size = 1.4) +
  ggrepel::geom_text_repel(data = lab, aes(label = short), hjust = 0,
                           nudge_x = 0.25, direction = "y",
                           segment.colour = NA, size = 5.5 / .pt,
                           family = "Helvetica", seed = 1,
                           show.legend = FALSE) +
  facet_wrap(~ family, nrow = 1) +
  scale_colour_manual(values = slot_col, guide = "none") +
  scale_x_continuous(breaks = seq_along(fpc_levels), labels = fpc_levels,
                     expand = expansion(add = c(0.3, 1.1))) +
  scale_y_log10(limits = c(3, 400),
                breaks = c(5, 10, 20, 50, 100, 200, 400),
                expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = "Fields per call",
       y = "Running time per record (min, log scale)") +
  theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(axis.line        = element_line(linewidth = lw, colour = "black"),
        axis.ticks       = element_line(linewidth = lw, colour = "black"),
        axis.ticks.length = unit(1, "mm"),
        axis.text        = element_text(size = 6, colour = "black"),
        axis.text.y      = element_text(margin = margin(r = 2)),
        axis.title       = element_text(size = 7, colour = "black"),
        strip.background = element_blank(),
        strip.text       = element_text(size = 7, colour = "black",
                                        margin = margin(b = 2)),
        panel.spacing.x  = unit(4, "mm"),
        plot.margin      = margin(3, 3, 3, 3))

ggsave(file.path(out_dir, "figS1_runtime.pdf"), fig,
       width = 183, height = 70, units = "mm", device = "pdf",
       useDingbats = FALSE)
ggsave(file.path(out_dir, "figS1_runtime.png"), fig,
       width = 183, height = 70, units = "mm", dpi = 600, bg = "white")

# ---- Console summary --------------------------------------------------------
wide <- reshape(as.data.frame(dat[c("model", "fpc", "minutes")]),
                                   idvar = "model", timevar = "fpc",
                                   direction = "wide")
print(wide, row.names = FALSE)
message("Figure written to ", out_dir)
