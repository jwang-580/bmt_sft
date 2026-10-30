#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# Supplementary Figure S2: accuracy for each CIBMTR Form 2400 question,
#   Gemma-4-12B (base) vs BMT-GRAFT-12B (fine-tuned), MIMIC and OSU.
#   100 questions shared by both runs x 20 records per cohort, 5 fields per
#   call, baseline prompt. Questions grouped by form section.
#
# Input : ../data/form_paired_cells.json  (matched cells, 2 x 2,000)
# Output: ../figs/figS2_question_accuracy.{pdf,png}
#         ../figs/figS2_question_accuracy_source_data.csv
#
# Notes
#   Accuracy = answer agreement with the reference (answer_match), the same
#   definition as the workbook (base 93.3% MIMIC, 93.9% OSU on these cells).
#   n = 20 records per question, so accuracy moves in 5-point steps; no CI.
#   Overlapping points = identical accuracy for both models.
#   The 29 questions run only by BMT-GRAFT-12B are not shown.
#
# Style: Nature figure guidelines - Helvetica, 5-7 pt text, 0.5 pt rules,
#   183 mm width, vector PDF. Colours from ggsci "npg".
#
# Run from anywhere:  Rscript Figs_scripts/figS2_question_accuracy.R
# Packages: jsonlite, dplyr, tidyr, ggplot2 (>= 3.4), ggsci
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  needed <- c("jsonlite", "dplyr", "tidyr", "ggplot2", "ggsci")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Install missing packages first: install.packages(c(",
         paste0('"', missing, '"', collapse = ", "), "))", call. = FALSE)
  }
  library(jsonlite)
  library(dplyr)
  library(tidyr)
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
json_path <- file.path(proj_dir, "data", "form_paired_cells.json")
out_dir   <- file.path(proj_dir, "figs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(json_path))

# ---- Data -------------------------------------------------------------------
raw <- fromJSON(json_path, simplifyVector = FALSE)

cells <- bind_rows(lapply(names(raw), function(co) {
  bind_rows(lapply(raw[[co]], function(x) tibble(
    cohort   = co,
    case     = x$case,
    field_id = as.integer(x$field_id),
    name     = x$field_name,
    section  = x$field_category,
    base     = isTRUE(x$base$answer_match),
    fine     = isTRUE(x$fine$answer_match))))
}))
stopifnot(nrow(cells) == 4000)

sections <- c(recipient_information = "Recipient",
              donor_information     = "Donor",
              hct                   = "HCT",
              recipient_status      = "Recipient\nstatus",
              comorbidity           = "Comorbidity",
              conditioning          = "Preparative\nregimen")
cohort_lab <- c(mimic = "MIMIC", osu = "OSU")

# Short question label: text before the first blank/colon/parenthesis
short_label <- function(s) {
  s <- sub("\\s*(__|:|\\(|\\?|--).*$", "", enc2utf8(s), perl = TRUE)
  s <- iconv(s, "UTF-8", "ASCII//TRANSLIT", sub = "")
  s <- trimws(gsub("\\s+", " ", s))
  ifelse(nchar(s) > 42, paste0(substr(s, 1, 40), "..."), s)
}

acc <- cells |>
  group_by(cohort, field_id, name, section) |>
  summarise(n = n(), base = mean(base) * 100, fine = mean(fine) * 100,
            .groups = "drop") |>
  mutate(label   = sprintf("Q%d  %s", field_id, short_label(name)),
         section = factor(sections[section], levels = sections),
         cohort  = factor(cohort_lab[cohort], levels = cohort_lab),
         change  = case_when(fine > base ~ "Improved",
                             fine < base ~ "Worse",
                             TRUE        ~ "Unchanged"))
stopifnot(all(acc$n == 20), n_distinct(acc$field_id) == 100)

lab_levels <- acc |> distinct(field_id, label) |> arrange(desc(field_id)) |>
  pull(label)
acc <- acc |> mutate(label = factor(label, levels = lab_levels))

acc |>
  transmute(cohort, question = field_id, question_text = name, section,
            n_records = n, accuracy_base = base, accuracy_finetuned = fine,
            difference = fine - base) |>
  arrange(cohort, question) |>
  write.csv(file.path(out_dir, "figS2_question_accuracy_source_data.csv"),
            row.names = FALSE)

# ---- Style ------------------------------------------------------------------
lw  <- 0.5 / .pt
npg <- pal_npg("nrc")(10)
seg_col <- c(Improved = npg[3], Worse = npg[1], Unchanged = "grey80")
mod_col <- c("Gemma-4-12B (base)" = "grey55", "BMT-GRAFT-12B" = npg[4])

pts <- acc |>
  pivot_longer(c(base, fine), names_to = "model", values_to = "accuracy") |>
  mutate(model = factor(if_else(model == "base", "Gemma-4-12B (base)",
                                "BMT-GRAFT-12B"), levels = names(mod_col)))

fig <- ggplot(acc, aes(y = label)) +
  geom_blank() +                      # trains y in question order first
  geom_vline(xintercept = seq(0, 100, 25), colour = "grey92",
             linewidth = 0.35 / .pt) +
  geom_segment(data = filter(acc, change != "Unchanged"),
               aes(x = base, xend = fine, yend = label, colour = change),
               linewidth = 0.9 / .pt,
               arrow = arrow(length = unit(0.9, "mm"), type = "closed")) +
  geom_point(data = pts, aes(x = accuracy, fill = model), shape = 21,
             size = 1.1, stroke = 0.3 / .pt, colour = "white") +
  facet_grid(section ~ cohort, scales = "free_y", space = "free_y",
             switch = "y") +
  scale_colour_manual(values = seg_col, breaks = c("Improved", "Worse"),
                      name = NULL) +
  scale_fill_manual(values = mod_col, name = NULL) +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 25),
                     expand = expansion(add = c(1, 2))) +
  labs(x = "Accuracy (%)", y = NULL) +
  guides(fill   = guide_legend(order = 1, override.aes = list(size = 2)),
         colour = guide_legend(order = 2)) +
  theme_classic(base_size = 7, base_family = "Helvetica") +
  theme(axis.line          = element_line(linewidth = lw, colour = "black"),
        axis.ticks         = element_line(linewidth = lw, colour = "black"),
        axis.ticks.y       = element_blank(),
        axis.ticks.length  = unit(1, "mm"),
        axis.text          = element_text(size = 5, colour = "black"),
        axis.text.y        = element_text(hjust = 0),
        axis.title         = element_text(size = 7, colour = "black"),
        strip.background   = element_blank(),
        strip.placement    = "outside",
        strip.text.x       = element_text(size = 7, face = "bold"),
        strip.text.y.left  = element_text(size = 6, angle = 0, hjust = 0,
                                          face = "bold", lineheight = 0.9),
        panel.spacing.y    = unit(1.5, "mm"),
        panel.spacing.x    = unit(4, "mm"),
        legend.position    = "top",
        legend.text        = element_text(size = 6),
        legend.key.size    = unit(3, "mm"),
        legend.margin      = margin(0, 0, 0, 0),
        plot.margin        = margin(3, 4, 3, 3))

ggsave(file.path(out_dir, "figS2_question_accuracy.pdf"), fig,
       width = 183, height = 240, units = "mm", device = "pdf",
       useDingbats = FALSE)
ggsave(file.path(out_dir, "figS2_question_accuracy.png"), fig,
       width = 183, height = 240, units = "mm", dpi = 600, bg = "white")

# ---- Console summary --------------------------------------------------------
acc |> group_by(cohort) |>
  summarise(perfect_base = sum(base == 100), perfect_fine = sum(fine == 100),
            base = mean(base), fine = mean(fine),
            improved = sum(change == "Improved"),
            worse = sum(change == "Worse"),
            unchanged = sum(change == "Unchanged")) |>
  as.data.frame() |> print(row.names = FALSE)
message("Figure written to ", out_dir)
