# ============================================================
# TDM: SuStaIn Stage vs Diagnosis (Plots + Nonparametric tests)
#
# Goal (TDM only)
# 1) Visualize how diagnoses distribute across ML_Stage
#    - Stage distribution per diagnosis (violin + box + jitter)
#    - Diagnosis composition per stage (stacked bars)
#    - Mean stage per diagnosis with 95% CI
# 2) Stats
#    - Global: Kruskal-Wallis (stage ~ diagnosis)
#    - Posthoc: Dunn test (BH/FDR)
#    - Effect size: epsilon2 (global), Cliff's delta (pairwise)
#
# Inputs
# - SuStaIn assignment file with at least: Proband/Name, ML_Stage, Group
#
# Outputs
# - results_tdm_stage_vs_diagnosis/
#   data/   : cleaned input + tables
#   plots/  : figures
#   stats/  : tests + effect sizes
#
# Notes
# - No GMV here.
# - Diagnosis mapping:
#   Group == 2      -> MDD
#   Group == 3      -> BD
#   Group in 4/5    -> SSD
#   Group in 7/8    -> ANX
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(rstatix)
})

# -----------------------------
# 0) Config
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

assign_path <- file.path(
  base_dir,
  "Output_sustainrun",
  "SuStaIn_assignments",
  "main_subject_subtype_assignment_3subtypes.csv"
)

out_root <- file.path(base_dir, "results_tdm_stage_vs_diagnosis")
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "data"),  showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "plots"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "stats"), showWarnings = FALSE, recursive = TRUE)

stage_col <- "ML_Stage"
group_col <- "Group"

MIN_N_STAGE_BIN <- 5

palette_paper <- c(
  "MDD" = "#BF3E39",
  "BD"  = "#CFB23F",
  "SSD" = "#4C90B5",
  "ANX" = "#A4C6D9"
)

# -----------------------------
# 1) Helpers
# -----------------------------
stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

norm_id <- function(x) {
  tolower(gsub("_", "-", trimws(as.character(x))))
}

safe_as_numeric_flexible <- function(x) {
  if (is.numeric(x)) return(x)
  xx <- as.character(x)
  xx <- trimws(xx)
  xx <- gsub(",", ".", xx, fixed = TRUE)
  suppressWarnings(as.numeric(xx))
}

detect_id_col <- function(df) {
  cand <- c("Proband", "Name", "participant_id", "Participant", "Subject", "subject", "ID", "Id", "names")
  hit <- cand[cand %in% names(df)]
  if (length(hit) > 0) return(hit[1])
  
  nm_low <- tolower(names(df))
  hit2 <- names(df)[nm_low %in% tolower(cand)]
  if (length(hit2) > 0) return(hit2[1])
  
  NULL
}

cliffs_delta <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  nx <- length(x)
  ny <- length(y)
  if (nx == 0 || ny == 0) return(NA_real_)
  cmp <- outer(x, y, FUN = "-")
  (sum(cmp > 0) - sum(cmp < 0)) / (nx * ny)
}

epsilon_sq_kruskal <- function(H, n, k) {
  if (!is.finite(H) || !is.finite(n) || !is.finite(k) || (n - k) <= 0) return(NA_real_)
  eps <- (H - k + 1) / (n - k)
  if (!is.finite(eps)) return(NA_real_)
  if (eps < 0) eps <- 0
  eps
}

safe_write_csv <- function(df, path) {
  readr::write_csv(df, path, na = "")
}

format_p_text <- function(p) {
  if (is.na(p) || !is.finite(p)) return(NA_character_)
  if (p < 2.23e-308) return("<2.23e-308")
  format(p, scientific = TRUE, digits = 3)
}

# -----------------------------
# 2) Load + clean TDM assignment
# -----------------------------
stop_if_missing(assign_path)
df <- read_csv(assign_path, show_col_types = FALSE)

id_col <- detect_id_col(df)
if (is.null(id_col)) stop("No ID column detected in assignment file.")
if (id_col != "Proband") df <- df %>% rename(Proband = all_of(id_col))

if (!(stage_col %in% names(df))) stop("ML_Stage missing in assignment file: ", assign_path)
if (!(group_col %in% names(df))) stop("Group missing in assignment file: ", assign_path)

df2 <- df %>%
  mutate(
    Proband = norm_id(Proband),
    ML_Stage = safe_as_numeric_flexible(.data[[stage_col]]),
    Group = safe_as_numeric_flexible(.data[[group_col]]),
    Diagnosis = case_when(
      Group == 2         ~ "MDD",
      Group == 3         ~ "BD",
      Group %in% c(4, 5) ~ "SSD",
      Group %in% c(7, 8) ~ "ANX",
      TRUE               ~ NA_character_
    ),
    Diagnosis = factor(Diagnosis, levels = c("MDD", "BD", "SSD", "ANX"))
  ) %>%
  filter(!is.na(Diagnosis), is.finite(ML_Stage))

if (nrow(df2) == 0) {
  stop("After filtering to MDD/BD/SSD/ANX, no rows remain. Check Group mapping.")
}

safe_write_csv(df2, file.path(out_root, "data", "input_tdm_stage_diagnosis_clean.csv"))

# -----------------------------
# 3) Basic tables
# -----------------------------
tab_stage_by_dx <- df2 %>%
  group_by(Diagnosis) %>%
  summarise(
    n = n(),
    stage_min = min(ML_Stage, na.rm = TRUE),
    stage_q25 = as.numeric(quantile(ML_Stage, 0.25, na.rm = TRUE)),
    stage_median = median(ML_Stage, na.rm = TRUE),
    stage_q75 = as.numeric(quantile(ML_Stage, 0.75, na.rm = TRUE)),
    stage_max = max(ML_Stage, na.rm = TRUE),
    mean_stage = mean(ML_Stage, na.rm = TRUE),
    sd_stage = sd(ML_Stage, na.rm = TRUE),
    se_stage = sd_stage / sqrt(n),
    ci_low = mean_stage - 1.96 * se_stage,
    ci_high = mean_stage + 1.96 * se_stage,
    .groups = "drop"
  )

safe_write_csv(tab_stage_by_dx, file.path(out_root, "stats", "desc_stage_by_diagnosis.csv"))

tab_counts <- df2 %>%
  mutate(Stage = as.integer(round(ML_Stage))) %>%
  group_by(Stage, Diagnosis) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Stage) %>%
  mutate(
    n_stage_total = sum(n),
    pct_within_stage = ifelse(n_stage_total > 0, 100 * n / n_stage_total, NA_real_)
  ) %>%
  ungroup()

safe_write_csv(tab_counts, file.path(out_root, "data", "counts_diagnosis_by_stage.csv"))

df_mean <- df2 %>%
  group_by(Diagnosis) %>%
  summarise(
    n = n(),
    mean_stage = mean(ML_Stage, na.rm = TRUE),
    sd_stage = sd(ML_Stage, na.rm = TRUE),
    se = sd_stage / sqrt(n),
    ci_low = mean_stage - 1.96 * se,
    ci_high = mean_stage + 1.96 * se,
    .groups = "drop"
  )

safe_write_csv(df_mean, file.path(out_root, "stats", "mean_stage_by_diagnosis_ci.csv"))

# -----------------------------
# 4) PLOTS
# -----------------------------

# Plot A: Violin + box + jitter
p1 <- ggplot(df2, aes(x = Diagnosis, y = ML_Stage, fill = Diagnosis, color = Diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.2, linewidth = 0.7) +
  geom_boxplot(
    width = 0.14,
    outlier.shape = NA,
    alpha = 0.9,
    linewidth = 0.7,
    color = "black"
  ) +
  geom_jitter(
    width = 0.14,
    height = 0,
    alpha = 0.25,
    size = 0.9
  ) +
  scale_fill_manual(values = palette_paper) +
  scale_color_manual(values = palette_paper) +
  labs(
    title = "SuStaIn stage distribution by diagnosis (TDM)",
    x = "Diagnosis",
    y = "ML Stage"
  ) +
  theme_minimal(base_size = 15) +
  theme(
    axis.text.x = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

ggsave(
  file.path(out_root, "plots", "plot_stage_by_diagnosis_violin_box.png"),
  p1,
  width = 10,
  height = 6.5,
  dpi = 400
)

# Plot B: Diagnosis composition within each stage
df_stack <- tab_counts %>%
  filter(n_stage_total >= MIN_N_STAGE_BIN) %>%
  mutate(Stage = factor(Stage, levels = sort(unique(Stage))))

p2 <- ggplot(df_stack, aes(x = Stage, y = pct_within_stage, fill = Diagnosis)) +
  geom_col(color = "black", linewidth = 0.25) +
  scale_fill_manual(values = palette_paper) +
  geom_text(
    data = df_stack %>% filter(!is.na(pct_within_stage) & pct_within_stage >= 5),
    aes(label = paste0(round(pct_within_stage, 1), "%")),
    position = position_stack(vjust = 0.5),
    size = 3.8,
    fontface = "bold"
  ) +
  labs(
    title = "Diagnosis composition within each SuStaIn stage (TDM)",
    x = "Stage",
    y = "% within stage"
  ) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.02))) +
  theme_minimal(base_size = 15) +
  theme(
    axis.text.x = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank()
  )

ggsave(
  file.path(out_root, "plots", "plot_diagnosis_within_stage_stacked100.png"),
  p2,
  width = 12,
  height = 6.5,
  dpi = 400
)

# Plot C: Mean stage per diagnosis with 95% CI
p3 <- ggplot(df_mean, aes(x = Diagnosis, y = mean_stage, color = Diagnosis)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15, linewidth = 0.8) +
  scale_color_manual(values = palette_paper) +
  labs(
    title = "Mean SuStaIn stage by diagnosis (TDM)",
    x = "Diagnosis",
    y = "Mean ML_Stage (95% CI)"
  ) +
  theme_minimal(base_size = 15) +
  theme(
    axis.text.x = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

ggsave(
  file.path(out_root, "plots", "plot_mean_stage_by_diagnosis_ci.png"),
  p3,
  width = 9,
  height = 6,
  dpi = 400
)

# -----------------------------
# 5) STATS: Stage ~ Diagnosis
# -----------------------------
kw <- kruskal.test(ML_Stage ~ Diagnosis, data = df2)

kw_p <- kw$p.value
if (!is.finite(kw_p) || kw_p == 0) kw_p <- 2.23e-308

kw_res <- tibble(
  method = "Kruskal-Wallis",
  H = unname(kw$statistic),
  df = unname(kw$parameter),
  p = kw_p,
  p_text = format_p_text(kw_p),
  n = nrow(df2),
  epsilon2 = epsilon_sq_kruskal(unname(kw$statistic), nrow(df2), nlevels(df2$Diagnosis))
)

safe_write_csv(kw_res, file.path(out_root, "stats", "kruskal_stage_by_diagnosis.csv"))

# -----------------------------
# 5b) Posthoc: Dunn + BH
# -----------------------------
group_sizes <- table(df2$Diagnosis)
group_sizes <- group_sizes[levels(df2$Diagnosis)]

dunn_df <- df2 %>%
  rstatix::dunn_test(ML_Stage ~ Diagnosis, p.adjust.method = "BH") %>%
  as_tibble() %>%
  mutate(
    group1 = as.character(group1),
    group2 = as.character(group2),
    p = ifelse(!is.finite(p) | p == 0, 2.23e-308, p),
    p.adj = ifelse(!is.finite(p.adj) | p.adj == 0, 2.23e-308, p.adj),
    n1 = as.numeric(group_sizes[group1]),
    n2 = as.numeric(group_sizes[group2])
  )

dunn_df$cliffs_delta <- mapply(
  FUN = function(a, b) {
    xa <- df2$ML_Stage[df2$Diagnosis == a]
    xb <- df2$ML_Stage[df2$Diagnosis == b]
    cliffs_delta(xa, xb)
  },
  dunn_df$group1,
  dunn_df$group2
)

post_df <- dunn_df %>%
  transmute(
    dx1 = group1,
    dx2 = group2,
    n1 = n1,
    n2 = n2,
    z = statistic,
    p = p,
    p_text = vapply(p, format_p_text, character(1)),
    q_BH = p.adj,
    q_BH_text = vapply(q_BH, format_p_text, character(1)),
    cliffs_delta_dx1_vs_dx2 = cliffs_delta
  ) %>%
  arrange(q_BH)

safe_write_csv(post_df, file.path(out_root, "stats", "posthoc_dunn_stage_by_diagnosis_BH.csv"))

# -----------------------------
# 6) Console output
# -----------------------------
cat("\nOK. TDM stage-vs-diagnosis finished.\n")
cat("Output in:\n", out_root, "\n\n")

cat("Global Kruskal-Wallis:\n")
print(kw_res)

cat("\nPosthoc Dunn:\n")
print(post_df)

graphics.off()
