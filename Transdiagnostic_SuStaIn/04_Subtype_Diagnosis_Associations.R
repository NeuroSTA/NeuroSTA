# ===============================
# SuStaIn Subtypes × Diagnosis Analysis – TDM cohort
# K subtypes + Plots + BH posthoc
#
# Diagnoses: MDD, BD, SSD, ANX  (no HC)
#
# IMPORTANT:
# - Uses ONLY the TDM assignment file: main_subject_subtype_assignment_Ksubtypes.csv
# - Multiple testing: BH/FDR (NOT Bonferroni)
# - ANX: Group 7 and 8 -> "ANX" (NOT merged to MDD)
#
# Plot labels:
# - Avoid "100.1%" artifacts by using sum-preserving rounding for labels
#   while keeping bar heights on exact (unrounded) proportions.
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
  library(rstatix)
  library(DescTools)   # Cramer's V
})

# ------------------------------
# 0) Settings
# ------------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))

K <- 3L  # final manuscript solution
alpha_fdr <- 0.05

output_dir <- file.path(base_dir, paste0("results_TDM_diagnosis_vs_subtypes_", K, "subtypes_BH"))
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

assign_path <- file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", paste0("main_subject_subtype_assignment_", K, "subtypes.csv"))
if (!file.exists(assign_path)) stop("Assignment file not found: ", assign_path)

# ------------------------------
# 1) Helper functions
# ------------------------------
as_proband_chr <- function(x) trimws(as.character(x))

derive_subtyp_from_ml <- function(df) {
  if (!("ML_Subtype" %in% names(df))) stop("ML_Subtype column not found.")
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype is all NA.")
  
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  ml_max <- suppressWarnings(max(df$ML_Subtype, na.rm = TRUE))
  
  if (!is.finite(ml_min) || !is.finite(ml_max)) stop("ML_Subtype min/max not finite.")
  
  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding. min=", ml_min, " max=", ml_max, " (expected start 0 or 1).")
  }
  df
}

round_preserve_sum <- function(x, digits = 1, target = 100) {
  # Sum-preserving rounding (prevents 100.1% artifacts in stacked labels)
  out <- rep(NA_real_, length(x))
  ok <- which(is.finite(x))
  if (length(ok) == 0) return(out)
  
  xs <- x[ok]
  fac <- 10^digits
  
  xs_int <- xs * fac
  floor_int <- floor(xs_int)
  remainder <- xs_int - floor_int
  
  target_int <- round(target * fac)
  need <- target_int - sum(floor_int)
  
  adj <- rep(0L, length(xs))
  if (need > 0) {
    idx <- order(remainder, decreasing = TRUE)
    adj[idx[seq_len(min(need, length(idx)))]] <- 1L
  } else if (need < 0) {
    idx <- order(remainder, decreasing = FALSE)
    adj[idx[seq_len(min(abs(need), length(idx)))]] <- -1L
  }
  
  out_vals <- (floor_int + adj) / fac
  out[ok] <- out_vals
  out
}

p_to_signif <- function(p) {
  dplyr::case_when(
    is.na(p) ~ NA_character_,
    p > 0.05 ~ "ns",
    p <= 0.05 & p > 0.01 ~ "*",
    p <= 0.01 & p > 0.001 ~ "**",
    p <= 0.001 & p > 0.0001 ~ "***",
    p <= 0.0001 ~ "****"
  )
}

# ------------------------------
# 2) Read assignment data
# ------------------------------
cluster_data <- readr::read_csv(assign_path, show_col_types = FALSE)

# normalize subject id column
if (!"Proband" %in% names(cluster_data)) {
  if ("Name"  %in% names(cluster_data)) cluster_data <- cluster_data %>% rename(Proband = Name)
  if ("names" %in% names(cluster_data)) cluster_data <- cluster_data %>% rename(Proband = names)
}
if (!"Proband" %in% names(cluster_data)) stop("No subject id column found (expected Proband or Name or names).")
if (!"ML_Subtype" %in% names(cluster_data)) stop("ML_Subtype not found in assignment file.")
if (!"Group" %in% names(cluster_data)) stop("Group not found in assignment file.")

cluster_data <- cluster_data %>%
  mutate(
    Proband = as_proband_chr(Proband),
    # safe normalization; do NOT over-normalize beyond what you need
    Proband = tolower(gsub("_", "-", Proband))
  ) %>%
  derive_subtyp_from_ml()

# ------------------------------
# 3) Map diagnoses (MDD, BD, SSD, ANX only)
# ------------------------------
cluster_data <- cluster_data %>%
  mutate(
    Diagnosis = case_when(
      Group == 2                 ~ "MDD",
      Group == 3                 ~ "BD",
      Group %in% c(4, 5)         ~ "SSD",
      Group %in% c(7, 8)         ~ "ANX",
      TRUE                       ~ NA_character_
    ),
    Diagnosis = factor(Diagnosis, levels = c("MDD", "BD", "SSD", "ANX")),
    Subtyp    = factor(Subtyp, levels = 1:K)
  ) %>%
  filter(!is.na(Diagnosis), !is.na(Subtyp))

if (nrow(cluster_data) == 0) stop("After filtering to MDD/BD/SSD/ANX, no rows remain. Check Group mapping.")

readr::write_csv(cluster_data, file.path(output_dir, "input_main_filtered.csv"))

# ------------------------------
# 4) Frequency table
# ------------------------------
group_cluster_table <- table(cluster_data$Subtyp, cluster_data$Diagnosis)
message(">>> Frequency table Subtyp × Diagnosis - TDM:")
print(group_cluster_table)

readr::write_csv(as.data.frame(group_cluster_table),
                 file.path(output_dir, "frequency_table.csv"))

# ------------------------------
# 5) Global test (Chi2 vs Fisher) + Cramer's V
# ------------------------------
chi_test <- suppressWarnings(chisq.test(group_cluster_table))

if (any(chi_test$expected < 5)) {
  global_test <- fisher.test(group_cluster_table)
  method_used <- "Fisher's Exact Test"
  stat_val <- NA_real_
  df_val   <- NA_real_
} else {
  global_test <- chi_test
  method_used <- "Chi-squared Test"
  stat_val <- unname(global_test$statistic)
  df_val   <- unname(global_test$parameter)
}

cramers_v <- DescTools::CramerV(group_cluster_table)

global_res <- tibble::tibble(
  method     = method_used,
  statistic  = stat_val,
  df         = df_val,
  p.value    = as.numeric(global_test$p.value),
  cramers_v  = as.numeric(cramers_v)
)

message(">>> Global test:")
print(global_res)

readr::write_csv(global_res, file.path(output_dir, "global_test.csv"))

# ------------------------------
# 6) Post-hoc: pairwise subtype comparisons (BH/FDR)
# ------------------------------
subtypes <- levels(cluster_data$Subtyp)
comparisons <- combn(subtypes, 2, simplify = FALSE)

posthoc_list <- lapply(comparisons, function(cmp) {
  sub_data <- cluster_data %>% filter(Subtyp %in% cmp)
  tab <- table(sub_data$Subtyp, sub_data$Diagnosis)
  
  # drop empty rows/cols (prevents NaN in Cramer's V)
  tab <- tab[rowSums(tab) > 0, colSums(tab) > 0, drop = FALSE]
  
  chi_tmp <- suppressWarnings(chisq.test(tab))
  use_fisher <- any(chi_tmp$expected < 5)
  
  test <- if (use_fisher) fisher.test(tab) else chi_tmp
  
  eff <- suppressWarnings(DescTools::CramerV(tab))
  if (!is.finite(eff)) eff <- NA_real_
  
  tibble::tibble(
    Subtyp1   = as.character(cmp[1]),
    Subtyp2   = as.character(cmp[2]),
    method    = if (use_fisher) "Fisher's Exact Test" else "Chi-squared Test",
    statistic = if (use_fisher) NA_real_ else unname(test$statistic),
    df        = if (use_fisher) NA_real_ else unname(test$parameter),
    p.value   = as.numeric(test$p.value),
    cramers_v = as.numeric(eff)
  )
})

posthoc_df <- dplyr::bind_rows(posthoc_list) %>%
  mutate(
    p.adj = p.adjust(p.value, method = "BH"),
    signif = p_to_signif(p.adj)
  ) %>%
  arrange(p.adj)

message(">>> Posthoc subtype-pair tests (BH/FDR) + Cramer's V:")
print(posthoc_df)

readr::write_csv(posthoc_df, file.path(output_dir, "posthoc_tests_BH.csv"))

# ------------------------------
# 7) Plots
# styled like acute vs. remittiert
# separate figures for poster, colored
# ------------------------------
df_plot <- as.data.frame(group_cluster_table) %>%
  rename(Subtyp = Var1, Diagnosis = Var2, Count = Freq) %>%
  mutate(
    Subtyp = factor(Subtyp, levels = as.character(1:K)),
    Diagnosis = factor(Diagnosis, levels = c("MDD", "BD", "SSD", "ANX"))
  )

palette_paper <- c(
  "MDD" = "#BF3E39",
  "BD" = "#CFB23F",
  "SSD" = "#4C90B5",
  "ANX" = "#A4C6D9"
)

# --- Plot 1: Counts (dodged) ---
p_counts <- ggplot(df_plot, aes(x = Subtyp, y = Count, fill = Diagnosis)) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.68,
    color = "black",
    linewidth = 0.3
  ) +
  scale_fill_manual(values = palette_paper, drop = FALSE) +
  labs(
    x = "SuStaIn subtype",
    y = "Count",
    title = "Diagnosis by SuStaIn subtype - TDM"
  ) +
  theme_classic(base_size = 18) +
  theme(
    legend.title = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    axis.text.x = element_text(size = 16, face = "bold", color = "black"),
    axis.text.y = element_text(size = 15, face = "bold", color = "black"),
    legend.text = element_text(size = 15, face = "bold"),
    panel.grid = element_blank()
  )

ggplot2::ggsave(
  file.path(output_dir, "barplot_counts_dodged.png"),
  p_counts,
  width = 8.5,
  height = 6.2,
  dpi = 300,
  bg = "white"
)

ggplot2::ggsave(
  file.path(output_dir, "barplot_counts_dodged.pdf"),
  p_counts,
  width = 8.5,
  height = 6.2,
  bg = "white"
)

# --- Plot 2: Within-subtype percentages (stacked to 100%) ---
df_within <- df_plot %>%
  group_by(Subtyp) %>%
  mutate(
    Total_subtyp = sum(Count),
    Percentage_within_raw = ifelse(Total_subtyp > 0, 100 * Count / Total_subtyp, NA_real_)
  ) %>%
  ungroup() %>%
  group_by(Subtyp) %>%
  mutate(
    Percentage_within_lab = round_preserve_sum(Percentage_within_raw, digits = 1, target = 100)
  ) %>%
  ungroup()

n_per_subtyp <- df_within %>%
  group_by(Subtyp) %>%
  summarise(n = unique(Total_subtyp), .groups = "drop")

p_within <- ggplot(df_within, aes(x = Subtyp, y = Percentage_within_raw, fill = Diagnosis)) +
  geom_col(
    width = 0.68,
    color = "black",
    linewidth = 0.3
  ) +
  geom_text(
    data = df_within %>% filter(!is.na(Percentage_within_lab) & Percentage_within_raw >= 3),
    aes(label = paste0(sprintf("%.1f", Percentage_within_lab), "%")),
    position = position_stack(vjust = 0.5),
    size = 5.2,
    fontface = "bold",
    color = "black"
  ) +
  geom_text(
    data = n_per_subtyp,
    inherit.aes = FALSE,
    aes(x = Subtyp, y = 103, label = paste0("n = ", n)),
    size = 5.2,
    fontface = "bold",
    color = "black"
  ) +
  scale_fill_manual(values = palette_paper, drop = FALSE) +
  scale_y_continuous(
    limits = c(0, 110),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = "SuStaIn subtype",
    y = "Percent within subtype",
    title = "Diagnosis distribution within SuStaIn subtypes - TDM"
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 18) +
  theme(
    legend.title = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    axis.text.x = element_text(size = 16, face = "bold", color = "black"),
    axis.text.y = element_text(size = 15, face = "bold", color = "black"),
    legend.text = element_text(size = 15, face = "bold"),
    panel.grid = element_blank()
  )

ggplot2::ggsave(
  file.path(output_dir, "barplot_withinSubtype_percent.png"),
  p_within,
  width = 8.5,
  height = 6.2,
  dpi = 300,
  bg = "white"
)

ggplot2::ggsave(
  file.path(output_dir, "barplot_withinSubtype_percent.pdf"),
  p_within,
  width = 8.5,
  height = 6.2,
  bg = "white"
)

# --- Plot 3: Across-subtypes as % of total sample (stacked, dual axis) ---
N_total <- nrow(cluster_data)

n_per_subtype_true <- cluster_data %>%
  count(Subtyp, name = "n_subtyp") %>%
  mutate(Subtyp_f = factor(Subtyp, levels = as.character(1:K)))

df_total <- df_plot %>%
  left_join(n_per_subtype_true %>% select(Subtyp = Subtyp_f, n_subtyp), by = "Subtyp") %>%
  mutate(
    Percentage_total_raw = 100 * Count / N_total,
    Subtyp_f = factor(Subtyp, levels = as.character(1:K)),
    target_stack = 100 * n_subtyp / N_total
  ) %>%
  group_by(Subtyp_f) %>%
  mutate(
    Percentage_total_lab = round_preserve_sum(
      Percentage_total_raw,
      digits = 1,
      target = unique(target_stack)
    )
  ) %>%
  ungroup()

max_sum_percent <- df_total %>%
  group_by(Subtyp_f) %>%
  summarise(total_pct = sum(Percentage_total_raw), .groups = "drop") %>%
  pull(total_pct) %>%
  max(na.rm = TRUE)

p_total <- ggplot(df_total, aes(x = Subtyp_f, y = Percentage_total_raw, fill = Diagnosis)) +
  geom_col(
    width = 0.68,
    color = "black",
    linewidth = 0.3
  ) +
  geom_text(
    data = df_total %>% filter(Percentage_total_raw >= 1),
    aes(label = paste0(sprintf("%.1f", Percentage_total_lab), "%")),
    position = position_stack(vjust = 0.5),
    size = 5.2,
    fontface = "bold",
    color = "black"
  ) +
  geom_text(
    data = n_per_subtype_true,
    inherit.aes = FALSE,
    aes(x = Subtyp_f, y = max_sum_percent + 1.0, label = paste0("n = ", n_subtyp)),
    size = 5.2,
    fontface = "bold",
    color = "black"
  ) +
  scale_fill_manual(values = palette_paper, drop = FALSE) +
  labs(
    x = "SuStaIn subtype",
    y = "% of total sample",
    title = "Diagnosis across SuStaIn subtypes - TDM)"
  ) +
  scale_y_continuous(
    limits = c(0, max_sum_percent * 1.18),
    sec.axis = sec_axis(~ . * N_total / 100, name = "Count")
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 18) +
  theme(
    legend.title = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    axis.title.y.right = element_text(size = 18, face = "bold"),
    axis.text.x = element_text(size = 16, face = "bold", color = "black"),
    axis.text.y = element_text(size = 15, face = "bold", color = "black"),
    axis.text.y.right = element_text(size = 15, face = "bold", color = "black"),
    legend.text = element_text(size = 15, face = "bold"),
    panel.grid = element_blank()
  )

ggplot2::ggsave(
  file.path(output_dir, "barplot_totalSample_percent_dualAxis.png"),
  p_total,
  width = 9.5,
  height = 6.8,
  dpi = 300,
  bg = "white"
)

ggplot2::ggsave(
  file.path(output_dir, "barplot_totalSample_percent_dualAxis.pdf"),
  p_total,
  width = 9.5,
  height = 6.8,
  bg = "white"
)
# --- Plot 3: Across-subtypes as % of total sample (stacked, dual axis) ---
N_total <- nrow(cluster_data)

n_per_subtype_true <- cluster_data %>%
  count(Subtyp, name = "n_subtyp") %>%
  mutate(Subtyp_f = factor(Subtyp, levels = as.character(1:K)))

df_total <- df_plot %>%
  left_join(n_per_subtype_true %>% select(Subtyp = Subtyp_f, n_subtyp), by = "Subtyp") %>%
  mutate(
    Percentage_total_raw = 100 * Count / N_total,
    Subtyp_f = factor(Subtyp, levels = as.character(1:K)),
    target_stack = 100 * n_subtyp / N_total
  ) %>%
  group_by(Subtyp_f) %>%
  mutate(
    Percentage_total_lab = round_preserve_sum(Percentage_total_raw, digits = 1, target = unique(target_stack))
  ) %>%
  ungroup()

max_sum_percent <- df_total %>%
  group_by(Subtyp_f) %>%
  summarise(total_pct = sum(Percentage_total_raw), .groups = "drop") %>%
  pull(total_pct) %>%
  max(na.rm = TRUE)

p_total <- ggplot(df_total, aes(x = Subtyp_f, y = Percentage_total_raw, fill = Diagnosis)) +
  geom_col(width = 0.65, color = "black", linewidth = 0.3) +
  geom_text(
    data = df_total %>% filter(Percentage_total_raw >= 1),
    aes(label = paste0(sprintf("%.1f", Percentage_total_lab), "%")),
    position = position_stack(vjust = 0.5),
    size = 4.5, fontface = "bold", color = "black"
  ) +
  geom_text(
    data = n_per_subtype_true,
    inherit.aes = FALSE,
    aes(x = Subtyp_f, y = max_sum_percent + 1.0, label = paste0("n = ", n_subtyp)),
    size = 5, fontface = "bold", color = "black"
  ) +
  scale_fill_manual(values = palette_paper) +
  labs(
    x = "SuStaIn Subtype",
    y = "% of total sample",
    title = "Diagnosis across SuStaIn subtypes - TDM"
  ) +
  scale_y_continuous(
    limits = c(0, max_sum_percent * 1.18),
    sec.axis = sec_axis(~ . * N_total / 100, name = "Participants (n)")
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x  = element_text(face = "bold", color = "black"),
    axis.text.y  = element_text(face = "bold", color = "black"),
    axis.text.y.right = element_text(face = "bold", color = "black"),
    axis.title   = element_text(face = "bold"),
    axis.title.y.right = element_text(face = "bold"),
    legend.title = element_blank(),
    legend.text  = element_text(face = "bold", color = "black"),
    plot.title   = element_text(face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

ggplot2::ggsave(file.path(output_dir, "barplot_totalSample_percent_dualAxis.png"),
                p_total, width = 12, height = 9, dpi = 400)

# ------------------------------
# 8) Quick sanity checks (optional, but good for paper QC)
# ------------------------------
# Within-subtype labels must sum to 100.0 per subtype (ignoring NA)
within_sums <- df_within %>%
  group_by(Subtyp) %>%
  summarise(sum_labels = sum(Percentage_within_lab, na.rm = TRUE), .groups = "drop")

# Total-sample labels must sum to the subtype share of total sample
total_sums <- df_total %>%
  group_by(Subtyp_f) %>%
  summarise(
    sum_labels = sum(Percentage_total_lab, na.rm = TRUE),
    target = unique(target_stack),
    .groups = "drop"
  )

readr::write_csv(within_sums, file.path(output_dir, "QC_withinSubtype_label_sums.csv"))
readr::write_csv(total_sums, file.path(output_dir, "QC_totalSample_label_sums.csv"))

message("[OK] TDM diagnosis-vs-subtypes analysis finished. Results in: ", output_dir)
graphics.off()
