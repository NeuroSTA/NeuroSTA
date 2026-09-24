# ===============================
# SuStaIn subtypes x clinical status (acute vs. remitted)
# Cohort-wise: TDM, MDD, Angst, BD, SZ
# - Global test: Chi-squared, Fisher if expected cell counts are small
# - Post hoc: Fisher 2x2
# - Omnibus BH family: six main symptom scales plus clinical status,
#   calculated in script 06 after this script writes the raw status p value.
# - Status post hoc: Benjamini-Hochberg (BH) within status comparisons.
# - HC excluded if identifiable
# - Outputs: write_csv2 (Excel-DE: ; and decimal comma)
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(purrr)
  library(DescTools)  # CramerV, Phi
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

output_root <- file.path(base_dir, "Output Scripte")
if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)

data_all_path <- getOption("SUSTAIN_CLINICAL_STATUS_FILE", Sys.getenv("SUSTAIN_CLINICAL_STATUS_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt_withGlobalRatings.csv")))

cohorts <- list(
  TDM  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

output_dir_global <- file.path(output_root, "status_acute_vs_remitted_all_cohorts")
if (!dir.exists(output_dir_global)) dir.create(output_dir_global, recursive = TRUE)

write_out <- function(df, path) readr::write_csv2(df, path, na = "")

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

norm_id <- function(x) {
  tolower(gsub("_", "-", trimws(as.character(x))))
}

derive_subtyp_from_ml <- function(df) {
  if (!("ML_Subtype" %in% names(df))) stop("ML_Subtype column not found in assignment file.")
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype is all NA.")

  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  if (!is.finite(ml_min)) stop("ML_Subtype min not finite.")

  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding. Expected start at 0 or 1. min=", ml_min)
  }
  df
}

rowmax_na <- function(...) {
  m <- cbind(...)
  out <- apply(m, 1, function(z) {
    if (all(is.na(z))) return(NA_real_)
    suppressWarnings(max(z, na.rm = TRUE))
  })
  as.numeric(out)
}

flag_gt <- function(x, thr) {
  ifelse(is.na(x), NA_real_, ifelse(x > thr, 1, 0))
}

flag_ge <- function(x, thr) {
  ifelse(is.na(x), NA_real_, ifelse(x >= thr, 1, 0))
}

minus99_to_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[x %in% c(-99, -2)] <- NA_real_
  x
}

safe_rowsum_na <- function(df, cols) {
  if (length(cols) == 0) return(rep(NA_real_, nrow(df)))
  mat <- as.matrix(df[, cols, drop = FALSE])

  out <- apply(mat, 1, function(z) {
    z <- suppressWarnings(as.numeric(z))
    if (all(is.na(z))) {
      NA_real_
    } else {
      sum(z, na.rm = TRUE)
    }
  })
  as.numeric(out)
}

p_adjust_bh_safe <- function(p) {
  out <- rep(NA_real_, length(p))
  idx <- which(is.finite(p))
  if (length(idx) > 0) out[idx] <- p.adjust(p[idx], method = "BH")
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

safe_cramers_v <- function(tab) {
  if (length(dim(tab)) != 2) return(NA_real_)
  if (any(dim(tab) < 2)) return(NA_real_)
  if (sum(tab) <= 0) return(NA_real_)
  out <- tryCatch(as.numeric(CramerV(tab)), error = function(e) NA_real_)
  out
}

safe_phi <- function(tab2x2) {
  if (!all(dim(tab2x2) == c(2, 2))) return(NA_real_)
  out <- tryCatch(as.numeric(Phi(tab2x2)), error = function(e) NA_real_)
  out
}

safe_fisher_2x2 <- function(tab2x2) {
  ft <- fisher.test(tab2x2)
  list(
    p = as.numeric(ft$p.value),
    or = as.numeric(unname(ft$estimate)),
    or_low = as.numeric(ft$conf.int[1]),
    or_high = as.numeric(ft$conf.int[2])
  )
}

hama_items <- function(df) grep("^HAMA[0-9]+$", names(df), value = TRUE)

ymrs_items <- function(df) grep("^YMRS[0-9]+$", names(df), value = TRUE)

hamd17_items <- function(df) {
  it <- paste0("HAMD", 1:17)
  it[it %in% names(df)]
}

# -----------------------------
# 1) Load master data
# -----------------------------
stop_if_missing(data_all_path)
status_data_md5 <- unname(tools::md5sum(data_all_path))
data_all <- read_csv(data_all_path, show_col_types = FALSE)

if (!"Proband" %in% names(data_all)) {
  if ("Name" %in% names(data_all)) data_all <- data_all %>% rename(Proband = Name)
  if ("names" %in% names(data_all)) data_all <- data_all %>% rename(Proband = names)
}
if (!"Proband" %in% names(data_all)) stop("Proband column missing in data_all.")

data_all <- data_all %>% mutate(Proband = norm_id(Proband))

# -----------------------------
# 2) Variables for status rule
# -----------------------------
trigger_vars_items <- c("SANS7","SANS12","SANS16","SANS21","SAPS7","SAPS20","SAPS25","SAPS34")
sum_targets <- c("HAMA_Sum","HAMD_Sum17","YMRS_Sum")

missing_trigger_cols <- setdiff(trigger_vars_items, names(data_all))
if (length(missing_trigger_cols) > 0) {
  stop(
    "Required SANS/SAPS global-rating columns are missing: ",
    paste(missing_trigger_cols, collapse = ", "),
    ". Check SUSTAIN_CLINICAL_STATUS_FILE."
  )
}

# -----------------------------
# 3) Run one cohort
# -----------------------------
run_one_cohort_status <- function(cohort_name, assign_path) {

  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")

  stop_if_missing(assign_path)

  out_dir <- file.path(output_dir_global, cohort_name)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  cluster_data <- read_csv(assign_path, show_col_types = FALSE)

  if (!"Proband" %in% names(cluster_data)) {
    if ("Name" %in% names(cluster_data)) cluster_data <- cluster_data %>% rename(Proband = Name)
    if ("names" %in% names(cluster_data)) cluster_data <- cluster_data %>% rename(Proband = names)
  }
  if (!"Proband" %in% names(cluster_data)) stop("Proband column missing in assignment file: ", assign_path)

  cluster_data <- cluster_data %>%
    mutate(Proband = norm_id(Proband)) %>%
    derive_subtyp_from_ml() %>%
    filter(Subtyp %in% 1:3) %>%
    mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
    dplyr::select(Proband, Subtyp)

  merged <- inner_join(cluster_data, data_all, by = "Proband")
  if (nrow(merged) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)

  # -----------------------------
  # Detect Group column if available
  # -----------------------------
  group_found <- FALSE
  if (!"Group" %in% names(merged)) {
    gcols <- grep("^group($|\\.|_x|_y)", names(merged), ignore.case = TRUE, value = TRUE)
    if (length(gcols) != 1L) stop("Exactly one Group column required after join; found: ", paste(gcols, collapse = ", "))
    merged <- merged %>% rename(Group = all_of(gcols))
  }
  group_found <- TRUE

  merged <- merged %>%
    mutate(Group_num = suppressWarnings(as.numeric(Group))) %>%
    mutate(is_HC = ifelse(is.finite(Group_num) & Group_num == 1, TRUE, FALSE))
  if (any(!is.finite(merged$Group_num))) stop("Missing or nonnumeric Group values after join.")

  # -----------------------------
  # Clean inputs and compute sums
  # -----------------------------
  missing_after_join <- setdiff(trigger_vars_items, names(merged))
  if (length(missing_after_join) > 0) {
    stop("Required SANS/SAPS trigger columns are missing after the join for cohort ", cohort_name, ": ",
         paste(missing_after_join, collapse = ", "))
  }
  merged <- merged %>%
    mutate(across(all_of(trigger_vars_items), minus99_to_na))

  missing_sum_targets <- setdiff(sum_targets, names(merged))
  if (length(missing_sum_targets) > 0L) {
    stop("Required direct status totals are missing: ", paste(missing_sum_targets, collapse = ", "))
  }
  merged <- merged %>%
    mutate(across(all_of(sum_targets), minus99_to_na))

  # -----------------------------
  # Missingness report
  # -----------------------------
  miss_vars <- unique(c(trigger_vars_items, sum_targets))

  missing_df <- tibble(
    cohort = cohort_name,
    variable = miss_vars,
    n_total = nrow(merged),
    n_missing = sapply(miss_vars, function(v) sum(!is.finite(merged[[v]]))),
    n_nonmissing = sapply(miss_vars, function(v) sum(is.finite(merged[[v]])))
  ) %>%
    mutate(missing_pct = round(100 * n_missing / n_total, 2))

  write_out(missing_df, file.path(out_dir, "missingness_status_inputs.csv"))

  # -----------------------------
  # Define status
  # acute if any trigger is positive
  # remitted only if all 11 triggers are observed and none is positive
  # NA if no trigger is positive but at least one required trigger is missing
  # -----------------------------
  f_SANS7  <- flag_gt(merged$SANS7, 2)
  f_SANS12 <- flag_gt(merged$SANS12, 2)
  f_SANS16 <- flag_gt(merged$SANS16, 2)
  f_SANS21 <- flag_gt(merged$SANS21, 2)

  f_SAPS7  <- flag_gt(merged$SAPS7, 2)
  f_SAPS20 <- flag_gt(merged$SAPS20, 2)
  f_SAPS25 <- flag_gt(merged$SAPS25, 2)
  f_SAPS34 <- flag_gt(merged$SAPS34, 2)

  f_HAMA <- flag_gt(merged$HAMA_Sum, 19)
  f_HAMD <- flag_gt(merged$HAMD_Sum17, 6)
  f_YMRS <- flag_ge(merged$YMRS_Sum, 4)  # Berk et al., 2008: remission = YMRS < 4; complement = >= 4

  akut_flag <- rowmax_na(
    f_SANS7, f_SANS12, f_SANS16, f_SANS21,
    f_SAPS7, f_SAPS20, f_SAPS25, f_SAPS34,
    f_HAMA, f_HAMD, f_YMRS
  )

  complete_flag <- apply(
    cbind(
      f_SANS7, f_SANS12, f_SANS16, f_SANS21,
      f_SAPS7, f_SAPS20, f_SAPS25, f_SAPS34,
      f_HAMA, f_HAMD, f_YMRS
    ),
    1,
    function(z) all(is.finite(z))
  )

  merged <- merged %>%
    mutate(
      akut_flag = akut_flag,
      complete_flag = complete_flag,
      Status = case_when(
        is_HC ~ NA_character_,
        akut_flag == 1 ~ "acute",
        complete_flag & akut_flag == 0 ~ "remitted",
        TRUE ~ NA_character_
      ),
      Status = factor(Status, levels = c("remitted", "acute"))
    )

  status_df <- merged %>%
    filter(!is.na(Subtyp), !is.na(Status)) %>%
    mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
    dplyr::select(Proband, Subtyp, Status)

  # -----------------------------
  # Descriptives
  # -----------------------------
  subtype_counts_after_join <- merged %>%
    count(Subtyp, name = "n_join") %>%
    complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n_join = 0)) %>%
    arrange(Subtyp)
  write_out(subtype_counts_after_join, file.path(out_dir, "subtype_counts_after_join.csv"))

  tab_long <- status_df %>%
    count(Subtyp, Status, name = "n") %>%
    group_by(Subtyp) %>%
    mutate(Percent = round(100 * n / sum(n), 1)) %>%
    ungroup() %>%
    complete(Subtyp, Status, fill = list(n = 0, Percent = 0))

  write_out(tab_long, file.path(out_dir, "freq_status_by_subtype_long.csv"))

  tab_wide_counts <- tab_long %>%
    dplyr::select(Subtyp, Status, n) %>%
    pivot_wider(names_from = Status, values_from = n, values_fill = 0)

  tab_wide_percent <- tab_long %>%
    dplyr::select(Subtyp, Status, Percent) %>%
    pivot_wider(names_from = Status, values_from = Percent, values_fill = 0)

  write_out(tab_wide_counts, file.path(out_dir, "counts_status_by_subtype_wide.csv"))
  write_out(tab_wide_percent, file.path(out_dir, "percent_status_by_subtype_wide.csv"))

  # -----------------------------
  # Global test: Chi-squared, Fisher if expected counts < 5
  # -----------------------------
  tbl <- xtabs(~ Status + Subtyp, data = status_df)

  chi_tmp <- suppressWarnings(chisq.test(tbl))
  use_fisher <- any(chi_tmp$expected < 5)

  if (use_fisher) {
    gt <- fisher.test(tbl)
    method_used <- "Fisher's Exact Test"
    statistic <- NA_real_
    df_val <- NA_real_
    p_val <- as.numeric(gt$p.value)
  } else {
    gt <- chi_tmp
    method_used <- "Chi-squared Test"
    statistic <- as.numeric(unname(gt$statistic))
    df_val <- as.numeric(unname(gt$parameter))
    p_val <- as.numeric(gt$p.value)
  }

  global_out <- tibble(
    cohort = cohort_name,
    method = method_used,
    statistic = statistic,
    df = df_val,
    p.value = p_val,
    cramers_v = safe_cramers_v(tbl),
    assignment_md5 = unname(tools::md5sum(assign_path)),
    status_data_md5 = status_data_md5
  )
  write_out(global_out, file.path(out_dir, "global_status_vs_subtype.csv"))

  # -----------------------------
  # Pairwise subtype comparisons: Fisher 2x2 + BH
  # -----------------------------
  subtypes <- levels(status_df$Subtyp)
  pair_idx <- combn(subtypes, 2, simplify = FALSE)

  posthoc_df <- purrr::map_dfr(pair_idx, function(pp) {
    sub <- status_df %>% filter(Subtyp %in% pp) %>% droplevels()

    tab2 <- xtabs(~ Status + Subtyp, data = sub)

    all_status <- c("remitted", "acute")
    all_sub <- pp
    tab2_full <- matrix(
      0,
      nrow = 2,
      ncol = 2,
      dimnames = list(Status = all_status, Subtyp = all_sub)
    )
    tab2_full[rownames(tab2), colnames(tab2)] <- tab2
    tab2_full <- as.table(tab2_full)

    ft <- safe_fisher_2x2(tab2_full)

    tibble(
      cohort = cohort_name,
      Subtyp1 = pp[1],
      Subtyp2 = pp[2],
      method = "Fisher's Exact Test",
      p.value = ft$p,
      odds_ratio = ft$or,
      or_low = ft$or_low,
      or_high = ft$or_high,
      phi = safe_phi(tab2_full)
    )
  })

  posthoc_df <- posthoc_df %>%
    mutate(
      p.adj.BH = p_adjust_bh_safe(p.value),
      p.adj.signif = p_to_signif(p.adj.BH)
    ) %>%
    arrange(p.adj.BH, p.value)

  write_out(posthoc_df, file.path(out_dir, "posthoc_status_pairs_BH.csv"))

  # -----------------------------
  # One-vs-all: Fisher 2x2 + BH
  # -----------------------------
  ova_df <- purrr::map_dfr(subtypes, function(s) {
    sub <- status_df %>%
      mutate(Subtyp_bin = factor(ifelse(Subtyp == s, s, "Rest"), levels = c("Rest", s))) %>%
      droplevels()

    tab2 <- xtabs(~ Status + Subtyp_bin, data = sub)

    all_status <- c("remitted", "acute")
    all_sub <- c("Rest", s)
    tab2_full <- matrix(
      0,
      nrow = 2,
      ncol = 2,
      dimnames = list(Status = all_status, Subtyp_bin = all_sub)
    )
    tab2_full[rownames(tab2), colnames(tab2)] <- tab2
    tab2_full <- as.table(tab2_full)

    ft <- safe_fisher_2x2(tab2_full)

    tibble(
      cohort = cohort_name,
      Subtyp = s,
      method = "Fisher's Exact Test",
      p.value = ft$p,
      odds_ratio = ft$or,
      or_low = ft$or_low,
      or_high = ft$or_high,
      phi = safe_phi(tab2_full)
    )
  })

  ova_df <- ova_df %>%
    mutate(
      p.adj.BH = p_adjust_bh_safe(p.value),
      p.adj.signif = p_to_signif(p.adj.BH)
    ) %>%
    arrange(p.adj.BH, p.value)

  write_out(ova_df, file.path(out_dir, "one_vs_all_status_BH.csv"))

  # -----------------------------
  # Plots
  # styled to match diagnosis plots
  # separate figures for poster, colored
  # -----------------------------
  palette_paper <- c(
    "acute" = "#BF3E39",
    "remitted" = "#CFB23F",
    "blue_main" = "#4C90B5",
    "blue_light" = "#A4C6D9"
  )

  tab_long <- tab_long %>%
    mutate(Status = factor(Status, levels = c("remitted", "acute")))

  plot_counts <- ggplot(tab_long, aes(x = Subtyp, y = n, fill = Status)) +
    geom_bar(
      stat = "identity",
      position = position_dodge(width = 0.78),
      width = 0.68,
      color = "black",
      linewidth = 0.3
    ) +
    scale_fill_manual(
      values = palette_paper[c("remitted", "acute")],
      drop = FALSE
    ) +
    labs(
      x = "SuStaIn subtype",
      y = "Count",
      title = paste0("Clinical status by subtype - ", cohort_name)
    ) +
    theme_classic(base_size = 18) +
    theme(
      legend.title = element_blank(),
      legend.position = "right",
      plot.title = element_text(size = 20, face = "bold", color = "black"),
      axis.title.x = element_text(size = 18, face = "bold", color = "black"),
      axis.title.y = element_text(size = 18, face = "bold", color = "black"),
      axis.text.x = element_text(size = 16, face = "bold", color = "black"),
      axis.text.y = element_text(size = 15, face = "bold", color = "black"),
      legend.text = element_text(size = 15, face = "bold", color = "black"),
      panel.grid = element_blank()
    )

  plot_percent <- ggplot(tab_long, aes(x = Subtyp, y = Percent, fill = Status)) +
    geom_bar(
      stat = "identity",
      position = position_dodge(width = 0.78),
      width = 0.68,
      color = "black",
      linewidth = 0.3
    ) +
    geom_text(
      aes(label = ifelse(Percent > 0, paste0(sprintf("%.1f", Percent), "%"), "")),
      position = position_dodge(width = 0.78),
      vjust = -0.35,
      size = 5.2,
      fontface = "bold",
      color = "black"
    ) +
    scale_fill_manual(
      values = palette_paper[c("remitted", "acute")],
      drop = FALSE
    ) +
    labs(
      x = "SuStaIn subtype",
      y = "Percent within subtype",
      title = paste0("Clinical status by subtype - ", cohort_name)
    ) +
    scale_y_continuous(
      limits = c(0, max(tab_long$Percent, na.rm = TRUE) * 1.12),
      expand = expansion(mult = c(0, 0.02))
    ) +
    coord_cartesian(clip = "off") +
    theme_classic(base_size = 18) +
    theme(
      legend.title = element_blank(),
      legend.position = "right",
      plot.title = element_text(size = 20, face = "bold", color = "black"),
      axis.title.x = element_text(size = 18, face = "bold", color = "black"),
      axis.title.y = element_text(size = 18, face = "bold", color = "black"),
      axis.text.x = element_text(size = 16, face = "bold", color = "black"),
      axis.text.y = element_text(size = 15, face = "bold", color = "black"),
      legend.text = element_text(size = 15, face = "bold", color = "black"),
      panel.grid = element_blank()
    )

  ggsave(
    file.path(out_dir, paste0("status_subtypes_", cohort_name, "_counts.png")),
    plot_counts,
    width = 8.5,
    height = 6.2,
    dpi = 300,
    bg = "white"
  )

  ggsave(
    file.path(out_dir, paste0("status_subtypes_", cohort_name, "_counts.pdf")),
    plot_counts,
    width = 8.5,
    height = 6.2,
    bg = "white"
  )

  ggsave(
    file.path(out_dir, paste0("status_subtypes_", cohort_name, "_percent.png")),
    plot_percent,
    width = 8.5,
    height = 6.2,
    dpi = 300,
    bg = "white"
  )

  ggsave(
    file.path(out_dir, paste0("status_subtypes_", cohort_name, "_percent.pdf")),
    plot_percent,
    width = 8.5,
    height = 6.2,
    bg = "white"
  )
  # -----------------------------
  # Cohort summary
  # -----------------------------
  summary_row <- tibble(
    cohort = cohort_name,
    n_join_total = nrow(merged),
    n_join_HC = sum(merged$is_HC, na.rm = TRUE),
    n_status_defined = nrow(status_df),
    n_remitted = sum(status_df$Status == "remitted"),
    n_acute = sum(status_df$Status == "acute"),
    global_method = method_used,
    global_p = p_val,
    global_cramers_v = safe_cramers_v(tbl)
  )

  write_out(summary_row, file.path(out_dir, "summary_cohort_status.csv"))

  message("[OK] Finished cohort: ", cohort_name)
  list(summary = summary_row)
}

# -----------------------------
# 4) Run all cohorts + end table
# -----------------------------
results <- purrr::imap(cohorts, ~run_one_cohort_status(cohort_name = .y, assign_path = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))

write_out(summary_all, file.path(output_dir_global, "summary_all_cohorts_status.csv"))

message("[OK] ALL COHORTS DONE. Output: ", output_dir_global)
