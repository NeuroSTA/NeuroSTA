# ============================================================
# PAPER PLOTS + STATS: ML Stage vs Clinical Status (acute vs remittiert)
# Cohorts: TDM, ANX, MDD, BD, SSD (SuStaIn assignments; 3 subtypes)
#
# Unified analysis strategy
#
# Status
# 1. Main nonparametric group comparison:
#      ML_Stage ~ Status
# 2. Descriptive follow-up group comparisons within subtype:
#      ML_Stage ~ Status
#
# Reported
# - H
# - df
# - p_value
# - q_BH
# - epsilon2
# - Cliff's delta
#
# Status rule:
# - acute if ANY trigger:
#   SANS7/12/16/21 > 2 OR SAPS7/20/25/34 > 2 OR HAMA_Sum > 19 OR HAMD_Sum17 > 6 OR YMRS_Sum >= 4 (Berk et al., 2008)
# - remittiert if no trigger == 1 but at least one trigger observed
# - HC excluded if Group == 1 detectable
# Outputs (per cohort):
#   plots/
#     plot_stageBins_vs_status_stackedProportion_bySubtype.png
#     plot_stageBins_vs_status_pAcute_CI_bySubtype.png
#     plot_stage_vs_status_box_bySubtype.png
#   data/
#     data_stage_vs_status_subject_level.csv
#     data_stageBins_vs_status_counts.csv
#   qc/
#     qc_stage_vs_status.csv
#     qc_summary_<cohort>.csv
#   stats/
#     desc_stage_<cohort>.csv
#     desc_status_<cohort>.csv
#     desc_stage_by_status_<cohort>.csv
#     status_main_test_<cohort>.csv
#     status_followup_tests_<cohort>.csv
#     status_TABLE_main_test_<cohort>.tsv
#     status_TABLE_followup_tests_<cohort>.tsv
#
# Notes:
# - write_csv() is used to preserve decimal point.
# - No median split is used for statistics.
# - Stage bins are used only for visualization.
# - No Dunn posthoc is used because status has only two groups.
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(purrr)
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
  TDM = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  ANX = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  MDD = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  BD  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SSD = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

output_dir_global <- file.path(output_root, "stage_vs_status_acute_vs_remittiert")
if (!dir.exists(output_dir_global)) dir.create(output_dir_global, recursive = TRUE)

K_SUBTYPES <- 3
stage_col <- "ML_Stage"
subtype_col <- "ML_Subtype"

MIN_N_PLOT <- 10
MIN_N_GROUP_TEST <- 10

APPLY_BH <- TRUE

# Binning only for plots
N_BINS <- 6
MIN_N_BIN <- 10

palette_subtypes <- c(
  "1" = "#F8766D",
  "2" = "#00BA38",
  "3" = "#619CFF"
)

status_palette <- c(
  "remittiert" = "#BDBDBD",
  "acute" = "#333333"
)

# -----------------------------
# 0b) Status trigger variables
# -----------------------------
trigger_vars_items <- c("SANS7","SANS12","SANS16","SANS21","SAPS7","SAPS20","SAPS25","SAPS34")
sum_targets <- c("HAMA_Sum","HAMD_Sum17","YMRS_Sum")

# -----------------------------
# Helpers
# -----------------------------
safe_write_csv <- function(df, path, na = "") {
  dirp <- dirname(path)
  if (!dir.exists(dirp)) dir.create(dirp, recursive = TRUE)
  readr::write_csv(df, path, na = na)
}

safe_write_tsv <- function(df, path, na = "") {
  dirp <- dirname(path)
  if (!dir.exists(dirp)) dir.create(dirp, recursive = TRUE)
  readr::write_tsv(df, path, na = na)
}

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

format_p_text <- function(p) {
  if (is.na(p) || !is.finite(p)) return(NA_character_)
  if (p <= 2.23e-308) return("<2.23e-308")
  format(p, scientific = TRUE, digits = 3)
}

safe_p_floor <- function(p) {
  if (is.na(p) || !is.finite(p)) return(NA_real_)
  if (p == 0) return(2.23e-308)
  max(p, 2.23e-308)
}

fmt_num <- function(x, digits = 4) {
  ifelse(
    is.na(x) | !is.finite(x),
    NA_character_,
    formatC(x, format = "f", digits = digits, decimal.mark = ".")
  )
}

fmt_p_sci <- function(p, digits = 2) {
  ifelse(
    is.na(p) | !is.finite(p),
    NA_character_,
    formatC(p, format = "e", digits = digits, decimal.mark = ".")
  )
}

force_text <- function(x) {
  ifelse(is.na(x) | x == "", "", paste0("'", x))
}

detect_id_col <- function(df) {
  cand <- c("Proband","Name","participant_id","Participant","Subject","subject","ID","Id","names")
  hit <- cand[cand %in% names(df)]
  if (length(hit) > 0) return(hit[1])
  nm_low <- tolower(names(df))
  hit2 <- names(df)[nm_low %in% tolower(cand)]
  if (length(hit2) > 0) return(hit2[1])
  NULL
}

derive_subtyp_from_ml <- function(df, k = 3) {
  if (!(subtype_col %in% names(df))) stop("ML_Subtype column not found in assignment file.")
  df <- df %>% mutate(ML_Subtype = safe_as_numeric_flexible(.data[[subtype_col]]))
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype all NA after numeric conversion.")
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  if (!is.finite(ml_min)) stop("ML_Subtype min not finite.")
  if (ml_min == 0) {
    df <- df %>% mutate(Subtype = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtype = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding (expected start 0 or 1). min=", ml_min)
  }
  if (any(df$Subtype < 1 | df$Subtype > k, na.rm = TRUE)) stop("Subtype out of range 1..", k)
  df %>% mutate(Subtype = factor(Subtype, levels = 1:k))
}

rowmax_na <- function(...) {
  m <- cbind(...)
  out <- apply(m, 1, function(z) {
    if (all(is.na(z))) return(NA_real_)
    suppressWarnings(max(z, na.rm = TRUE))
  })
  as.numeric(out)
}

flag_gt <- function(x, thr) ifelse(is.na(x), NA_real_, ifelse(x > thr, 1, 0))
flag_ge <- function(x, thr) ifelse(is.na(x), NA_real_, ifelse(x >= thr, 1, 0))

minus99_to_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[x %in% c(-99, -2)] <- NA_real_
  x
}

hama_items <- function(df) grep("^HAMA[0-9]+$", names(df), value = TRUE)
ymrs_items <- function(df) grep("^YMRS[0-9]+$", names(df), value = TRUE)
hamd17_items <- function(df) {
  it <- paste0("HAMD", 1:17)
  it[it %in% names(df)]
}

desc_paper <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(tibble(
      n = 0,
      median = NA_real_,
      q25 = NA_real_,
      q75 = NA_real_,
      iqr = NA_real_,
      min = NA_real_,
      max = NA_real_,
      mean = NA_real_,
      sd = NA_real_
    ))
  }
  q <- stats::quantile(x, probs = c(0.25, 0.75), names = FALSE, type = 7)
  tibble(
    n = length(x),
    median = stats::median(x),
    q25 = q[1],
    q75 = q[2],
    iqr = q[2] - q[1],
    min = min(x),
    max = max(x),
    mean = mean(x),
    sd = sd(x)
  )
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

initialize_p_columns <- function(df, family_label = NA_character_) {
  df %>%
    mutate(
      family_BH = family_label,
      q_BH = NA_real_,
      p_for_table = p_value
    )
}

apply_bh_subset <- function(df, subset_idx, family_label) {
  out <- df
  if (!APPLY_BH) {
    out$family_BH[subset_idx] <- family_label
    out$p_for_table[subset_idx] <- out$p_value[subset_idx]
    return(out)
  }

  if (sum(subset_idx, na.rm = TRUE) > 0) {
    pvals <- out$p_value[subset_idx]
    out$family_BH[subset_idx] <- family_label
    out$q_BH[subset_idx] <- p.adjust(pvals, method = "BH")
    out$p_for_table[subset_idx] <- out$q_BH[subset_idx]
  }
  out
}

run_status_kruskal <- function(dat, cohort_name, group_label, min_n = 10) {
  dd <- dat %>%
    filter(is.finite(ML_Stage), !is.na(Status)) %>%
    mutate(Status = droplevels(factor(Status, levels = c("remittiert", "acute"))))

  n_total <- nrow(dd)
  n_rem <- sum(dd$Status == "remittiert", na.rm = TRUE)
  n_acute <- sum(dd$Status == "acute", na.rm = TRUE)

  if (n_total < min_n || n_rem < 2 || n_acute < 2 || nlevels(dd$Status) != 2) {
    return(tibble(
      cohort = cohort_name,
      model = group_label,
      n = n_total,
      n_remittiert = n_rem,
      n_acute = n_acute,
      H = NA_real_,
      df = NA_real_,
      p_value = NA_real_,
      epsilon2 = NA_real_,
      cliffs_delta_acute_vs_remittiert = NA_real_
    ))
  }

  kw <- kruskal.test(ML_Stage ~ Status, data = dd)
  p <- safe_p_floor(kw$p.value)
  H <- unname(kw$statistic)
  df_kw <- unname(kw$parameter)
  eps <- epsilon_sq_kruskal(H, n_total, nlevels(dd$Status))
  dlt <- cliffs_delta(
    dd$ML_Stage[dd$Status == "acute"],
    dd$ML_Stage[dd$Status == "remittiert"]
  )

  tibble(
    cohort = cohort_name,
    model = group_label,
    n = n_total,
    n_remittiert = n_rem,
    n_acute = n_acute,
    H = H,
    df = df_kw,
    p_value = p,
    epsilon2 = eps,
    cliffs_delta_acute_vs_remittiert = dlt
  )
}

make_compact_status_table <- function(df_terms) {
  df_terms %>%
    transmute(
      cohort = cohort,
      model = model,
      n = force_text(as.character(n)),
      n_remittiert = force_text(as.character(n_remittiert)),
      n_acute = force_text(as.character(n_acute)),
      H = force_text(fmt_num(H, 3)),
      df = force_text(fmt_num(df, 0)),
      p = force_text(fmt_p_sci(p_for_table, 2)),
      p_raw = force_text(fmt_p_sci(p_value, 2)),
      q_BH = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
      epsilon2 = force_text(fmt_num(epsilon2, 3)),
      cliffs_delta = force_text(fmt_num(cliffs_delta_acute_vs_remittiert, 3))
    )
}

make_stage_bins <- function(x, n_bins = 6) {
  x <- as.numeric(x)
  if (sum(is.finite(x)) < 5) return(rep(NA_character_, length(x)))
  probs <- seq(0, 1, length.out = n_bins + 1)
  qs <- suppressWarnings(stats::quantile(x, probs = probs, na.rm = TRUE, type = 7))
  qs <- unique(qs)
  if (length(qs) < 3) return(rep(NA_character_, length(x)))
  cut(x, breaks = qs, include.lowest = TRUE, right = TRUE)
}

wilson_ci <- function(k, n, conf = 0.95) {
  if (!is.finite(k) || !is.finite(n) || n <= 0) return(c(NA_real_, NA_real_))
  z <- qnorm(1 - (1 - conf) / 2)
  phat <- k / n
  denom <- 1 + (z^2) / n
  center <- (phat + (z^2) / (2 * n)) / denom
  half <- (z * sqrt((phat * (1 - phat) + (z^2) / (4 * n)) / n)) / denom
  c(max(0, center - half), min(1, center + half))
}

# -----------------------------
# 1) Load data_all
# -----------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)

id_col_all <- detect_id_col(data_all)
if (is.null(id_col_all)) stop("No ID column detected in data_all.")
if (id_col_all != "Proband") data_all <- data_all %>% rename(Proband = all_of(id_col_all))
data_all <- data_all %>% mutate(Proband = norm_id(Proband))

missing_trigger_cols <- setdiff(trigger_vars_items, names(data_all))
if (length(missing_trigger_cols) > 0) {
  stop(
    "Required SANS/SAPS global-rating columns are missing: ",
    paste(missing_trigger_cols, collapse = ", "),
    ". Check SUSTAIN_CLINICAL_STATUS_FILE."
  )
}

# -----------------------------
# 2) Run one cohort
# -----------------------------
run_one_cohort_stage_status <- function(cohort_name, assign_path) {

  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")

  stop_if_missing(assign_path)

  out_dir <- file.path(output_dir_global, cohort_name)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  out_plots_dir <- file.path(out_dir, "plots")
  out_data_dir  <- file.path(out_dir, "data")
  out_qc_dir    <- file.path(out_dir, "qc")
  out_stats_dir <- file.path(out_dir, "stats")
  for (p in c(out_plots_dir, out_data_dir, out_qc_dir, out_stats_dir)) {
    if (!dir.exists(p)) dir.create(p, recursive = TRUE)
  }

  # assignments
  asg <- read_csv(assign_path, show_col_types = FALSE)

  id_col_asg <- detect_id_col(asg)
  if (is.null(id_col_asg)) stop("No ID column detected in assignment file: ", assign_path)
  if (id_col_asg != "Proband") asg <- asg %>% rename(Proband = all_of(id_col_asg))
  if (!(stage_col %in% names(asg))) stop("ML_Stage missing in assignment file: ", assign_path)
  if (!(subtype_col %in% names(asg))) stop("ML_Subtype missing in assignment file: ", assign_path)

  asg2 <- asg %>%
    mutate(
      Proband = norm_id(Proband),
      ML_Stage = safe_as_numeric_flexible(.data[[stage_col]]),
      ML_Subtype = safe_as_numeric_flexible(.data[[subtype_col]])
    ) %>%
    derive_subtyp_from_ml(k = K_SUBTYPES) %>%
    dplyr::select(Proband, Subtype, ML_Stage) %>%
    filter(!is.na(Proband), !is.na(Subtype), is.finite(ML_Stage))

  merged <- inner_join(asg2, data_all, by = "Proband")
  if (nrow(merged) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)

  # Group required for reliable HC exclusion.
  if (!"Group" %in% names(merged)) {
    gcols <- grep("^group($|\\.|_x|_y)", names(merged), ignore.case = TRUE, value = TRUE)
    if (length(gcols) != 1L) stop("Exactly one Group column required after join; found: ", paste(gcols, collapse = ", "))
    merged <- merged %>% rename(Group = all_of(gcols))
  }

  merged <- merged %>%
    mutate(Group_num = suppressWarnings(as.numeric(Group))) %>%
    mutate(is_HC = ifelse(is.finite(Group_num) & Group_num == 1, TRUE, FALSE))
  if (any(!is.finite(merged$Group_num))) stop("Missing or nonnumeric Group values after join.")

  # Confirm that all status items remain available after the join.
  missing_after_join <- setdiff(trigger_vars_items, names(merged))
  if (length(missing_after_join) > 0) {
    stop("Required SANS/SAPS trigger columns are missing after the join for cohort ", cohort_name, ": ",
         paste(missing_after_join, collapse = ", "))
  }
  merged <- merged %>% mutate(across(all_of(trigger_vars_items), minus99_to_na))

  # Status verwendet ausschliesslich die vorhandenen direkten Summenspalten.
  missing_sum_targets <- setdiff(sum_targets, names(merged))
  if (length(missing_sum_targets) > 0L) {
    stop("Required direct status totals are missing: ", paste(missing_sum_targets, collapse = ", "))
  }
  merged <- merged %>%
    mutate(across(all_of(sum_targets), minus99_to_na))

  # Missingness inputs
  miss_vars <- unique(c(trigger_vars_items, sum_targets))
  missing_df <- tibble(
    cohort = cohort_name,
    variable = miss_vars,
    n_total = nrow(merged),
    n_missing = sapply(miss_vars, function(v) sum(!is.finite(merged[[v]]))),
    n_nonmissing = sapply(miss_vars, function(v) sum(is.finite(merged[[v]])))
  ) %>%
    mutate(missing_pct = round(100 * n_missing / n_total, 2))
  safe_write_csv(missing_df, file.path(out_stats_dir, "missingness_status_inputs.csv"))

  # Status define: acute bei mindestens einem positiven Trigger; remittiert
  # nur bei vollstaendigen 11 Triggern ohne Schwellenueberschreitung.
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

  merged2 <- merged %>%
    mutate(
      akut_flag = akut_flag,
      complete_flag = complete_flag,
      Status = case_when(
        is_HC ~ NA_character_,
        akut_flag == 1 ~ "acute",
        complete_flag & akut_flag == 0 ~ "remittiert",
        TRUE ~ NA_character_
      ),
      Status = factor(Status, levels = c("remittiert", "acute"))
    ) %>%
    filter(!is.na(Subtype), is.finite(ML_Stage), !is.na(Status))

  if (nrow(merged2) == 0) stop("No rows with defined Status. Cohort: ", cohort_name)

  # Export subject-level data
  out_csv_subj <- file.path(out_data_dir, "data_stage_vs_status_subject_level.csv")
  safe_write_csv(
    merged2 %>% dplyr::select(Proband, Subtype, ML_Stage, Status),
    out_csv_subj
  )

  # -----------------
  # Descriptives
  # -----------------
  desc_stage <- bind_rows(
    desc_paper(merged2$ML_Stage) %>% mutate(group = "overall"),
    merged2 %>%
      group_by(Subtype) %>%
      summarise(desc_paper(ML_Stage), .groups = "drop") %>%
      mutate(group = paste0("subtype_", as.character(Subtype))) %>%
      dplyr::select(-Subtype)
  ) %>%
    mutate(cohort = cohort_name) %>%
    dplyr::select(cohort, group, everything())
  safe_write_csv(desc_stage, file.path(out_stats_dir, paste0("desc_stage_", cohort_name, ".csv")))

  desc_status <- merged2 %>%
    count(Subtype, Status, name = "n") %>%
    group_by(Subtype) %>%
    mutate(pct_within_subtype = 100 * n / sum(n)) %>%
    ungroup() %>%
    mutate(
      cohort = cohort_name,
      group = paste0("Subtype ", as.character(Subtype))
    ) %>%
    dplyr::select(cohort, group, Status, n, pct_within_subtype)
  safe_write_csv(desc_status, file.path(out_stats_dir, paste0("desc_status_", cohort_name, ".csv")))

  desc_stage_by_status <- bind_rows(
    merged2 %>%
      group_by(Status) %>%
      summarise(desc_paper(ML_Stage), .groups = "drop") %>%
      mutate(group = ifelse(Status == "acute", "overall_acute", "overall_remittiert")) %>%
      dplyr::select(-Status),
    merged2 %>%
      group_by(Subtype, Status) %>%
      summarise(desc_paper(ML_Stage), .groups = "drop") %>%
      mutate(group = paste0("subtype_", as.character(Subtype), "_", as.character(Status))) %>%
      dplyr::select(-Subtype, -Status)
  ) %>%
    mutate(cohort = cohort_name) %>%
    dplyr::select(cohort, group, n, median, q25, q75, iqr, min, max, mean, sd)
  safe_write_csv(desc_stage_by_status, file.path(out_stats_dir, paste0("desc_stage_by_status_", cohort_name, ".csv")))

  # -----------------
  # Plots: stage bins only for visualization
  # -----------------
  df_bins <- merged2 %>%
    group_by(Subtype) %>%
    mutate(StageBin = make_stage_bins(ML_Stage, n_bins = N_BINS)) %>%
    ungroup() %>%
    filter(!is.na(StageBin))

  df_bin_meta <- df_bins %>%
    group_by(Subtype, StageBin) %>%
    summarise(
      n_bin = n(),
      stage_min = min(ML_Stage, na.rm = TRUE),
      stage_max = max(ML_Stage, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(n_bin >= MIN_N_BIN) %>%
    group_by(Subtype) %>%
    arrange(stage_min, .by_group = TRUE) %>%
    mutate(
      StageBinLabel = paste0("[", sprintf("%.1f", stage_min), ", ", sprintf("%.1f", stage_max), "] | n=", n_bin),
      StageBinLabel = factor(StageBinLabel, levels = StageBinLabel)
    ) %>%
    ungroup() %>%
    dplyr::select(Subtype, StageBin, StageBinLabel, n_bin, stage_min, stage_max)

  df_counts <- df_bins %>%
    inner_join(df_bin_meta, by = c("Subtype", "StageBin")) %>%
    group_by(Subtype, StageBinLabel, Status) %>%
    summarise(n = n(), n_bin = first(n_bin), .groups = "drop") %>%
    group_by(Subtype, StageBinLabel) %>%
    mutate(prop = ifelse(n_bin > 0, n / n_bin, NA_real_)) %>%
    ungroup() %>%
    mutate(Status = factor(Status, levels = c("remittiert", "acute")))

  out_csv_bins <- file.path(out_data_dir, "data_stageBins_vs_status_counts.csv")
  safe_write_csv(df_counts %>% dplyr::select(Subtype, StageBinLabel, Status, n, n_bin, prop), out_csv_bins)

  # PLOT 1: stacked proportions
  out_png1 <- file.path(out_plots_dir, "plot_stageBins_vs_status_stackedProportion_bySubtype.png")
  if (nrow(df_counts) < MIN_N_PLOT) {
    png(out_png1, width = 1600, height = 1100, res = 200)
    plot.new()
    title(main = paste0("Not enough binned data for stacked bars: ", cohort_name))
    dev.off()
  } else {
    gg1 <- ggplot(df_counts, aes(x = StageBinLabel, y = prop, fill = Status)) +
      geom_col(width = 0.75, color = "black", linewidth = 0.20) +
      facet_wrap(~ Subtype, nrow = 1, labeller = labeller(Subtype = function(x) paste0("Subtype ", x))) +
      scale_fill_manual(values = status_palette) +
      scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
      theme_minimal(base_size = 12) +
      labs(
        title = paste0("Clinical status across ML Stage bins - ", cohort_name),
        x = "ML Stage bin (within subtype, low to high)",
        y = "Proportion within bin",
        fill = "Status"
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.0),
        axis.text.x = element_text(angle = 30, hjust = 1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "top",
        legend.title = element_blank()
      )
    ggsave(out_png1, gg1, width = 13.8, height = 4.6, dpi = 400)
  }

  # PLOT 2: p(acute) with Wilson CI
  out_png2 <- file.path(out_plots_dir, "plot_stageBins_vs_status_pAcute_CI_bySubtype.png")
  df_p <- df_counts %>%
    filter(Status == "acute") %>%
    group_by(Subtype, StageBinLabel) %>%
    summarise(
      n_bin = first(n_bin),
      k_acute = sum(n, na.rm = TRUE),
      p_acute = ifelse(n_bin > 0, k_acute / n_bin, NA_real_),
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      ci_low = wilson_ci(k_acute, n_bin, conf = 0.95)[1],
      ci_high = wilson_ci(k_acute, n_bin, conf = 0.95)[2]
    ) %>%
    ungroup()

  if (nrow(df_p) < MIN_N_PLOT) {
    png(out_png2, width = 1600, height = 1100, res = 200)
    plot.new()
    title(main = paste0("Not enough binned data for p(acute) CI plot: ", cohort_name))
    dev.off()
  } else {
    gg2 <- ggplot(df_p, aes(x = StageBinLabel, y = p_acute)) +
      geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15, linewidth = 0.6) +
      geom_point(aes(size = n_bin), alpha = 0.9) +
      facet_wrap(~ Subtype, nrow = 1, labeller = labeller(Subtype = function(x) paste0("Subtype ", x))) +
      scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
      theme_minimal(base_size = 12) +
      labs(
        title = paste0("P(acute) across ML Stage bins - ", cohort_name),
        x = "ML Stage bin (within subtype, low to high)",
        y = "P(acute) with 95% CI",
        size = "n per bin"
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.0),
        axis.text.x = element_text(angle = 30, hjust = 1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "right"
      )
    ggsave(out_png2, gg2, width = 13.8, height = 4.6, dpi = 400)
  }

  # PLOT 3: boxplot by subtype
  out_png3 <- file.path(out_plots_dir, "plot_stage_vs_status_box_bySubtype.png")
  if (nrow(merged2) < MIN_N_PLOT) {
    png(out_png3, width = 1600, height = 1100, res = 200)
    plot.new()
    title(main = paste0("Not enough data for status boxplot: ", cohort_name))
    dev.off()
  } else {
    gg3 <- ggplot(
      merged2 %>% mutate(Status = factor(Status, levels = c("remittiert", "acute"))),
      aes(x = Status, y = ML_Stage, fill = Status)
    ) +
      geom_boxplot(width = 0.65, color = "black", linewidth = 0.25, outlier.size = 1.0) +
      facet_wrap(~ Subtype, nrow = 1, labeller = labeller(Subtype = function(x) paste0("Subtype ", x))) +
      scale_fill_manual(values = status_palette) +
      theme_minimal(base_size = 12) +
      labs(
        title = paste0("ML Stage by clinical status - ", cohort_name),
        x = "Clinical status",
        y = "ML Stage",
        fill = "Status"
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.0),
        legend.position = "top",
        legend.title = element_blank()
      )
    ggsave(out_png3, gg3, width = 10.5, height = 4.8, dpi = 400)
  }

  # -----------------
  # Statistics: main + follow-up Kruskal
  # -----------------
  status_main_tbl <- run_status_kruskal(merged2, cohort_name, "status_main_test") %>%
    initialize_p_columns(family_label = NA_character_)

  status_follow_tbl <- bind_rows(
    lapply(levels(merged2$Subtype), function(s) {
      ds <- merged2 %>% filter(Subtype == s)
      run_status_kruskal(ds, cohort_name, paste0("status_followup_subtype_", s))
    })
  ) %>%
    initialize_p_columns(family_label = NA_character_)

  idx_follow <- rep(TRUE, nrow(status_follow_tbl))
  status_follow_tbl <- apply_bh_subset(
    status_follow_tbl,
    subset_idx = idx_follow,
    family_label = paste0(cohort_name, "_status_followup_terms")
  )

  safe_write_csv(status_main_tbl,   file.path(out_stats_dir, paste0("status_main_test_", cohort_name, ".csv")))
  safe_write_csv(status_follow_tbl, file.path(out_stats_dir, paste0("status_followup_tests_", cohort_name, ".csv")))

  status_table_main <- make_compact_status_table(status_main_tbl)
  status_table_follow <- status_follow_tbl %>%
    mutate(model = gsub("status_followup_subtype_", "Subtype ", model)) %>%
    make_compact_status_table()

  safe_write_tsv(status_table_main,   file.path(out_stats_dir, paste0("status_TABLE_main_test_", cohort_name, ".tsv")))
  safe_write_tsv(status_table_follow, file.path(out_stats_dir, paste0("status_TABLE_followup_tests_", cohort_name, ".tsv")))

  # QC
  qc <- tibble(
    cohort = cohort_name,
    n_assign = nrow(asg2),
    n_join_total = nrow(merged),
    n_join_HC = sum(merged$is_HC, na.rm = TRUE),
    n_status_defined = nrow(merged2),
    n_bins_rows = nrow(df_counts),
    out_png_main = out_png1,
    out_png_pacute = out_png2,
    out_png_box = out_png3,
    out_csv_subject_level = out_csv_subj,
    out_csv_bins = out_csv_bins
  )

  safe_write_csv(qc, file.path(out_qc_dir, paste0("qc_summary_", cohort_name, ".csv")))
  safe_write_csv(qc, file.path(out_qc_dir, "qc_stage_vs_status.csv"))

  message("[OK] Finished cohort: ", cohort_name)

  invisible(list(
    qc = qc,
    status_main = status_main_tbl,
    status_follow = status_follow_tbl
  ))
}

# -----------------------------
# 3) Run all cohorts
# -----------------------------
res_list <- purrr::imap(cohorts, ~ run_one_cohort_stage_status(cohort_name = .y, assign_path = .x))

qc_all <- purrr::map_dfr(res_list, ~ .x$qc) %>%
  arrange(match(cohort, names(cohorts)))
safe_write_csv(qc_all, file.path(output_dir_global, "qc_all_cohorts.csv"))

status_main_all <- purrr::map_dfr(res_list, ~ .x$status_main) %>%
  arrange(match(cohort, names(cohorts)))
safe_write_csv(status_main_all, file.path(output_dir_global, "status_main_test_all_cohorts.csv"))

status_follow_all <- purrr::map_dfr(res_list, ~ .x$status_follow) %>%
  arrange(match(cohort, names(cohorts)), q_BH)
safe_write_csv(status_follow_all, file.path(output_dir_global, "status_followup_tests_all_cohorts.csv"))

message("[OK] ALL COHORTS DONE. Output: ", output_dir_global)
