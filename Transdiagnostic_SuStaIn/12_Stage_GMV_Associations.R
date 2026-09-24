# ============================================================
# GMV vs SuStaIn Stage
#
# Final analysis strategy
# 1. Main regression model:
#      GMV_index_std ~ ML_Stage_std
# 2. Interaction model:
#      GMV_index_std ~ ML_Stage_std * ML_Subtype
# 3. Descriptive follow-up models within subtype:
#      GMV_index_std ~ ML_Stage_std
#
# Reported for all models:
#   - beta_std
#   - 95%-CI
#   - p_value
#   - q_BH
#   - R2 / adj. R2
#
# Runs for all cohorts:
#   - TDM
#   - ANX
#   - MDD
#   - BD
#   - SSD
#
# Uses:
#   - assignment CSVs exported from Output_sustainrun
#   - GMV tables
#
# Outputs per dataset:
#   - merged analysis tables
#   - main regression model tables
#   - interaction model tables
#   - subtype-specific follow-up tables
#   - compact manuscript-ready tables
#   - plots
#
# Notes
# - paper_z is assumed to be negated SuStaIn input and is flipped back
#   once via * -1.
# - gmv_raw is assumed to already be in normal orientation.
# - After flip-back:
#     ROI_value > 0 = higher GMV
#     ROI_value < 0 = lower GMV
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(ggplot2)
})

# -----------------------------
# 0) Config
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

cfg <- list(
  paper_z_path = file.path(base_dir, "HPC", "input", "residuals_z_scored_corrected_allGroups.csv"),
  gmv_raw_path = getOption("SUSTAIN_GMV_RAW_FILE", file.path(base_dir, "private_data", "merged_data_avg_only.csv")),
  
  flip_sign_undo_negation_paper = TRUE,
  flip_sign_undo_negation_raw   = FALSE,
  
  cohorts = list(
    TDM = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
    ANX = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
    MDD = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
    BD  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
    SSD = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
  ),
  
  K_SUBTYPES = 3,
  stage_col = "ML_Stage",
  subtype_col = "ML_Subtype",
  
  MIN_N_PER_SUBTYPE_MODEL = 20,
  
  APPLY_BH = TRUE,
  RUN_RAW_GMV_SENSITIVITY = getOption("SUSTAIN_RUN_RAW_GMV", FALSE),
  
  PLOT_SHOW_SUBTYPE_LM_LINES = TRUE,
  PLOT_SAVE_FACETS = TRUE
)

out_root <- file.path(getwd(), "Output Scripte", "gmv_vs_stage_models_no_recovariates")
if (!dir.exists(out_root)) dir.create(out_root, recursive = TRUE)

# -----------------------------
# 1) Helpers
# -----------------------------
stop_if_missing_file <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

read_any_csv <- function(path) {
  l1 <- readLines(path, n = 1, warn = FALSE)
  
  if (length(l1) == 1 && grepl(";", l1, fixed = TRUE)) {
    x <- tryCatch(
      readr::read_csv2(
        path,
        show_col_types = FALSE,
        progress = FALSE,
        locale = readr::locale(decimal_mark = ".", grouping_mark = ",")
      ),
      error = function(e) NULL
    )
    if (!is.null(x)) return(x)
    
    x <- tryCatch(
      readr::read_csv2(
        path,
        show_col_types = FALSE,
        progress = FALSE,
        locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
      ),
      error = function(e) NULL
    )
    if (!is.null(x)) return(x)
  }
  
  x <- tryCatch(
    readr::read_csv(
      path,
      show_col_types = FALSE,
      progress = FALSE,
      locale = readr::locale(decimal_mark = ".", grouping_mark = ",")
    ),
    error = function(e) NULL
  )
  if (!is.null(x)) return(x)
  
  x <- tryCatch(
    readr::read_csv(
      path,
      show_col_types = FALSE,
      progress = FALSE,
      locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
    ),
    error = function(e) NULL
  )
  if (!is.null(x)) return(x)
  
  stop("Could not read CSV: ", path)
}

write_csv_out <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(df, path, na = "")
}

write_tsv_out <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  readr::write_tsv(df, path, na = "")
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

zscore_safe <- function(x) {
  x <- safe_as_numeric_flexible(x)
  s <- sd(x, na.rm = TRUE)
  m <- mean(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(NA_real_, length(x)))
  (x - m) / s
}

detect_id_col <- function(df) {
  cand <- c("Name", "Proband", "participant_id", "Participant", "Subject", "subject", "ID", "Id", "names")
  hit <- cand[cand %in% names(df)]
  if (length(hit) > 0) return(hit[1])
  
  nm_low <- tolower(names(df))
  hit2 <- names(df)[nm_low %in% tolower(cand)]
  if (length(hit2) > 0) return(hit2[1])
  
  n <- nrow(df)
  if (n == 0) return(NULL)
  
  char_cols <- names(df)[vapply(df, is.character, logical(1))]
  if (length(char_cols) > 0) {
    uniq <- vapply(df[char_cols], dplyr::n_distinct, integer(1))
    best <- char_cols[which.max(uniq)]
    if (max(uniq) >= ceiling(0.80 * n)) return(best)
  }
  
  num_cols <- names(df)[vapply(df, is.numeric, logical(1))]
  if (length(num_cols) > 0) {
    uniq <- vapply(df[num_cols], dplyr::n_distinct, integer(1))
    best <- num_cols[which.max(uniq)]
    if (max(uniq) >= ceiling(0.95 * n)) return(best)
  }
  
  NULL
}

detect_roi_cols_by_suffix_any <- function(df, id_col) {
  drop_pat <- "(^Group$|^Alter$|^Age$|^Sex$|^Geschlecht$|^Site$|^Scanner$|^TIV$|^Sum_MED$|^MED$|^ML_Stage$|^ML_Subtype$|^Prob_)"
  cand <- setdiff(names(df), id_col)
  cand <- cand[!grepl(drop_pat, cand)]
  
  pats <- c("_avg_resid_final$", "_avg$", "_avg_resid$", "_resid_final$", "_resid$")
  roi <- character(0)
  for (p in pats) {
    roi <- cand[grepl(p, cand)]
    if (length(roi) >= 10) break
  }
  roi
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
  if (!isTRUE(cfg$APPLY_BH)) {
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

# -----------------------------
# 1b) Config checks
# -----------------------------
check_cfg_paths <- function(cfg) {
  missing_assign <- names(cfg$cohorts)[!file.exists(unlist(cfg$cohorts))]
  if (length(missing_assign) > 0) {
    stop(
      "Missing assignment files for: ",
      paste(missing_assign, collapse = ", "),
      "\nPaths:\n",
      paste(unlist(cfg$cohorts)[missing_assign], collapse = "\n")
    )
  }
  
  if (!file.exists(cfg$paper_z_path)) stop("Missing paper_z_path: ", cfg$paper_z_path)
  if (!file.exists(cfg$gmv_raw_path)) stop("Missing gmv_raw_path: ", cfg$gmv_raw_path)
}

check_cfg_paths(cfg)

# -----------------------------
# 2) Plot helpers
# -----------------------------
plot_scatter_stage_by_subtype <- function(df, out_png, title_txt, show_lines = TRUE, facet = FALSE) {
  df <- df %>% mutate(ML_Subtype = factor(ML_Subtype, levels = 1:cfg$K_SUBTYPES))
  
  gg <- ggplot(df, aes(x = ML_Stage, y = GMV_index, color = ML_Subtype)) +
    geom_point(alpha = 0.7) +
    theme_minimal(base_size = 13) +
    labs(title = title_txt, x = "ML Stage", y = "GMV", color = "Subtype")
  
  if (isTRUE(show_lines)) {
    gg <- gg + geom_smooth(method = "lm", se = FALSE)
  } else {
    gg <- gg + geom_smooth(method = "lm", se = TRUE, color = "black")
  }
  
  if (isTRUE(facet)) {
    gg <- gg + facet_wrap(~ ML_Subtype, ncol = 2) + guides(color = "none")
  }
  
  ggsave(out_png, gg, width = 7.5, height = 5.5, dpi = 300)
}

# -----------------------------
# 3) GMV dataset loader
# -----------------------------
make_gmv_dataset <- function(path, flip_sign_undo_negation, dataset_tag, out_dir) {
  stop_if_missing_file(path)
  raw <- read_any_csv(path)
  
  id_col <- detect_id_col(raw)
  if (is.null(id_col)) stop("Could not detect subject id column in GMV file: ", path)
  
  df <- raw %>%
    rename(Name = all_of(id_col)) %>%
    mutate(Name = norm_id(Name))
  
  roi_cols <- detect_roi_cols_by_suffix_any(df, id_col = "Name")
  if (length(roi_cols) < 10) stop("ROI columns cannot be identified by the expected suffix in: ", path)
  if (dataset_tag == "paper_z" &&
      (length(roi_cols) != 62L || !all(grepl("_avg_resid_final$", roi_cols))))
    stop("Expected exactly 62 paper-z ROI columns with suffix _avg_resid_final in: ", path)
  
  if (length(roi_cols) < 10) {
    write_csv_out(tibble(col = names(df)), file.path(out_dir, paste0("debug_cols_", dataset_tag, ".csv")))
    stop("Too few ROI columns detected in: ", path, ". Wrote debug_cols_*.")
  }
  
  df <- df %>% mutate(across(all_of(roi_cols), safe_as_numeric_flexible))
  
  if (isTRUE(flip_sign_undo_negation)) {
    df <- df %>% mutate(across(all_of(roi_cols), ~ .x * (-1)))
  }
  
  gmv_index <- df %>%
    transmute(
      Name,
      GMV_index = rowMeans(across(all_of(roi_cols)), na.rm = TRUE),
      n_roi_used = rowSums(is.finite(as.matrix(select(., all_of(roi_cols)))))
    )
  
  roi_qc <- df %>%
    select(Name, all_of(roi_cols)) %>%
    pivot_longer(
      cols = all_of(roi_cols),
      names_to = "ROI",
      values_to = "ROI_value"
    ) %>%
    group_by(ROI) %>%
    summarise(
      mean_roi_value = mean(ROI_value, na.rm = TRUE),
      n = sum(is.finite(ROI_value)),
      .groups = "drop"
    )
  
  write_csv_out(tibble(roi_col = roi_cols), file.path(out_dir, paste0("roi_cols_used_", dataset_tag, ".csv")))
  write_csv_out(roi_qc, file.path(out_dir, paste0("roi_qc_orientation_", dataset_tag, ".csv")))
  
  list(
    gmv_index = gmv_index,
    roi_cols = roi_cols
  )
}

# -----------------------------
# 4) Load GMV datasets
# -----------------------------
dir_paper <- file.path(out_root, "paper_z")
dir_raw   <- file.path(out_root, "gmv_raw")
dir.create(dir_paper, showWarnings = FALSE, recursive = TRUE)
dir.create(dir_raw, showWarnings = FALSE, recursive = TRUE)

gmv_paper <- make_gmv_dataset(
  path = cfg$paper_z_path,
  flip_sign_undo_negation = cfg$flip_sign_undo_negation_paper,
  dataset_tag = "paper_z",
  out_dir = dir_paper
)

gmv_raw <- NULL
if (isTRUE(cfg$RUN_RAW_GMV_SENSITIVITY)) {
  gmv_raw <- make_gmv_dataset(
    path = cfg$gmv_raw_path,
    flip_sign_undo_negation = cfg$flip_sign_undo_negation_raw,
    dataset_tag = "gmv_raw",
    out_dir = dir_raw
  )
}

# -----------------------------
# 5) Standardization + model helpers
# -----------------------------
prepare_standardized_model_data <- function(df) {
  dd <- df
  
  dd$GMV_index <- safe_as_numeric_flexible(dd$GMV_index)
  dd$ML_Stage  <- safe_as_numeric_flexible(dd$ML_Stage)
  dd$GMV_index_std <- zscore_safe(dd$GMV_index)
  dd$ML_Stage_std  <- zscore_safe(dd$ML_Stage)
  dd$ML_Subtype <- factor(dd$ML_Subtype, levels = 1:cfg$K_SUBTYPES)
  
  dd
}

extract_lm_table_with_ci <- function(fit, cohort_name, dataset_tag, model_name) {
  sm <- summary(fit)
  cf <- as.data.frame(sm$coefficients)
  cf$term <- rownames(cf)
  rownames(cf) <- NULL
  names(cf) <- c("beta_std", "std_error", "t_value", "p_value", "term")
  
  ci <- suppressMessages(confint(fit))
  ci_df <- as.data.frame(ci)
  ci_df$term <- rownames(ci_df)
  rownames(ci_df) <- NULL
  names(ci_df) <- c("CI_lower", "CI_upper", "term")
  
  out <- cf %>%
    left_join(ci_df, by = "term") %>%
    mutate(
      cohort = cohort_name,
      dataset = dataset_tag,
      model = model_name,
      r_squared = sm$r.squared,
      adj_r_squared = sm$adj.r.squared,
      n = nobs(fit)
    ) %>%
    select(
      cohort, dataset, model, n, term,
      beta_std, std_error, t_value, p_value,
      CI_lower, CI_upper, r_squared, adj_r_squared
    )
  
  out
}

make_compact_model_table <- function(df_terms, keep_terms = NULL) {
  x <- df_terms
  if (!is.null(keep_terms)) {
    x <- x %>% filter(term %in% keep_terms)
  }
  
  x %>%
    transmute(
      cohort = cohort,
      dataset = dataset,
      model = model,
      term = term,
      n = force_text(as.character(n)),
      beta_std = force_text(fmt_num(beta_std, 4)),
      CI_95 = force_text(paste0("[", fmt_num(CI_lower, 4), ", ", fmt_num(CI_upper, 4), "]")),
      p = force_text(fmt_p_sci(p_for_table, 2)),
      p_raw = force_text(fmt_p_sci(p_value, 2)),
      q_BH = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
      R2 = force_text(fmt_num(r_squared, 3)),
      adj_R2 = force_text(fmt_num(adj_r_squared, 3))
    )
}

# -----------------------------
# 6) One cohort analysis
# -----------------------------
analyze_cohort_one_dataset <- function(assign_path, cohort_name, gmv_dataset, dataset_tag, out_base) {
  out_dir <- file.path(out_base, cohort_name)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  stop_if_missing_file(assign_path)
  asg <- read_any_csv(assign_path)
  
  id_col <- detect_id_col(asg)
  if (is.null(id_col)) stop("Could not detect ID column in assignment file: ", assign_path)
  
  asg <- asg %>%
    rename(Name = all_of(id_col)) %>%
    mutate(Name = norm_id(Name))
  
  req <- c("ML_Stage", "ML_Subtype")
  miss_req <- setdiff(req, names(asg))
  if (length(miss_req) > 0) stop("Missing required columns in assignment file: ", paste(miss_req, collapse = ", "))
  
  asg$ML_Stage   <- safe_as_numeric_flexible(asg$ML_Stage)
  asg$ML_Subtype <- safe_as_numeric_flexible(asg$ML_Subtype)
  
  st_min <- suppressWarnings(min(asg$ML_Subtype, na.rm = TRUE))
  if (is.finite(st_min) && st_min == 0) asg$ML_Subtype <- asg$ML_Subtype + 1
  asg$ML_Subtype <- factor(as.integer(asg$ML_Subtype), levels = 1:cfg$K_SUBTYPES)
  
  merged <- asg %>%
    inner_join(gmv_dataset$gmv_index, by = "Name") %>%
    mutate(
      ML_Stage = safe_as_numeric_flexible(ML_Stage),
      GMV_index = safe_as_numeric_flexible(GMV_index),
      ML_Subtype = factor(ML_Subtype, levels = 1:cfg$K_SUBTYPES)
    ) %>%
    filter(is.finite(ML_Stage), is.finite(GMV_index))
  
  if (nrow(merged) == 0) {
    stop("No rows left after joining assignment file with GMV dataset for cohort ", cohort_name, " and dataset ", dataset_tag)
  }
  
  write_csv_out(merged, file.path(out_dir, paste0("merged_analysis_table_", dataset_tag, ".csv")))
  
  dmod <- prepare_standardized_model_data(merged)
  
  # Main model
  formula_main <- "GMV_index_std ~ ML_Stage_std"
  
  fit_main <- lm(as.formula(formula_main), data = dmod)
  tbl_main <- extract_lm_table_with_ci(fit_main, cohort_name, dataset_tag, "main_model") %>%
    initialize_p_columns(family_label = NA_character_) %>%
    mutate(
      term_group = ifelse(term == "ML_Stage_std", "primary_stage_term", "other_terms")
    )
  
  write_csv_out(tbl_main, file.path(out_dir, paste0("main_model_terms_", dataset_tag, ".csv")))
  
  # Interaction model
  formula_int <- "GMV_index_std ~ ML_Stage_std * ML_Subtype"
  
  fit_int <- lm(as.formula(formula_int), data = dmod)
  tbl_int <- extract_lm_table_with_ci(fit_int, cohort_name, dataset_tag, "interaction_model") %>%
    initialize_p_columns(family_label = NA_character_) %>%
    mutate(
      term_group = case_when(
        grepl("^ML_Stage_std:ML_Subtype", term) ~ "interaction_terms",
        term == "ML_Stage_std" ~ "reference_stage_term",
        grepl("^ML_Subtype", term) ~ "subtype_main_effect_terms",
        TRUE ~ "other_terms"
      )
    )
  
  idx_int_local <- grepl("^ML_Stage_std:ML_Subtype", tbl_int$term)
  tbl_int <- apply_bh_subset(
    tbl_int,
    subset_idx = idx_int_local,
    family_label = paste0(dataset_tag, "_", cohort_name, "_interaction_terms")
  )
  
  write_csv_out(tbl_int, file.path(out_dir, paste0("interaction_model_terms_", dataset_tag, ".csv")))
  
  # Follow-up models
  tbl_follow <- dmod %>%
    group_by(ML_Subtype) %>%
    group_modify(function(dd, key) {
      nsub <- nrow(dd)
      
      if (nsub < cfg$MIN_N_PER_SUBTYPE_MODEL) {
        return(tibble(
          cohort = cohort_name,
          dataset = dataset_tag,
          model = paste0("followup_subtype_", as.character(key$ML_Subtype)),
          n = nsub,
          term = "ML_Stage_std",
          beta_std = NA_real_,
          std_error = NA_real_,
          t_value = NA_real_,
          p_value = NA_real_,
          CI_lower = NA_real_,
          CI_upper = NA_real_,
          r_squared = NA_real_,
          adj_r_squared = NA_real_
        ))
      }
      
      formula_sub <- "GMV_index_std ~ ML_Stage_std"
      
      fit_sub <- lm(as.formula(formula_sub), data = dd)
      extract_lm_table_with_ci(
        fit = fit_sub,
        cohort_name = cohort_name,
        dataset_tag = dataset_tag,
        model_name = paste0("followup_subtype_", as.character(key$ML_Subtype))
      )
    }) %>%
    ungroup() %>%
    initialize_p_columns(family_label = NA_character_) %>%
    mutate(
      term_group = ifelse(term == "ML_Stage_std", "followup_stage_term", "other_terms")
    )
  
  idx_follow_local <- tbl_follow$term == "ML_Stage_std"
  tbl_follow <- apply_bh_subset(
    tbl_follow,
    subset_idx = idx_follow_local,
    family_label = paste0(dataset_tag, "_", cohort_name, "_followup_stage_terms")
  )
  
  write_csv_out(tbl_follow, file.path(out_dir, paste0("followup_subtype_models_", dataset_tag, ".csv")))
  
  # Compact tables -> TSV + forced text
  compact_main <- make_compact_model_table(
    tbl_main %>% filter(term == "ML_Stage_std"),
    keep_terms = "ML_Stage_std"
  )
  
  compact_int <- make_compact_model_table(
    tbl_int %>% filter(term %in% c("ML_Stage_std", "ML_Subtype2", "ML_Subtype3", "ML_Stage_std:ML_Subtype2", "ML_Stage_std:ML_Subtype3")),
    keep_terms = c("ML_Stage_std", "ML_Subtype2", "ML_Subtype3", "ML_Stage_std:ML_Subtype2", "ML_Stage_std:ML_Subtype3")
  )
  
  compact_follow <- tbl_follow %>%
    filter(term == "ML_Stage_std") %>%
    mutate(subtype = gsub("followup_subtype_", "Subtype ", model)) %>%
    transmute(
      cohort = cohort,
      dataset = dataset,
      model = subtype,
      term = term,
      n = force_text(as.character(n)),
      beta_std = force_text(fmt_num(beta_std, 4)),
      CI_95 = force_text(paste0("[", fmt_num(CI_lower, 4), ", ", fmt_num(CI_upper, 4), "]")),
      p = force_text(fmt_p_sci(p_for_table, 2)),
      p_raw = force_text(fmt_p_sci(p_value, 2)),
      q_BH = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
      R2 = force_text(fmt_num(r_squared, 3)),
      adj_R2 = force_text(fmt_num(adj_r_squared, 3))
    )
  
  write_tsv_out(compact_main,  file.path(out_dir, paste0("TABLE_main_model_compact_", dataset_tag, ".tsv")))
  write_tsv_out(compact_int,   file.path(out_dir, paste0("TABLE_interaction_model_compact_", dataset_tag, ".tsv")))
  write_tsv_out(compact_follow,file.path(out_dir, paste0("TABLE_followup_subtype_models_compact_", dataset_tag, ".tsv")))
  
  # Plots
  plot_scatter_stage_by_subtype(
    merged,
    out_png = file.path(out_dir, paste0("plot_scatter_stage_vs_gmv_bySubtype_", dataset_tag, ".png")),
    title_txt = paste0("Stage vs GMV by Subtype (", cohort_name, ")"),
    show_lines = cfg$PLOT_SHOW_SUBTYPE_LM_LINES,
    facet = FALSE
  )
  
  if (isTRUE(cfg$PLOT_SAVE_FACETS)) {
    plot_scatter_stage_by_subtype(
      merged,
      out_png = file.path(out_dir, paste0("plot_scatter_stage_vs_gmv_bySubtype_FACET_", dataset_tag, ".png")),
      title_txt = paste0("Stage vs GMV - ", cohort_name),
      show_lines = TRUE,
      facet = TRUE
    )
  }
  
  invisible(list(
    main_terms = tbl_main,
    interaction_terms = tbl_int,
    followup_terms = tbl_follow
  ))
}

# -----------------------------
# 7) Run all cohorts
# -----------------------------
run_all <- function(gmv_dataset, dataset_tag, out_base) {
  res <- purrr::imap(cfg$cohorts, function(assign_path, cohort_name) {
    analyze_cohort_one_dataset(
      assign_path = assign_path,
      cohort_name = cohort_name,
      gmv_dataset = gmv_dataset,
      dataset_tag = dataset_tag,
      out_base = out_base
    )
  })
  
  main_all        <- purrr::map_dfr(res, "main_terms")
  interaction_all <- purrr::map_dfr(res, "interaction_terms")
  followup_all    <- purrr::map_dfr(res, "followup_terms")
  
  compact_main_all <- make_compact_model_table(
    main_all %>% filter(term == "ML_Stage_std"),
    keep_terms = "ML_Stage_std"
  )
  
  compact_int_all <- make_compact_model_table(
    interaction_all %>% filter(term %in% c("ML_Stage_std", "ML_Subtype2", "ML_Subtype3", "ML_Stage_std:ML_Subtype2", "ML_Stage_std:ML_Subtype3")),
    keep_terms = c("ML_Stage_std", "ML_Subtype2", "ML_Subtype3", "ML_Stage_std:ML_Subtype2", "ML_Stage_std:ML_Subtype3")
  )
  
  compact_follow_all <- followup_all %>%
    filter(term == "ML_Stage_std") %>%
    mutate(subtype = gsub("followup_subtype_", "Subtype ", model)) %>%
    transmute(
      cohort = cohort,
      dataset = dataset,
      model = subtype,
      term = term,
      n = force_text(as.character(n)),
      beta_std = force_text(fmt_num(beta_std, 4)),
      CI_95 = force_text(paste0("[", fmt_num(CI_lower, 4), ", ", fmt_num(CI_upper, 4), "]")),
      p = force_text(fmt_p_sci(p_for_table, 2)),
      p_raw = force_text(fmt_p_sci(p_value, 2)),
      q_BH = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
      R2 = force_text(fmt_num(r_squared, 3)),
      adj_R2 = force_text(fmt_num(adj_r_squared, 3))
    )
  
  # Write outputs
  if (nrow(main_all) > 0) {
    write_csv_out(main_all, file.path(out_base, paste0("ALL_cohorts_main_model_terms_", dataset_tag, ".csv")))
  }
  if (nrow(interaction_all) > 0) {
    write_csv_out(interaction_all, file.path(out_base, paste0("ALL_cohorts_interaction_model_terms_", dataset_tag, ".csv")))
  }
  if (nrow(followup_all) > 0) {
    write_csv_out(followup_all, file.path(out_base, paste0("ALL_cohorts_followup_subtype_models_", dataset_tag, ".csv")))
  }
  
  if (nrow(compact_main_all) > 0) {
    write_tsv_out(compact_main_all, file.path(out_base, paste0("ALL_cohorts_TABLE_main_model_compact_", dataset_tag, ".tsv")))
  }
  if (nrow(compact_int_all) > 0) {
    write_tsv_out(compact_int_all, file.path(out_base, paste0("ALL_cohorts_TABLE_interaction_model_compact_", dataset_tag, ".tsv")))
  }
  if (nrow(compact_follow_all) > 0) {
    write_tsv_out(compact_follow_all, file.path(out_base, paste0("ALL_cohorts_TABLE_followup_subtype_models_compact_", dataset_tag, ".tsv")))
  }
  
  invisible(list(
    results = res,
    main_all = main_all,
    interaction_all = interaction_all,
    followup_all = followup_all,
    compact_main_all = compact_main_all,
    compact_int_all = compact_int_all,
    compact_follow_all = compact_follow_all
  ))
}

# -----------------------------
# 8) Run
# -----------------------------
res_paper <- run_all(gmv_paper, "paper_z", dir_paper)
res_raw <- NULL
if (isTRUE(cfg$RUN_RAW_GMV_SENSITIVITY)) {
  res_raw <- run_all(gmv_raw, "gmv_raw", dir_raw)
}

cat("Done. Outputs written to:\n", out_root, "\n")
