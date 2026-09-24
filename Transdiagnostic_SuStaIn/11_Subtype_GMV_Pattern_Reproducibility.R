# ===============================
# GMV Subtype Pattern Reproducibility + Bootstrap CIs (paper-like)
# - Canonical GMV space: residuals_z_scored_negated_paper.csv
# - Compare cohort runs (Angst, MDD, BD, SZ) vs reference run (main)
# - Compute subtype patterns (mean ROI per subtype) and match labels (permutation)
# - Similarity: Pearson or Spearman correlations of ROI pattern vectors
# - Bootstrap (stratified within subtype): CI for matched subtype correlations and mean reproducibility
#
# SECTION 6 (NEW):
# - Pairwise diagnosis similarity: all 6 pairs of diagnosis cohorts
# - Full K×K matrix per pair (3×3=9 correlations), total 54 correlations
# - Best label match + bootstrap CIs with a prespecified mapping
#
# Outputs (in out_root):
#   settings_used.csv
#   roi_columns_used.csv
#   join_qc.csv
#   corr_matrix_<cohort>_vs_main.csv
#   best_mapping_<cohort>_to_main.csv
#   matched_diag_corr_<cohort>_to_main.csv
#   pattern_<cohort>_ROI_by_subtype.csv
#   pattern_<cohort>_aligned_to_main.csv
#   pattern_diff_<cohort>_minus_main.csv
#   summary_pattern_reproducibility_vs_main.csv
#   bootstrap_diag_raw_<cohort>_to_main.csv
#   bootstrap_ci_per_subtype_<cohort>_to_main.csv
#   bootstrap_ci_mean_<cohort>_to_main.csv
#   bootstrap_ci_per_subtype_all_cohorts.csv
#   bootstrap_ci_mean_all_cohorts.csv
#
#   diagnosis_pairwise/
#     corr_matrix_full_<A>_vs_<B>.csv      (full 3×3 matrix per pair)
#     all_corr_matrices_full_6pairs.csv     (all 54 correlations combined)
#     best_mapping_<A>_vs_<B>.csv
#     all_best_mappings_6pairs.csv
#     matched_diag_corr_<A>_vs_<B>.csv
#     summary_diagnosis_pairwise_similarity.csv
#     bootstrap_diag_raw_<A>_vs_<B>.csv
#     bootstrap_ci_per_subtype_<A>_vs_<B>.csv
#     bootstrap_ci_mean_<A>_vs_<B>.csv
#     bootstrap_ci_per_subtype_all_6pairs.csv
#     bootstrap_ci_mean_all_6pairs.csv
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(tibble)
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

gmv_z_path <- file.path(base_dir, "HPC", "input", "residuals_z_scored_corrected_allGroups.csv")

out_root <- file.path(base_dir, "Output Scripte", "gmv_pattern_reproducibility_3subtypes_bootstrap")
if (!dir.exists(out_root)) dir.create(out_root, recursive = TRUE)

REF_COHORT <- "main"
K_SUBTYPES <- 3

COR_METHOD <- "pearson"   # "pearson" or "spearman"

# If TRUE: multiply all ROI columns by -1 once (undo global negation)
flip_sign_undo_negation <- TRUE

# Minimum subjects per subtype for pattern estimation (warning if below)
MIN_N_PER_SUBTYPE <- 10

# Prespecified ROI selection pattern
ROI_NAME_PATTERN <- "_avg_resid_final$"

# Bootstrap
BOOT_REPS <- 2000
BOOT_CI   <- c(0.025, 0.975)
BOOT_SEED <- 42

# Cohort -> assignment file
cohorts <- list(
  main  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

# -----------------------------
# 1) Helpers
# -----------------------------
stop_if_missing_file <- function(path) {
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

q_ <- function(x, p) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  as.numeric(stats::quantile(x, probs = p, names = FALSE, type = 7, na.rm = TRUE))
}

write_csv2_out <- function(df, path) {
  readr::write_csv2(df, path, na = "")
}

read_any_csv <- function(path) {
  l1 <- readLines(path, n = 1, warn = FALSE)
  if (length(l1) == 1 && grepl(";", l1, fixed = TRUE)) {
    return(readr::read_csv2(
      path,
      show_col_types = FALSE,
      progress = FALSE,
      locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
    ))
  }
  readr::read_csv(
    path,
    show_col_types = FALSE,
    progress = FALSE,
    locale = readr::locale(decimal_mark = ".", grouping_mark = ",")
  )
}

detect_id_col <- function(df) {
  cand <- c("Proband", "participant_id", "Participant", "Subject", "subject",
            "ID", "Id", "Name", "names")
  hit <- cand[cand %in% names(df)]
  if (length(hit) > 0) return(hit[1])
  nm_low <- tolower(names(df))
  hit2 <- names(df)[nm_low %in% tolower(cand)]
  if (length(hit2) > 0) return(hit2[1])
  char_cols <- names(df)[vapply(df, is.character, logical(1))]
  if (length(char_cols) > 0) {
    uniq <- vapply(df[char_cols], dplyr::n_distinct, integer(1))
    best <- char_cols[which.max(uniq)]
    if (max(uniq) >= ceiling(0.80 * nrow(df))) return(best)
  }
  num_cols <- names(df)[vapply(df, is.numeric, logical(1))]
  if (length(num_cols) > 0) {
    uniq <- vapply(df[num_cols], dplyr::n_distinct, integer(1))
    best <- num_cols[which.max(uniq)]
    if (max(uniq) >= ceiling(0.95 * nrow(df))) return(best)
  }
  NULL
}

derive_subtyp_from_ml <- function(df) {
  if (!("ML_Subtype" %in% names(df))) stop("ML_Subtype column not found.")
  df <- df %>% mutate(ML_Subtype = safe_as_numeric_flexible(ML_Subtype))
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype is all NA after numeric conversion.")
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  ml_max <- suppressWarnings(max(df$ML_Subtype, na.rm = TRUE))
  if (!is.finite(ml_min) || !is.finite(ml_max)) stop("ML_Subtype min/max not finite.")
  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding. min=", ml_min, " max=", ml_max)
  }
  df
}

detect_roi_cols <- function(df, pattern = "_avg_resid_final$") {
  roi <- names(df)[grepl(pattern, names(df))]
  drop_pat <- "(^Group$|^Alter$|^Age$|^Sex$|^Geschlecht$|^Site$|^Scanner$|^TIV$)"
  roi[!grepl(drop_pat, roi)]
}

all_perms <- function(v) {
  if (length(v) == 1) return(list(v))
  out <- list()
  for (i in seq_along(v)) {
    rest <- v[-i]
    subp <- all_perms(rest)
    for (sp in subp) out[[length(out) + 1]] <- c(v[i], sp)
  }
  out
}

compute_subtype_patterns <- function(df_join, roi_cols, k = 3) {
  df_join <- df_join %>% mutate(Subtyp = factor(Subtyp, levels = seq_len(k)))
  n_tbl <- df_join %>%
    count(Subtyp, name = "n_subj") %>%
    complete(Subtyp = factor(seq_len(k), levels = seq_len(k)), fill = list(n_subj = 0)) %>%
    arrange(Subtyp)
  pat <- df_join %>%
    group_by(Subtyp) %>%
    summarise(across(all_of(roi_cols), function(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
    complete(Subtyp = factor(seq_len(k), levels = seq_len(k))) %>%
    arrange(Subtyp)
  mat <- as.matrix(pat %>% select(all_of(roi_cols)))
  mat <- t(mat)
  colnames(mat) <- paste0("Subtyp", seq_len(k))
  rownames(mat) <- roi_cols
  list(pattern_mat = mat, n_tbl = n_tbl)
}

cor_mat_subtypes <- function(matA, matB, method = "pearson") {
  kA <- ncol(matA); kB <- ncol(matB)
  out <- matrix(NA_real_, nrow = kA, ncol = kB)
  for (i in seq_len(kA)) {
    for (j in seq_len(kB)) {
      out[i, j] <- suppressWarnings(stats::cor(
        matA[, i], matB[, j],
        use = "pairwise.complete.obs",
        method = method
      ))
    }
  }
  rownames(out) <- colnames(matA)
  colnames(out) <- colnames(matB)
  out
}

best_label_match <- function(corr_mat) {
  k <- nrow(corr_mat)
  perms <- all_perms(seq_len(k))
  best_score <- -Inf
  best_perm  <- NULL
  best_diag  <- rep(NA_real_, k)
  for (p in perms) {
    d <- diag(corr_mat[, p, drop = FALSE])
    score <- mean(d, na.rm = TRUE)
    if (is.finite(score) && score > best_score) {
      best_score <- score
      best_perm  <- p
      best_diag  <- d
    }
  }
  tib_map <- tibble(
    cohort_subtyp_index      = seq_len(k),
    matched_ref_subtyp_index = best_perm,
    cohort_subtyp            = paste0("Subtyp", seq_len(k)),
    matched_ref_subtyp       = paste0("Subtyp", best_perm),
    corr                     = as.numeric(best_diag)
  )
  list(best_mean_diag_corr = best_score, perm = best_perm, mapping = tib_map)
}

inv_perm_from_perm <- function(p) {
  sapply(seq_along(p), function(j) which(p == j))
}

bootstrap_resample_stratified <- function(df_join, k) {
  df_join %>%
    mutate(Subtyp = factor(Subtyp, levels = seq_len(k))) %>%
    group_by(Subtyp) %>%
    group_modify(function(.x, .y) {
      n <- nrow(.x)
      if (n <= 1) return(.x)
      .x[sample.int(n, size = n, replace = TRUE), , drop = FALSE]
    }) %>%
    ungroup()
}

# -----------------------------
# 2) Load canonical GMV file
# -----------------------------
stop_if_missing_file(gmv_z_path)
gmv_z_raw <- read_any_csv(gmv_z_path)

id_col <- detect_id_col(gmv_z_raw)
if (is.null(id_col)) {
  l1 <- readLines(gmv_z_path, n = 2, warn = FALSE)
  write_csv2_out(tibble(line = l1), file.path(out_root, "debug_first_lines_gmv.csv"))
  stop("Could not identify subject id column in GMV file.")
}

gmv_z <- gmv_z_raw %>%
  rename(Proband = all_of(id_col)) %>%
  mutate(Proband = norm_id(Proband))

roi_cols <- detect_roi_cols(gmv_z, pattern = ROI_NAME_PATTERN)
if (length(roi_cols) < 5) {
  write_csv2_out(
    tibble(col = names(gmv_z),
           class = vapply(gmv_z, function(x) paste(class(x), collapse = ","), character(1))),
    file.path(out_root, "debug_gmv_column_classes.csv")
  )
  stop("Too few ROI columns detected using pattern: ", ROI_NAME_PATTERN)
}

gmv_z <- gmv_z %>%
  mutate(across(all_of(roi_cols), safe_as_numeric_flexible))

if ("Group" %in% roi_cols) stop("ROI selection included 'Group'. Check ROI_NAME_PATTERN.")

if (isTRUE(flip_sign_undo_negation)) {
  gmv_z <- gmv_z %>% mutate(across(all_of(roi_cols), function(x) x * (-1)))
}

write_csv2_out(
  tibble(
    gmv_file                = gmv_z_path,
    detected_id_col         = id_col,
    n_subjects              = nrow(gmv_z),
    n_roi_cols              = length(roi_cols),
    roi_name_pattern        = ROI_NAME_PATTERN,
    corr_method             = COR_METHOD,
    flip_sign_undo_negation = flip_sign_undo_negation,
    ref_cohort              = REF_COHORT,
    k_subtypes              = K_SUBTYPES,
    boot_reps               = BOOT_REPS,
    boot_ci_low             = BOOT_CI[1],
    boot_ci_high            = BOOT_CI[2],
    boot_seed               = BOOT_SEED
  ),
  file.path(out_root, "settings_used.csv")
)

write_csv2_out(tibble(ROI = roi_cols), file.path(out_root, "roi_columns_used.csv"))

# -----------------------------
# 3) Load assignments and compute observed patterns per cohort
# -----------------------------
load_assignments <- function(path) {
  stop_if_missing_file(path)
  a <- read_any_csv(path)
  if (!"Proband" %in% names(a)) {
    if ("names" %in% names(a)) a <- a %>% rename(Proband = names)
    if ("Name"  %in% names(a)) a <- a %>% rename(Proband = Name)
  }
  if (!"Proband" %in% names(a)) stop("No Proband column in assignment file: ", path)
  a %>%
    mutate(Proband = norm_id(Proband)) %>%
    derive_subtyp_from_ml() %>%
    filter(Subtyp %in% seq_len(K_SUBTYPES)) %>%
    mutate(Subtyp = factor(Subtyp, levels = seq_len(K_SUBTYPES))) %>%
    select(Proband, Subtyp)
}

patterns     <- list()
n_tables     <- list()
join_qc      <- list()
joined_data  <- list()   # stored for bootstrap reuse in Sections 5 + 6

for (nm in names(cohorts)) {
  message("Processing cohort: ", nm)
  a <- load_assignments(cohorts[[nm]])
  df_join <- inner_join(a, gmv_z, by = "Proband")
  
  join_qc[[nm]] <- tibble(
    cohort                       = nm,
    n_assignments                = nrow(a),
    n_after_join                 = nrow(df_join),
    n_unique_subjects_after_join = dplyr::n_distinct(df_join$Proband)
  )
  joined_data[[nm]] <- df_join
  
  if (nrow(df_join) == 0) {
    patterns[[nm]] <- NULL
    n_tables[[nm]] <- NULL
    next
  }
  
  res <- compute_subtype_patterns(df_join, roi_cols, k = K_SUBTYPES)
  patterns[[nm]] <- res$pattern_mat
  n_tables[[nm]] <- res$n_tbl %>% mutate(cohort = nm) %>% relocate(cohort)
  
  pat_df <- as.data.frame(res$pattern_mat) %>%
    tibble::rownames_to_column("ROI") %>%
    mutate(cohort = nm) %>%
    relocate(cohort, ROI)
  write_csv2_out(pat_df,
                 file.path(out_root, paste0("pattern_", nm, "_ROI_by_subtype.csv")))
  write_csv2_out(n_tables[[nm]],
                 file.path(out_root, paste0("n_per_subtype_", nm, ".csv")))
}

join_qc_df <- bind_rows(join_qc)
write_csv2_out(join_qc_df, file.path(out_root, "join_qc.csv"))

if (is.null(patterns[[REF_COHORT]])) {
  stop("Reference cohort patterns not available: ", REF_COHORT)
}
ref_mat <- patterns[[REF_COHORT]]

main_pat_df <- as.data.frame(ref_mat) %>%
  tibble::rownames_to_column("ROI") %>%
  mutate(cohort = REF_COHORT) %>%
  relocate(cohort, ROI)
write_csv2_out(main_pat_df, file.path(out_root, "pattern_main_ROI_by_subtype.csv"))

ref_n <- n_tables[[REF_COHORT]] %>%
  transmute(ref_subtyp_index = as.integer(as.character(Subtyp)), n_ref = n_subj)

# -----------------------------
# 4) Observed reproducibility vs main: full K×K + best label match
# -----------------------------
summary_list  <- list()
mapping_list  <- list()
diag_list     <- list()
warnings_list <- list()

for (nm in names(patterns)) {
  if (nm == REF_COHORT) next
  
  mat <- patterns[[nm]]
  if (is.null(mat)) {
    summary_list[[nm]] <- tibble(
      cohort = nm, ref_cohort = REF_COHORT, status = "no_data_after_join",
      best_mean_diag_corr = NA_real_, mean_diag_corr = NA_real_,
      min_diag_corr = NA_real_,       max_diag_corr  = NA_real_
    )
    next
  }
  
  cm   <- cor_mat_subtypes(mat, ref_mat, method = COR_METHOD)
  best <- best_label_match(cm)
  
  cm_df <- as.data.frame(cm) %>%
    tibble::rownames_to_column("cohort_subtype") %>%
    mutate(cohort = nm, ref_cohort = REF_COHORT) %>%
    relocate(cohort, ref_cohort, cohort_subtype)
  write_csv2_out(cm_df,
                 file.path(out_root, paste0("corr_matrix_", nm, "_vs_", REF_COHORT, ".csv")))
  
  map_df <- best$mapping %>%
    mutate(cohort = nm, ref_cohort = REF_COHORT) %>%
    relocate(cohort, ref_cohort)
  write_csv2_out(map_df,
                 file.path(out_root, paste0("best_mapping_", nm, "_to_", REF_COHORT, ".csv")))
  
  n_coh <- n_tables[[nm]] %>%
    transmute(cohort_subtyp_index = as.integer(as.character(Subtyp)), n_cohort = n_subj)
  
  diag_corr <- best$mapping %>%
    mutate(cohort = nm, ref_cohort = REF_COHORT) %>%
    relocate(cohort, ref_cohort) %>%
    left_join(n_coh, by = "cohort_subtyp_index") %>%
    left_join(ref_n, by = c("matched_ref_subtyp_index" = "ref_subtyp_index"))
  write_csv2_out(diag_corr,
                 file.path(out_root, paste0("matched_diag_corr_", nm, "_to_", REF_COHORT, ".csv")))
  
  low_n <- n_tables[[nm]] %>% filter(n_subj > 0, n_subj < MIN_N_PER_SUBTYPE)
  if (nrow(low_n) > 0) {
    warnings_list[[nm]] <- low_n %>%
      mutate(cohort = nm, warning = paste0("Subtype n < ", MIN_N_PER_SUBTYPE)) %>%
      relocate(cohort, warning, Subtyp, n_subj)
  }
  
  p    <- best$perm
  invp <- inv_perm_from_perm(p)
  mat_aligned <- mat[, invp, drop = FALSE]
  colnames(mat_aligned) <- paste0("Subtyp", seq_len(K_SUBTYPES))
  
  aligned_df <- as.data.frame(mat_aligned) %>%
    tibble::rownames_to_column("ROI") %>%
    mutate(cohort = nm, aligned_to = REF_COHORT) %>%
    relocate(cohort, aligned_to, ROI)
  write_csv2_out(aligned_df,
                 file.path(out_root, paste0("pattern_", nm, "_aligned_to_", REF_COHORT, ".csv")))
  
  diff_df <- as.data.frame(mat_aligned - ref_mat) %>%
    tibble::rownames_to_column("ROI") %>%
    mutate(cohort = nm, diff_vs = REF_COHORT) %>%
    relocate(cohort, diff_vs, ROI)
  write_csv2_out(diff_df,
                 file.path(out_root, paste0("pattern_diff_", nm, "_minus_", REF_COHORT, ".csv")))
  
  summary_list[[nm]] <- tibble(
    cohort              = nm,
    ref_cohort          = REF_COHORT,
    status              = "ok",
    best_mean_diag_corr = as.numeric(best$best_mean_diag_corr),
    mean_diag_corr      = mean(diag_corr$corr, na.rm = TRUE),
    min_diag_corr       = min(diag_corr$corr, na.rm = TRUE),
    max_diag_corr       = max(diag_corr$corr, na.rm = TRUE)
  )
  mapping_list[[nm]] <- map_df
  diag_list[[nm]]    <- diag_corr
}

summary_df <- bind_rows(summary_list) %>% arrange(match(cohort, names(cohorts)))
write_csv2_out(summary_df,
               file.path(out_root, "summary_pattern_reproducibility_vs_main.csv"))
write_csv2_out(bind_rows(mapping_list),
               file.path(out_root, "all_best_mappings_vs_main.csv"))
write_csv2_out(bind_rows(diag_list),
               file.path(out_root, "all_matched_diag_corr_vs_main.csv"))

if (length(warnings_list) > 0) {
  write_csv2_out(bind_rows(warnings_list),
                 file.path(out_root, "warnings_low_n_per_subtype.csv"))
} else {
  write_csv2_out(tibble(note = "No low-n warnings."),
                 file.path(out_root, "warnings_low_n_per_subtype.csv"))
}

# -----------------------------
# 5) Bootstrap CIs vs main (prespecified mapping)
# -----------------------------
set.seed(BOOT_SEED)

boot_ci_per_sub_list  <- list()
boot_ci_mean_list     <- list()

for (nm in names(cohorts)) {
  if (nm == REF_COHORT) next
  
  message("Bootstrap vs main — cohort: ", nm, " (", BOOT_REPS, " reps)")
  
  df_join <- joined_data[[nm]]
  if (is.null(df_join) || nrow(df_join) == 0) next
  
  map_path <- file.path(out_root, paste0("best_mapping_", nm, "_to_", REF_COHORT, ".csv"))
  stop_if_missing_file(map_path)
  obs_map <- read_any_csv(map_path) %>%
    transmute(
      cohort_subtyp_index      = as.integer(cohort_subtyp_index),
      matched_ref_subtyp_index = as.integer(matched_ref_subtyp_index),
      matched_ref_subtyp       = as.character(matched_ref_subtyp)
    ) %>%
    arrange(cohort_subtyp_index)
  
  p <- obs_map$matched_ref_subtyp_index
  if (length(p) != K_SUBTYPES || any(!p %in% seq_len(K_SUBTYPES))) {
    stop("Observed mapping invalid for cohort ", nm)
  }
  
  boot_diag <- matrix(NA_real_, nrow = BOOT_REPS, ncol = K_SUBTYPES)
  colnames(boot_diag) <- paste0("Subtyp", seq_len(K_SUBTYPES))
  
  for (b in seq_len(BOOT_REPS)) {
    df_b    <- bootstrap_resample_stratified(df_join, k = K_SUBTYPES)
    res_b   <- compute_subtype_patterns(df_b, roi_cols, k = K_SUBTYPES)
    cm_b    <- cor_mat_subtypes(res_b$pattern_mat, ref_mat, method = COR_METHOD)
    boot_diag[b, ] <- as.numeric(diag(cm_b[, p, drop = FALSE]))
  }
  
  boot_raw <- as.data.frame(boot_diag) %>%
    mutate(rep = seq_len(BOOT_REPS),
           mean_diag = apply(boot_diag, 1, function(x) mean(x, na.rm = TRUE))) %>%
    relocate(rep, mean_diag)
  write_csv2_out(boot_raw,
                 file.path(out_root, paste0("bootstrap_diag_raw_", nm, "_to_", REF_COHORT, ".csv")))
  
  per_sub_ci <- tibble(
    cohort               = nm,
    ref_cohort           = REF_COHORT,
    cohort_subtyp_index  = seq_len(K_SUBTYPES),
    cohort_subtyp        = paste0("Subtyp", seq_len(K_SUBTYPES)),
    corr_mean    = apply(boot_diag, 2, function(x) mean(x, na.rm = TRUE)),
    corr_median  = apply(boot_diag, 2, function(x) q_(x, 0.50)),
    corr_ci_low  = apply(boot_diag, 2, function(x) q_(x, BOOT_CI[1])),
    corr_ci_high = apply(boot_diag, 2, function(x) q_(x, BOOT_CI[2])),
    corr_min     = apply(boot_diag, 2, function(x) min(x, na.rm = TRUE)),
    corr_max     = apply(boot_diag, 2, function(x) max(x, na.rm = TRUE))
  ) %>%
    left_join(obs_map, by = "cohort_subtyp_index") %>%
    relocate(cohort, ref_cohort, cohort_subtyp_index, cohort_subtyp,
             matched_ref_subtyp_index, matched_ref_subtyp,
             corr_mean, corr_median, corr_ci_low, corr_ci_high, corr_min, corr_max)
  write_csv2_out(per_sub_ci,
                 file.path(out_root, paste0("bootstrap_ci_per_subtype_", nm, "_to_", REF_COHORT, ".csv")))
  
  mean_vec <- boot_raw$mean_diag
  mean_ci  <- tibble(
    cohort = nm, ref_cohort = REF_COHORT,
    metric = "mean_matched_diag_corr",
    mean   = mean(mean_vec, na.rm = TRUE),
    median = q_(mean_vec, 0.50),
    ci_low = q_(mean_vec, BOOT_CI[1]),
    ci_high= q_(mean_vec, BOOT_CI[2]),
    min    = min(mean_vec, na.rm = TRUE),
    max    = max(mean_vec, na.rm = TRUE),
    reps        = BOOT_REPS,
    corr_method = COR_METHOD
  )
  write_csv2_out(mean_ci,
                 file.path(out_root, paste0("bootstrap_ci_mean_", nm, "_to_", REF_COHORT, ".csv")))
  
  boot_ci_per_sub_list[[nm]] <- per_sub_ci
  boot_ci_mean_list[[nm]]    <- mean_ci
}

if (length(boot_ci_per_sub_list) > 0) {
  write_csv2_out(bind_rows(boot_ci_per_sub_list),
                 file.path(out_root, "bootstrap_ci_per_subtype_all_cohorts.csv"))
} else {
  write_csv2_out(tibble(note = "No bootstrap per-subtype results."),
                 file.path(out_root, "bootstrap_ci_per_subtype_all_cohorts.csv"))
}

if (length(boot_ci_mean_list) > 0) {
  write_csv2_out(bind_rows(boot_ci_mean_list),
                 file.path(out_root, "bootstrap_ci_mean_all_cohorts.csv"))
} else {
  write_csv2_out(tibble(note = "No bootstrap mean results."),
                 file.path(out_root, "bootstrap_ci_mean_all_cohorts.csv"))
}

# ===============================
# 6) Pairwise diagnosis similarity
# For each of the 6 diagnosis pairs:
#   - Full K×K correlation matrix (3×3 = 9 correlations)
#   - Best label match via permutation
#   - Bootstrap CIs with a prespecified mapping (both cohorts resampled independently)
#
# Total observed correlations: 6 × 9 = 54
# ===============================

# Map internal cohort names to diagnosis labels
diagnosis_label_map <- c(
  Angst = "ANX",
  MDD   = "MDD",
  BD    = "BD",
  SZ    = "SSD"
)

diag_cohorts <- names(diagnosis_label_map)   # Angst, MDD, BD, SZ
diag_pairs   <- combn(diag_cohorts, 2, simplify = FALSE)

message("\n======================================")
message("SECTION 6: Pairwise diagnosis similarity")
message(length(diag_pairs), " pairs x ", K_SUBTYPES, "x", K_SUBTYPES,
        " = ", length(diag_pairs) * K_SUBTYPES^2, " correlations total")
message("======================================")

out_diag <- file.path(out_root, "diagnosis_pairwise")
if (!dir.exists(out_diag)) dir.create(out_diag, recursive = TRUE)

# -----------------------------
# 6a) Observed full K×K matrices + best label match
# -----------------------------
diag_summary_list <- list()
diag_mapping_list <- list()
diag_fullmat_list <- list()

for (pp in diag_pairs) {
  nm_a <- pp[1];  nm_b <- pp[2]
  lab_a <- diagnosis_label_map[nm_a]
  lab_b <- diagnosis_label_map[nm_b]
  pair_id <- paste0(lab_a, "_vs_", lab_b)
  
  message("  Pair: ", pair_id)
  
  mat_a <- patterns[[nm_a]]
  mat_b <- patterns[[nm_b]]
  
  if (is.null(mat_a) || is.null(mat_b)) {
    diag_summary_list[[pair_id]] <- tibble(
      pair = pair_id, cohort_A = lab_a, cohort_B = lab_b,
      status = "no_data",
      best_mean_diag_corr = NA_real_, mean_diag_corr = NA_real_,
      min_diag_corr = NA_real_,       max_diag_corr  = NA_real_
    )
    next
  }
  
  # Full K×K: rows = subtypes of A, columns = subtypes of B
  cm_ab <- cor_mat_subtypes(mat_a, mat_b, method = COR_METHOD)
  
  cm_df <- as.data.frame(cm_ab) %>%
    tibble::rownames_to_column("subtype_A") %>%
    mutate(pair = pair_id, cohort_A = lab_a, cohort_B = lab_b) %>%
    relocate(pair, cohort_A, cohort_B, subtype_A)
  write_csv2_out(cm_df,
                 file.path(out_diag, paste0("corr_matrix_full_", pair_id, ".csv")))
  diag_fullmat_list[[pair_id]] <- cm_df
  
  best <- best_label_match(cm_ab)
  
  map_df <- best$mapping %>%
    rename(
      subtype_A_index    = cohort_subtyp_index,
      subtype_A          = cohort_subtyp,
      best_match_B_index = matched_ref_subtyp_index,
      best_match_B       = matched_ref_subtyp
    ) %>%
    mutate(pair = pair_id, cohort_A = lab_a, cohort_B = lab_b) %>%
    relocate(pair, cohort_A, cohort_B)
  write_csv2_out(map_df,
                 file.path(out_diag, paste0("best_mapping_", pair_id, ".csv")))
  diag_mapping_list[[pair_id]] <- map_df
  
  n_a <- n_tables[[nm_a]] %>%
    transmute(subtype_A_index    = as.integer(as.character(Subtyp)), n_A = n_subj)
  n_b <- n_tables[[nm_b]] %>%
    transmute(best_match_B_index = as.integer(as.character(Subtyp)), n_B = n_subj)
  
  matched_df <- map_df %>%
    left_join(n_a, by = "subtype_A_index") %>%
    left_join(n_b, by = "best_match_B_index")
  write_csv2_out(matched_df,
                 file.path(out_diag, paste0("matched_diag_corr_", pair_id, ".csv")))
  
  diag_summary_list[[pair_id]] <- tibble(
    pair                = pair_id,
    cohort_A            = lab_a,
    cohort_B            = lab_b,
    status              = "ok",
    best_mean_diag_corr = as.numeric(best$best_mean_diag_corr),
    mean_diag_corr      = mean(best$mapping$corr, na.rm = TRUE),
    min_diag_corr       = min(best$mapping$corr, na.rm = TRUE),
    max_diag_corr       = max(best$mapping$corr, na.rm = TRUE)
  )
}

# All 54 correlations in one file
write_csv2_out(bind_rows(diag_fullmat_list),
               file.path(out_diag, "all_corr_matrices_full_6pairs.csv"))

diag_summary_df <- bind_rows(diag_summary_list) %>%
  arrange(desc(best_mean_diag_corr))
write_csv2_out(diag_summary_df,
               file.path(out_diag, "summary_diagnosis_pairwise_similarity.csv"))
write_csv2_out(bind_rows(diag_mapping_list),
               file.path(out_diag, "all_best_mappings_6pairs.csv"))

message("[Section 6a] Observed matrices done:")
print(diag_summary_df %>%
        select(pair, best_mean_diag_corr, min_diag_corr, max_diag_corr, status))

# -----------------------------
# 6b) Bootstrap CIs with a prespecified mapping; both cohorts resampled independently
# -----------------------------
message("\n[Section 6b] Bootstrap CIs for diagnosis pairs (", BOOT_REPS, " reps each)")

# Offset seed to distinguish from Section 5 bootstrap
set.seed(BOOT_SEED + 1L)

boot_diag_pair_sub_list  <- list()
boot_diag_pair_mean_list <- list()

for (pp in diag_pairs) {
  nm_a <- pp[1];  nm_b <- pp[2]
  lab_a <- diagnosis_label_map[nm_a]
  lab_b <- diagnosis_label_map[nm_b]
  pair_id <- paste0(lab_a, "_vs_", lab_b)
  
  message("  Bootstrap: ", pair_id)
  
  df_a <- joined_data[[nm_a]]
  df_b <- joined_data[[nm_b]]
  
  if (is.null(df_a) || nrow(df_a) == 0 ||
      is.null(df_b) || nrow(df_b) == 0) {
    stop("Missing joined data for required reproducibility pair: ", pair_id)
  }
  
  # Load the observed mapping
  map_path <- file.path(out_diag, paste0("best_mapping_", pair_id, ".csv"))
  if (!file.exists(map_path)) {
    stop("Missing mapping file for required reproducibility pair: ", pair_id)
  }
  obs_map_pair <- read_any_csv(map_path) %>%
    transmute(
      subtype_A_index    = as.integer(subtype_A_index),
      best_match_B_index = as.integer(best_match_B_index)
    ) %>%
    arrange(subtype_A_index)
  
  p_pair <- obs_map_pair$best_match_B_index
  if (length(p_pair) != K_SUBTYPES || any(!p_pair %in% seq_len(K_SUBTYPES))) {
    stop("Invalid mapping for required reproducibility pair: ", pair_id)
  }
  
  boot_diag_pair <- matrix(NA_real_, nrow = BOOT_REPS, ncol = K_SUBTYPES)
  colnames(boot_diag_pair) <- paste0("Subtyp", seq_len(K_SUBTYPES))
  
  for (b in seq_len(BOOT_REPS)) {
    # Both cohorts resampled independently (stratified within subtype)
    df_a_b <- bootstrap_resample_stratified(df_a, k = K_SUBTYPES)
    df_b_b <- bootstrap_resample_stratified(df_b, k = K_SUBTYPES)
    res_a  <- compute_subtype_patterns(df_a_b, roi_cols, k = K_SUBTYPES)
    res_b  <- compute_subtype_patterns(df_b_b, roi_cols, k = K_SUBTYPES)
    cm_b   <- cor_mat_subtypes(res_a$pattern_mat, res_b$pattern_mat, method = COR_METHOD)
    boot_diag_pair[b, ] <- as.numeric(diag(cm_b[, p_pair, drop = FALSE]))
  }
  
  boot_raw_pair <- as.data.frame(boot_diag_pair) %>%
    mutate(rep       = seq_len(BOOT_REPS),
           mean_diag = apply(boot_diag_pair, 1, function(x) mean(x, na.rm = TRUE))) %>%
    relocate(rep, mean_diag)
  write_csv2_out(boot_raw_pair,
                 file.path(out_diag, paste0("bootstrap_diag_raw_", pair_id, ".csv")))
  
  per_sub_ci_pair <- tibble(
    pair               = pair_id,
    cohort_A           = lab_a,
    cohort_B           = lab_b,
    subtype_A_index    = seq_len(K_SUBTYPES),
    subtype_A          = paste0("Subtyp", seq_len(K_SUBTYPES)),
    best_match_B_index = p_pair,
    best_match_B       = paste0("Subtyp", p_pair),
    corr_mean    = apply(boot_diag_pair, 2, function(x) mean(x, na.rm = TRUE)),
    corr_median  = apply(boot_diag_pair, 2, function(x) q_(x, 0.50)),
    corr_ci_low  = apply(boot_diag_pair, 2, function(x) q_(x, BOOT_CI[1])),
    corr_ci_high = apply(boot_diag_pair, 2, function(x) q_(x, BOOT_CI[2])),
    corr_min     = apply(boot_diag_pair, 2, function(x) min(x, na.rm = TRUE)),
    corr_max     = apply(boot_diag_pair, 2, function(x) max(x, na.rm = TRUE))
  )
  write_csv2_out(per_sub_ci_pair,
                 file.path(out_diag, paste0("bootstrap_ci_per_subtype_", pair_id, ".csv")))
  boot_diag_pair_sub_list[[pair_id]] <- per_sub_ci_pair
  
  mean_vec_pair <- boot_raw_pair$mean_diag
  mean_ci_pair <- tibble(
    pair        = pair_id,
    cohort_A    = lab_a,
    cohort_B    = lab_b,
    metric      = "mean_matched_diag_corr",
    mean        = mean(mean_vec_pair, na.rm = TRUE),
    median      = q_(mean_vec_pair, 0.50),
    ci_low      = q_(mean_vec_pair, BOOT_CI[1]),
    ci_high     = q_(mean_vec_pair, BOOT_CI[2]),
    min         = min(mean_vec_pair, na.rm = TRUE),
    max         = max(mean_vec_pair, na.rm = TRUE),
    reps        = BOOT_REPS,
    corr_method = COR_METHOD
  )
  write_csv2_out(mean_ci_pair,
                 file.path(out_diag, paste0("bootstrap_ci_mean_", pair_id, ".csv")))
  boot_diag_pair_mean_list[[pair_id]] <- mean_ci_pair
}

if (length(boot_diag_pair_sub_list) > 0) {
  write_csv2_out(bind_rows(boot_diag_pair_sub_list),
                 file.path(out_diag, "bootstrap_ci_per_subtype_all_6pairs.csv"))
}

if (length(boot_diag_pair_mean_list) > 0) {
  boot_all_pairs_mean <- bind_rows(boot_diag_pair_mean_list)
  write_csv2_out(boot_all_pairs_mean,
                 file.path(out_diag, "bootstrap_ci_mean_all_6pairs.csv"))
  message("\n[Section 6b] Bootstrap mean CIs:")
  print(boot_all_pairs_mean %>% select(pair, mean, ci_low, ci_high))
}

message("[Section 6] Done. Output: ", out_diag)

# ===============================
message("\n[OK] ALL SECTIONS DONE.")
message("[OK] Output root: ", out_root)
message("[OK] Key Section 1-5 file: summary_pattern_reproducibility_vs_main.csv")
message("[OK] Key Section 6 files: diagnosis_pairwise/summary_diagnosis_pairwise_similarity.csv")
message("[OK]                       diagnosis_pairwise/all_corr_matrices_full_6pairs.csv (54 correlations)")
message("[OK]                       diagnosis_pairwise/bootstrap_ci_mean_all_6pairs.csv")
