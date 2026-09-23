# ===============================
# SuStaIn Subtypes x Duration + Episode Count variables
# Duration:       DurCurEp, DurHosp, DurMan, DurDep, DurPsych
# Episode counts: ManEp, DepEp, PsychEp
# - 3 subtypes, all cohorts: main, MDD, Angst, BD, SZ
# - All 8 variables form ONE BH correction family per cohort
#   (same overarching question: illness course differences across subtypes)
# - Cleaning: -99 -> NA; for counts additionally negative -> NA
# - Units: months (duration) / weeks (DurHosp) / count (episode counts)
# - QC filter applied ONLY to duration variables (not to counts):
#   * Exclude_age (DEFAULT: TRUE): duration > age in matching unit
#   * Exclude_IQR (DEFAULT: FALSE): IQR outlier filter (1.5*IQR)
# - Global: Kruskal-Wallis
# - Post hoc: Dunn pairwise with BH
# - Pairwise effect sizes: r (from z) + Cliff's delta
# - Descriptives after filtering
# - Output: write_csv2 (Excel-DE compatible)
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(rstatix)
  library(tibble)
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

output_root <- file.path(base_dir, "Output Scripte", "durations_episoden_3subtypes_all_cohorts")
if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

cohorts <- list(
  main  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

alpha_fdr <- 0.05

# Filter switches
EXCLUDE_EXCEEDS_AGE <- TRUE
EXCLUDE_IQR_OUTLIERS <- FALSE

# Duration variables + units (age-limit filter applies)
duration_meta <- tibble::tibble(
  Variable = c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp"),
  Unit     = c("months", "months", "months", "months", "weeks"),
  is_count = FALSE
)

# Episode count variables (age-limit filter does NOT apply)
episode_meta <- tibble::tibble(
  Variable = c("ManEp", "DepEp", "PsychEp"),
  Unit     = c("count", "count", "count"),
  is_count = TRUE
)

# Combined: all 8 variables form ONE BH family per cohort
all_vars_meta <- bind_rows(duration_meta, episode_meta)

# Optional: if 0 is coded as missing for specific variables
zero_is_missing_vars <- character(0)

# -----------------------------
# Helpers
# -----------------------------
stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

write_out <- function(df, path) {
  readr::write_csv2(df, path, na = "")
}

norm_id <- function(x) {
  tolower(gsub("_", "-", trimws(as.character(x))))
}

safe_as_numeric <- function(x) suppressWarnings(as.numeric(x))

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
    stop("Unexpected ML_Subtype coding (expected start 0 or 1). min=", ml_min)
  }
  df
}

epsilon2_kw <- function(H, n, k) {
  if (!is.finite(H) || !is.finite(n) || !is.finite(k) || n <= k) return(NA_real_)
  as.numeric((H - k + 1) / (n - k))
}

q_ <- function(x, p) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  as.numeric(stats::quantile(x, probs = p, names = FALSE, type = 7, na.rm = TRUE))
}

iqr_bounds <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4) return(c(lower = NA_real_, upper = NA_real_))
  q1 <- q_(x, 0.25)
  q3 <- q_(x, 0.75)
  iqr <- q3 - q1
  c(lower = q1 - 1.5 * iqr, upper = q3 + 1.5 * iqr)
}

safe_cliffs_delta_pairs <- function(df, value_col = "Value", group_col = "Subtyp") {
  df[[group_col]] <- factor(df[[group_col]])
  gr <- levels(df[[group_col]])
  if (length(gr) < 2) return(tibble())
  
  pairs <- combn(gr, 2, simplify = FALSE)
  out <- purrr::map_dfr(pairs, function(pp) {
    sub <- df %>% filter(.data[[group_col]] %in% pp) %>% droplevels()
    if (nrow(sub) == 0) {
      return(tibble(group1 = pp[1], group2 = pp[2], cliffs_delta = NA_real_, delta_magnitude = NA_character_))
    }
    cd <- tryCatch(
      rstatix::cliffs_delta(sub, as.formula(paste0(value_col, " ~ ", group_col))),
      error = function(e) NULL
    )
    if (is.null(cd) || nrow(cd) == 0) {
      return(tibble(group1 = pp[1], group2 = pp[2], cliffs_delta = NA_real_, delta_magnitude = NA_character_))
    }
    cd %>%
      transmute(
        group1 = as.character(group1),
        group2 = as.character(group2),
        cliffs_delta = as.numeric(effsize),
        delta_magnitude = as.character(magnitude)
      )
  })
  out
}

age_limit_in_unit <- function(age_years, unit) {
  if (!is.finite(age_years)) return(NA_real_)
  if (unit == "months") return(age_years * 12)
  if (unit == "weeks")  return(age_years * 52.1775)
  NA_real_
}

# -----------------------------
# 1) Load data_all once
# -----------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)

if (!"Proband" %in% names(data_all)) {
  if ("Name" %in% names(data_all)) data_all <- data_all %>% rename(Proband = Name)
  if ("names" %in% names(data_all)) data_all <- data_all %>% rename(Proband = names)
}
if (!"Proband" %in% names(data_all)) stop("Proband column missing in data_all.")
if (!"Alter" %in% names(data_all)) stop("Alter column missing in data_all.")

data_all <- data_all %>%
  mutate(
    Proband = norm_id(Proband),
    Alter = safe_as_numeric(Alter)
  )

required_course_vars <- c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp", "ManEp", "DepEp", "PsychEp")
missing_course_vars <- setdiff(required_course_vars, names(data_all))
if (length(missing_course_vars) > 0L) stop("Required clinical-course variables missing: ", paste(missing_course_vars, collapse = ", "))
base_vars <- c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp")

if (length(base_vars) > 0) {
  data_all <- data_all %>%
    mutate(across(all_of(base_vars), ~ ifelse(.x == -99, NA, .x))) %>%
    mutate(across(all_of(base_vars), safe_as_numeric))
}

# Episode counts: -99 -> NA; negative values impossible -> NA
ep_base_vars <- c("ManEp", "DepEp", "PsychEp")
if (length(ep_base_vars) > 0) {
  data_all <- data_all %>%
    mutate(across(all_of(ep_base_vars), ~ {
      x <- safe_as_numeric(.x)
      x[x == -99] <- NA_real_
      x[x < 0]    <- NA_real_
      x
    }))
}

missing_ep <- setdiff(c("ManEp", "DepEp", "PsychEp"), names(data_all))
if (length(missing_ep) > 0)
  message("[WARN] Episode count variables absent from database: ",
          paste(missing_ep, collapse = ", "))

if (length(zero_is_missing_vars) > 0) {
  zvars <- intersect(zero_is_missing_vars, base_vars)
  if (length(zvars) > 0) {
    data_all <- data_all %>% mutate(across(all_of(zvars), ~ ifelse(.x == 0, NA, .x)))
  }
}

# -----------------------------
# 2) QC / filter per variable
# -----------------------------
qc_and_filter_variable <- function(df_sub, var, unit) {
  x <- safe_as_numeric(df_sub[[var]])
  age <- df_sub$Alter
  lim <- purrr::map_dbl(age, ~ age_limit_in_unit(.x, unit))
  
  flag_negative <- is.finite(x) & (x < 0)
  flag_exceeds_age <- is.finite(x) & is.finite(lim) & (x > lim)
  
  b <- iqr_bounds(x)
  flag_iqr <- is.finite(x) & is.finite(b["lower"]) & is.finite(b["upper"]) &
    (x < b["lower"] | x > b["upper"])
  
  flag_exclude <- FALSE
  if (EXCLUDE_EXCEEDS_AGE) flag_exclude <- flag_exclude | flag_exceeds_age
  if (EXCLUDE_IQR_OUTLIERS) flag_exclude <- flag_exclude | flag_iqr
  flag_exclude <- flag_exclude | flag_negative
  
  df_sub %>%
    mutate(
      Value = x,
      Unit = unit,
      age_limit = lim,
      flag_negative = flag_negative,
      flag_exceeds_age = flag_exceeds_age,
      flag_iqr = flag_iqr,
      flag_exclude = flag_exclude
    )
}

# -----------------------------
# 3) Run one cohort
# -----------------------------
run_one_cohort_durations <- function(cohort_name, assign_path) {
  
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing(assign_path)
  
  out_dir <- file.path(output_root, cohort_name)
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
    select(Proband, Subtyp)
  
  merged <- inner_join(cluster_data, data_all, by = "Proband")
  if (nrow(merged) == 0) {
    note <- tibble::tibble(note = "Join produced 0 rows. Check ID harmonization.")
    write_out(note, file.path(out_dir, "note_join_zero_rows.csv"))
    return(list(summary = tibble(cohort = cohort_name, n_total_after_join = 0L, n_vars_tested = 0L, n_vars_sig_BH = 0L)))
  }
  
  subtype_counts <- merged %>%
    count(Subtyp, name = "n") %>%
    complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n = 0)) %>%
    arrange(Subtyp)
  write_out(subtype_counts, file.path(out_dir, "subtype_counts_after_join.csv"))
  
  vars_here <- all_vars_meta %>% filter(Variable %in% names(merged))
  if (nrow(vars_here) == 0) {
    writeLines("No duration or episode variables present in merged data.", con = file.path(out_dir, "note_no_vars.txt"))
    return(list(summary = tibble(cohort = cohort_name, n_total_after_join = nrow(merged), n_vars_tested = 0L, n_vars_sig_BH = 0L)))
  }
  
  # Build long_qc: duration variables use age-limit QC filter;
  # episode counts only get -99/negative filter (already done globally above)
  dur_vars_here <- vars_here %>% filter(!is_count)
  ep_vars_here  <- vars_here %>% filter(is_count)
  
  long_qc_parts <- list()
  
  # Duration variables — apply age-limit filter as before
  if (nrow(dur_vars_here) > 0) {
    long_qc_parts[["duration"]] <- dur_vars_here %>%
      mutate(tmp = purrr::map2(Variable, Unit, function(var_name, unit_name) {
        df0 <- merged %>% select(Proband, Subtyp, Alter, all_of(var_name))
        qc_and_filter_variable(df0, var = var_name, unit = unit_name) %>%
          mutate(Variable = var_name)
      })) %>%
      select(tmp) %>%
      tidyr::unnest(tmp) %>%
      mutate(cohort = cohort_name)
  }
  
  # Episode count variables — no age-limit filter; flag_exclude only for negative/missing
  if (nrow(ep_vars_here) > 0) {
    long_qc_parts[["counts"]] <- ep_vars_here %>%
      mutate(tmp = purrr::map2(Variable, Unit, function(var_name, unit_name) {
        vals <- safe_as_numeric(merged[[var_name]])
        merged %>%
          select(Proband, Subtyp, Alter) %>%
          mutate(
            Value            = vals,
            Unit             = unit_name,
            age_limit        = NA_real_,
            flag_negative    = is.finite(vals) & vals < 0,
            flag_exceeds_age = FALSE,
            flag_iqr         = FALSE,
            flag_exclude     = is.finite(vals) & vals < 0,
            Variable         = var_name
          )
      })) %>%
      select(tmp) %>%
      tidyr::unnest(tmp) %>%
      mutate(cohort = cohort_name)
  }
  
  long_qc <- bind_rows(long_qc_parts) %>%
    relocate(cohort, Variable, Unit, Proband, Subtyp, Alter, Value, age_limit,
             flag_negative, flag_exceeds_age, flag_iqr, flag_exclude)
  
  qc_overall <- long_qc %>%
    group_by(cohort, Variable, Unit) %>%
    summarise(
      N_total = n(),
      N_nonmissing = sum(is.finite(Value)),
      N_negative = sum(flag_negative, na.rm = TRUE),
      N_exclude_exceeds_age = sum(flag_exceeds_age, na.rm = TRUE),
      N_exclude_iqr = sum(flag_iqr, na.rm = TRUE),
      N_excluded_total = sum(flag_exclude, na.rm = TRUE),
      N_used_in_tests = sum(is.finite(Value) & !flag_exclude),
      Min = ifelse(sum(is.finite(Value) & !flag_exclude) > 0, min(Value[is.finite(Value) & !flag_exclude], na.rm = TRUE), NA_real_),
      Q05 = q_(Value[is.finite(Value) & !flag_exclude], 0.05),
      Q25 = q_(Value[is.finite(Value) & !flag_exclude], 0.25),
      Q50 = q_(Value[is.finite(Value) & !flag_exclude], 0.50),
      Q75 = q_(Value[is.finite(Value) & !flag_exclude], 0.75),
      Q95 = q_(Value[is.finite(Value) & !flag_exclude], 0.95),
      Max = ifelse(sum(is.finite(Value) & !flag_exclude) > 0, max(Value[is.finite(Value) & !flag_exclude], na.rm = TRUE), NA_real_),
      .groups = "drop"
    ) %>%
    mutate(
      exclude_age_enabled = EXCLUDE_EXCEEDS_AGE,
      exclude_iqr_enabled = EXCLUDE_IQR_OUTLIERS
    )
  write_out(qc_overall, file.path(out_dir, "qc_overall_used_for_tests.csv"))
  
  excluded_cases <- long_qc %>%
    filter(is.finite(Value), flag_exclude) %>%
    group_by(cohort, Variable) %>%
    arrange(desc(flag_exceeds_age), desc(abs(Value))) %>%
    slice_head(n = 200) %>%
    ungroup()
  write_out(excluded_cases, file.path(out_dir, "excluded_cases_age_or_iqr.csv"))
  
  descriptives_long <- long_qc %>%
    filter(is.finite(Value), !flag_exclude) %>%
    group_by(cohort, Variable, Unit, Subtyp) %>%
    summarise(
      n = n(),
      mean = mean(Value, na.rm = TRUE),
      sd = sd(Value, na.rm = TRUE),
      median = median(Value, na.rm = TRUE),
      iqr = IQR(Value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    complete(
      cohort = cohort_name,
      Variable = vars_here$Variable,
      Subtyp = factor(1:3, levels = 1:3),
      fill = list(n = 0, mean = NA_real_, sd = NA_real_, median = NA_real_, iqr = NA_real_)
    ) %>%
    arrange(Variable, Unit, Subtyp)
  
  descriptives_wide <- descriptives_long %>%
    mutate(
      Mean_SD = ifelse(n > 0, sprintf("%.3f (%.3f)", mean, sd), NA_character_),
      Median_IQR = ifelse(n > 0, sprintf("%.3f [%.3f]", median, iqr), NA_character_)
    ) %>%
    select(cohort, Variable, Unit, Subtyp, n, Mean_SD, Median_IQR) %>%
    pivot_wider(
      names_from = Subtyp,
      values_from = c(n, Mean_SD, Median_IQR),
      names_glue = "{.value}_Subtyp{Subtyp}"
    )
  
  write_out(descriptives_long, file.path(out_dir, "descriptives_long_post_filter.csv"))
  write_out(descriptives_wide, file.path(out_dir, "descriptives_wide_post_filter.csv"))
  
  # Global + post hoc
  global_list <- list()
  posthoc_list <- list()
  decision_list <- list()
  
  for (v in vars_here$Variable) {
    
    unit_v <- vars_here$Unit[vars_here$Variable == v][1]
    
    df_use <- long_qc %>%
      filter(Variable == v, is.finite(Value), !flag_exclude) %>%
      mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
      select(Subtyp, Value)
    
    n_by <- df_use %>% count(Subtyp, name = "n")
    enough_groups <- sum(n_by$n >= 2, na.rm = TRUE) >= 2
    
    if (!enough_groups || nrow(df_use) == 0) {
      decision_list[[v]] <- tibble(
        cohort = cohort_name, Variable = v, Unit = unit_v,
        decision = "Skipped (not enough data after filtering)"
      )
      next
    }
    kw <- tryCatch(kruskal.test(Value ~ Subtyp, data = df_use), error = function(e) NULL)
    
    if (is.null(kw) || !is.finite(as.numeric(kw$statistic)) || !is.finite(as.numeric(kw$p.value))) {
      decision_list[[v]] <- tibble(
        cohort = cohort_name,
        Variable = v,
        Unit = unit_v,
        decision = "Skipped (insufficient variability / excessive ties / non-informative variable)"
      )
      next
    }
    kw <- kruskal.test(Value ~ Subtyp, data = df_use)
    n <- nrow(df_use)
    k <- nlevels(df_use$Subtyp)
    eps2 <- epsilon2_kw(as.numeric(kw$statistic), n, k)
    
    global_list[[v]] <- tibble(
      cohort = cohort_name,
      Variable = v,
      Unit = unit_v,
      method = "Kruskal-Wallis",
      statistic = as.numeric(kw$statistic),
      df = as.numeric(kw$parameter),
      p.value = as.numeric(kw$p.value),
      effect_name = "epsilon2",
      effect_value = eps2,
      N = n
    )
    
    dunn <- tryCatch(
      rstatix::dunn_test(df_use, Value ~ Subtyp, p.adjust.method = "BH"),
      error = function(e) tibble()
    )
    
    if (nrow(dunn) == 0) {
      decision_list[[v]] <- tibble(
        cohort = cohort_name, Variable = v, Unit = unit_v,
        decision = "Kruskal-Wallis ok, Dunn failed or empty"
      )
      next
    }
    
    dunn <- dunn %>%
      rename(p_adj = p.adj) %>%
      mutate(
        cohort = cohort_name,
        Variable = v,
        Unit = unit_v,
        group1 = as.character(group1),
        group2 = as.character(group2),
        z_value = suppressWarnings(as.numeric(statistic))
      )
    
    if (!all(c("n1","n2") %in% names(dunn))) {
      n_lookup <- df_use %>% count(Subtyp, name = "n_sub") %>% mutate(Subtyp = as.character(Subtyp))
      dunn <- dunn %>%
        left_join(n_lookup, by = c("group1" = "Subtyp")) %>%
        rename(n1 = n_sub) %>%
        left_join(n_lookup, by = c("group2" = "Subtyp"), suffix = c("", "_g2")) %>%
        rename(n2 = n_sub_g2)
    }
    
    dunn <- dunn %>%
      mutate(
        n1 = suppressWarnings(as.numeric(n1)),
        n2 = suppressWarnings(as.numeric(n2)),
        N_pair = n1 + n2,
        r = z_value / sqrt(N_pair)
      )
    
    cd_tbl <- safe_cliffs_delta_pairs(df_use, value_col = "Value", group_col = "Subtyp")
    
    dunn2 <- dunn %>%
      left_join(cd_tbl, by = c("group1", "group2")) %>%
      mutate(
        signif = case_when(
          is.na(p_adj) ~ NA_character_,
          p_adj <= 0.0001 ~ "****",
          p_adj <= 0.001  ~ "***",
          p_adj <= 0.01   ~ "**",
          p_adj <= 0.05   ~ "*",
          TRUE ~ "ns"
        ),
        test = "Dunn_BH"
      )
    
    posthoc_list[[v]] <- dunn2
    
    decision_list[[v]] <- tibble(
      cohort = cohort_name,
      Variable = v,
      Unit = unit_v,
      decision = paste0(
        "Kruskal-Wallis + Dunn (BH), filter: -99->NA",
        if (unit_v == "count") " + negative->NA (count variable; no age-limit filter)"
        else paste0(
          if (EXCLUDE_EXCEEDS_AGE) " + exceeds_age" else "",
          if (EXCLUDE_IQR_OUTLIERS) " + IQR_outliers" else ""
        )
      )
    )
  }
  
  global_df <- bind_rows(global_list)
  posthoc_df <- bind_rows(posthoc_list)
  decision_df <- bind_rows(decision_list)
  
  global_df2 <- global_df %>%
    mutate(p_adj_BH_across_vars = p.adjust(p.value, method = "BH")) %>%
    arrange(p_adj_BH_across_vars)
  
  write_out(global_df,  file.path(out_dir, "global_tests_kw.csv"))
  write_out(global_df2, file.path(out_dir, "global_tests_kw_with_BH_across_vars.csv"))
  write_out(posthoc_df, file.path(out_dir, "posthoc_dunn_BH.csv"))
  write_out(decision_df, file.path(out_dir, "decision_table.csv"))
  
  summary_row <- tibble(
    cohort = cohort_name,
    n_total_after_join = nrow(merged),
    n_vars_available = nrow(vars_here),
    n_duration_vars  = sum(!vars_here$is_count),
    n_episode_vars   = sum(vars_here$is_count),
    n_vars_tested = nrow(global_df2),
    n_vars_sig_BH = sum(global_df2$p_adj_BH_across_vars < alpha_fdr, na.rm = TRUE),
    BH_family = paste0("all_", nrow(vars_here), "_illness_course_vars"),
    exclude_age_enabled = EXCLUDE_EXCEEDS_AGE,
    exclude_iqr_enabled = EXCLUDE_IQR_OUTLIERS
  )
  write_out(summary_row, file.path(out_dir, "summary_cohort.csv"))
  
  message("[OK] Finished cohort: ", cohort_name)
  list(summary = summary_row)
}

# -----------------------------
# 4) Run all cohorts
# -----------------------------
results <- purrr::imap(cohorts, ~run_one_cohort_durations(cohort_name = .y, assign_path = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))

write_out(summary_all, file.path(output_root, "summary_all_cohorts.csv"))

message("[OK] ALL COHORTS DONE. Output root: ", output_root)
message("[OK] End table written: summary_all_cohorts.csv")
