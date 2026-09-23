# ===============================
# PAPER VERSION: ML Stage vs duration + episode count variables
# Duration:       DurCurEp, DurHosp, DurMan, DurDep, DurPsych
# Episode counts: ManEp, DepEp, PsychEp
#
# All 8 variables form ONE BH correction family per cohort per model type
# (same overarching question: illness course ~ stage)
#
# Unified analysis strategy
#
# Main model
# 1. Value_std ~ ML_Stage_std
#
# Interaction model
# 2. Value_std ~ ML_Stage_std * Subtype
#
# Descriptive follow-up models within subtype
# 3. Value_std ~ ML_Stage_std
#
# Reported
# - beta_std
# - 95%-CI
# - p_value
# - q_BH
# - R2 / adj. R2
#
# QC / Cleaning
# - Duration vars: -99 -> NA, negative excluded, optional age plausibility filter
# - Episode counts: -99 -> NA, negative -> NA (no age-limit filter applies)
# - DurHosp is analyzed only in weeks
#
# BH families (per cohort):
#   * main slopes: all 8 ML_Stage_std terms across main models
#   * interaction terms: all ML_Stage_std:Subtype terms across interaction models
#   * follow-up slopes: all ML_Stage_std terms across subtype follow-up models
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(tibble)
  library(ggplot2)
  library(patchwork)
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

output_root <- file.path(base_dir, "Output Scripte", "durations_episoden_stage_3subtypes_all_cohorts_unified")
if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

cohorts <- list(
  TDM = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  MDD = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  ANX = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  BD  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SSD = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

K_SUBTYPES <- 3

stage_col <- "ML_Stage"
subtype_col <- "ML_Subtype"

MIN_N_MODEL <- 20
MIN_N_PLOT <- 25

EXCLUDE_EXCEEDS_AGE <- TRUE
EXCLUDE_IQR_OUTLIERS <- FALSE
APPLY_BH <- TRUE

palette_subtypes <- c(
  "1" = "#F8766D",
  "2" = "#00BA38",
  "3" = "#619CFF"
)

subtype_labels <- c(
  "1" = "Subtype 1",
  "2" = "Subtype 2",
  "3" = "Subtype 3"
)

duration_meta <- tibble(
  Variable = c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp"),
  Unit     = c("months",   "months", "months", "months", "weeks"),
  Derived  = c(FALSE,      FALSE,    FALSE,    FALSE,    FALSE),
  is_count = FALSE
)

# Episode count variables — no age-limit filter
episode_meta <- tibble(
  Variable = c("ManEp", "DepEp", "PsychEp"),
  Unit     = c("count", "count", "count"),
  Derived  = FALSE,
  is_count = TRUE
)

# Combined: 8 variables, one BH family per model type
all_vars_meta <- bind_rows(duration_meta, episode_meta)

duration_labels <- c(
  DurCurEp = "Duration current episode",
  DurMan   = "Duration mania",
  DurDep   = "Duration depression",
  DurPsych = "Duration psychosis",
  DurHosp  = "Duration hospitalization",
  ManEp    = "Manic episodes (n)",
  DepEp    = "Depressive episodes (n)",
  PsychEp  = "Psychotic episodes (n)"
)

# Duration variables first in combined plot, then episode counts
duration_order <- c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp",
                    "ManEp", "DepEp", "PsychEp")

zero_is_missing_vars <- character(0)

# -----------------------------
# 1) Helpers
# -----------------------------
stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

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
  cand <- c("Proband", "Name", "participant_id", "Participant", "Subject", "subject", "ID", "Id", "names")
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
    stop("Unexpected ML_Subtype coding (expected start 0 or 1). min = ", ml_min)
  }
  
  if (any(df$Subtype < 1 | df$Subtype > k, na.rm = TRUE)) {
    stop("Subtype out of range 1..", k)
  }
  
  df %>% mutate(Subtype = factor(Subtype, levels = 1:k))
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

safe_p_floor <- function(p) {
  if (is.na(p) || !is.finite(p)) return(NA_real_)
  if (p == 0) return(2.23e-308)
  max(p, 2.23e-308)
}

format_p_text <- function(p) {
  if (is.na(p) || !is.finite(p)) return(NA_character_)
  if (p <= 2.23e-308) return("<2.23e-308")
  format(p, scientific = TRUE, digits = 3)
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

age_limit_in_unit <- function(age_years, unit) {
  if (!is.finite(age_years)) return(NA_real_)
  if (unit == "months") return(age_years * 12)
  if (unit == "weeks") return(age_years * 52.1775)
  NA_real_
}

extract_lm_table_with_ci <- function(fit, cohort_name, variable_name, unit_name, model_name) {
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
  
  cf %>%
    left_join(ci_df, by = "term") %>%
    mutate(
      cohort = cohort_name,
      variable = variable_name,
      unit = unit_name,
      model = model_name,
      r_squared = sm$r.squared,
      adj_r_squared = sm$adj.r.squared,
      n = nobs(fit)
    ) %>%
    select(
      cohort, variable, unit, model, n, term,
      beta_std, std_error, t_value, p_value,
      CI_lower, CI_upper, r_squared, adj_r_squared
    )
}

make_compact_duration_table <- function(df_terms, keep_terms = NULL) {
  x <- df_terms
  if (!is.null(keep_terms)) {
    x <- x %>% filter(term %in% keep_terms)
  }
  
  x %>%
    transmute(
      cohort = cohort,
      variable = variable,
      unit = unit,
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

prepare_model_data <- function(df) {
  df %>%
    mutate(
      ML_Stage = safe_as_numeric_flexible(ML_Stage),
      Value = safe_as_numeric_flexible(Value),
      ML_Stage_std = zscore_safe(ML_Stage),
      Value_std = zscore_safe(Value),
      Subtype = factor(Subtype, levels = 1:K_SUBTYPES)
    )
}

qc_and_filter_variable <- function(df_sub, var, unit) {
  x <- safe_as_numeric_flexible(df_sub[[var]])
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
# Plot functions
# -----------------------------
make_scatter_plot <- function(df, cohort_name, variable_name, unit_name, out_png) {
  df_plot <- df %>%
    filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtype)) %>%
    mutate(Subtype = factor(as.character(Subtype), levels = c("1", "2", "3")))
  
  dirp <- dirname(out_png)
  if (!dir.exists(dirp)) dir.create(dirp, recursive = TRUE)
  
  values_finite <- df_plot$Value[is.finite(df_plot$Value)]
  
  if (nrow(df_plot) < MIN_N_PLOT) return(invisible(NULL))
  if (length(unique(values_finite)) < 2) return(invisible(NULL))
  if (is.na(stats::sd(values_finite)) || stats::sd(values_finite) == 0) return(invisible(NULL))
  if (all(values_finite == 0)) return(invisible(NULL))
  
  y_lab <- ifelse(
    variable_name %in% names(duration_labels),
    paste0(duration_labels[[variable_name]], " (", unit_name, ")"),
    paste0(variable_name, " (", unit_name, ")")
  )
  
  gg <- ggplot(df_plot, aes(x = ML_Stage, y = Value, color = Subtype)) +
    geom_point(
      alpha = 0.20,
      size = 1.0,
      position = position_jitter(width = 0.10, height = 0)
    ) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
    facet_wrap(
      ~ Subtype,
      nrow = 1,
      labeller = labeller(Subtype = subtype_labels)
    ) +
    scale_color_manual(
      values = palette_subtypes,
      breaks = c("1", "2", "3"),
      labels = subtype_labels
    ) +
    labs(
      title = paste0(
        "ML Stage vs ",
        ifelse(variable_name %in% names(duration_labels), duration_labels[[variable_name]], variable_name),
        " (", cohort_name, ")"
      ),
      x = "ML Stage",
      y = y_lab,
      color = "Subtype"
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0),
      axis.title = element_text(face = "bold"),
      legend.position = "none",
      plot.margin = margin(5, 5, 5, 5)
    )
  
  ggsave(out_png, gg, width = 12, height = 4.3, dpi = 300, bg = "white")
  invisible(NULL)
}

make_duration_panel <- function(df, variable_name) {
  df_plot <- df %>%
    filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtype)) %>%
    mutate(Subtype = factor(as.character(Subtype), levels = c("1", "2", "3")))
  
  values_finite <- df_plot$Value[is.finite(df_plot$Value)]
  
  if (nrow(df_plot) < MIN_N_PLOT) return(NULL)
  if (length(unique(values_finite)) < 2) return(NULL)
  if (is.na(stats::sd(values_finite)) || stats::sd(values_finite) == 0) return(NULL)
  if (all(values_finite == 0)) return(NULL)
  
  unit_name <- unique(df_plot$Unit)
  if (length(unit_name) == 0) unit_name <- NA_character_
  unit_name <- unit_name[1]
  
  y_lab <- ifelse(
    variable_name %in% names(duration_labels),
    paste0(duration_labels[[variable_name]], " (", unit_name, ")"),
    paste0(variable_name, " (", unit_name, ")")
  )
  
  ggplot(df_plot, aes(x = ML_Stage, y = Value, color = Subtype)) +
    geom_point(
      alpha = 0.20,
      size = 1.0,
      position = position_jitter(width = 0.10, height = 0)
    ) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
    scale_color_manual(
      values = palette_subtypes,
      breaks = c("1", "2", "3"),
      labels = subtype_labels
    ) +
    labs(
      x = "ML Stage",
      y = y_lab,
      color = "Subtype"
    ) +
    theme_classic(base_size = 12) +
    theme(
      axis.title = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      plot.margin = margin(5, 5, 5, 5)
    )
}

make_combined_duration_plot <- function(plot_data_list, cohort_name, out_png) {
  available_vars <- intersect(duration_order, names(plot_data_list))
  
  if (length(available_vars) == 0) {
    message("[SKIP] No duration data available for combined plot: ", cohort_name)
    return(invisible(NULL))
  }
  
  panel_list <- lapply(available_vars, function(v) {
    make_duration_panel(
      df = plot_data_list[[v]],
      variable_name = v
    )
  })
  
  keep <- !vapply(panel_list, is.null, logical(1))
  panel_list <- panel_list[keep]
  
  if (length(panel_list) == 0) {
    message("[SKIP] No plottable duration variables for combined plot: ", cohort_name)
    return(invisible(NULL))
  }
  
  combined_plot <- wrap_plots(panel_list, ncol = 3, guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      plot.margin = margin(8, 8, 8, 8)
    )
  
  combined_plot <- combined_plot +
    plot_annotation(
      title = paste0("ML Stage vs duration variables (", cohort_name, ")"),
      tag_levels = "A",
      theme = theme(
        plot.title = element_text(face = "bold", hjust = 0, size = 16),
        plot.margin = margin(8, 8, 8, 8)
      )
    )
  
  ggsave(
    filename = out_png,
    plot = combined_plot,
    width = 12,
    height = 8,
    dpi = 300,
    bg = "white"
  )
  
  invisible(out_png)
}

# -----------------------------
# 2) Load data_all once
# -----------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)

id_col_all <- detect_id_col(data_all)
if (is.null(id_col_all)) stop("No ID column detected in data_all.")
if (id_col_all != "Proband") data_all <- data_all %>% rename(Proband = all_of(id_col_all))

if (!("Alter" %in% names(data_all))) stop("Column 'Alter' missing in data_all.")

data_all <- data_all %>%
  mutate(
    Proband = norm_id(Proband),
    Alter = safe_as_numeric_flexible(Alter)
  )

required_course_vars <- c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp", "ManEp", "DepEp", "PsychEp")
missing_course_vars <- setdiff(required_course_vars, names(data_all))
if (length(missing_course_vars) > 0L) stop("Required clinical-course variables missing: ", paste(missing_course_vars, collapse = ", "))
base_vars <- c("DurCurEp", "DurMan", "DurDep", "DurPsych", "DurHosp")

if (length(base_vars) > 0) {
  data_all <- data_all %>%
    mutate(across(all_of(base_vars), ~ ifelse(.x == -99, NA, .x))) %>%
    mutate(across(all_of(base_vars), safe_as_numeric_flexible))
}

if (length(zero_is_missing_vars) > 0) {
  zvars <- intersect(zero_is_missing_vars, base_vars)
  if (length(zvars) > 0) {
    data_all <- data_all %>%
      mutate(across(all_of(zvars), ~ ifelse(.x == 0, NA, .x)))
  }
}

# Episode counts: -99 -> NA; negative values impossible -> NA
ep_base_vars <- c("ManEp", "DepEp", "PsychEp")
if (length(ep_base_vars) > 0) {
  data_all <- data_all %>%
    mutate(across(all_of(ep_base_vars), ~ {
      x <- safe_as_numeric_flexible(.x)
      x[x == -99] <- NA_real_
      x[x < 0]    <- NA_real_
      x
    }))
}

missing_ep <- setdiff(c("ManEp", "DepEp", "PsychEp"), names(data_all))
if (length(missing_ep) > 0)
  message("[WARN] Episode count variables absent from database: ",
          paste(missing_ep, collapse = ", "))

# -----------------------------
# 3) Run one cohort
# -----------------------------
run_one_cohort_durations_stage <- function(cohort_name, assign_path) {
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing(assign_path)
  
  out_dir <- file.path(output_root, cohort_name)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  dir_data  <- file.path(out_dir, "data")
  dir_plots <- file.path(out_dir, "plots")
  dir_qc    <- file.path(out_dir, "qc")
  dir_stats <- file.path(out_dir, "stats")
  for (d in c(dir_data, dir_plots, dir_qc, dir_stats)) {
    if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  }
  
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
    select(Proband, Subtype, ML_Stage) %>%
    filter(!is.na(Proband), is.finite(ML_Stage), !is.na(Subtype))
  
  merged <- inner_join(asg2, data_all, by = "Proband")
  
  if (nrow(merged) == 0) {
    note <- tibble(note = "Join produced 0 rows. Check ID harmonization.")
    safe_write_csv(note, file.path(dir_qc, "note_join_zero_rows.csv"))
    return(list(summary = tibble(cohort = cohort_name, n_total_after_join = 0L)))
  }
  
  safe_write_csv(
    merged %>% select(Proband, Subtype, ML_Stage, Alter),
    file.path(dir_data, "merged_ids_stage_age.csv")
  )
  
  subtype_counts <- merged %>%
    count(Subtype, name = "n") %>%
    complete(Subtype = factor(1:3, levels = 1:3), fill = list(n = 0)) %>%
    arrange(Subtype)
  
  safe_write_csv(subtype_counts, file.path(dir_qc, "subtype_counts_after_join.csv"))
  
  vars_here <- all_vars_meta %>% filter(Variable %in% names(merged))
  if (nrow(vars_here) == 0) {
    writeLines("No duration or episode variables present in merged data.",
               con = file.path(dir_qc, "note_no_vars.txt"))
    return(list(summary = tibble(cohort = cohort_name, n_total_after_join = nrow(merged), n_vars_available = 0L)))
  }
  
  dur_vars_here <- vars_here %>% filter(!is_count)
  ep_vars_here  <- vars_here %>% filter(is_count)
  
  long_qc_parts <- list()
  
  # Duration variables — age-limit QC filter applies
  if (nrow(dur_vars_here) > 0) {
    long_qc_parts[["duration"]] <- dur_vars_here %>%
      mutate(tmp = purrr::map2(Variable, Unit, function(var_name, unit_name) {
        df0 <- merged %>% select(Proband, Subtype, ML_Stage, Alter, all_of(var_name))
        qc_and_filter_variable(df0, var = var_name, unit = unit_name) %>%
          mutate(Variable = var_name)
      })) %>%
      select(tmp) %>%
      unnest(tmp) %>%
      mutate(cohort = cohort_name)
  }
  
  # Episode count variables — no age-limit filter; only negative/missing exclusion
  if (nrow(ep_vars_here) > 0) {
    long_qc_parts[["counts"]] <- ep_vars_here %>%
      mutate(tmp = purrr::map2(Variable, Unit, function(var_name, unit_name) {
        vals <- safe_as_numeric_flexible(merged[[var_name]])
        merged %>%
          select(Proband, Subtype, ML_Stage, Alter) %>%
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
      unnest(tmp) %>%
      mutate(cohort = cohort_name)
  }
  
  long_qc <- bind_rows(long_qc_parts) %>%
    relocate(cohort, Variable, Unit, Proband, Subtype, ML_Stage, Alter, Value, age_limit,
             flag_negative, flag_exceeds_age, flag_iqr, flag_exclude)
  
  safe_write_csv(long_qc, file.path(dir_data, "long_qc_table.csv"))
  
  qc_overall_pre <- long_qc %>%
    group_by(cohort, Variable, Unit) %>%
    summarise(
      N_total = n(),
      N_nonmissing = sum(is.finite(Value)),
      N_missing = N_total - N_nonmissing,
      Missing_pct = 100 * N_missing / N_total,
      N_negative = sum(flag_negative, na.rm = TRUE),
      Min = ifelse(sum(is.finite(Value)) > 0, min(Value, na.rm = TRUE), NA_real_),
      Q05 = q_(Value, 0.05),
      Q25 = q_(Value, 0.25),
      Q50 = q_(Value, 0.50),
      Q75 = q_(Value, 0.75),
      Q95 = q_(Value, 0.95),
      Max = ifelse(sum(is.finite(Value)) > 0, max(Value, na.rm = TRUE), NA_real_),
      IQR_lower = iqr_bounds(Value)["lower"],
      IQR_upper = iqr_bounds(Value)["upper"],
      .groups = "drop"
    )
  
  safe_write_csv(qc_overall_pre, file.path(dir_qc, "qc_overall_pre_filter.csv"))
  
  qc_used <- long_qc %>%
    group_by(cohort, Variable, Unit) %>%
    summarise(
      N_nonmissing = sum(is.finite(Value)),
      N_exclude_negative = sum(flag_negative, na.rm = TRUE),
      N_exclude_exceeds_age = sum(flag_exceeds_age, na.rm = TRUE),
      N_exclude_iqr = sum(flag_iqr, na.rm = TRUE),
      N_excluded_total = sum(flag_exclude, na.rm = TRUE),
      N_used_in_tests = sum(is.finite(Value) & !flag_exclude & is.finite(ML_Stage)),
      Used_pct_of_nonmissing = ifelse(N_nonmissing > 0, 100 * N_used_in_tests / N_nonmissing, NA_real_),
      .groups = "drop"
    ) %>%
    mutate(
      exclude_age_enabled = EXCLUDE_EXCEEDS_AGE,
      exclude_iqr_enabled = EXCLUDE_IQR_OUTLIERS
    )
  
  safe_write_csv(qc_used, file.path(dir_qc, "qc_exclusions_used_for_tests.csv"))
  
  excluded_cases <- long_qc %>%
    filter(is.finite(Value), flag_exclude) %>%
    group_by(cohort, Variable) %>%
    arrange(desc(flag_exceeds_age), desc(abs(Value))) %>%
    slice_head(n = 200) %>%
    ungroup()
  
  safe_write_csv(excluded_cases, file.path(dir_qc, "excluded_cases_age_or_iqr.csv"))
  
  desc_stage <- bind_rows(
    desc_paper(merged$ML_Stage) %>% mutate(group = "Overall"),
    merged %>%
      group_by(Subtype) %>%
      summarise(desc_paper(ML_Stage), .groups = "drop") %>%
      mutate(group = paste0("Subtype ", as.character(Subtype))) %>%
      select(-Subtype)
  ) %>%
    mutate(cohort = cohort_name) %>%
    select(cohort, group, everything())
  
  safe_write_csv(desc_stage, file.path(dir_stats, paste0("desc_stage_", cohort_name, ".csv")))
  
  desc_rows <- list()
  main_rows <- list()
  interaction_rows <- list()
  followup_rows <- list()
  plot_data_list <- list()
  
  for (v in vars_here$Variable) {
    unit_v <- vars_here$Unit[vars_here$Variable == v][1]
    
    df_use <- long_qc %>%
      filter(Variable == v, is.finite(Value), !flag_exclude, is.finite(ML_Stage)) %>%
      mutate(Subtype = factor(Subtype, levels = 1:3)) %>%
      select(Proband, Subtype, ML_Stage, Value) %>%
      mutate(Unit = unit_v)
    
    subtype_counts_current <- df_use %>%
      count(Subtype, name = "n_subtype")
    
    valid_subtypes <- subtype_counts_current %>%
      filter(n_subtype >= MIN_N_MODEL) %>%
      pull(Subtype) %>%
      as.character()
    
    if (length(valid_subtypes) < 2) {
      message("[SKIP] ", cohort_name, " | ", v, ": <2 valid subtypes (n >= ", MIN_N_MODEL, ")")
      next
    }
    
    df_use <- df_use %>%
      filter(Subtype %in% valid_subtypes)
    
    plot_data_list[[v]] <- df_use
    
    desc_one <- bind_rows(
      desc_paper(df_use$Value) %>% mutate(group = "Overall"),
      df_use %>%
        group_by(Subtype) %>%
        summarise(desc_paper(Value), .groups = "drop") %>%
        mutate(group = paste0("Subtype ", as.character(Subtype))) %>%
        select(-Subtype)
    ) %>%
      mutate(cohort = cohort_name, variable = v, unit = unit_v) %>%
      select(cohort, variable, unit, group, everything())
    
    desc_rows[[length(desc_rows) + 1]] <- desc_one
    
    model_dat <- prepare_model_data(df_use) %>%
      filter(is.finite(Value_std), is.finite(ML_Stage_std))
    
    if (nrow(model_dat) < MIN_N_MODEL) next
    
    # Main model
    fit_main <- lm(Value_std ~ ML_Stage_std, data = model_dat)
    tbl_main <- extract_lm_table_with_ci(
      fit = fit_main,
      cohort_name = cohort_name,
      variable_name = v,
      unit_name = unit_v,
      model_name = "main_model"
    ) %>%
      initialize_p_columns(family_label = NA_character_) %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "primary_stage_term", "other_terms"))
    
    main_rows[[length(main_rows) + 1]] <- tbl_main
    
    # Interaction model
    fit_int <- lm(Value_std ~ ML_Stage_std * Subtype, data = model_dat)
    tbl_int <- extract_lm_table_with_ci(
      fit = fit_int,
      cohort_name = cohort_name,
      variable_name = v,
      unit_name = unit_v,
      model_name = "interaction_model"
    ) %>%
      initialize_p_columns(family_label = NA_character_) %>%
      mutate(
        term_group = case_when(
          grepl("^ML_Stage_std:Subtype", term) ~ "interaction_terms",
          term == "ML_Stage_std" ~ "reference_stage_term",
          grepl("^Subtype", term) ~ "subtype_main_effect_terms",
          TRUE ~ "other_terms"
        )
      )
    
    interaction_rows[[length(interaction_rows) + 1]] <- tbl_int
    
    # Follow-up models
    tbl_follow <- model_dat %>%
      group_by(Subtype) %>%
      group_modify(function(dd, key) {
        nsub <- nrow(dd)
        
        if (nsub < MIN_N_MODEL) {
          return(tibble(
            cohort = cohort_name,
            variable = v,
            unit = unit_v,
            model = paste0("followup_subtype_", as.character(key$Subtype)),
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
        
        fit_sub <- lm(Value_std ~ ML_Stage_std, data = dd)
        extract_lm_table_with_ci(
          fit = fit_sub,
          cohort_name = cohort_name,
          variable_name = v,
          unit_name = unit_v,
          model_name = paste0("followup_subtype_", as.character(key$Subtype))
        )
      }) %>%
      ungroup() %>%
      initialize_p_columns(family_label = NA_character_) %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "followup_stage_term", "other_terms"))
    
    followup_rows[[length(followup_rows) + 1]] <- tbl_follow
    
    if (nrow(df_use %>% filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtype))) > 0) {
      out_png <- file.path(dir_plots, paste0("scatter_stage_vs_", v, "_bySubtype.png"))
      make_scatter_plot(df_use, cohort_name, v, unit_v, out_png)
    }
  }
  
  out_png_combined <- file.path(dir_plots, paste0("figure_combined_MLStage_vs_duration_", cohort_name, ".png"))
  make_combined_duration_plot(plot_data_list, cohort_name, out_png_combined)
  
  desc_duration <- bind_rows(desc_rows)
  main_df <- bind_rows(main_rows)
  interaction_df <- bind_rows(interaction_rows)
  followup_df <- bind_rows(followup_rows)
  
  # -----------------------------
  # BH corrections
  # -----------------------------
  if (nrow(main_df) > 0) {
    idx_main <- main_df$term == "ML_Stage_std"
    main_df <- apply_bh_subset(
      main_df,
      subset_idx = idx_main,
      family_label = paste0(cohort_name, "_duration_episode_main_terms")
    )
  }
  
  if (nrow(interaction_df) > 0) {
    idx_int <- grepl("^ML_Stage_std:Subtype", interaction_df$term)
    interaction_df <- apply_bh_subset(
      interaction_df,
      subset_idx = idx_int,
      family_label = paste0(cohort_name, "_duration_episode_interaction_terms")
    )
  }
  
  if (nrow(followup_df) > 0) {
    idx_follow <- followup_df$term == "ML_Stage_std"
    followup_df <- apply_bh_subset(
      followup_df,
      subset_idx = idx_follow,
      family_label = paste0(cohort_name, "_duration_episode_followup_terms")
    )
  }
  
  safe_write_csv(desc_duration, file.path(dir_stats, paste0("desc_duration_", cohort_name, ".csv")))
  safe_write_csv(main_df, file.path(dir_stats, paste0("duration_main_model_", cohort_name, ".csv")))
  safe_write_csv(interaction_df, file.path(dir_stats, paste0("duration_interaction_model_", cohort_name, ".csv")))
  safe_write_csv(followup_df, file.path(dir_stats, paste0("duration_followup_models_", cohort_name, ".csv")))
  
  duration_table_main <- make_compact_duration_table(
    main_df %>% filter(term == "ML_Stage_std"),
    keep_terms = "ML_Stage_std"
  )
  
  duration_table_int <- make_compact_duration_table(
    interaction_df %>% filter(term %in% c("ML_Stage_std", "Subtype2", "Subtype3", "ML_Stage_std:Subtype2", "ML_Stage_std:Subtype3")),
    keep_terms = c("ML_Stage_std", "Subtype2", "Subtype3", "ML_Stage_std:Subtype2", "ML_Stage_std:Subtype3")
  )
  
  duration_table_follow <- followup_df %>%
    filter(term == "ML_Stage_std") %>%
    mutate(model = gsub("followup_subtype_", "Subtype ", model)) %>%
    make_compact_duration_table()
  
  safe_write_tsv(duration_table_main, file.path(dir_stats, paste0("duration_TABLE_main_model_", cohort_name, ".tsv")))
  safe_write_tsv(duration_table_int, file.path(dir_stats, paste0("duration_TABLE_interaction_model_", cohort_name, ".tsv")))
  safe_write_tsv(duration_table_follow, file.path(dir_stats, paste0("duration_TABLE_followup_models_", cohort_name, ".tsv")))
  
  summary_row <- tibble(
    cohort = cohort_name,
    n_total_after_join  = nrow(merged),
    n_vars_available    = nrow(vars_here),
    n_duration_vars     = sum(!vars_here$is_count),
    n_episode_vars      = sum(vars_here$is_count),
    BH_family           = paste0("all_", nrow(vars_here), "_illness_course_vars_per_model_type"),
    durhosp_unit        = "weeks",
    exclude_age_enabled = EXCLUDE_EXCEEDS_AGE,
    exclude_iqr_enabled = EXCLUDE_IQR_OUTLIERS,
    min_n_model         = MIN_N_MODEL
  )
  
  safe_write_csv(summary_row, file.path(dir_stats, "summary_cohort.csv"))
  
  message("[OK] Finished cohort: ", cohort_name)
  list(summary = summary_row)
}

# -----------------------------
# 4) Run all cohorts
# -----------------------------
results <- purrr::imap(cohorts, ~ run_one_cohort_durations_stage(cohort_name = .y, assign_path = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))

safe_write_csv(summary_all, file.path(output_root, "summary_all_cohorts.csv"))

message("[OK] ALL COHORTS DONE. Output root: ", output_root)
message("[OK] End table written: summary_all_cohorts.csv")
