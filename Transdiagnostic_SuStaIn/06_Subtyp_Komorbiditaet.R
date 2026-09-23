# ===============================
# SuStaIn subtypes x Comorbidity (Komorbid: 0 = No, 1 = Yes)
# Cohort-wise: TDM, MDD, Angst, BD, SZ
#
# STEP 1 — Global test (χ² or Fisher's exact if expected cell < 5) + Cramer's V
# STEP 2 — Post hoc pairwise Fisher 2×2 + BH correction
# STEP 3 — Diagnosis-adjusted multinomial logistic regression
#            Subtyp ~ Komorbid + Diagnosis   (TDM cohort only; Dx is nuisance covariate)
#            Reported: OR + 95% CI + p-value (BH-adjusted) per subtype contrast
#
# Output: write_csv2 (Excel-DE: ; separator, decimal comma)
# Plots:  PNG + PDF (counts + percent, matching paper style)
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(purrr)
  library(DescTools)   # CramerV, Phi
  library(nnet)        # multinom
  library(broom)       # tidy.multinom
})

# -----------------------------------------------------------------------
# 0) Setup
# -----------------------------------------------------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
set.seed(20260916L) # Reproducible simulated Fisher p-values, if used.
setwd(base_dir)

output_root <- file.path(base_dir, "Output Scripte", "comorbidity_subtypes_all_cohorts")
if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

cohorts <- list(
  TDM   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

# -----------------------------------------------------------------------
# 1) Helpers
# -----------------------------------------------------------------------
write_out <- function(df, path) readr::write_csv2(df, path, na = "")

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

norm_id <- function(x) tolower(gsub("_", "-", trimws(as.character(x))))

derive_subtyp_from_ml <- function(df) {
  if (!("ML_Subtype" %in% names(df))) stop("ML_Subtype column not found in assignment file.")
  if (all(is.na(df$ML_Subtype)))      stop("ML_Subtype is all NA.")
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  if (!is.finite(ml_min)) stop("ML_Subtype min not finite.")
  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding. min=", ml_min)
  }
  df
}

p_adjust_bh_safe <- function(p) {
  out <- rep(NA_real_, length(p))
  idx <- which(is.finite(p))
  if (length(idx) > 0) out[idx] <- p.adjust(p[idx], method = "BH")
  out
}

p_to_signif <- function(p) {
  dplyr::case_when(
    is.na(p)              ~ NA_character_,
    p > 0.05              ~ "ns",
    p <= 0.05 & p > 0.01  ~ "*",
    p <= 0.01 & p > 0.001 ~ "**",
    p <= 0.001 & p > 0.0001 ~ "***",
    p <= 0.0001           ~ "****"
  )
}

safe_cramers_v <- function(tab) {
  if (length(dim(tab)) != 2) return(NA_real_)
  if (any(dim(tab) < 2))     return(NA_real_)
  if (sum(tab) <= 0)         return(NA_real_)
  tryCatch(as.numeric(CramerV(tab)), error = function(e) NA_real_)
}

safe_phi <- function(tab2x2) {
  if (!all(dim(tab2x2) == c(2, 2))) return(NA_real_)
  tryCatch(as.numeric(Phi(tab2x2)), error = function(e) NA_real_)
}

safe_fisher_2x2 <- function(tab2x2) {
  if (any(rowSums(tab2x2) == 0) || any(colSums(tab2x2) == 0))
    return(list(p = NA_real_, or = NA_real_, or_low = NA_real_, or_high = NA_real_))
  ft <- fisher.test(tab2x2)
  list(
    p      = as.numeric(ft$p.value),
    or     = as.numeric(unname(ft$estimate)),
    or_low  = as.numeric(ft$conf.int[1]),
    or_high = as.numeric(ft$conf.int[2])
  )
}

# Diagnosis mapping (consistent with all other scripts in this project)
map_diagnosis <- function(group_num) {
  dplyr::case_when(
    group_num == 2L            ~ "MDD",
    group_num == 3L            ~ "BD",
    group_num %in% c(4L, 5L)  ~ "SSD",
    group_num %in% c(7L, 8L)  ~ "ANX",
    TRUE                       ~ NA_character_
  )
}

# -----------------------------------------------------------------------
# 2) Load master database once
# -----------------------------------------------------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)

# Harmonise Proband column name
if (!"Proband" %in% names(data_all)) {
  if ("Name"  %in% names(data_all)) data_all <- data_all %>% rename(Proband = Name)
  if ("names" %in% names(data_all)) data_all <- data_all %>% rename(Proband = names)
}
if (!"Proband" %in% names(data_all)) stop("Proband column missing in data_all.")

data_all <- data_all %>%
  mutate(
    Proband  = norm_id(Proband),
    Komorbid = suppressWarnings(as.numeric(trimws(as.character(Komorbid)))),
    Komorbid = ifelse(Komorbid == -99, NA_real_, Komorbid),
    Group    = suppressWarnings(as.integer(as.numeric(trimws(as.character(Group)))))
  )

# Derive diagnosis label for use in multinomial regression
data_all <- data_all %>%
  mutate(Diagnosis = map_diagnosis(Group))

# -----------------------------------------------------------------------
# 3) Per-cohort runner
# -----------------------------------------------------------------------
run_one_cohort <- function(cohort_name, assign_path) {
  
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing(assign_path)
  
  out_dir <- file.path(output_root, cohort_name)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  # --- 3a) Load and harmonise assignment file ---
  cluster_data <- read_csv(assign_path, show_col_types = FALSE)
  
  if (!"Proband" %in% names(cluster_data)) {
    if ("Name"  %in% names(cluster_data)) cluster_data <- cluster_data %>% rename(Proband = Name)
    if ("names" %in% names(cluster_data)) cluster_data <- cluster_data %>% rename(Proband = names)
  }
  if (!"Proband" %in% names(cluster_data)) stop("Proband column missing: ", assign_path)
  
  cluster_data <- cluster_data %>%
    mutate(Proband = norm_id(Proband)) %>%
    derive_subtyp_from_ml() %>%
    filter(Subtyp %in% 1:3) %>%
    mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
    select(Proband, Subtyp)
  
  # --- 3b) Merge with database ---
  merged <- inner_join(cluster_data, data_all, by = "Proband")
  if (nrow(merged) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)
  
  # Exclude healthy controls (Group == 1)
  merged <- merged %>% filter(!(!is.na(Group) & Group == 1L))
  
  message("N after merge (patients only): ", nrow(merged))
  
  # --- 3c) Working dataset: keep Komorbid complete cases only ---
  komorbid_df <- merged %>%
    filter(!is.na(Komorbid)) %>%
    mutate(
      Komorbid_label = factor(
        ifelse(Komorbid == 1, "Yes", "No"),
        levels = c("No", "Yes")
      )
    )
  
  if (nrow(komorbid_df) == 0) {
    message("[SKIP] No non-missing Komorbid values for cohort: ", cohort_name)
    return(invisible(NULL))
  }
  
  # --- 3d) Descriptives: counts + percent per subtype ---
  desc_df <- komorbid_df %>%
    count(Subtyp, Komorbid_label, .drop = FALSE) %>%
    group_by(Subtyp) %>%
    mutate(
      Total   = sum(n),
      Percent = ifelse(Total > 0, 100 * n / Total, NA_real_)
    ) %>%
    ungroup()
  
  write_out(desc_df, file.path(out_dir, "desc_comorbidity_by_subtype.csv"))
  
  # Wide format: percent per subtype
  desc_wide <- desc_df %>%
    select(Subtyp, Komorbid_label, n, Percent) %>%
    pivot_wider(
      names_from  = Komorbid_label,
      values_from = c(n, Percent),
      names_sep   = "_"
    )
  write_out(desc_wide, file.path(out_dir, "desc_comorbidity_by_subtype_wide.csv"))
  
  # --- 3e) STEP 1: Global test (χ² or Fisher) + Cramer's V ---
  tbl <- xtabs(~ Komorbid_label + Subtyp, data = komorbid_df)
  
  chi_tmp    <- suppressWarnings(chisq.test(tbl))
  use_fisher <- any(chi_tmp$expected < 5)
  
  if (use_fisher) {
    testable <- sum(rowSums(tbl) > 0) >= 2L && sum(colSums(tbl) > 0) >= 2L
    gt <- if (testable) fisher.test(tbl, simulate.p.value = TRUE, B = 1e6) else NULL
    method_used <- if (testable) "Fisher's Exact Test (simulated)" else "Not estimable: fewer than two nonempty levels"
    statistic   <- NA_real_
    df_val      <- NA_real_
    p_val       <- if (testable) as.numeric(gt$p.value) else NA_real_
  } else {
    gt          <- chi_tmp
    method_used <- "Chi-squared Test"
    statistic   <- as.numeric(unname(gt$statistic))
    df_val      <- as.numeric(unname(gt$parameter))
    p_val       <- as.numeric(gt$p.value)
  }
  
  global_out <- tibble(
    cohort     = cohort_name,
    method     = method_used,
    statistic  = statistic,
    df         = df_val,
    p.value    = p_val,
    cramers_v  = safe_cramers_v(tbl),
    N_analyzed = nrow(komorbid_df),
    N_Yes      = sum(komorbid_df$Komorbid_label == "Yes"),
    N_No       = sum(komorbid_df$Komorbid_label == "No")
  )
  write_out(global_out, file.path(out_dir, "global_comorbidity_vs_subtype.csv"))
  
  message(sprintf("[Global] method=%s  statistic=%.3f  df=%s  p=%.4f  V=%.3f",
                  method_used,
                  ifelse(is.na(statistic), NA_real_, statistic),
                  ifelse(is.na(df_val), "sim", as.character(df_val)),
                  ifelse(is.na(p_val), NA_real_, p_val),
                  ifelse(is.na(safe_cramers_v(tbl)), NA_real_, safe_cramers_v(tbl))
  ))
  
  # --- 3f) STEP 2: Pairwise post hoc (Fisher 2×2) + BH ---
  subtypes  <- levels(komorbid_df$Subtyp)
  pair_idx  <- combn(subtypes, 2, simplify = FALSE)
  
  posthoc_df <- purrr::map_dfr(pair_idx, function(pp) {
    sub <- komorbid_df %>% filter(Subtyp %in% pp) %>% droplevels()
    tab2 <- xtabs(~ Komorbid_label + Subtyp, data = sub)
    
    # Ensure 2×2 structure even if one cell is empty
    tab2_full <- matrix(
      0L, nrow = 2, ncol = 2,
      dimnames = list(
        Komorbid_label = c("No", "Yes"),
        Subtyp         = pp
      )
    )
    tab2_full[rownames(tab2), colnames(tab2)] <- tab2
    tab2_full <- as.table(tab2_full)
    
    ft <- safe_fisher_2x2(tab2_full)
    
    tibble(
      cohort    = cohort_name,
      Subtyp1   = pp[1],
      Subtyp2   = pp[2],
      method    = "Fisher's Exact Test",
      p.value   = ft$p,
      OR        = ft$or,
      OR_low    = ft$or_low,
      OR_high   = ft$or_high,
      phi       = safe_phi(tab2_full)
    )
  })
  
  posthoc_df <- posthoc_df %>%
    mutate(
      p.adj.BH     = p_adjust_bh_safe(p.value),
      p.adj.signif = p_to_signif(p.adj.BH)
    ) %>%
    arrange(p.adj.BH, p.value)
  
  write_out(posthoc_df, file.path(out_dir, "posthoc_comorbidity_pairs_BH.csv"))
  
  # --- 3g) STEP 3: Diagnosis-adjusted multinomial logistic regression ---
  # Only meaningful for TDM (transdiagnostic) cohort where Diagnosis varies.
  # For diagnosis-specific cohorts (MDD, BD, etc.) Diagnosis is constant
  # -> we run a simple (unadjusted) multinom as a sensitivity check.
  
  # Prepare regression data
  reg_df <- komorbid_df %>%
    filter(!is.na(Komorbid_label), !is.na(Subtyp)) %>%
    mutate(
      Subtyp = factor(Subtyp, levels = 1:3),   # reference: Subtype 1
      Komorbid_bin = as.integer(Komorbid_label == "Yes")  # 0/1 numeric
    )
  
  has_diagnosis_variation <- length(unique(na.omit(reg_df$Diagnosis))) > 1
  
  if (has_diagnosis_variation) {
    # TDM: include Diagnosis as covariate
    reg_df <- reg_df %>%
      filter(!is.na(Diagnosis)) %>%
      mutate(Diagnosis = factor(Diagnosis, levels = c("MDD", "BD", "SSD", "ANX")))  # MDD = reference
    
    formula_multinom <- Subtyp ~ Komorbid_bin + Diagnosis
    
  } else {
    # Diagnosis-specific cohort: unadjusted model
    formula_multinom <- Subtyp ~ Komorbid_bin
  }
  
  # A constant comorbidity predictor has no estimable effect. Do not fit
  # a model and do not report coefficients for such a cohort.
  can_fit_multinom <- length(unique(reg_df$Komorbid_bin)) >= 2L
  multinom_model <- if (can_fit_multinom) {
    nnet::multinom(formula_multinom, data = reg_df, trace = FALSE)
  } else {
    message("[NOT ESTIMABLE] Comorbidity has only one observed level in ", cohort_name)
    NULL
  }
  
  if (!is.null(multinom_model)) {
    
    # tidy coefficients (log-OR scale)
    coef_tbl <- broom::tidy(multinom_model, conf.int = TRUE, conf.level = 0.95) %>%
      rename(
        subtype_contrast = y.level,
        log_OR           = estimate,
        log_OR_low       = conf.low,
        log_OR_high      = conf.high
      ) %>%
      mutate(
        OR       = exp(log_OR),
        OR_low   = exp(log_OR_low),
        OR_high  = exp(log_OR_high),
        cohort   = cohort_name,
        model    = ifelse(has_diagnosis_variation, "Diagnosis-adjusted", "Unadjusted (single Dx)")
      )
    
    # BH correction across ALL terms in this cohort
    coef_tbl <- coef_tbl %>%
      mutate(
        p.adj.BH     = p_adjust_bh_safe(p.value),
        p.adj.signif = p_to_signif(p.adj.BH)
      ) %>%
      select(
        cohort, model, subtype_contrast, term,
        log_OR, log_OR_low, log_OR_high,
        OR, OR_low, OR_high,
        statistic, p.value, p.adj.BH, p.adj.signif
      )
    
    write_out(coef_tbl, file.path(out_dir, "multinom_comorbidity_subtype.csv"))
    
    # Komorbid_bin rows only (main effect of interest)
    komorbid_coefs <- coef_tbl %>% filter(term == "Komorbid_bin")
    write_out(komorbid_coefs, file.path(out_dir, "multinom_komorbid_maineffect.csv"))
    
    message("[Multinom] Model fitted. Komorbid_bin effects:")
    print(komorbid_coefs %>% select(subtype_contrast, OR, OR_low, OR_high, p.value, p.adj.BH, p.adj.signif))
    
  } else {
    message("[SKIP] Multinomial model not fitted for cohort: ", cohort_name)
  }
  
  # -----------------------------------------------------------------------
  # 4) Plots
  # -----------------------------------------------------------------------
  palette_paper <- c(
    "No"  = "#A4C6D9",
    "Yes" = "#BF3E39"
  )
  
  plot_data <- desc_df %>%
    mutate(
      Komorbid_label = factor(Komorbid_label, levels = c("No", "Yes")),
      Subtyp = factor(Subtyp, levels = 1:3)
    )
  
  # -- Plot A: Counts (dodged) --
  p_counts <- ggplot(plot_data, aes(x = Subtyp, y = n, fill = Komorbid_label)) +
    geom_col(
      position = position_dodge(width = 0.78),
      width = 0.68,
      color = "black",
      linewidth = 0.3
    ) +
    scale_fill_manual(values = palette_paper, drop = FALSE) +
    labs(
      x     = "SuStaIn subtype",
      y     = "Count",
      fill  = "Comorbidity",
      title = paste0("Comorbidity by subtype — ", cohort_name)
    ) +
    theme_classic(base_size = 18) +
    theme(
      legend.title    = element_text(size = 15, face = "bold"),
      legend.position = "right",
      plot.title      = element_text(size = 20, face = "bold", color = "black"),
      axis.title.x    = element_text(size = 18, face = "bold", color = "black"),
      axis.title.y    = element_text(size = 18, face = "bold", color = "black"),
      axis.text.x     = element_text(size = 16, face = "bold", color = "black"),
      axis.text.y     = element_text(size = 15, face = "bold", color = "black"),
      legend.text     = element_text(size = 15, face = "bold", color = "black"),
      panel.grid      = element_blank()
    )
  
  # -- Plot B: Percent (dodged + labels) --
  p_percent <- ggplot(plot_data, aes(x = Subtyp, y = Percent, fill = Komorbid_label)) +
    geom_col(
      position = position_dodge(width = 0.78),
      width = 0.68,
      color = "black",
      linewidth = 0.3
    ) +
    geom_text(
      aes(label = ifelse(Percent > 0,
                         paste0(sprintf("%.1f", Percent), "%"), "")),
      position = position_dodge(width = 0.78),
      vjust = -0.35,
      size  = 5.2,
      fontface = "bold",
      color = "black"
    ) +
    scale_fill_manual(values = palette_paper, drop = FALSE) +
    scale_y_continuous(
      limits = c(0, max(plot_data$Percent, na.rm = TRUE) * 1.14),
      expand = expansion(mult = c(0, 0.02))
    ) +
    coord_cartesian(clip = "off") +
    labs(
      x     = "SuStaIn subtype",
      y     = "Percent within subtype",
      fill  = "Comorbidity",
      title = paste0("Comorbidity by subtype — ", cohort_name)
    ) +
    theme_classic(base_size = 18) +
    theme(
      legend.title    = element_text(size = 15, face = "bold"),
      legend.position = "right",
      plot.title      = element_text(size = 20, face = "bold", color = "black"),
      axis.title.x    = element_text(size = 18, face = "bold", color = "black"),
      axis.title.y    = element_text(size = 18, face = "bold", color = "black"),
      axis.text.x     = element_text(size = 16, face = "bold", color = "black"),
      axis.text.y     = element_text(size = 15, face = "bold", color = "black"),
      legend.text     = element_text(size = 15, face = "bold", color = "black"),
      panel.grid      = element_blank()
    )
  
  ggsave(
    file.path(out_dir, paste0("comorbidity_subtype_", cohort_name, "_counts.png")),
    p_counts, width = 8.5, height = 6.2, dpi = 300, bg = "white"
  )
  ggsave(
    file.path(out_dir, paste0("comorbidity_subtype_", cohort_name, "_counts.pdf")),
    p_counts, width = 8.5, height = 6.2, bg = "white"
  )
  ggsave(
    file.path(out_dir, paste0("comorbidity_subtype_", cohort_name, "_percent.png")),
    p_percent, width = 8.5, height = 6.2, dpi = 300, bg = "white"
  )
  ggsave(
    file.path(out_dir, paste0("comorbidity_subtype_", cohort_name, "_percent.pdf")),
    p_percent, width = 8.5, height = 6.2, bg = "white"
  )
  
  # -----------------------------------------------------------------------
  # 5) Cohort summary row
  # -----------------------------------------------------------------------
  summary_row <- tibble(
    cohort              = cohort_name,
    N_merged            = nrow(merged),
    N_Komorbid_analyzed = nrow(komorbid_df),
    N_Yes               = sum(komorbid_df$Komorbid_label == "Yes"),
    N_No                = sum(komorbid_df$Komorbid_label == "No"),
    global_method       = method_used,
    global_statistic    = statistic,
    global_df           = df_val,
    global_p            = p_val,
    cramers_v           = safe_cramers_v(tbl),
    multinom_model      = ifelse(!is.null(multinom_model),
                                 ifelse(has_diagnosis_variation,
                                        "Diagnosis-adjusted", "Unadjusted"),
                                 "Not fitted")
  )
  
  write_out(summary_row, file.path(out_dir, "summary_cohort_comorbidity.csv"))
  message("[OK] Finished cohort: ", cohort_name)
  
  list(summary = summary_row)
}

# -----------------------------------------------------------------------
# 4) Run all cohorts
# -----------------------------------------------------------------------
results <- purrr::imap(cohorts, ~ run_one_cohort(cohort_name = .y, assign_path = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))

write_out(summary_all, file.path(output_root, "summary_all_cohorts_comorbidity.csv"))

message("\n[DONE] All cohorts finished. Output: ", output_root)
