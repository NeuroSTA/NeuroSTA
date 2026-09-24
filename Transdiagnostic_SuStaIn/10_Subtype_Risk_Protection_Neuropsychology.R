# ===============================
# Script 1 (UPDATED):
# Early/Current risk & protective factors + Neuropsych performance ~ SuStaIn subtypes
#
# - 3 subtypes per cohort: main, MDD, Angst, BD, SZ
# - Metric: Kruskal-Wallis + Dunn
# - Nominal: Chi2 or Fisher + pairwise Fisher
#
# DOMAIN LOGIC (paper-conform):
# - Domains: Early, Current, Neuropsych
#   * Early includes early risk + early protective + familial genetic risk
#   * Current includes LEQ (events) + resilience/social support
#
# MULTIPLE TESTING (domain-specific, per cohort):
# - Global tests: BH within domain across ALL variables (metric + nominal)
# - Posthoc tests: BH within domain across ALL pairwise tests (Dunn + Fisher pairs)
#
# DESCRIPTIVES:
# - Metric: Mean(SD) + Median(IQR) per subtype
# - Nominal (GenRisk_*): counts + percent per category per subtype
#
# Output: write_csv2 (Excel-DE: ; and decimal comma)
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(rstatix)
  library(DescTools)
  library(purrr)
  library(stringr)
  library(car)
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

output_root <- file.path(base_dir, "Output Scripte", "early_current_neuropsych_nonparam_3subtypes_all_cohorts")
if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

cohorts <- list(
  main  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

# -----------------------------
# Helpers
# -----------------------------
stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

norm_id <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- tolower(x)
  x <- gsub("_", "-", x)
  x
}

write_out <- function(df, path) {
  readr::write_csv2(df, path, na = "")
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
    stop("Unexpected ML_Subtype coding (expected start 0 or 1). min=", ml_min)
  }
  df
}

safe_shapiro_p <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  if (isTRUE(sd(x) == 0)) return(NA_real_)
  out <- tryCatch(shapiro.test(x)$p.value, error = function(e) NA_real_)
  as.numeric(out)
}

safe_levene_p <- function(df, y, g = "Subtyp") {
  out <- tryCatch({
    lv <- car::leveneTest(df[[y]] ~ df[[g]])
    as.numeric(lv$`Pr(>F)`[1])
  }, error = function(e) NA_real_)
  as.numeric(out)
}

epsilon2_from_kw <- function(H, n, k) {
  if (!is.finite(H) || !is.finite(n) || !is.finite(k) || n <= 1) return(NA_real_)
  val <- (H - k + 1) / (n - 1)
  val <- as.numeric(val)
  if (!is.finite(val)) return(NA_real_)
  if (val < 0) val <- 0
  val
}

z_to_r <- function(z, n1, n2) {
  N <- n1 + n2
  if (!is.finite(z) || N <= 0) return(NA_real_)
  as.numeric(z) / sqrt(N)
}

cliffs_delta <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  n1 <- length(x); n2 <- length(y)
  if (n1 == 0 || n2 == 0) return(NA_real_)
  cmp <- outer(x, y, FUN = "-")
  gt <- sum(cmp > 0)
  lt <- sum(cmp < 0)
  as.numeric((gt - lt) / (n1 * n2))
}

cliffs_magnitude <- function(delta) {
  if (!is.finite(delta)) return(NA_character_)
  ad <- abs(delta)
  if (ad < 0.147) return("negligible")
  if (ad < 0.33)  return("small")
  if (ad < 0.474) return("medium")
  "large"
}

p_stars <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p <= 0.0001) return("****")
  if (p <= 0.001)  return("***")
  if (p <= 0.01)   return("**")
  if (p <= 0.05)   return("*")
  "ns"
}

safe_fisher <- function(tab) {
  fisher.test(tab)
}

safe_p_adjust <- function(p, method = "BH") {
  p <- as.numeric(p)
  if (length(p) == 0) return(p)
  ok <- is.finite(p)
  out <- rep(NA_real_, length(p))
  if (sum(ok) > 0) out[ok] <- p.adjust(p[ok], method = method)
  out
}

# -----------------------------
# 1) Variables + DOMAIN MAPPING (paper-conform)
# -----------------------------
# Metric vars
early_metric_vars <- c(
  "CTQ_Sum","UrbanicityScore","AlterVaterBeiGeburt","Geburtsgewicht","SSW_Geburt",
  "FEB_FuersorgeMutter","FEB_FuersorgeVater"
)

current_metric_vars <- c(
  "LEQ_TotalEventScore","LEQ_NegativeEventScore","LEQ_PositiveEventScore","FSozU_Sum","RS25_Sum"
)

neuropsych_metric_vars <- c(
  "VLMT_Sum_Richtige","TMT_Differenz","BZT_Sum","D2_KL",
  "RWT_Tiere_SumCorrected","RWT_P_SumCorrected","RWT_Altern_SumCorrected",
  "ZST_Sum","Blockspanne_Gesamt"
)

metric_vars <- c(early_metric_vars, current_metric_vars, neuropsych_metric_vars)

# Nominal (familial/genetic risk): belongs to EARLY domain for BH family as requested
nominal_vars <- c("GenRisiko_Affektiv1","GenRisiko_Psycho1")

domain_of_metric <- function(v) {
  if (v %in% early_metric_vars) return("Early")
  if (v %in% current_metric_vars) return("Current")
  if (v %in% neuropsych_metric_vars) return("Neuropsych")
  "Other"
}

domain_of_nominal <- function(v) {
  if (v %in% nominal_vars) return("Early")
  "Nominal_Other"
}

# -----------------------------
# 2) Load data
# -----------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)

if (!"Proband" %in% names(data_all)) {
  if ("Name" %in% names(data_all)) data_all <- data_all %>% rename(Proband = Name)
  if ("names" %in% names(data_all)) data_all <- data_all %>% rename(Proband = names)
}
if (!"Proband" %in% names(data_all)) stop("Fehler: Proband-Spalte fehlt in data_all.")
data_all <- data_all %>% mutate(Proband = norm_id(Proband))

# -----------------------------
# 3) Cohort runner
# -----------------------------
run_one_cohort_early_current_neuro <- function(cohort_name, assign_path) {
  
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
  if (!"Proband" %in% names(cluster_data)) stop("Fehler: Proband-Spalte fehlt in assignment file: ", assign_path)
  
  cluster_data <- cluster_data %>%
    mutate(Proband = norm_id(Proband)) %>%
    derive_subtyp_from_ml() %>%
    filter(Subtyp %in% 1:3) %>%
    mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
    select(Proband, Subtyp)
  
  merged <- inner_join(cluster_data, data_all, by = "Proband")
  if (nrow(merged) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)
  
  subtype_counts <- merged %>%
    count(Subtyp, name = "n") %>%
    tidyr::complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n = 0)) %>%
    arrange(Subtyp)
  write_out(subtype_counts, file.path(out_dir, "subtype_counts_after_join.csv"))
  
  present_metrics  <- intersect(metric_vars, names(merged))
  present_nominals <- intersect(nominal_vars, names(merged))
  missing_analysis_vars <- setdiff(c(metric_vars, nominal_vars), names(merged))
  if (length(missing_analysis_vars) > 0L) stop("Required phenotype variables missing: ", paste(missing_analysis_vars, collapse = ", "))
  
  merged <- merged %>%
    mutate(across(all_of(c(present_metrics, present_nominals)), ~ replace(.x, .x == -99, NA))) %>%
    mutate(across(all_of(present_metrics), ~ suppressWarnings(as.numeric(.x))))
  
  if (length(present_nominals) > 0) {
    merged <- merged %>% mutate(across(all_of(present_nominals), ~ droplevels(as.factor(.x))))
  }
  
  # -----------------------------
  # Missingness
  # -----------------------------
  missing_overall <- tibble(Variable = c(present_metrics, present_nominals)) %>%
    mutate(
      cohort = cohort_name,
      Domain = ifelse(Variable %in% present_metrics,
                      vapply(Variable, domain_of_metric, character(1)),
                      vapply(Variable, domain_of_nominal, character(1))),
      N_total = nrow(merged),
      N_nonmissing = map_int(Variable, ~ sum(!is.na(merged[[.x]]))),
      N_missing = map_int(Variable, ~ sum(is.na(merged[[.x]]))),
      Missing_pct = ifelse(N_total > 0, 100 * N_missing / N_total, NA_real_)
    ) %>%
    select(cohort, Domain, Variable, N_total, N_nonmissing, N_missing, Missing_pct)
  
  missing_by_subtype <- tibble(Variable = c(present_metrics, present_nominals)) %>%
    tidyr::expand_grid(Subtyp = factor(1:3, levels = 1:3)) %>%
    mutate(
      cohort = cohort_name,
      Domain = ifelse(Variable %in% present_metrics,
                      vapply(Variable, domain_of_metric, character(1)),
                      vapply(Variable, domain_of_nominal, character(1))),
      N_in_subtype = map_int(Subtyp, ~ sum(merged$Subtyp == .x, na.rm = TRUE)),
      N_missing = map2_int(Variable, Subtyp, ~ {
        idx <- merged$Subtyp == .y
        sum(is.na(merged[[.x]][idx]))
      }),
      Missing_pct = ifelse(N_in_subtype > 0, 100 * N_missing / N_in_subtype, NA_real_)
    ) %>%
    select(cohort, Domain, Variable, Subtyp, N_in_subtype, N_missing, Missing_pct)
  
  write_out(missing_overall, file.path(out_dir, "missingness_by_variable.csv"))
  write_out(missing_by_subtype, file.path(out_dir, "missingness_by_variable_and_subtype.csv"))
  
  # -----------------------------
  # Descriptives (metrics)
  # -----------------------------
  if (length(present_metrics) > 0) {
    descriptives_long <- merged %>%
      select(Subtyp, all_of(present_metrics)) %>%
      pivot_longer(-Subtyp, names_to = "Variable", values_to = "Value") %>%
      mutate(Domain = vapply(Variable, domain_of_metric, character(1))) %>%
      group_by(Domain, Variable, Subtyp) %>%
      summarise(
        n = sum(is.finite(Value)),
        mean = mean(Value, na.rm = TRUE),
        sd = sd(Value, na.rm = TRUE),
        median = median(Value, na.rm = TRUE),
        iqr = IQR(Value, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        cohort = cohort_name,
        Mean_SD = ifelse(n > 0, sprintf("%.2f (%.2f)", mean, sd), NA_character_),
        Median_IQR = ifelse(n > 0, sprintf("%.2f (%.2f)", median, iqr), NA_character_)
      ) %>%
      select(cohort, Domain, Variable, Subtyp, n, Mean_SD, Median_IQR)
    
    descriptives_wide <- descriptives_long %>%
      select(cohort, Domain, Variable, Subtyp, Mean_SD) %>%
      pivot_wider(names_from = Subtyp, values_from = Mean_SD, names_prefix = "Subtyp_")
    
    write_out(descriptives_long, file.path(out_dir, "descriptives_metrics_long.csv"))
    write_out(descriptives_wide, file.path(out_dir, "descriptives_metrics_wide.csv"))
  }
  
  # -----------------------------
  # Descriptives (nominal): counts + percent per category per subtype
  # -----------------------------
  if (length(present_nominals) > 0) {
    nominal_desc_long <- merged %>%
      select(Subtyp, all_of(present_nominals)) %>%
      pivot_longer(-Subtyp, names_to = "Variable", values_to = "Value") %>%
      filter(!is.na(Subtyp)) %>%
      mutate(
        Domain = vapply(Variable, domain_of_nominal, character(1)),
        Value = droplevels(as.factor(Value))
      ) %>%
      filter(!is.na(Value)) %>%
      count(Domain, Variable, Subtyp, Value, name = "n") %>%
      group_by(Domain, Variable, Subtyp) %>%
      mutate(
        n_subtype_nonmiss = sum(n),
        Percent = ifelse(n_subtype_nonmiss > 0, round(100 * n / n_subtype_nonmiss, 1), NA_real_)
      ) %>%
      ungroup() %>%
      mutate(cohort = cohort_name) %>%
      select(cohort, Domain, Variable, Subtyp, Category = Value, n, n_subtype_nonmiss, Percent)
    
    write_out(nominal_desc_long, file.path(out_dir, "descriptives_nominal_long.csv"))
  }
  
  # -----------------------------
  # Assumptions (metrics only; reporting)
  # -----------------------------
  shapiro_all <- list()
  levene_all <- list()
  decision_all <- list()
  
  for (var in present_metrics) {
    dat <- merged %>% select(Subtyp, all_of(var)) %>% rename(Value = all_of(var))
    dat <- dat %>% filter(!is.na(Subtyp), is.finite(Value))
    if (nrow(dat) == 0) next
    
    sh <- dat %>%
      group_by(Subtyp) %>%
      summarise(n = n(), shapiro_p = safe_shapiro_p(Value), .groups = "drop") %>%
      tidyr::complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n = 0, shapiro_p = NA_real_)) %>%
      mutate(cohort = cohort_name, Domain = domain_of_metric(var), Variable = var)
    
    lev_p <- safe_levene_p(dat %>% mutate(Subtyp = factor(Subtyp, levels = 1:3)), y = "Value", g = "Subtyp")
    
    shapiro_all[[var]] <- sh
    levene_all[[var]] <- tibble(cohort = cohort_name, Domain = domain_of_metric(var), Variable = var, levene_p = lev_p)
    
    decision_all[[var]] <- tibble(
      cohort = cohort_name,
      Domain = domain_of_metric(var),
      Variable = var,
      normality_ok = all(sh$shapiro_p > 0.05, na.rm = TRUE),
      homogeneity_ok = ifelse(is.finite(lev_p), lev_p > 0.05, NA),
      Test_selected = "Kruskal + Dunn"
    )
  }
  
  write_out(bind_rows(shapiro_all),  file.path(out_dir, "assumptions_shapiro_by_subtype.csv"))
  write_out(bind_rows(levene_all),   file.path(out_dir, "assumptions_levene.csv"))
  write_out(bind_rows(decision_all), file.path(out_dir, "assumptions_decision_table.csv"))
  
  # -----------------------------
  # Metric globals + posthoc (raw p)
  # -----------------------------
  metric_global <- list()
  metric_posthoc <- list()
  
  for (var in present_metrics) {
    dat <- merged %>% select(Subtyp, all_of(var)) %>% rename(Value = all_of(var))
    dat <- dat %>% filter(!is.na(Subtyp), is.finite(Value)) %>% mutate(Subtyp = factor(Subtyp, levels = 1:3))
    
    n_by <- dat %>% group_by(Subtyp) %>% summarise(n = n(), .groups = "drop")
    if (sum(n_by$n >= 2, na.rm = TRUE) < 2) next
    
    kw <- kruskal.test(Value ~ Subtyp, data = dat)
    H <- as.numeric(kw$statistic)
    k <- as.numeric(kw$parameter) + 1
    n <- nrow(dat)
    
    metric_global[[var]] <- tibble(
      cohort = cohort_name,
      Domain = domain_of_metric(var),
      Variable = var,
      Test = "Kruskal-Wallis",
      statistic = H,
      df = as.numeric(kw$parameter),
      p_global = as.numeric(kw$p.value),
      effect_name = "epsilon2",
      effect_value = epsilon2_from_kw(H, n, k),
      N = n
    )
    
    dunn_raw <- rstatix::dunn_test(dat, Value ~ Subtyp, p.adjust.method = "none") %>%
      rename(z = statistic) %>%
      rowwise() %>%
      mutate(
        n1 = sum(dat$Subtyp == group1),
        n2 = sum(dat$Subtyp == group2),
        N = n1 + n2,
        r = z_to_r(z, n1, n2),
        cliffs_delta = cliffs_delta(
          dat %>% filter(Subtyp == group1) %>% pull(Value),
          dat %>% filter(Subtyp == group2) %>% pull(Value)
        ),
        delta_magnitude = cliffs_magnitude(cliffs_delta)
      ) %>%
      ungroup() %>%
      mutate(
        cohort = cohort_name,
        Domain = domain_of_metric(var),
        Variable = var,
        method = "Dunn"
      ) %>%
      select(cohort, Domain, Variable, method, group1, group2, n1, n2, N, z, p, r, cliffs_delta, delta_magnitude)
    
    metric_posthoc[[var]] <- dunn_raw
  }
  
  metric_global_df <- bind_rows(metric_global)
  metric_posthoc_df <- bind_rows(metric_posthoc)
  
  # -----------------------------
  # Nominal globals + posthoc (raw p)
  # -----------------------------
  nominal_global <- list()
  nominal_posthoc <- list()
  
  for (var in present_nominals) {
    dat <- merged %>% select(Subtyp, all_of(var)) %>% rename(Value = all_of(var))
    dat <- dat %>% filter(!is.na(Subtyp), !is.na(Value)) %>%
      mutate(Subtyp = factor(Subtyp, levels = 1:3), Value = droplevels(as.factor(Value)))
    if (nrow(dat) == 0) next
    
    tab <- table(dat$Subtyp, dat$Value)
    chi_try <- suppressWarnings(chisq.test(tab))
    use_fisher <- if (!is.null(chi_try$expected)) any(chi_try$expected < 5) else TRUE
    
    cv <- suppressWarnings(CramerV(tab))
    
    if (use_fisher) {
      ft <- safe_fisher(tab)
      pval <- as.numeric(ft$p.value)
      method_used <- if (isTRUE(ft$simulate.p.value)) "Fisher_simulated" else "Fisher"
      stat_val <- NA_real_
      df_val <- NA_real_
    } else {
      pval <- as.numeric(chi_try$p.value)
      method_used <- "Chi2"
      stat_val <- as.numeric(chi_try$statistic)
      df_val <- as.numeric(chi_try$parameter)
    }
    
    nominal_global[[var]] <- tibble(
      cohort = cohort_name,
      Domain = domain_of_nominal(var),  # Early
      Variable = var,
      Test = method_used,
      statistic = stat_val,
      df = df_val,
      p_global = pval,
      effect_name = "CramersV",
      effect_value = as.numeric(cv),
      N = sum(tab)
    )
    
    subs <- levels(dat$Subtyp)
    pairs <- combn(subs, 2, simplify = FALSE)
    
    ph_list <- lapply(pairs, function(pp) {
      subdat <- dat %>% filter(Subtyp %in% pp) %>% droplevels()
      tab2 <- table(subdat$Subtyp, subdat$Value)
      if (any(rowSums(tab2) == 0) || any(colSums(tab2) == 0)) return(NULL)
      
      ft2 <- safe_fisher(tab2)
      p2 <- as.numeric(ft2$p.value)
      
      eff <- if (nrow(tab2) == 2 && ncol(tab2) == 2) suppressWarnings(Phi(tab2)) else suppressWarnings(CramerV(tab2))
      eff_label <- if (nrow(tab2) == 2 && ncol(tab2) == 2) "Phi" else "CramersV"
      
      tibble(group1 = pp[1], group2 = pp[2], p = p2, effect_name = eff_label, effect_value = as.numeric(eff), N = sum(tab2))
    })
    
    ph <- bind_rows(ph_list)
    if (nrow(ph) > 0) {
      ph <- ph %>%
        mutate(
          cohort = cohort_name,
          Domain = domain_of_nominal(var),
          Variable = var,
          method = "Fisher pairwise"
        ) %>%
        select(cohort, Domain, Variable, method, group1, group2, N, p, effect_name, effect_value)
      nominal_posthoc[[var]] <- ph
    }
  }
  
  nominal_global_df <- bind_rows(nominal_global)
  nominal_posthoc_df <- bind_rows(nominal_posthoc)
  
  # -----------------------------
  # DOMAIN-specific BH across ALL globals in domain (metric + nominal together)
  # -----------------------------
  global_all <- bind_rows(
    metric_global_df %>% mutate(type = "metric"),
    nominal_global_df %>% mutate(type = "nominal")
  )
  
  if (nrow(global_all) > 0) {
    global_all <- global_all %>%
      group_by(Domain) %>%
      mutate(
        p_global_BH_within_domain = safe_p_adjust(p_global, method = "BH"),
        p_global_adj_signif = vapply(p_global_BH_within_domain, p_stars, character(1))
      ) %>%
      ungroup()
  }
  
  metric_global_df <- global_all %>% filter(type == "metric") %>% select(-type)
  nominal_global_df <- global_all %>% filter(type == "nominal") %>% select(-type)
  
  # -----------------------------
  # DOMAIN-specific BH across ALL posthoc tests in domain (Dunn + Fisher pairs together)
  # -----------------------------
  posthoc_all <- bind_rows(
    metric_posthoc_df %>% mutate(type = "metric"),
    nominal_posthoc_df %>% mutate(type = "nominal")
  )
  
  if (nrow(posthoc_all) > 0) {
    posthoc_all <- posthoc_all %>%
      group_by(Domain) %>%
      mutate(
        p_pair_BH_within_domain = safe_p_adjust(p, method = "BH"),
        p_pair_adj_signif = vapply(p_pair_BH_within_domain, p_stars, character(1))
      ) %>%
      ungroup()
  }
  
  metric_posthoc_df <- posthoc_all %>% filter(type == "metric") %>% select(-type)
  nominal_posthoc_df <- posthoc_all %>% filter(type == "nominal") %>% select(-type)
  
  # -----------------------------
  # Write outputs (domain-specific tables)
  # -----------------------------
  if (nrow(metric_global_df) > 0) {
    write_out(metric_global_df, file.path(out_dir, "metric_global_tests_kruskal.csv"))
    write_out(metric_global_df %>% filter(Domain == "Early"), file.path(out_dir, "metric_global_tests_Early.csv"))
    write_out(metric_global_df %>% filter(Domain == "Current"), file.path(out_dir, "metric_global_tests_Current.csv"))
    write_out(metric_global_df %>% filter(Domain == "Neuropsych"), file.path(out_dir, "metric_global_tests_Neuropsych.csv"))
  }
  
  if (nrow(nominal_global_df) > 0) {
    write_out(nominal_global_df, file.path(out_dir, "nominal_global_tests.csv"))
    write_out(nominal_global_df %>% filter(Domain == "Early"), file.path(out_dir, "nominal_global_tests_Early.csv"))
  }
  
  if (nrow(metric_posthoc_df) > 0) {
    write_out(metric_posthoc_df, file.path(out_dir, "metric_posthoc_dunn_domainBH.csv"))
    write_out(metric_posthoc_df %>% filter(Domain == "Early"), file.path(out_dir, "metric_posthoc_Early.csv"))
    write_out(metric_posthoc_df %>% filter(Domain == "Current"), file.path(out_dir, "metric_posthoc_Current.csv"))
    write_out(metric_posthoc_df %>% filter(Domain == "Neuropsych"), file.path(out_dir, "metric_posthoc_Neuropsych.csv"))
  }
  
  if (nrow(nominal_posthoc_df) > 0) {
    write_out(nominal_posthoc_df, file.path(out_dir, "nominal_posthoc_pairs_domainBH.csv"))
    write_out(nominal_posthoc_df %>% filter(Domain == "Early"), file.path(out_dir, "nominal_posthoc_Early.csv"))
  }
  
  # unified summary for paper (global only)
  summary_tbl <- bind_rows(
    metric_global_df %>%
      transmute(cohort, Domain, Variable, Test, N, statistic, df, p_global,
                p_global_adj = p_global_BH_within_domain, effect_name, effect_value),
    nominal_global_df %>%
      transmute(cohort, Domain, Variable, Test, N, statistic, df, p_global,
                p_global_adj = p_global_BH_within_domain, effect_name, effect_value)
  ) %>% arrange(Domain, p_global_adj)
  
  write_out(summary_tbl, file.path(out_dir, "summary_for_paper_global_tests.csv"))
  
  # Cohort summary row
  st_counts_wide <- subtype_counts %>%
    mutate(Subtyp = as.character(Subtyp)) %>%
    pivot_wider(names_from = Subtyp, values_from = n, names_prefix = "n_subtyp_") %>%
    mutate(across(everything(), ~ as.integer(.)))
  
  if (!"n_subtyp_1" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_1 <- 0L
  if (!"n_subtyp_2" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_2 <- 0L
  if (!"n_subtyp_3" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_3 <- 0L
  
  cohort_summary <- tibble(
    cohort = cohort_name,
    n_total = nrow(merged),
    n_metric_tested = length(unique(metric_global_df$Variable)),
    n_nominal_tested = length(unique(nominal_global_df$Variable)),
    n_domains = length(unique(c(metric_global_df$Domain, nominal_global_df$Domain)))
  ) %>% bind_cols(st_counts_wide %>% select(n_subtyp_1, n_subtyp_2, n_subtyp_3))
  
  write_out(cohort_summary, file.path(out_dir, "summary_cohort.csv"))
  
  message("[OK] Finished cohort: ", cohort_name)
  list(summary = cohort_summary)
}

# -----------------------------
# 4) Run all cohorts + end table
# -----------------------------
results <- purrr::imap(cohorts, ~run_one_cohort_early_current_neuro(cohort_name = .y, assign_path = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))

write_out(summary_all, file.path(output_root, "summary_all_cohorts.csv"))

message("[OK] ALL COHORTS DONE. Output root: ", output_root)
message("[OK] End table written: summary_all_cohorts.csv")
