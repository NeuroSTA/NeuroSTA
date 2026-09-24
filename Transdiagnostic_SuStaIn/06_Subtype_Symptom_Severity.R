# ===============================
# SuStaIn Subtypen x Symptom-Scores - WHITELIST, 3 Subtypen, alle Cohorts
# Cohorts: main, MDD, Angst, BD, SZ
# Omnibus BH/FDR family per cohort: six main scales plus clinical status
# from script 08. Run script 08 first.
# Pairwise symptom-scale tests retain their existing within-scale BH correction.
# Output: write_csv2 (Excel-DE kompatibel: ; und Dezimalkomma)
#
# Regel (cohort-level):
# - Wenn in einer Cohort auch nur eine Skala nicht-parametrisch sein muss
#   (Normalitaet verletzt/NA oder Levene fail oder zu kleine Gruppen),
#   dann werden ALLE Skalen dieser Cohort nicht-parametrisch ausgewertet
#   (Kruskal-Wallis + Dunn, BH).
# HAM-D analyses use the prespecified HAMD_Sum17 variable. HAMA, SANS,
# SAPS, and YMRS scores are calculated from the item whitelists below.
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(car)
  library(broom)
  library(purrr)
  library(rstatix)
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

output_root <- file.path(base_dir, "Output Scripte", "symptom_scores_3subtypes_all_cohorts")
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

minus99_to_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[x %in% c(-99, -2)] <- NA_real_
  x
}

safe_shapiro_p <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  if (isTRUE(sd(x) == 0)) return(NA_real_)
  out <- tryCatch(shapiro.test(x)$p.value, error = function(e) NA_real_)
  as.numeric(out)
}

safe_levene_p <- function(df, y = "Sum", g = "Subtyp") {
  out <- tryCatch({
    lv <- car::leveneTest(df[[y]] ~ df[[g]])
    as.numeric(lv$`Pr(>F)`[1])
  }, error = function(e) NA_real_)
  as.numeric(out)
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

p_to_signif <- function(p) {
  case_when(
    is.na(p) ~ NA_character_,
    p <= 0.0001 ~ "****",
    p <= 0.001  ~ "***",
    p <= 0.01   ~ "**",
    p <= 0.05   ~ "*",
    TRUE ~ "ns"
  )
}

t_to_r <- function(tval, n1, n2) {
  N <- n1 + n2
  if (!is.finite(tval) || N <= 0) return(NA_real_)
  as.numeric(tval) / sqrt(N)
}

# Manual pairwise t-tests (BH)
pairwise_t_manual_BH <- function(df, y_col = "Sum", g_col = "Subtyp") {
  df <- df %>% filter(is.finite(.data[[y_col]]), !is.na(.data[[g_col]]))
  df[[g_col]] <- factor(df[[g_col]])
  
  gr <- levels(df[[g_col]])
  if (length(gr) < 2) return(tibble())
  
  pairs <- combn(gr, 2, simplify = FALSE)
  
  res <- purrr::map_dfr(pairs, function(pr) {
    g1 <- pr[1]; g2 <- pr[2]
    x1 <- df %>% filter(.data[[g_col]] == g1) %>% pull(.data[[y_col]])
    x2 <- df %>% filter(.data[[g_col]] == g2) %>% pull(.data[[y_col]])
    n1 <- length(x1); n2 <- length(x2)
    N  <- n1 + n2
    
    if (n1 < 2 || n2 < 2) {
      return(tibble(
        group1 = g1, group2 = g2, n1 = n1, n2 = n2, N = N,
        statistic = NA_real_, df = NA_real_, p = NA_real_, r = NA_real_
      ))
    }
    
    tt <- tryCatch(t.test(x1, x2, var.equal = TRUE), error = function(e) NULL)
    if (is.null(tt)) {
      return(tibble(
        group1 = g1, group2 = g2, n1 = n1, n2 = n2, N = N,
        statistic = NA_real_, df = NA_real_, p = NA_real_, r = NA_real_
      ))
    }
    
    tval  <- as.numeric(tt$statistic)
    dfree <- as.numeric(tt$parameter)
    pval  <- as.numeric(tt$p.value)
    
    tibble(
      group1 = g1, group2 = g2, n1 = n1, n2 = n2, N = N,
      statistic = tval, df = dfree, p = pval,
      r = t_to_r(tval, n1, n2)
    )
  })
  
  if (nrow(res) == 0) return(res)
  
  res %>%
    mutate(
      p_adj = p.adjust(p, method = "BH"),
      p.signif = p_to_signif(p),
      p.adj.signif = p_to_signif(p_adj),
      test_stat_name = "t"
    )
}

# Dunn (BH)
dunn_manual_BH <- function(df, y_col = "Sum", g_col = "Subtyp") {
  df <- df %>% filter(is.finite(.data[[y_col]]), !is.na(.data[[g_col]]))
  df[[g_col]] <- factor(df[[g_col]])
  
  n_by <- df %>% count(.data[[g_col]], name = "n")
  if (sum(n_by$n >= 2, na.rm = TRUE) < 2) return(tibble())
  
  dunn <- rstatix::dunn_test(df, as.formula(paste0(y_col, " ~ ", g_col)), p.adjust.method = "BH")
  
  dunn %>%
    rename(p_adj = p.adj) %>%
    rowwise() %>%
    mutate(
      n1 = sum(df[[g_col]] == group1),
      n2 = sum(df[[g_col]] == group2),
      N = n1 + n2,
      df = NA_real_,
      r = ifelse(N > 0, as.numeric(statistic) / sqrt(N), NA_real_),
      p.signif = p_to_signif(p),
      p.adj.signif = p_to_signif(p_adj),
      test_stat_name = "Z"
    ) %>%
    ungroup()
}

eta2_from_anova <- function(anova_tbl) {
  row_sub <- anova_tbl %>% filter(term == "Subtyp")
  ss_tot <- sum(anova_tbl$sumsq, na.rm = TRUE)
  ss_bt <- row_sub$sumsq[1]
  as.numeric(ss_bt / ss_tot)
}

epsilon2_from_kw <- function(H, n, k) {
  if (!is.finite(H) || n <= k) return(NA_real_)
  as.numeric((H - k + 1) / (n - k))
}

# -----------------------------
# Item whitelists for HAMA/SANS/SAPS/YMRS
# -----------------------------
questionnaire_vars <- c(
  "HAMD1","HAMD2","HAMDa1","HAMD3","HAMD4","HAMDa2","HAMDa3","HAMDa5","HAMD6","HAMD7",
  "HAMD8","HAMDa6","HAMD9_SF","HAMDa7","HAMD10","HAMD11","HAMD12","HAMD13","HAMD14",
  "HAMD15","HAMD16","HAMD17","HAMD18","HAMDa8","HAMD19","HAMD20","HAMD21",
  "SANS1","SANS2","SANS3","SANS4","SANS5","SANS6","SANS8","SANS9","SANS10","SANS11",
  "SANS13","SANS14","SANS15","SANS17","SANS18","SANS19","SANS20","SANS22","SANS23",
  "SAPS1","SAPS2","SAPS3","SAPS4","SAPS5","SAPS6","SAPS8","SAPS9","SAPS10","SAPS11",
  "SAPS12","SAPS13","SAPS14","SAPS15","SAPS16","SAPS17","SAPS18","SAPS19","SAPS21",
  "SAPS22","SAPS23","SAPS24","SAPS26","SAPS27","SAPS28","SAPS29","SAPS30","SAPS31",
  "SAPS32","SAPS33","SAPS35",
  "HAMA1","HAMA2","HAMA3","HAMA4","HAMA5","HAMA6","HAMA7","HAMA8","HAMA9","HAMA10",
  "HAMA11","HAMA12","HAMA13","HAMA14",
  "YMRS1","YMRS2","YMRS3","YMRS4","YMRS5","YMRS6","YMRS7","YMRS8","YMRS9","YMRS10","YMRS11"
)

# HAMD_Sum17 is read directly in make_hamd_df().
items_by_scale <- list(
  HAMA = questionnaire_vars[grepl("^HAMA", questionnaire_vars)],
  SANS = questionnaire_vars[grepl("^SANS", questionnaire_vars)],
  SAPS = questionnaire_vars[grepl("^SAPS", questionnaire_vars)],
  YMRS = questionnaire_vars[grepl("^YMRS", questionnaire_vars)]
)

make_sum_from_cols <- function(df, cols, scale_name) {
  missing_cols <- setdiff(cols, names(df))
  if (length(missing_cols) > 0L) {
    stop("Missing prespecified ", scale_name, " items: ", paste(missing_cols, collapse = ", "))
  }
  
  out <- df %>%
    select(Proband, Subtyp, all_of(cols)) %>%
    mutate(across(all_of(cols), minus99_to_na)) %>%
    mutate(
      n_items_observed = rowSums(!is.na(across(all_of(cols)))),
      Sum = ifelse(
        n_items_observed >= 1L,
        rowSums(across(all_of(cols)), na.rm = TRUE),
        NA_real_
      )
    ) %>%
    select(Proband, Subtyp, Sum)
  
  attr(out, "n_items") <- length(cols)
  out
}

# Read the prespecified HAM-D 17-item total.
make_hamd_df <- function(df) {
  if (!"HAMD_Sum17" %in% names(df)) {
    stop("Required HAMD_Sum17 column is missing; HAM-D analysis cannot run.")
  }
  out <- df %>%
    select(Proband, Subtyp, HAMD_Sum17) %>%
    mutate(Sum = minus99_to_na(HAMD_Sum17)) %>%
    filter(is.finite(Sum)) %>%
    select(Proband, Subtyp, Sum)
  attr(out, "n_items") <- 17
  out
}

plot_boxplot <- function(data, title, ylab_text = "Sum score") {
  data <- data %>% mutate(Subtyp = factor(Subtyp, levels = 1:3))
  cluster_colors <- gray.colors(4, start = 0.3, end = 0.9)
  ggplot(data, aes(x = Subtyp, y = Sum, fill = Subtyp)) +
    geom_boxplot(outlier.shape = 16, outlier.size = 1.6, width = 0.6, color = "black", linewidth = 0.25) +
    scale_fill_manual(values = cluster_colors) +
    labs(x = "SuStaIn subtypes", y = ylab_text, title = title) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", hjust = 0.0))
}

# -----------------------------
# Load data_all once
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
# Run one cohort
# -----------------------------
run_one_cohort_symptoms <- function(cohort_name, assign_path) {
  
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing(assign_path)
  out_dir <- file.path(output_root, cohort_name)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  # assignments
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
  
  merged_data <- inner_join(cluster_data, data_all, by = "Proband")
  if (nrow(merged_data) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)
  
  # subtype counts after join
  subtype_counts <- merged_data %>%
    count(Subtyp, name = "n") %>%
    complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n = 0)) %>%
    arrange(Subtyp)
  write_out(subtype_counts, file.path(out_dir, "subtype_counts_after_join.csv"))
  
  # build score dfs
  hamd_df <- make_hamd_df(merged_data)
  hama_df <- make_sum_from_cols(merged_data, items_by_scale$HAMA, "HAMA")
  sans_df <- make_sum_from_cols(merged_data, items_by_scale$SANS, "SANS")
  saps_df <- make_sum_from_cols(merged_data, items_by_scale$SAPS, "SAPS")
  ymrs_df <- make_sum_from_cols(merged_data, items_by_scale$YMRS, "YMRS")
  
  gaf_df <- NULL
  if ("GAFscore" %in% names(merged_data)) {
    gaf_df <- merged_data %>%
      select(Proband, Subtyp, GAFscore) %>%
      mutate(GAFscore = suppressWarnings(as.numeric(GAFscore))) %>%
      rename(Sum = GAFscore) %>%
      filter(is.finite(Sum), Sum != -99) %>%
      select(Proband, Subtyp, Sum)
  }
  
  score_list <- list(HAMD = hamd_df, HAMA = hama_df, SANS = sans_df, SAPS = saps_df, YMRS = ymrs_df, GAF = gaf_df)
  score_list <- score_list[!vapply(score_list, is.null, logical(1))]
  
  used_scales <- tibble(
    Scale = names(score_list),
    n_items = sapply(names(score_list), function(nm) if (nm == "GAF") NA_integer_ else attr(score_list[[nm]], "n_items"))
  )
  write_out(used_scales, file.path(out_dir, "used_scales_and_items.csv"))
  
  # -----------------------------
  # Missingness per scale
  # -----------------------------
  n_total_join <- nrow(merged_data)
  
  missing_scale_overall <- purrr::map_dfr(names(score_list), function(sc) {
    df <- score_list[[sc]] %>% filter(is.finite(Sum))
    n_valid <- nrow(df)
    tibble(
      cohort = cohort_name,
      Scale = sc,
      n_total_join = n_total_join,
      n_valid = n_valid,
      n_missing = n_total_join - n_valid,
      missing_pct = 100 * (n_total_join - n_valid) / n_total_join
    )
  })
  
  missing_scale_by_subtype <- purrr::map_dfr(names(score_list), function(sc) {
    df <- score_list[[sc]] %>% filter(is.finite(Sum)) %>% mutate(Subtyp = factor(Subtyp, levels = 1:3))
    valid_counts <- df %>%
      count(Subtyp, name = "n_valid") %>%
      complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n_valid = 0)) %>%
      arrange(Subtyp)
    
    total_counts <- merged_data %>%
      mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
      count(Subtyp, name = "n_total") %>%
      complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n_total = 0)) %>%
      arrange(Subtyp)
    
    left_join(total_counts, valid_counts, by = "Subtyp") %>%
      mutate(
        cohort = cohort_name,
        Scale = sc,
        n_missing = n_total - n_valid,
        missing_pct = ifelse(n_total > 0, 100 * (n_total - n_valid) / n_total, NA_real_)
      ) %>%
      select(cohort, Scale, Subtyp, n_total, n_valid, n_missing, missing_pct)
  })
  
  write_out(missing_scale_overall, file.path(out_dir, "missingness_by_scale_overall.csv"))
  write_out(missing_scale_by_subtype, file.path(out_dir, "missingness_by_scale_by_subtype.csv"))
  
  # -----------------------------
  # A) First pass: diagnostics for ALL scales (to decide cohort-level method)
  # -----------------------------
  diagnostics_list <- list()
  
  for (sc in names(score_list)) {
    df <- score_list[[sc]] %>%
      mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
      filter(is.finite(Sum))
    
    if (nrow(df) < 3) {
      diagnostics_list[[sc]] <- tibble(
        cohort = cohort_name, Scale = sc,
        enough_groups = FALSE,
        levene_ok = FALSE,
        all_groups_normal = FALSE
      )
      next
    }
    
    shapiro_res <- df %>%
      group_by(Subtyp) %>%
      summarise(n = n(), shapiro_p = safe_shapiro_p(Sum), .groups = "drop") %>%
      complete(Subtyp = factor(1:3, levels = 1:3))
    
    # Missing diagnostics count as insufficient evidence for normality.
    group_normal_flag <- shapiro_res %>%
      mutate(is_normal = ifelse(is.finite(shapiro_p) & shapiro_p > 0.05, TRUE, FALSE)) %>%
      pull(is_normal)
    all_normal <- all(group_normal_flag)
    
    levene_p <- safe_levene_p(df, y = "Sum", g = "Subtyp")
    levene_ok <- is.finite(levene_p) && levene_p > 0.05
    
    n_by_group <- df %>% count(Subtyp, name = "n")
    enough_groups <- sum(n_by_group$n >= 2, na.rm = TRUE) >= 2
    
    diagnostics_list[[sc]] <- tibble(
      cohort = cohort_name, Scale = sc,
      enough_groups = enough_groups,
      levene_ok = levene_ok,
      all_groups_normal = all_normal
    )
  }
  
  diagnostics_df <- bind_rows(diagnostics_list)
  write_out(diagnostics_df, file.path(out_dir, "diagnostics_scale_level_for_cohort_decision.csv"))
  
  # Cohort-level rule:
  # If ANY scale fails parametric prerequisites -> force nonparam for ALL scales in this cohort
  any_scale_fails_param <- any(
    (!diagnostics_df$enough_groups) |
      (!diagnostics_df$levene_ok) |
      (!diagnostics_df$all_groups_normal),
    na.rm = TRUE
  )
  
  force_nonparam_cohort <- isTRUE(any_scale_fails_param)
  
  write_out(
    tibble(
      cohort = cohort_name,
      force_nonparam_cohort = force_nonparam_cohort,
      reason_any_scale_failed = any_scale_fails_param
    ),
    file.path(out_dir, "cohort_level_method_decision.csv")
  )
  
  # -----------------------------
  # B) Second pass: run analyses with cohort-level enforced method
  # -----------------------------
  shapiro_all <- list()
  levene_all <- list()
  global_all <- list()
  posthoc_all <- list()
  means_all <- list()
  decisions_all <- list()
  
  for (sc in names(score_list)) {
    
    df <- score_list[[sc]] %>%
      mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
      filter(is.finite(Sum))
    
    if (nrow(df) < 3) next
    
    # Save shapiro + levene anyway (for reporting)
    shapiro_res <- df %>%
      group_by(Subtyp) %>%
      summarise(
        n = n(),
        shapiro_p = safe_shapiro_p(Sum),
        .groups = "drop"
      ) %>%
      complete(Subtyp = factor(1:3, levels = 1:3)) %>%
      mutate(cohort = cohort_name, Scale = sc)
    shapiro_all[[sc]] <- shapiro_res
    
    levene_p <- safe_levene_p(df, y = "Sum", g = "Subtyp")
    levene_all[[sc]] <- tibble(cohort = cohort_name, Scale = sc, levene_p = levene_p)
    
    n_by_group <- df %>% count(Subtyp, name = "n")
    enough_groups <- sum(n_by_group$n >= 2, na.rm = TRUE) >= 2
    
    decisions_all[[sc]] <- tibble(
      cohort = cohort_name,
      Scale = sc,
      forced_nonparam_by_cohort = force_nonparam_cohort,
      enough_groups = enough_groups
    )
    
    if (!enough_groups) next
    
    if (!force_nonparam_cohort) {
      # Parametric for ALL scales in cohort
      
      aov_mod <- aov(Sum ~ Subtyp, data = df)
      anova_tbl <- broom::tidy(anova(aov_mod))
      row_sub <- anova_tbl %>% filter(term == "Subtyp")
      row_res <- anova_tbl %>% filter(term == "Residuals")
      
      eta2 <- eta2_from_anova(anova_tbl)
      
      global_all[[sc]] <- tibble(
        cohort = cohort_name,
        Scale = sc,
        method_global = "ANOVA",
        statistic_global = as.numeric(row_sub$statistic[1]),
        df1 = as.numeric(row_sub$df[1]),
        df2 = as.numeric(row_res$df[1]),
        p.value = as.numeric(row_sub$p.value[1]),
        effect_name = "eta2",
        effect_value = eta2
      )
      
      pw <- pairwise_t_manual_BH(df, y_col = "Sum", g_col = "Subtyp") %>%
        mutate(
          cohort = cohort_name,
          Scale = sc,
          method_posthoc = "pairwise_t_BH"
        )
      
      d_pairs <- tryCatch({
        rstatix::cohens_d(df, Sum ~ Subtyp, var.equal = TRUE) %>%
          select(group1, group2, effsize, magnitude) %>%
          rename(cohens_d = effsize, d_magnitude = magnitude)
      }, error = function(e) tibble(group1 = character(), group2 = character(), cohens_d = numeric(), d_magnitude = character()))
      
      posthoc_all[[sc]] <- pw %>%
        left_join(d_pairs, by = c("group1", "group2"))
      
    } else {
      # Nonparametric for ALL scales in cohort
      
      kw <- kruskal.test(Sum ~ Subtyp, data = df)
      n <- nrow(df)
      k <- nlevels(df$Subtyp)
      epsilon2 <- epsilon2_from_kw(as.numeric(kw$statistic), n, k)
      
      global_all[[sc]] <- tibble(
        cohort = cohort_name,
        Scale = sc,
        method_global = "Kruskal-Wallis",
        statistic_global = as.numeric(kw$statistic),
        df1 = as.numeric(kw$parameter),
        df2 = NA_real_,
        p.value = as.numeric(kw$p.value),
        effect_name = "epsilon2",
        effect_value = epsilon2
      )
      
      dunn <- dunn_manual_BH(df, y_col = "Sum", g_col = "Subtyp") %>%
        mutate(
          cohort = cohort_name,
          Scale = sc,
          method_posthoc = "dunn_BH",
          cohens_d = NA_real_,
          d_magnitude = NA_character_
        )
      
      posthoc_all[[sc]] <- dunn
    }
    
    means_all[[sc]] <- df %>%
      group_by(Subtyp) %>%
      summarise(
        n = n(),
        mean = mean(Sum, na.rm = TRUE),
        sd = sd(Sum, na.rm = TRUE),
        median = median(Sum, na.rm = TRUE),
        iqr = IQR(Sum, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      complete(Subtyp = factor(1:3, levels = 1:3),
               fill = list(n = 0, mean = NA_real_, sd = NA_real_, median = NA_real_, iqr = NA_real_)) %>%
      mutate(cohort = cohort_name, Scale = sc)
  }
  
  shapiro_df   <- bind_rows(shapiro_all)
  levene_df    <- bind_rows(levene_all)
  global_df    <- bind_rows(global_all)
  posthoc_df   <- bind_rows(posthoc_all)
  means_df     <- bind_rows(means_all)
  decisions_df <- bind_rows(decisions_all)
  
  # Clinical status is derived from the symptom ratings and belongs to the
  # same omnibus BH family. Script 08 must use the same assignment and
  # clinical-status input file.
  status_cohort <- if (cohort_name == "main") "TDM" else cohort_name
  status_path <- file.path(base_dir, "Output Scripte",
                           "status_acute_vs_remitted_all_cohorts",
                           status_cohort, "global_status_vs_subtype.csv")
  status_data_path <- getOption("SUSTAIN_CLINICAL_STATUS_FILE", Sys.getenv("SUSTAIN_CLINICAL_STATUS_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt_withGlobalRatings.csv")))
  stop_if_missing(status_path)
  stop_if_missing(status_data_path)
  status_test <- readr::read_csv2(status_path, show_col_types = FALSE)
  required_status_cols <- c("cohort", "method", "statistic", "df", "p.value",
                            "cramers_v", "assignment_md5", "status_data_md5")
  if (nrow(status_test) != 1L ||
      length(setdiff(required_status_cols, names(status_test))) > 0L ||
      !identical(status_test$cohort[[1]], status_cohort) ||
      !identical(status_test$assignment_md5[[1]], unname(tools::md5sum(assign_path))) ||
      !identical(status_test$status_data_md5[[1]], unname(tools::md5sum(status_data_path))) ||
      !is.finite(status_test$p.value[[1]]) ||
      status_test$p.value[[1]] < 0 || status_test$p.value[[1]] > 1) {
    stop("Clinical-status result is missing, incomplete, or does not match the current inputs: ",
         status_path, ". Run script 08 with the current data first.")
  }
  if (nrow(global_df) != 6L || anyNA(global_df$p.value) ||
      any(!is.finite(global_df$p.value)) ||
      !setequal(global_df$Scale, c("HAMD", "HAMA", "SANS", "SAPS", "YMRS", "GAF"))) {
    stop("Expected exactly six finite main-scale omnibus p values for cohort ", cohort_name)
  }
  family_q <- p.adjust(c(global_df$p.value, status_test$p.value[[1]]), method = "BH")
  global_df$p.adj_BH_scales <- family_q[seq_len(nrow(global_df))]
  status_family_row <- tibble(
    cohort = cohort_name, Scale = "Clinical status",
    method_global = status_test$method[[1]],
    statistic_global = status_test$statistic[[1]],
    df1 = status_test$df[[1]], df2 = NA_real_,
    p.value = status_test$p.value[[1]],
    effect_name = "CramersV", effect_value = status_test$cramers_v[[1]],
    p.adj_BH_scales = family_q[[length(family_q)]]
  )
  global_family_df <- bind_rows(global_df, status_family_row) %>%
    arrange(p.adj_BH_scales, p.value) %>%
    mutate(BH_family = "Six main scales plus clinical status",
           BH_family_n = 7L)
  
  write_out(shapiro_df,   file.path(out_dir, "shapiro_all.csv"))
  write_out(levene_df,    file.path(out_dir, "levene_all.csv"))
  write_out(global_family_df, file.path(out_dir, "global_tests_all_with_BH_across_scales.csv"))
  write_out(posthoc_df,   file.path(out_dir, "posthoc_all_BH.csv"))
  write_out(means_df,     file.path(out_dir, "descriptives_means_all.csv"))
  write_out(decisions_df, file.path(out_dir, "decisions_per_scale_flags.csv"))
  
  # Plots
  plot_list <- list()
  for (sc in names(score_list)) {
    dfp <- score_list[[sc]]
    if (is.null(dfp) || nrow(dfp) == 0) next
    dfp <- dfp %>% mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>% filter(is.finite(Sum))
    if (nrow(dfp) < 3) next
    ylab <- if (sc == "GAF") "Value" else "Sum score"
    plot_list[[sc]] <- plot_boxplot(dfp, title = paste0(sc, " - ", cohort_name), ylab_text = ylab)
  }
  
  if (length(plot_list) > 0) {
    fig <- ggpubr::ggarrange(plotlist = plot_list, ncol = 2,
                             nrow = ceiling(length(plot_list) / 2))
    ggsave(file.path(out_dir, "boxplots_symptom_scores.png"), fig, width = 12, height = 10, dpi = 300)
  } else {
    writeLines("No plots created (no valid scales).", con = file.path(out_dir, "plot_note.txt"))
  }
  
  st_counts_wide <- subtype_counts %>%
    mutate(Subtyp = as.character(Subtyp)) %>%
    pivot_wider(names_from = Subtyp, values_from = n, names_prefix = "n_subtyp_") %>%
    mutate(across(everything(), ~ as.integer(.)))
  
  if (!"n_subtyp_1" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_1 <- 0L
  if (!"n_subtyp_2" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_2 <- 0L
  if (!"n_subtyp_3" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_3 <- 0L
  
  summary_row <- tibble(
    cohort = cohort_name,
    n_total_join = nrow(merged_data),
    n_scales_tested = length(unique(global_df$Scale)),
    n_scales_sig_BH = sum(global_df$p.adj_BH_scales < alpha_fdr, na.rm = TRUE),
    n_omnibus_family_tests = nrow(global_family_df),
    n_omnibus_family_sig_BH = sum(global_family_df$p.adj_BH_scales < alpha_fdr),
    force_nonparam_cohort = force_nonparam_cohort
  ) %>%
    bind_cols(st_counts_wide %>% select(n_subtyp_1, n_subtyp_2, n_subtyp_3))
  
  write_out(summary_row, file.path(out_dir, "summary_cohort.csv"))
  
  message("[OK] Finished cohort: ", cohort_name)
  
  list(
    summary = summary_row,
    missing_overall = missing_scale_overall
  )
}

# -----------------------------
# Run all cohorts + end tables
# -----------------------------
results <- purrr::imap(cohorts, ~run_one_cohort_symptoms(cohort_name = .y, assign_path = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))
write_out(summary_all, file.path(output_root, "summary_all_cohorts.csv"))

missing_all <- purrr::map_dfr(results, "missing_overall") %>%
  arrange(match(cohort, names(cohorts)), Scale)
write_out(missing_all, file.path(output_root, "missingness_overall_all_cohorts.csv"))

message("[OK] ALL COHORTS DONE. Output root: ", output_root)
message("[OK] End tables written: summary_all_cohorts.csv, missingness_overall_all_cohorts.csv")
