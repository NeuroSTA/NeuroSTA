# ===============================
# SuStaIn Subtypen × ENIGMA-Subskalen (SANS/SAPS) – 3 Subtypen, alle Cohorts
# Cohorts: main, MDD, Angst, BD, SZ
# Tests: global (ANOVA oder Kruskal, datengetrieben) + pairwise (BH)
# Missing columns: wird protokolliert (NA)
# Output: CSV + PNG
# ===============================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(rstatix)
  library(car)
  library(broom)
  library(purrr)
})

# -----------------------------
# 0) Setup
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

cohorts <- list(
  main  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "MDD_subject_subtype_assignment_3subtypes.csv"),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "Angst_subject_subtype_assignment_3subtypes.csv"),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "BD_subject_subtype_assignment_3subtypes.csv"),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", "SZ_subject_subtype_assignment_3subtypes.csv")
)

out_root <- file.path(base_dir, "Output Scripte", "symptom_subscales_ENIGMA_3subtypes_all_cohorts")
if (!dir.exists(out_root)) dir.create(out_root, recursive = TRUE)

alpha_fdr <- 0.05

# -----------------------------
# 1) Variables (subscales)
# -----------------------------
subscale_vars <- c(
  "SANS_Affektverflachung_ENIGMA",
  "SANS_AlogieParalogie_ENIGMA",
  "SANS_AbulieApathie_ENIGMA",
  "SANS_Anhedonie_ENIGMA",
  "SANS_Aufmerksamkeit_ENIGMA",
  "SAPS_Halluzinationen_ENIGMA",
  "SAPS_Wahn_ENIGMA",
  "SAPS_PositiveFormaleDenkstoerung_ENIGMA",
  "SAPS_BizarresVerhalten_ENIGMA"
)

# -----------------------------
# 2) Helpers
# -----------------------------
stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

norm_id <- function(x) {
  tolower(gsub("_", "-", trimws(as.character(x))))
}

safe_shapiro_p <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  out <- tryCatch(shapiro.test(x)$p.value, error = function(e) NA_real_)
  as.numeric(out)
}

derive_subtyp_from_ml <- function(df) {
  if (!("ML_Subtype" %in% names(df))) stop("ML_Subtype column not found in assignment file.")
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype is all NA.")
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  if (is.infinite(ml_min)) stop("ML_Subtype min not finite.")
  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding (expected start 0 or 1). min=", ml_min)
  }
  df
}

# -----------------------------
# 3) Load data_all once
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
# 4) Run one cohort
# -----------------------------
run_one_cohort_subscales <- function(cohort_name, assign_path) {
  
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing(assign_path)
  
  out_dir <- file.path(out_root, cohort_name)
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
  
  merged <- inner_join(cluster_data, data_all, by = "Proband")
  if (nrow(merged) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)
  
  # subtype counts after join
  subtype_counts <- merged %>%
    count(Subtyp, name = "n") %>%
    tidyr::complete(Subtyp = factor(1:3, levels = 1:3), fill = list(n = 0)) %>%
    arrange(Subtyp)
  write_csv(subtype_counts, file.path(out_dir, "subtype_counts_after_join.csv"))
  
  # present/missing variables
  present_vars <- intersect(subscale_vars, names(merged))
  missing_vars <- setdiff(subscale_vars, names(merged))
  if (length(missing_vars) > 0L) stop("Required ENIGMA subscales missing: ", paste(missing_vars, collapse = ", "))
  
  write_csv(
    tibble(variable = subscale_vars,
           present = subscale_vars %in% present_vars),
    file.path(out_dir, "variables_present_flag.csv")
  )
  
  if (length(present_vars) == 0) {
    writeLines("No ENIGMA subscale variables found in data_all for this cohort.",
               con = file.path(out_dir, "NOTE_no_variables_found.txt"))
    return(list(
      summary = tibble(cohort = cohort_name, n_total = nrow(merged), n_vars_present = 0L),
      global = tibble()
    ))
  }
  
  # Long format
  long_df <- merged %>%
    select(Proband, Subtyp, all_of(present_vars)) %>%
    pivot_longer(cols = all_of(present_vars), names_to = "Variable", values_to = "Value") %>%
    mutate(Value = suppressWarnings(as.numeric(Value))) %>%
    filter(!is.na(Subtyp))
  
  # descriptives
  desc <- long_df %>%
    group_by(Variable, Subtyp) %>%
    summarise(
      n = sum(is.finite(Value)),
      mean = mean(Value, na.rm = TRUE),
      sd = sd(Value, na.rm = TRUE),
      median = median(Value, na.rm = TRUE),
      iqr = IQR(Value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::complete(Variable = present_vars,
                    Subtyp = factor(1:3, levels = 1:3),
                    fill = list(n = 0, mean = NA_real_, sd = NA_real_, median = NA_real_, iqr = NA_real_)) %>%
    mutate(cohort = cohort_name)
  
  write_csv(desc, file.path(out_dir, "descriptives_by_variable_and_subtype.csv"))
  
  # global tests per variable + pairwise posthoc per variable (BH)
  global_list <- list()
  posthoc_list <- list()
  qc_list <- list()
  
  for (v in present_vars) {
    
    dfv <- long_df %>% filter(Variable == v) %>% filter(is.finite(Value))
    if (nrow(dfv) < 3) next
    
    # QC normality per subtype (safe) + Levene
    sh <- dfv %>%
      group_by(Subtyp) %>%
      summarise(
        n = sum(is.finite(Value)),
        shapiro_p = safe_shapiro_p(Value),
        .groups = "drop"
      ) %>% tidyr::complete(Subtyp = factor(1:3, levels = 1:3))
    
    lev_p <- tryCatch({
      lv <- car::leveneTest(Value ~ Subtyp, data = dfv)
      as.numeric(lv$`Pr(>F)`[1])
    }, error = function(e) NA_real_)
    
    qc_list[[v]] <- tibble(
      cohort = cohort_name,
      Variable = v,
      levene_p = lev_p,
      min_shapiro_p = suppressWarnings(min(sh$shapiro_p, na.rm = TRUE))
    )
    
    all_normal <- all(sh$shapiro_p > 0.05, na.rm = TRUE)
    levene_ok <- is.finite(lev_p) && lev_p > 0.05
    n_by_group <- dfv %>% group_by(Subtyp) %>% summarise(n = sum(is.finite(Value)), .groups = "drop")
    enough_groups <- sum(n_by_group$n >= 2, na.rm = TRUE) >= 2
    
    if (all_normal && levene_ok && enough_groups) {
      
      # ANOVA global + eta2
      aov_mod <- aov(Value ~ Subtyp, data = dfv)
      an_tbl <- broom::tidy(anova(aov_mod))
      row_sub <- an_tbl %>% filter(term == "Subtyp")
      row_res <- an_tbl %>% filter(term == "Residuals")
      
      ss_tot <- sum(an_tbl$sumsq, na.rm = TRUE)
      ss_bt <- row_sub$sumsq[1]
      eta2 <- as.numeric(ss_bt / ss_tot)
      
      global_list[[v]] <- tibble(
        cohort = cohort_name,
        Variable = v,
        method = "ANOVA",
        statistic = as.numeric(row_sub$statistic[1]),
        df1 = as.numeric(row_sub$df[1]),
        df2 = as.numeric(row_res$df[1]),
        p.value = as.numeric(row_sub$p.value[1]),
        effect_name = "eta2",
        effect_value = eta2
      )
      
      # Pairwise t with BH (within variable)
      pw <- rstatix::pairwise_t_test(dfv, Value ~ Subtyp, p.adjust.method = "BH", pool.sd = TRUE) %>%
        rename(p_adj = p.adj) %>%
        mutate(cohort = cohort_name, Variable = v)
      
      d_pairs <- rstatix::cohens_d(dfv, Value ~ Subtyp, var.equal = TRUE) %>%
        select(group1, group2, effsize, magnitude) %>%
        rename(cohens_d = effsize, d_magnitude = magnitude)
      
      posthoc_list[[v]] <- pw %>% left_join(d_pairs, by = c("group1", "group2"))
      
    } else {
      
      # Kruskal global + epsilon2
      kw <- kruskal.test(Value ~ Subtyp, data = dfv)
      n <- nrow(dfv); k <- nlevels(dfv$Subtyp)
      eps2 <- (as.numeric(kw$statistic) - k + 1) / (n - k)
      
      global_list[[v]] <- tibble(
        cohort = cohort_name,
        Variable = v,
        method = "Kruskal-Wallis",
        statistic = as.numeric(kw$statistic),
        df1 = as.numeric(kw$parameter),
        df2 = NA_real_,
        p.value = as.numeric(kw$p.value),
        effect_name = "epsilon2",
        effect_value = as.numeric(eps2)
      )
      
      # Dunn with BH (within variable)
      dunn <- rstatix::dunn_test(dfv, Value ~ Subtyp, p.adjust.method = "BH") %>%
        rename(p_adj = p.adj) %>%
        rowwise() %>%
        mutate(
          n1 = sum(dfv$Subtyp == group1),
          n2 = sum(dfv$Subtyp == group2),
          N = n1 + n2,
          r = ifelse(N > 0, statistic / sqrt(N), NA_real_)
        ) %>%
        ungroup() %>%
        mutate(cohort = cohort_name, Variable = v)
      
      posthoc_list[[v]] <- dunn
    }
  }
  
  global_df <- bind_rows(global_list)
  posthoc_df <- bind_rows(posthoc_list)
  qc_df <- bind_rows(qc_list)
  
  # BH across variables (global p-values)
  global_df <- global_df %>%
    mutate(p.adj_BH_across_vars = p.adjust(p.value, method = "BH")) %>%
    arrange(p.adj_BH_across_vars)
  
  write_csv(qc_df, file.path(out_dir, "qc_normality_levene_by_variable.csv"))
  write_csv(global_df, file.path(out_dir, "global_tests_by_variable_with_BH_across_vars.csv"))
  write_csv(posthoc_df, file.path(out_dir, "posthoc_pairwise_by_variable_BH_within_variable.csv"))
  
  # -----------------------------
  # Plots: boxplots per variable
  # -----------------------------
  plot_list <- list()
  for (v in present_vars) {
    dfv <- long_df %>%
      filter(Variable == v) %>%
      mutate(Subtyp = factor(Subtyp, levels = 1:3)) %>%
      mutate(Value = suppressWarnings(as.numeric(Value))) %>%
      filter(is.finite(Value))
    
    if (nrow(dfv) < 3) next
    
    cluster_colors <- gray.colors(4, start = 0.3, end = 0.9)
    p <- ggplot(dfv, aes(x = Subtyp, y = Value, fill = Subtyp)) +
      geom_boxplot(outlier.shape = 16, outlier.size = 1.4, width = 0.6, color = "black", linewidth = 0.25) +
      scale_fill_manual(values = cluster_colors) +
      labs(title = paste0(v, " - ", cohort_name), x = "SuStaIn subtype (1-3)", y = v) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "none",
            plot.title = element_text(face = "bold", hjust = 0))
    
    plot_list[[v]] <- p
  }
  
  if (length(plot_list) > 0) {
    fig <- ggpubr::ggarrange(plotlist = plot_list, ncol = 2, nrow = ceiling(length(plot_list)/2))
    ggsave(file.path(out_dir, "boxplots_ENIGMA_subscales.png"), fig, width = 14, height = 10, dpi = 300)
  } else {
    writeLines("No plots created (no valid variables with data).", con = file.path(out_dir, "plot_note.txt"))
  }
  
  # cohort summary row
  st_counts_wide <- subtype_counts %>%
    mutate(Subtyp = as.character(Subtyp)) %>%
    pivot_wider(names_from = Subtyp, values_from = n, names_prefix = "n_subtyp_") %>%
    mutate(across(everything(), ~as.integer(.)))
  
  if (!"n_subtyp_1" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_1 <- 0L
  if (!"n_subtyp_2" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_2 <- 0L
  if (!"n_subtyp_3" %in% names(st_counts_wide)) st_counts_wide$n_subtyp_3 <- 0L
  
  summary_row <- tibble(
    cohort = cohort_name,
    n_total = nrow(merged),
    n_vars_present = length(present_vars),
    n_vars_missing = length(missing_vars),
    n_global_sig_BH = sum(global_df$p.adj_BH_across_vars < alpha_fdr, na.rm = TRUE)
  ) %>%
    bind_cols(st_counts_wide %>% select(n_subtyp_1, n_subtyp_2, n_subtyp_3))
  
  write_csv(summary_row, file.path(out_dir, "summary_cohort.csv"))
  
  list(summary = summary_row, global = global_df)
}

# -----------------------------
# 5) Run all cohorts + end table
# -----------------------------
stop_if_missing(data_all_path)
summary_list <- list()
global_list <- list()

for (nm in names(cohorts)) {
  res <- run_one_cohort_subscales(nm, cohorts[[nm]])
  summary_list[[nm]] <- res$summary
  global_list[[nm]] <- res$global
}

summary_all <- bind_rows(summary_list) %>%
  arrange(match(cohort, names(cohorts)))
write_csv(summary_all, file.path(out_root, "summary_all_cohorts.csv"))

global_all <- bind_rows(global_list) %>%
  arrange(match(cohort, names(cohorts)), p.adj_BH_across_vars)
write_csv(global_all, file.path(out_root, "global_tests_all_cohorts_long.csv"))

message("[OK] Done. Output root: ", out_root)
message("[OK] End tables: summary_all_cohorts.csv, global_tests_all_cohorts_long.csv")
