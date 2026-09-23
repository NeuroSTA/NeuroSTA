# Deskriptive Statistik + Gruppenvergleiche (PATIENTEN: MDD, BD, SZ, Angst)
# ADD-ONs:
#   1) AgeOfOnset + DurationOfIllness = Alter - AgeOfOnset
#   2) Clinical status: acute vs remittiert (domain-trigger rule)
#
# Multiple testing: Benjamini-Hochberg FDR (alpha=0.05)
# Post-hoc only if global (FDR) significant:
#   - Continuous: Dunn (BH) after Kruskal
#   - Categorical: Pairwise Fisher (BH) after global Chi^2 with Monte Carlo p
#
# Groups in column "Group":
#   1 = HC (excluded)
#   2 = MDD
#   3 = BD
#   4 + 5 = SZ
#   8 = Angst
#
# -----------------------------------------------------------------------------------

base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

required_packages <- c(
  "dplyr","readr","tidyr","tibble","purrr","stringr",
  "rstatix","car","moments","flextable"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Missing required R packages: ", paste(missing_packages, collapse = ", "),
    ". Install the versions listed in r-requirements.txt before running the analysis."
  )
}
invisible(lapply(required_packages, library, character.only = TRUE))

alpha_fdr <- 0.05
output_dir <- file.path(base_dir, "Output Scripte", "descriptives_status")
if(!dir.exists(output_dir)) dir.create(output_dir, recursive=TRUE)

# -----------------------------
# 1) Load data
# -----------------------------
data_all_path <- getOption("SUSTAIN_CLINICAL_STATUS_FILE", Sys.getenv("SUSTAIN_CLINICAL_STATUS_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt_withGlobalRatings.csv")))
if(!file.exists(data_all_path)){
  stop(
    "Clinical-status input file not found: ", data_all_path
  )
}
data_all <- readr::read_csv(data_all_path, show_col_types = FALSE)

trigger_vars_items <- c("SANS7","SANS12","SANS16","SANS21","SAPS7","SAPS20","SAPS25","SAPS34")

# Required demographic, clinical, and status variables.
required_cols <- c(
  "Proband","Group","Alter","Geschlecht","Bildungsjahre","TIV","Komorbid","AgeOfOnset",
  "HAMA_Sum","HAMD_Sum17","YMRS_Sum",
  trigger_vars_items
)
missing_cols <- setdiff(required_cols, names(data_all))
if(length(missing_cols) > 0){
  stop(paste("Fehlende Pflichtspalten:", paste(missing_cols, collapse=", ")))
}

# -----------------------------
# 2) Variables
# -----------------------------
questionnaire_vars <- c(
  "HAMD1","HAMD2","HAMDa1","HAMD3","HAMD4","HAMDa2","HAMDa3","HAMDa5","HAMD6","HAMD7","HAMD8","HAMDa6","HAMD9_SF",
  "HAMDa7","HAMD10","HAMD11","HAMD12","HAMD13","HAMD14","HAMD15","HAMD16","HAMD17","HAMD18","HAMDa8","HAMD19","HAMD20","HAMD21",
  "SANS1","SANS2","SANS3","SANS4","SANS5","SANS6","SANS8","SANS9","SANS10","SANS11","SANS13","SANS14","SANS15","SANS17",
  "SANS18","SANS19","SANS20","SANS22","SANS23",
  "SAPS1","SAPS2","SAPS3","SAPS4","SAPS5","SAPS6","SAPS8","SAPS9","SAPS10",
  "SAPS11","SAPS12","SAPS13","SAPS14","SAPS15","SAPS16","SAPS17","SAPS18","SAPS19","SAPS21","SAPS22","SAPS23","SAPS24",
  "SAPS26","SAPS27","SAPS28","SAPS29","SAPS30","SAPS31","SAPS32","SAPS33","SAPS35",
  "HAMA1","HAMA2","HAMA3","HAMA4","HAMA5","HAMA6","HAMA7","HAMA8","HAMA9","HAMA10","HAMA11","HAMA12","HAMA13","HAMA14",
  "YMRS1","YMRS2","YMRS3","YMRS4","YMRS5","YMRS6","YMRS7","YMRS8","YMRS9","YMRS10","YMRS11"
)

missing_q <- setdiff(questionnaire_vars, names(data_all))
if(length(missing_q) > 0){
  stop(paste("Fehlende Fragebogen-Variablen:", paste(missing_q, collapse=", ")))
}

groups_patients <- c("MDD","BD","SZ","Angst")

# -----------------------------
# 3) Clean + group mapping
# -----------------------------
data_all <- data_all %>%
  mutate(
    Group = suppressWarnings(as.integer(as.numeric(trimws(as.character(Group))))),
    Geschlecht = suppressWarnings(as.numeric(trimws(as.character(Geschlecht)))),
    Komorbid = suppressWarnings(as.numeric(trimws(as.character(Komorbid)))),
    Alter = suppressWarnings(as.numeric(trimws(as.character(Alter)))),
    Bildungsjahre = suppressWarnings(as.numeric(trimws(as.character(Bildungsjahre)))),
    TIV = suppressWarnings(as.numeric(trimws(as.character(TIV)))),
    AgeOfOnset = suppressWarnings(as.numeric(trimws(as.character(AgeOfOnset)))),
    HAMA_Sum = suppressWarnings(as.numeric(trimws(as.character(HAMA_Sum)))),
    HAMD_Sum17 = suppressWarnings(as.numeric(trimws(as.character(HAMD_Sum17)))),
    YMRS_Sum = suppressWarnings(as.numeric(trimws(as.character(YMRS_Sum))))
  ) %>%
  mutate(
    across(
      c("Alter","Geschlecht","Bildungsjahre","TIV","Komorbid","AgeOfOnset",
        "HAMA_Sum","HAMD_Sum17","YMRS_Sum"),
      ~ ifelse(. %in% c(-99, -2), NA, .)
    ),
    across(all_of(questionnaire_vars), ~ ifelse(. %in% c(-99, -2), NA, .)),
    across(all_of(trigger_vars_items), ~ ifelse(. %in% c(-99, -2), NA, .))
  )

cat("\n--- Debug: Verteilung Group (inkl. NA) ---\n")
print(table(data_all$Group, useNA = "ifany"))

data_patients <- data_all %>%
  filter(Group %in% c(2L,3L,4L,5L,8L)) %>%
  mutate(
    gruppe_diag = dplyr::case_when(
      Group == 2L ~ "MDD",
      Group == 3L ~ "BD",
      Group %in% c(4L,5L) ~ "SZ",
      Group == 8L ~ "Angst",
      TRUE ~ NA_character_
    ),
    gruppe_diag = factor(gruppe_diag, levels = groups_patients),
    Geschlecht_label = factor(Geschlecht, levels=c(1,2), labels=c("Male","Female"))
  ) %>%
  filter(!is.na(gruppe_diag))

cat("\n--- Debug: Gruppengroessen (nur Patienten) ---\n")
print(table(data_patients$gruppe_diag, useNA = "ifany"))

# -----------------------------
# 3b) Duration of illness (years)
# -----------------------------
data_patients <- data_patients %>%
  mutate(
    DurationOfIllness = Alter - AgeOfOnset,
    DurationOfIllness = ifelse(is.na(DurationOfIllness), NA_real_, DurationOfIllness),
    DurationOfIllness = ifelse(DurationOfIllness < 0, NA_real_, DurationOfIllness)
  )

cat("\n--- Debug: AgeOfOnset / DurationOfIllness Quick QC ---\n")
print(data_patients %>%
        summarise(
          N_AgeOfOnset_nonmiss = sum(!is.na(AgeOfOnset)),
          N_Duration_nonmiss   = sum(!is.na(DurationOfIllness)),
          N_Duration_negative  = sum((Alter - AgeOfOnset) < 0, na.rm=TRUE)
        ))

# -----------------------------
# 4) Sum scores
#
# - alle Items einer Skala fehlen -> Summenscore bleibt NA
# - mindestens ein Item vorhanden -> Summe der vorhandenen Items
# -----------------------------
row_sum_observed <- function(data, vars){
  vars <- intersect(vars, names(data))
  if(length(vars) == 0) return(rep(NA_real_, nrow(data)))

  item_data <- dplyr::select(data, dplyr::all_of(vars))
  n_observed <- rowSums(!is.na(item_data))
  score <- rowSums(item_data, na.rm = TRUE)
  score[n_observed == 0] <- NA_real_
  as.numeric(score)
}

sum_for_prefix <- function(prefix, data){
  vars <- questionnaire_vars[grepl(paste0("^", prefix), questionnaire_vars)]
  row_sum_observed(data, vars)
}

data_patients <- data_patients %>%
  mutate(
    Sum_HAMD = HAMD_Sum17,
    Sum_SANS = sum_for_prefix("SANS", .),
    Sum_SAPS = sum_for_prefix("SAPS", .),
    Sum_HAMA = sum_for_prefix("HAMA", .),
    Sum_YMRS = sum_for_prefix("YMRS", .)
  ) %>%
  mutate(
    Sum_Questionnaire = row_sum_observed(
      ., c("Sum_HAMD", "Sum_SANS", "Sum_SAPS", "Sum_HAMA", "Sum_YMRS")
    )
  )

# -----------------------------
# 4b) Acute vs remittiert status
# Rule: acute if ANY trigger:
#   - SANS7/SANS12/SANS16/SANS21 > 2
#   - SAPS7/SAPS20/SAPS25/SAPS34 > 2
#   - HAMA_Sum > 19
#   - HAMD_Sum17 > 6
#   - YMRS_Sum >= 4   (Berk et al., 2008: remission = YMRS < 4; complement = >= 4)
# Remittiert: alle 11 Trigger beobachtet und kein Trigger positiv.
# Wenn kein Trigger positiv ist, aber mindestens einer fehlt: Status NA.
# -----------------------------

minus99_to_na <- function(x){
  x <- suppressWarnings(as.numeric(x))
  x[x %in% c(-99, -2)] <- NA_real_
  x
}

rowmax_na <- function(...){
  m <- cbind(...)
  out <- apply(m, 1, function(z){
    if(all(is.na(z))) return(NA_real_)
    suppressWarnings(max(z, na.rm = TRUE))
  })
  as.numeric(out)
}

flag_gt <- function(x, thr){
  ifelse(is.na(x), NA_real_, ifelse(x > thr, 1, 0))
}

flag_ge <- function(x, thr){
  ifelse(is.na(x), NA_real_, ifelse(x >= thr, 1, 0))
}

# Convert documented missing-value codes before status classification.
data_patients <- data_patients %>%
  mutate(
    across(all_of(trigger_vars_items), minus99_to_na),
    HAMA_Sum = minus99_to_na(HAMA_Sum),
    HAMD_Sum17 = minus99_to_na(HAMD_Sum17),
    YMRS_Sum = minus99_to_na(YMRS_Sum)
  )

# Status flags
f_SANS7  <- flag_gt(data_patients$SANS7, 2)
f_SANS12 <- flag_gt(data_patients$SANS12, 2)
f_SANS16 <- flag_gt(data_patients$SANS16, 2)
f_SANS21 <- flag_gt(data_patients$SANS21, 2)

f_SAPS7  <- flag_gt(data_patients$SAPS7, 2)
f_SAPS20 <- flag_gt(data_patients$SAPS20, 2)
f_SAPS25 <- flag_gt(data_patients$SAPS25, 2)
f_SAPS34 <- flag_gt(data_patients$SAPS34, 2)

f_HAMA <- flag_gt(data_patients$HAMA_Sum, 19)
f_HAMD <- flag_gt(data_patients$HAMD_Sum17, 6)
f_YMRS <- flag_ge(data_patients$YMRS_Sum, 4)

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

data_patients <- data_patients %>%
  mutate(
    akut_flag = akut_flag,
    complete_flag = complete_flag,
    Status = dplyr::case_when(
      akut_flag == 1 ~ "acute",
      complete_flag & akut_flag == 0 ~ "remittiert",
      TRUE ~ NA_character_
    ),
    Status = factor(Status, levels = c("remittiert","acute"))
  )

# Missingness summary for status inputs
status_input_vars <- unique(c(trigger_vars_items, "HAMA_Sum","HAMD_Sum17","YMRS_Sum"))
status_missing_df <- purrr::map_dfr(status_input_vars, function(v){
  data_patients %>%
    group_by(gruppe_diag) %>%
    summarise(
      N_total = n(),
      Missing_N = sum(!is.finite(.data[[v]])),
      Missing_pct = round(100 * Missing_N / N_total, 2),
      .groups="drop"
    ) %>%
    mutate(Variable = v)
}) %>%
  tidyr::pivot_wider(
    names_from = gruppe_diag,
    values_from = c(N_total, Missing_N, Missing_pct),
    values_fill = 0
  )
readr::write_csv(status_missing_df, file.path(output_dir, "QC_StatusInputs_Missingness_PATIENTS.csv"))

cat("\n--- Debug: Status counts (defined only) ---\n")
print(table(data_patients$Status, useNA="ifany"))

# -----------------------------
# 5) Variables for tests/tables
# -----------------------------
continuous_vars  <- c(
  "Alter","AgeOfOnset","DurationOfIllness","Bildungsjahre","TIV",
  "HAMD_Sum17","Sum_SANS","Sum_SAPS","Sum_HAMA","Sum_YMRS"
)
categorical_vars <- c("Geschlecht_label","Komorbid","Status")

# -----------------------------
# 6) Helpers
# -----------------------------
fmt_q <- function(q){
  if(is.na(q)) return("q = NA")
  if(q < .001) return("q < .001")
  paste0("q = ", format(round(q,3), nsmall=3))
}

fmt_q_posthoc <- function(q){
  out <- rep(NA_character_, length(q))
  out[is.na(q)] <- "q = NA"
  out[!is.na(q) & q < .001] <- "q < 0.001"
  out[!is.na(q) & q >= .001] <- paste0("q = ", format(round(q[!is.na(q) & q >= .001], 3), nsmall = 3))
  out
}

fmt_statline_q_row <- function(q, stat_value, test_name){
  stat_lab <- if (test_name %in% c("ANOVA","Welch")) "F" else if (test_name == "Kruskal") "H" else "Chi^2"
  paste0(fmt_q(q), " (", stat_lab, " = ", round(as.numeric(stat_value), 2), "; ", test_name, ")")
}

complete_table <- function(df, row_var, col_var, col_levels=NULL){
  r <- factor(df[[row_var]])
  c <- if(!is.null(col_levels)) factor(df[[col_var]], levels=col_levels) else factor(df[[col_var]])
  table(r, c)
}

label_category <- function(varname, x){
  if(varname == "Geschlecht_label"){
    as.character(x)
  } else if(varname == "Komorbid"){
    dplyr::case_when(
      is.na(x) ~ NA_character_,
      x == 0 ~ "No",
      x == 1 ~ "Yes",
      TRUE ~ paste0("Value_", as.character(x))
    )
  } else if(varname == "Status"){
    dplyr::case_when(
      is.na(x) ~ NA_character_,
      as.character(x) %in% c("remittiert","acute") ~ as.character(x),
      TRUE ~ as.character(x)
    )
  } else {
    as.character(x)
  }
}

safe_shapiro_p <- function(x){
  x <- x[!is.na(x)]
  if(length(x) < 3) return(NA_real_)
  out <- tryCatch(stats::shapiro.test(x)$p.value, error=function(e) NA_real_)
  as.numeric(out)
}

# -----------------------------
# 7) Continuous: descriptives + diagnostics + global Kruskal + FDR + Posthoc (Dunn, BH)
# -----------------------------
global_cont_list <- list()
cont_rows_list <- list()
posthoc_cont_list <- list()
cont_diag_list <- list()

for(var in continuous_vars){

  stats <- data_patients %>%
    group_by(gruppe_diag) %>%
    summarise(
      N = sum(!is.na(.data[[var]])),
      Mean = round(mean(.data[[var]], na.rm = TRUE), 2),
      SD   = round(sd(.data[[var]], na.rm = TRUE), 2),
      Median = round(median(.data[[var]], na.rm = TRUE), 2),
      IQR = round(IQR(.data[[var]], na.rm = TRUE), 2),
      .groups = "drop"
    ) %>%
    mutate(Formatted = paste0("N=", N, "; M=", Mean, " (SD=", SD, "); Median=", Median, " [IQR=", IQR, "]"))

  stats_wide <- stats %>%
    dplyr::select(gruppe_diag, Formatted) %>%
    tidyr::pivot_wider(names_from = gruppe_diag, values_from = Formatted)

  for (g in groups_patients){
    if(!(g %in% names(stats_wide))) stats_wide[[g]] <- NA_character_
  }
  stats_wide <- stats_wide %>% mutate(Variable = var) %>% dplyr::select(Variable, all_of(groups_patients))

  diag_by_group <- data_patients %>%
    group_by(gruppe_diag) %>%
    summarise(
      Variable = var,
      N_nonmissing = sum(!is.na(.data[[var]])),
      Shapiro_p = safe_shapiro_p(.data[[var]]),
      Skewness = suppressWarnings(moments::skewness(.data[[var]], na.rm = TRUE)),
      Kurtosis = suppressWarnings(moments::kurtosis(.data[[var]], na.rm = TRUE)),
      .groups = "drop"
    )

  tmp_nonmiss <- data_patients %>% dplyr::select(gruppe_diag, value = all_of(var)) %>% filter(!is.na(value))

  levene_p <- tryCatch({
    lv <- car::leveneTest(value ~ gruppe_diag, data = tmp_nonmiss)
    as.numeric(lv$`Pr(>F)`[1])
  }, error = function(e) NA_real_)

  fligner_p <- tryCatch({
    as.numeric(stats::fligner.test(value ~ gruppe_diag, data = tmp_nonmiss)$p.value)
  }, error = function(e) NA_real_)

  all_groups_shapiro_ok <- all(diag_by_group$Shapiro_p > 0.05, na.rm = TRUE)
  homogeneity_ok <- (!is.na(levene_p) && levene_p > 0.05) && (!is.na(fligner_p) && fligner_p > 0.05)

  recommended_global_test <- if (all_groups_shapiro_ok && homogeneity_ok) "ANOVA_candidate" else "Kruskal_preferred"
  decision_reason <- if (recommended_global_test == "ANOVA_candidate") {
    "All groups approx normal (Shapiro p > 0.05) and homogeneity ok (Levene and Fligner p > 0.05)."
  } else {
    "At least one normality or homogeneity criterion failed (supports nonparametric choice)."
  }

  cont_diag_list[[var]] <- diag_by_group %>%
    mutate(
      Levene_p = levene_p,
      Fligner_p = fligner_p,
      AllGroups_Shapiro_OK = all_groups_shapiro_ok,
      Homogeneity_OK = homogeneity_ok,
      Recommended_Global_Test = recommended_global_test,
      Decision_Reason = decision_reason
    )

  res <- kruskal.test(as.formula(paste(var, "~ gruppe_diag")), data=data_patients)
  stat_value <- as.numeric(res$statistic)
  p_val <- as.numeric(res$p.value)
  test_type <- "Kruskal"

  global_cont_list[[var]] <- tibble::tibble(
    Variable = var, test_type = test_type, stat_value = stat_value, p_global_raw = p_val
  )

  cont_rows_list[[var]] <- stats_wide %>%
    mutate(test_type=test_type, stat_value=stat_value, p_global_raw=p_val)
}

global_cont_df <- bind_rows(global_cont_list) %>%
  mutate(
    q_global_fdr = p.adjust(p_global_raw, method="BH"),
    global_significant_fdr = q_global_fdr < alpha_fdr
  )

sig_cont_vars <- global_cont_df %>% filter(global_significant_fdr) %>% pull(Variable)

if(length(sig_cont_vars) > 0){
  for(var in sig_cont_vars){
    ph <- data_patients %>%
      rstatix::dunn_test(as.formula(paste(var, "~ gruppe_diag")), p.adjust.method="BH") %>%
      mutate(Variable = var)
    posthoc_cont_list[[var]] <- ph
  }
}

posthoc_cont_df <- if(length(posthoc_cont_list) > 0) bind_rows(posthoc_cont_list) else tibble::tibble()

posthoc_cont_sig_summary <- tibble::tibble(Variable=continuous_vars, posthoc_sig="")

if(nrow(posthoc_cont_df) > 0){
  ph_sig <- posthoc_cont_df %>% filter(!is.na(p.adj) & p.adj < alpha_fdr)
  if(nrow(ph_sig) > 0){
    ph_sum <- ph_sig %>%
      mutate(pair = paste0(group1, " vs ", group2, " (q=", format(round(p.adj,3), nsmall=3), ")")) %>%
      group_by(Variable) %>%
      summarise(posthoc_sig = paste(pair, collapse="; "), .groups="drop")
    posthoc_cont_sig_summary <- posthoc_cont_sig_summary %>%
      dplyr::select(Variable) %>%
      left_join(ph_sum, by="Variable") %>%
      mutate(posthoc_sig = ifelse(is.na(posthoc_sig), "", posthoc_sig))
  }
}

cont_table_df <- bind_rows(cont_rows_list) %>%
  left_join(global_cont_df %>% dplyr::select(Variable, q_global_fdr), by="Variable") %>%
  left_join(posthoc_cont_sig_summary, by="Variable") %>%
  mutate(posthoc_sig = ifelse(is.na(posthoc_sig), "", posthoc_sig))

if(nrow(posthoc_cont_df) > 0){
  readr::write_csv(posthoc_cont_df, file.path(output_dir, "Posthoc_Continuous_Dunn_BH_PATIENTS.csv"))
}

cont_diag_df <- if(length(cont_diag_list) > 0) bind_rows(cont_diag_list) else tibble::tibble()
readr::write_csv(cont_diag_df, file.path(output_dir, "QC_Continuous_Distribution_Homoscedasticity_PATIENTS.csv"))

# -----------------------------
# 8) Categorical: global Chi^2 (Monte Carlo p) + FDR + Posthoc pairwise Fisher (BH)
# -----------------------------
set.seed(1)
B_mc <- 20000L

global_cat_list <- list()
cat_rows_list <- list()
posthoc_cat_list <- list()

for (var in categorical_vars){

  # Clinical status is an acute-vs-remitted variable. Indeterminate cases
  # remain missing and are reported in the separate missingness output, but
  # they must not form a third display category or enter the percentage
  # denominator. Other categorical variables retain the prior explicit NA
  # display for descriptive completeness.
  stats_cat_source <- data_patients %>%
    mutate(cat_label_disp = label_category(var, .data[[var]]))

  if (var == "Status") {
    stats_cat_source <- stats_cat_source %>%
      filter(!is.na(cat_label_disp))
  } else {
    stats_cat_source <- stats_cat_source %>%
      mutate(cat_label_disp = ifelse(is.na(cat_label_disp), "NA", cat_label_disp))
  }

  stats_cat_long <- stats_cat_source %>%
    group_by(gruppe_diag, cat_label_disp) %>%
    summarise(Freq = n(), .groups = "drop") %>%
    group_by(gruppe_diag) %>%
    mutate(
      Percent = round(Freq / sum(Freq) * 100, 2),
      Formatted = paste0(cat_label_disp, ": n=", Freq, " (", Percent, "%)")
    ) %>%
    summarise(Formatted = paste(Formatted, collapse="; "), .groups="drop")

  stats_wide <- stats_cat_long %>%
    tidyr::pivot_wider(names_from = gruppe_diag, values_from = Formatted)

  for(g in groups_patients){
    if(!(g %in% names(stats_wide))) stats_wide[[g]] <- NA_character_
  }
  stats_wide <- stats_wide %>% mutate(Variable=var) %>% dplyr::select(Variable, all_of(groups_patients))

  df_tmp <- data_patients %>%
    mutate(cat_label = label_category(var, .data[[var]])) %>%
    filter(!is.na(cat_label))

  T <- complete_table(df_tmp, row_var="cat_label", col_var="gruppe_diag", col_levels=groups_patients)

  if (nrow(T) < 2 || ncol(T) < 2) {

    global_cat_list[[var]] <- tibble::tibble(
      Variable=var,
      test_type="Chi^2 (Monte Carlo p)",
      stat_value=NA_real_,
      df=NA_real_,
      B_mc=B_mc,
      p_global_raw=NA_real_,
      note="Degenerate contingency table (<2 categories or <2 groups after excluding NA)."
    )

    cat_rows_list[[var]] <- stats_wide %>%
      mutate(test_type="Chi^2 (Monte Carlo p)",
             stat_value=NA_real_, df=NA_real_, B_mc=B_mc, p_global_raw=NA_real_)

  } else {

    chi_res <- suppressWarnings(chisq.test(T, simulate.p.value=TRUE, B=B_mc))

    global_cat_list[[var]] <- tibble::tibble(
      Variable=var,
      test_type="Chi^2 (Monte Carlo p)",
      stat_value=as.numeric(chi_res$statistic),
      df=as.numeric(chi_res$parameter),
      B_mc=B_mc,
      p_global_raw=as.numeric(chi_res$p.value),
      note=""
    )

    cat_rows_list[[var]] <- stats_wide %>%
      mutate(test_type="Chi^2 (Monte Carlo p)",
             stat_value=as.numeric(chi_res$statistic),
             df=as.numeric(chi_res$parameter),
             B_mc=B_mc,
             p_global_raw=as.numeric(chi_res$p.value))
  }
}

global_cat_df <- bind_rows(global_cat_list) %>%
  mutate(
    q_global_fdr = p.adjust(p_global_raw, method="BH"),
    global_significant_fdr = q_global_fdr < alpha_fdr
  )

sig_cat_vars <- global_cat_df %>% filter(global_significant_fdr) %>% pull(Variable)

if (length(sig_cat_vars) > 0) {

  pairs <- combn(groups_patients, 2, simplify=FALSE)

  for (var in sig_cat_vars){

    df_tmp <- data_patients %>%
      mutate(cat_label = label_category(var, .data[[var]])) %>%
      filter(!is.na(cat_label))

    for (pr in pairs){

      sub <- df_tmp %>%
        filter(gruppe_diag %in% pr) %>%
        mutate(gruppe_diag = factor(gruppe_diag, levels=pr))

      T2 <- table(sub$cat_label, sub$gruppe_diag)

      if(nrow(T2) < 2 || ncol(T2) < 2){
        posthoc_cat_list[[paste0(var,"_",pr[1],"_",pr[2])]] <- tibble::tibble(
          Variable=var, Group1=pr[1], Group2=pr[2],
          test="Fisher", p=1.0,
          note="Degenerate 1-category table (variable constant within this pair)."
        )
      } else {
        ft <- fisher.test(T2)
        posthoc_cat_list[[paste0(var,"_",pr[1],"_",pr[2])]] <- tibble::tibble(
          Variable=var, Group1=pr[1], Group2=pr[2],
          test="Fisher", p=as.numeric(ft$p.value),
          note=""
        )
      }
    }
  }
}

posthoc_cat_df <- if(length(posthoc_cat_list) > 0) bind_rows(posthoc_cat_list) else tibble::tibble()
posthoc_cat_sig_summary <- tibble::tibble(Variable=categorical_vars, posthoc_sig="")

if(nrow(posthoc_cat_df) > 0){

  posthoc_cat_df <- posthoc_cat_df %>%
    group_by(Variable) %>%
    mutate(p_adj = p.adjust(p, method="BH")) %>%
    ungroup()

  readr::write_csv(posthoc_cat_df, file.path(output_dir, "Posthoc_Categorical_Fisher_BH_PATIENTS.csv"))

  ph_sig <- posthoc_cat_df %>% filter(!is.na(p_adj) & p_adj < alpha_fdr)
  if(nrow(ph_sig) > 0){
    ph_sum <- ph_sig %>%
      mutate(pair = paste0(Group1, " vs ", Group2, " (", fmt_q_posthoc(p_adj), ")")) %>%
      group_by(Variable) %>%
      summarise(posthoc_sig = paste(pair, collapse="; "), .groups="drop")

    posthoc_cat_sig_summary <- posthoc_cat_sig_summary %>%
      dplyr::select(Variable) %>%
      left_join(ph_sum, by="Variable") %>%
      mutate(posthoc_sig = ifelse(is.na(posthoc_sig), "", posthoc_sig))
  }
}

cat_table_df <- bind_rows(cat_rows_list) %>%
  left_join(global_cat_df %>% dplyr::select(Variable, q_global_fdr), by="Variable") %>%
  left_join(posthoc_cat_sig_summary, by="Variable") %>%
  mutate(posthoc_sig = ifelse(is.na(posthoc_sig), "", posthoc_sig))

# -----------------------------
# 9) Missing values report (updated includes AgeOfOnset + Duration + Status)
# -----------------------------
missing_df <- purrr::map_dfr(c(continuous_vars, categorical_vars), function(v){
  data_patients %>%
    group_by(gruppe_diag) %>%
    summarise(
      N_total = n(),
      Missing_N = sum(is.na(.data[[v]])),
      Missing_pct = round(Missing_N / N_total * 100, 2),
      .groups="drop"
    ) %>%
    mutate(Variable = v)
}) %>%
  tidyr::pivot_wider(
    names_from = gruppe_diag,
    values_from = c(N_total, Missing_N, Missing_pct),
    values_fill = 0
  )

readr::write_csv(missing_df, file.path(output_dir, "Missing_Values_Report_PATIENTS.csv"))

# -----------------------------
# 10) Final table + export (CSV + HTML)
# -----------------------------
combined_all_df <- bind_rows(cont_table_df, cat_table_df)

if(!("q_global_fdr" %in% names(combined_all_df))) combined_all_df$q_global_fdr <- NA_real_
if(!("posthoc_sig" %in% names(combined_all_df))) combined_all_df$posthoc_sig <- ""

combined_all_df <- combined_all_df %>%
  mutate(
    Group_Comparison_FDR = mapply(
      FUN = fmt_statline_q_row,
      q = q_global_fdr,
      stat_value = stat_value,
      test_name = test_type
    ),
    posthoc_sig = ifelse(is.na(posthoc_sig), "", posthoc_sig)
  )

readr::write_csv(combined_all_df, file.path(output_dir, "Descriptive_Statistics_PATIENTS_FULL.csv"))
readr::write_csv(global_cont_df, file.path(output_dir, "GlobalTests_Continuous_PATIENTS_FDR_BH.csv"))
readr::write_csv(global_cat_df,  file.path(output_dir, "GlobalTests_Categorical_PATIENTS_FDR_BH.csv"))

final_out <- combined_all_df %>%
  dplyr::select(Variable, all_of(groups_patients), Group_Comparison_FDR, posthoc_sig)

table_combined <- flextable::flextable(final_out) %>%
  flextable::set_header_labels(
    Variable = "Variable",
    MDD = "MDD", BD = "BD", SZ = "SZ", Angst = "Angst",
    Group_Comparison_FDR = paste0("Group Comparison (FDR BH, alpha=", alpha_fdr, ")"),
    posthoc_sig = "Post-hoc significant (FDR BH)"
  ) %>%
  flextable::add_header_lines(values="Descriptive Statistics and Group Comparisons (PATIENTS) with FDR (Benjamini-Hochberg)") %>%
  flextable::autofit() %>%
  flextable::theme_booktabs()

if (requireNamespace("rmarkdown", quietly = TRUE) && rmarkdown::pandoc_available()) {
  flextable::save_as_html(
    table_combined,
    path = file.path(output_dir, "Descriptive_Statistics_by_Group_PATIENTS_FDR_BH.html")
  )
} else {
  warning("HTML export skipped because pandoc is unavailable; CSV outputs are complete.")
}

cat("\nFertig. Exporte unter 'Tabellen/':\n")
cat(" - Descriptive_Statistics_by_Group_PATIENTS_FDR_BH.html\n")
cat(" - Descriptive_Statistics_PATIENTS_FULL.csv\n")
cat(" - Missing_Values_Report_PATIENTS.csv\n")
cat(" - GlobalTests_Continuous_PATIENTS_FDR_BH.csv\n")
cat(" - GlobalTests_Categorical_PATIENTS_FDR_BH.csv\n")
cat(" - Posthoc_Continuous_Dunn_BH_PATIENTS.csv (falls globale Tests signifikant)\n")
cat(" - Posthoc_Categorical_Fisher_BH_PATIENTS.csv (falls globale Tests signifikant)\n")
cat(" - QC_Continuous_Distribution_Homoscedasticity_PATIENTS.csv\n")
cat(" - QC_StatusInputs_Missingness_PATIENTS.csv\n")



# -----------------------------
# ADD-ON: Missing IDs per Variable (patient-only)
# Writes one compact overview + optional per-variable files
# Place this AFTER data_patients is fully built (after Status is created),
# and AFTER continuous_vars / categorical_vars are defined.
# -----------------------------

# Safety: ensure Proband exists
if(!("Proband" %in% names(data_patients))){
  stop("Proband column not found in data_patients. Cannot list missing IDs.")
}

# Output folder
missing_id_dir <- file.path(output_dir, "Missing_IDs")
if(!dir.exists(missing_id_dir)) dir.create(missing_id_dir, recursive = TRUE)

# Helper: treat NA as missing (you already converted -99 to NA earlier)
is_missing_value <- function(x){
  if (is.factor(x) || is.character(x)) {
    return(is.na(x) | trimws(as.character(x)) == "")
  }
  is.na(x)
}

vars_to_check <- unique(c(continuous_vars, categorical_vars))

# 1) Overview: per variable, per diagnosis: N_missing + comma-separated ID list
missing_ids_overview <- purrr::map_dfr(vars_to_check, function(v){

  if(!(v %in% names(data_patients))){
    return(tibble::tibble(
      Variable = v,
      gruppe_diag = NA_character_,
      N_total = nrow(data_patients),
      N_missing = NA_integer_,
      Missing_pct = NA_real_,
      Missing_IDs = NA_character_,
      note = "Variable not present in data_patients."
    ))
  }

  df <- data_patients %>%
    dplyr::select(Proband, gruppe_diag, value = all_of(v)) %>%
    mutate(is_missing = is_missing_value(value)) %>%
    group_by(gruppe_diag) %>%
    summarise(
      N_total = n(),
      N_missing = sum(is_missing),
      Missing_pct = round(100 * N_missing / N_total, 2),
      Missing_IDs = paste(Proband[is_missing], collapse = ","),
      .groups = "drop"
    ) %>%
    mutate(
      Variable = v,
      note = ""
    ) %>%
    dplyr::select(Variable, gruppe_diag, N_total, N_missing, Missing_pct, Missing_IDs, note)

  df
})

readr::write_csv(
  missing_ids_overview,
  file.path(missing_id_dir, "Missing_IDs_Overview_byVariable_byGroup_PATIENTS.csv")
)

# 2) Optional: wide summary table (counts only) for quick scanning
missing_counts_wide <- missing_ids_overview %>%
  dplyr::select(Variable, gruppe_diag, N_missing, Missing_pct) %>%
  tidyr::pivot_wider(
    names_from = gruppe_diag,
    values_from = c(N_missing, Missing_pct),
    values_fill = 0
  )

readr::write_csv(
  missing_counts_wide,
  file.path(missing_id_dir, "Missing_IDs_CountsWide_PATIENTS.csv")
)

# 3) Optional: one file per variable with missing IDs (long, one ID per row)
write_per_variable_files <- FALSE

if(isTRUE(write_per_variable_files)){
  purrr::walk(vars_to_check, function(v){
    if(!(v %in% names(data_patients))) return(NULL)

    dfv <- data_patients %>%
      dplyr::select(Proband, gruppe_diag, value = all_of(v)) %>%
      mutate(is_missing = is_missing_value(value)) %>%
      filter(is_missing) %>%
      dplyr::select(Proband, gruppe_diag) %>%
      arrange(gruppe_diag, Proband) %>%
      mutate(Variable = v) %>%
      dplyr::select(Variable, gruppe_diag, Proband)

    out_path <- file.path(missing_id_dir, paste0("Missing_IDs_", v, "_PATIENTS.csv"))
    readr::write_csv(dfv, out_path)
  })
}

cat("\n[OK] Missing-ID reports written to:\n")
cat(" - ", missing_id_dir, "\n", sep = "")
cat("Files:\n")
cat(" - Missing_IDs_Overview_byVariable_byGroup_PATIENTS.csv\n")
cat(" - Missing_IDs_CountsWide_PATIENTS.csv\n")
if(isTRUE(write_per_variable_files)){
  cat(" - Missing_IDs_<Variable>_PATIENTS.csv (per variable)\n")
}


# -----------------------------
# ADD-ON (APPENDIX): Comorbidity table (0/1 lifetime diagnoses) by primary group
# Goal: For each primary diagnosis group (MDD, BD, SZ, Angst),
#       report N_total, N_nonmissing, N_yes (==1), Percent_yes.
#
# Place this AFTER data_patients is created (with gruppe_diag) and cleaned (-99 -> NA etc.).
# Writes CSV for appendix.
# -----------------------------

comorbidity_vars <- c(
  "Generalized_Anxiety_Disorder_Lifetime",
  "Panic_Disorder_Lifetime",
  "Social_Anxiety_Disorder_Lifetime",
  "Specific_phobia_lifetime",
  "Eating_Disorder_lifetime",
  "Agoraphobia_lifetime",
  "OCD_Lifetime",
  "PTSD_Lifetime",
  "Alcohol_Use_Disorder_Lifetime",
  "Other_Substance_Use_Disorder_Lifetime",
  "Psychotic_lifetime",
  "MDD_Lifetime",
  "BD_Lifetime"
)

# check presence
missing_comorb_vars <- setdiff(comorbidity_vars, names(data_patients))
if(length(missing_comorb_vars) > 0){
  stop(paste0(
    "Missing comorbidity variables in data_patients: ",
    paste(missing_comorb_vars, collapse = ", ")
  ))
}

# Coerce + clean: keep only 0/1, everything else -> NA (incl -99)
to01_na <- function(x){
  x <- suppressWarnings(as.numeric(trimws(as.character(x))))
  x[x == -99] <- NA_real_
  x[!(x %in% c(0,1))] <- NA_real_
  x
}

data_patients <- data_patients %>%
  dplyr::mutate(dplyr::across(dplyr::all_of(comorbidity_vars), to01_na))

# Long table with counts
comorb_long <- tidyr::pivot_longer(
  data_patients,
  cols = dplyr::all_of(comorbidity_vars),
  names_to = "Comorbidity",
  values_to = "Value"
) %>%
  dplyr::group_by(gruppe_diag, Comorbidity) %>%
  dplyr::summarise(
    N_total = dplyr::n(),
    N_nonmissing = sum(!is.na(Value)),
    N_yes = sum(Value == 1, na.rm = TRUE),
    Percent_yes = ifelse(N_nonmissing > 0, round(100 * N_yes / N_nonmissing, 2), NA_real_),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    Display = paste0("n=", N_yes, " (", Percent_yes, "%); N_nonmiss=", N_nonmissing)
  )

# Wide appendix table (one row per comorbidity; one column per primary group)
comorb_wide_appendix <- comorb_long %>%
  dplyr::select(Comorbidity, gruppe_diag, Display) %>%
  tidyr::pivot_wider(
    names_from = gruppe_diag,
    values_from = Display,
    values_fill = ""
  ) %>%
  dplyr::arrange(Comorbidity)

# Also provide numeric wide versions (optional, useful for QC)
comorb_wide_counts <- comorb_long %>%
  dplyr::select(Comorbidity, gruppe_diag, N_yes) %>%
  tidyr::pivot_wider(names_from = gruppe_diag, values_from = N_yes, values_fill = 0) %>%
  dplyr::arrange(Comorbidity)

comorb_wide_percent <- comorb_long %>%
  dplyr::select(Comorbidity, gruppe_diag, Percent_yes) %>%
  tidyr::pivot_wider(names_from = gruppe_diag, values_from = Percent_yes, values_fill = NA_real_) %>%
  dplyr::arrange(Comorbidity)

# Export
appendix_dir <- file.path(output_dir, "Appendix")
if(!dir.exists(appendix_dir)) dir.create(appendix_dir, recursive = TRUE)

readr::write_csv(comorb_long, file.path(appendix_dir, "Appendix_Comorbidity_Lifetime_Long_PATIENTS.csv"))
readr::write_csv(comorb_wide_appendix, file.path(appendix_dir, "Appendix_Comorbidity_Lifetime_WideDisplay_PATIENTS.csv"))
readr::write_csv(comorb_wide_counts, file.path(appendix_dir, "Appendix_Comorbidity_Lifetime_WideCounts_PATIENTS.csv"))
readr::write_csv(comorb_wide_percent, file.path(appendix_dir, "Appendix_Comorbidity_Lifetime_WidePercent_PATIENTS.csv"))

cat("\n[OK] Appendix comorbidity tables written to:\n")
cat(" - ", appendix_dir, "\n", sep = "")
cat("Files:\n")
cat(" - Appendix_Comorbidity_Lifetime_WideDisplay_PATIENTS.csv (for manuscript appendix)\n")
cat(" - Appendix_Comorbidity_Lifetime_Long_PATIENTS.csv (long QC)\n")
cat(" - Appendix_Comorbidity_Lifetime_WideCounts_PATIENTS.csv\n")
cat(" - Appendix_Comorbidity_Lifetime_WidePercent_PATIENTS.csv\n")
