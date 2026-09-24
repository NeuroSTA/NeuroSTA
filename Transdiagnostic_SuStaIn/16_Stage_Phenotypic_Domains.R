# ============================================================
# Unified paper-style script:
# ML Stage vs Early / Current / Neuropsych / Psychopathology variables
# HAM-D analyses use the prespecified HAMD_Sum17 variable. HAMA, SANS,
# SAPS, and YMRS scores are calculated from the item whitelists below.
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(ggplot2)
  library(patchwork)
})

# -----------------------------
# 0) Config
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
setwd(base_dir)

out_root <- file.path(base_dir, "Output Scripte", "paper_stage_vs_domains_unified_separate_psychopathology_BH")
if (!dir.exists(out_root)) dir.create(out_root, recursive = TRUE)

assignment_dir <- file.path(
  base_dir, "Output_sustainrun", "SuStaIn_assignments"
)

cohorts <- list(
  TDM = file.path(assignment_dir, "main_subject_subtype_assignment_3subtypes.csv"),
  ANX = file.path(assignment_dir, "Angst_subject_subtype_assignment_3subtypes.csv"),
  MDD = file.path(assignment_dir, "MDD_subject_subtype_assignment_3subtypes.csv"),
  BD  = file.path(assignment_dir, "BD_subject_subtype_assignment_3subtypes.csv"),
  SSD = file.path(assignment_dir, "SZ_subject_subtype_assignment_3subtypes.csv")
)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

K_SUBTYPES <- 3
stage_col <- "ML_Stage"
subtype_col <- "ML_Subtype"

MIN_N_PLOT <- 10
MIN_N_MODEL <- 20
MIN_N_GROUP_TEST <- 10

APPLY_BH <- TRUE
PLOT_SAVE_FACETS <- TRUE
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

# -----------------------------
# 1) Variable definitions
# -----------------------------

early_metric_vars <- c(
  "CTQ_Sum",
  "UrbanicityScore",
  "AlterVaterBeiGeburt",
  "Geburtsgewicht",
  "SSW_Geburt",
  "FEB_FuersorgeMutter",
  "FEB_FuersorgeVater"
)

current_metric_vars <- c(
  "LEQ_TotalEventScore",
  "LEQ_NegativeEventScore",
  "LEQ_PositiveEventScore",
  "FSozU_Sum",
  "RS25_Sum"
)

neuropsych_metric_vars <- c(
  "VLMT_Sum_Richtige",
  "TMT_Differenz",
  "BZT_Sum",
  "D2_KL",
  "RWT_Tiere_SumCorrected",
  "RWT_P_SumCorrected",
  "RWT_Altern_SumCorrected",
  "ZST_Sum",
  "Blockspanne_Gesamt"
)

nominal_vars <- c(
  "GenRisiko_Affektiv1",
  "GenRisiko_Psycho1"
)

# Main symptom scale items (still used for HAMA/SANS/SAPS/YMRS; HAMD
# now sourced directly from HAMD_Sum17)
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

# NOTE: "HAMD" removed -- sourced directly from HAMD_Sum17 (see section 3).
items_by_scale <- list(
  HAMA = questionnaire_vars[grepl("^HAMA", questionnaire_vars)],
  SANS = questionnaire_vars[grepl("^SANS", questionnaire_vars)],
  SAPS = questionnaire_vars[grepl("^SAPS", questionnaire_vars)],
  YMRS = questionnaire_vars[grepl("^YMRS", questionnaire_vars)]
)

main_symptom_scales <- c("HAMD", "HAMA", "SANS", "SAPS", "YMRS", "GAF")

enigma_vars <- c(
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

var_labels <- c(
  CTQ_Sum = "CTQ total score",
  UrbanicityScore = "Urbanicity score",
  AlterVaterBeiGeburt = "Paternal age at birth",
  Geburtsgewicht = "Birth weight",
  SSW_Geburt = "Gestational age",
  FEB_FuersorgeMutter = "Maternal care",
  FEB_FuersorgeVater = "Paternal care",
  LEQ_TotalEventScore = "LEQ total event score",
  LEQ_NegativeEventScore = "LEQ negative event score",
  LEQ_PositiveEventScore = "LEQ positive event score",
  FSozU_Sum = "FSozU total score",
  RS25_Sum = "RS-25 total score",
  VLMT_Sum_Richtige = "VLMT total correct",
  TMT_Differenz = "Trail Making Test B-A difference",
  BZT_Sum = "Letter-number span",
  D2_KL = "d2 attention test",
  RWT_Tiere_SumCorrected = "Verbal fluency - semantic fluency",
  RWT_P_SumCorrected = "Verbal fluency - letter fluency",
  RWT_Altern_SumCorrected = "Verbal fluency - switching",
  ZST_Sum = "Symbol coding",
  Blockspanne_Gesamt = "Spatial span",
  GenRisiko_Affektiv1 = "Familial affective risk",
  GenRisiko_Psycho1 = "Familial psychotic risk",
  HAMD = "HAM-D",
  HAMA = "HAM-A",
  SANS = "SANS",
  SAPS = "SAPS",
  YMRS = "YMRS",
  GAF = "GAF",
  SANS_Affektverflachung_ENIGMA = "SANS Affective flattening",
  SANS_AlogieParalogie_ENIGMA = "SANS Alogia-paralogia",
  SANS_AbulieApathie_ENIGMA = "SANS Avolition-apathy",
  SANS_Anhedonie_ENIGMA = "SANS Anhedonia",
  SANS_Aufmerksamkeit_ENIGMA = "SANS Attention",
  SAPS_Halluzinationen_ENIGMA = "SAPS Hallucinations",
  SAPS_Wahn_ENIGMA = "SAPS Delusions",
  SAPS_PositiveFormaleDenkstoerung_ENIGMA = "SAPS Positive formal thought disorder",
  SAPS_BizarresVerhalten_ENIGMA = "SAPS Bizarre behavior"
)

domain_of_variable <- function(v) {
  if (v %in% early_metric_vars) return("Early")
  if (v %in% nominal_vars) return("Early")
  if (v %in% current_metric_vars) return("Current")
  if (v %in% neuropsych_metric_vars) return("Neuropsych")
  if (v %in% main_symptom_scales) return("Psychopathology")
  if (v %in% enigma_vars) return("Psychopathology")
  "Other"
}

correction_family_of_variable <- function(v) {
  if (v %in% main_symptom_scales) return("Psychopathology_main_scales")
  if (v %in% enigma_vars) return("Psychopathology_subscales")
  domain_of_variable(v)
}

# -----------------------------
# 2) Helpers
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
  x <- as.character(x); x <- trimws(x); x <- tolower(x); gsub("_", "-", x)
}

safe_as_numeric_flexible <- function(x) {
  if (is.numeric(x)) return(x)
  xx <- as.character(x); xx <- trimws(xx); xx <- gsub(",", ".", xx, fixed = TRUE)
  suppressWarnings(as.numeric(xx))
}

zscore_safe <- function(x) {
  x <- safe_as_numeric_flexible(x)
  s <- sd(x, na.rm = TRUE); m <- mean(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(NA_real_, length(x)))
  (x - m) / s
}

minus99_to_na_num <- function(x) {
  x <- safe_as_numeric_flexible(x)
  x[x %in% c(-99, -2)] <- NA_real_
  x
}

minus99_to_na_chr <- function(x) {
  xx <- as.character(x); xx <- trimws(xx)
  xx[xx %in% c("-99", "-2", "", "NA", "NaN")] <- NA_character_
  xx
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

derive_subtyp_from_ml <- function(df, k) {
  if (!(subtype_col %in% names(df))) stop("ML_Subtype column not found in assignment file.")
  df <- df %>% mutate(ML_Subtype = safe_as_numeric_flexible(.data[[subtype_col]]))
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype all NA after numeric conversion.")
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  if (!is.finite(ml_min)) stop("ML_Subtype min not finite.")
  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop("Unexpected ML_Subtype coding (expected start 0 or 1). min=", ml_min)
  }
  if (any(df$Subtyp < 1 | df$Subtyp > k, na.rm = TRUE)) stop("Subtyp out of range 1..", k)
  df %>% mutate(Subtyp = factor(Subtyp, levels = 1:k))
}

desc_paper <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(tibble(n = 0, median = NA_real_, q25 = NA_real_, q75 = NA_real_, iqr = NA_real_,
                  min = NA_real_, max = NA_real_, mean = NA_real_, sd = NA_real_))
  }
  q <- stats::quantile(x, probs = c(0.25, 0.75), names = FALSE, type = 7)
  tibble(n = length(x), median = stats::median(x), q25 = q[1], q75 = q[2], iqr = q[2] - q[1],
         min = min(x), max = max(x), mean = mean(x), sd = sd(x))
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
  ifelse(is.na(x) | !is.finite(x), NA_character_, formatC(x, format = "f", digits = digits, decimal.mark = "."))
}

fmt_p_sci <- function(p, digits = 2) {
  ifelse(is.na(p) | !is.finite(p), NA_character_, formatC(p, format = "e", digits = digits, decimal.mark = "."))
}

force_text <- function(x) {
  ifelse(is.na(x) | x == "", "", paste0("'", x))
}

initialize_p_columns <- function(df, family_label = NA_character_) {
  df %>% mutate(family_BH = family_label, q_BH = NA_real_, p_for_table = p_value)
}

apply_bh_subset_domain <- function(df, subset_idx, family_label_prefix) {
  out <- df
  if (!("correction_family" %in% names(out))) out$correction_family <- out$domain
  if (!APPLY_BH) {
    out$family_BH[subset_idx] <- paste0(family_label_prefix, "_", out$correction_family[subset_idx])
    out$p_for_table[subset_idx] <- out$p_value[subset_idx]
    return(out)
  }
  idx_all <- which(subset_idx)
  if (length(idx_all) == 0) return(out)
  correction_families <- unique(out$correction_family[idx_all])
  for (fam in correction_families) {
    idx_fam <- subset_idx & out$correction_family == fam
    pvals <- out$p_value[idx_fam]
    ok <- is.finite(pvals)
    out$family_BH[idx_fam] <- paste0(family_label_prefix, "_", fam)
    if (sum(ok) > 0) {
      qvals <- rep(NA_real_, length(pvals))
      qvals[ok] <- p.adjust(pvals[ok], method = "BH")
      out$q_BH[idx_fam] <- qvals
      out$p_for_table[idx_fam] <- qvals
    }
  }
  out
}

epsilon_sq_kruskal <- function(H, n, k) {
  if (!is.finite(H) || !is.finite(n) || !is.finite(k) || (n - k) <= 0) return(NA_real_)
  eps <- (H - k + 1) / (n - k)
  if (!is.finite(eps)) return(NA_real_)
  if (eps < 0) eps <- 0
  eps
}

cliffs_delta <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  nx <- length(x); ny <- length(y)
  if (nx == 0 || ny == 0) return(NA_real_)
  cmp <- outer(x, y, FUN = "-")
  (sum(cmp > 0) - sum(cmp < 0)) / (nx * ny)
}

recode_binary_nominal <- function(x) {
  xx <- as.character(x); xx <- trimws(xx); xx_low <- tolower(xx)
  out <- rep(NA_real_, length(xx_low))
  out[xx_low %in% c("absent", "0", "no", "nein")] <- 0
  out[xx_low %in% c("present", "1", "yes", "ja")] <- 1
  out
}

prepare_continuous_model_data <- function(df) {
  df %>%
    mutate(
      ML_Stage = safe_as_numeric_flexible(ML_Stage),
      Value = safe_as_numeric_flexible(Value),
      ML_Stage_std = zscore_safe(ML_Stage),
      Value_std = zscore_safe(Value),
      Subtyp = factor(Subtyp, levels = 1:K_SUBTYPES)
    )
}

extract_lm_table_with_ci <- function(fit, cohort_name, variable_name, domain_name, model_name, correction_family_name = NA_character_) {
  sm <- summary(fit)
  cf <- as.data.frame(sm$coefficients)
  cf$term <- rownames(cf); rownames(cf) <- NULL
  names(cf) <- c("beta_std", "std_error", "t_value", "p_value", "term")
  ci <- suppressMessages(confint(fit))
  ci_df <- as.data.frame(ci)
  ci_df$term <- rownames(ci_df); rownames(ci_df) <- NULL
  names(ci_df) <- c("CI_lower", "CI_upper", "term")
  cf %>%
    left_join(ci_df, by = "term") %>%
    mutate(cohort = cohort_name, domain = domain_name, correction_family = correction_family_name,
           variable = variable_name, model = model_name, r_squared = sm$r.squared,
           adj_r_squared = sm$adj.r.squared, n = nobs(fit)) %>%
    select(cohort, domain, correction_family, variable, model, n, term,
           beta_std, std_error, t_value, p_value, CI_lower, CI_upper, r_squared, adj_r_squared)
}

run_nominal_kruskal <- function(df, cohort_name, variable_name, domain_name, correction_family_name, model_name) {
  dd <- df %>% filter(is.finite(ML_Stage), !is.na(Category)) %>% mutate(Category = droplevels(factor(Category)))
  n_total <- nrow(dd)
  grp_levels <- levels(dd$Category)
  if (n_total < MIN_N_GROUP_TEST || length(grp_levels) != 2) {
    return(tibble(cohort = cohort_name, domain = domain_name, correction_family = correction_family_name,
                  variable = variable_name, model = model_name, n = n_total,
                  n_group1 = ifelse(length(grp_levels) >= 1, sum(dd$Category == grp_levels[1]), NA_real_),
                  n_group2 = ifelse(length(grp_levels) >= 2, sum(dd$Category == grp_levels[2]), NA_real_),
                  group1 = ifelse(length(grp_levels) >= 1, grp_levels[1], NA_character_),
                  group2 = ifelse(length(grp_levels) >= 2, grp_levels[2], NA_character_),
                  H = NA_real_, df = NA_real_, p_value = NA_real_, epsilon2 = NA_real_, cliffs_delta = NA_real_))
  }
  n1 <- sum(dd$Category == grp_levels[1], na.rm = TRUE)
  n2 <- sum(dd$Category == grp_levels[2], na.rm = TRUE)
  if (n1 < 2 || n2 < 2) {
    return(tibble(cohort = cohort_name, domain = domain_name, correction_family = correction_family_name,
                  variable = variable_name, model = model_name, n = n_total, n_group1 = n1, n_group2 = n2,
                  group1 = grp_levels[1], group2 = grp_levels[2],
                  H = NA_real_, df = NA_real_, p_value = NA_real_, epsilon2 = NA_real_, cliffs_delta = NA_real_))
  }
  kw <- kruskal.test(ML_Stage ~ Category, data = dd)
  H <- unname(kw$statistic); df_kw <- unname(kw$parameter); p <- safe_p_floor(kw$p.value)
  eps <- epsilon_sq_kruskal(H, n_total, 2)
  dlt <- cliffs_delta(dd$ML_Stage[dd$Category == grp_levels[2]], dd$ML_Stage[dd$Category == grp_levels[1]])
  tibble(cohort = cohort_name, domain = domain_name, correction_family = correction_family_name,
         variable = variable_name, model = model_name, n = n_total, n_group1 = n1, n_group2 = n2,
         group1 = grp_levels[1], group2 = grp_levels[2], H = H, df = df_kw, p_value = p,
         epsilon2 = eps, cliffs_delta = dlt)
}

make_sum_from_cols <- function(df, cols) {
  missing_cols <- setdiff(cols, names(df))
  if (length(missing_cols) > 0L) {
    stop("Missing prespecified symptom items: ", paste(missing_cols, collapse = ", "))
  }
  tmp <- df %>%
    select(Proband, all_of(cols)) %>%
    mutate(across(all_of(cols), minus99_to_na_num))
  n_items_total <- length(cols)
  out <- tmp %>%
    mutate(
      n_items_present = rowSums(!is.na(across(all_of(cols)))),
      Sum = ifelse(
        n_items_present >= 1L,
        rowSums(across(all_of(cols)), na.rm = TRUE),
        NA_real_
      )
    ) %>%
    select(Proband, n_items_present, Sum)
  attr(out, "n_items") <- n_items_total
  attr(out, "min_items_required") <- 1L
  out
}

# Read the prespecified HAM-D 17-item total.
make_hamd_from_existing <- function(df) {
  if (!"HAMD_Sum17" %in% names(df)) {
    stop("Required HAMD_Sum17 column is missing; HAM-D analysis cannot run.")
  }
  out <- df %>%
    select(Proband, HAMD_Sum17) %>%
    mutate(
      HAMD_Sum17 = minus99_to_na_num(HAMD_Sum17),
      n_items_present = ifelse(is.finite(HAMD_Sum17), 17L, 0L),
      Sum = HAMD_Sum17
    ) %>%
    select(Proband, n_items_present, Sum)
  attr(out, "n_items") <- 17
  attr(out, "min_items_required") <- 17
  out
}

make_scatter_plot <- function(df, cohort_name, var_name, out_png, facet = FALSE) {
  df_plot <- df %>%
    filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtyp)) %>%
    mutate(Subtyp = factor(as.character(Subtyp), levels = c("1", "2", "3")),
           Label = ifelse(var_name %in% names(var_labels), var_labels[[var_name]], var_name))
  dirp <- dirname(out_png)
  if (!dir.exists(dirp)) dir.create(dirp, recursive = TRUE)
  if (nrow(df_plot) < MIN_N_PLOT) {
    png(out_png, width = 1600, height = 1100, res = 200)
    plot.new()
    title(main = paste0("Not enough data: ", cohort_name, " | ", var_name, " (n=", nrow(df_plot), ")"))
    dev.off()
    return(df_plot)
  }
  gg <- ggplot(df_plot, aes(x = ML_Stage, y = Value, color = Subtyp)) +
    geom_point(alpha = 0.22, size = 1.2, position = position_jitter(width = 0.08, height = 0.08)) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 0.9) +
    scale_color_manual(values = palette_subtypes, labels = subtype_labels, breaks = c("1", "2", "3")) +
    labs(title = paste0("ML Stage vs ", ifelse(var_name %in% names(var_labels), var_labels[[var_name]], var_name), " (", cohort_name, ")"),
         x = "ML Stage", y = ifelse(var_name %in% names(var_labels), var_labels[[var_name]], var_name), color = "Subtype") +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(face = "bold", hjust = 0), legend.position = "top",
          legend.title = element_text(face = "bold"), axis.title = element_text(face = "bold"))
  if (isTRUE(facet)) {
    gg <- gg + facet_wrap(~ Subtyp, ncol = 3, labeller = labeller(Subtyp = subtype_labels)) + guides(color = "none")
  }
  ggsave(out_png, gg, width = 7.8, height = 5.6, dpi = 300, bg = "white")
  df_plot
}

make_combined_panel <- function(plot_data_list, var_subset, cohort_name, out_png, title_text) {
  available_vars <- intersect(var_subset, names(plot_data_list))
  if (length(available_vars) == 0) return(invisible(NULL))
  panel_list <- lapply(seq_along(available_vars), function(i) {
    v <- available_vars[i]
    dfp <- plot_data_list[[v]] %>%
      filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtyp)) %>%
      mutate(Subtyp = factor(as.character(Subtyp), levels = c("1", "2", "3")))
    ggplot(dfp, aes(x = ML_Stage, y = Value, color = Subtyp)) +
      geom_point(alpha = 0.20, size = 1.0, position = position_jitter(width = 0.10, height = 0)) +
      geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
      scale_color_manual(values = palette_subtypes, breaks = c("1", "2", "3"), labels = subtype_labels) +
      labs(x = "ML Stage", y = ifelse(v %in% names(var_labels), var_labels[[v]], v), color = "Subtype") +
      theme_classic(base_size = 12) +
      theme(axis.title = element_text(face = "bold"), legend.position = if (i == 1) "bottom" else "none",
            legend.title = element_text(face = "bold"))
  })
  combined_plot <- wrap_plots(panel_list, ncol = 3, guides = "collect") +
    plot_annotation(title = paste0(title_text, " (", cohort_name, ")"), tag_levels = "A")
  ggsave(out_png, combined_plot, width = 12, height = 8, dpi = 300, bg = "white")
  invisible(out_png)
}

# -----------------------------
# 3) Load data_all
# -----------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)
required_phenotype_vars <- c(early_metric_vars, current_metric_vars, neuropsych_metric_vars,
                             nominal_vars, enigma_vars, "HAMD_Sum17", "GAFscore")
missing_phenotype_vars <- setdiff(required_phenotype_vars, names(data_all))
if (length(missing_phenotype_vars) > 0L) stop("Required phenotype variables missing: ", paste(missing_phenotype_vars, collapse = ", "))

id_col_all <- detect_id_col(data_all)
if (is.null(id_col_all)) stop("No ID column detected in data_all.")
if (id_col_all != "Proband") data_all <- data_all %>% rename(Proband = all_of(id_col_all))
data_all <- data_all %>% mutate(Proband = norm_id(Proband))

present_metric_vars_all <- intersect(c(early_metric_vars, current_metric_vars, neuropsych_metric_vars), names(data_all))
metric_tbl <- data_all %>%
  select(Proband, all_of(present_metric_vars_all)) %>%
  mutate(across(all_of(setdiff(present_metric_vars_all, "TMT_Differenz")), minus99_to_na_num),
         # A negative TMT B-A difference is possible. In particular, -2 is
         # valid for participant 3249 (TMT_Ausschluss = 0), not a missing code.
         TMT_Differenz = {
           x <- safe_as_numeric_flexible(TMT_Differenz)
           x[x %in% -99] <- NA_real_
           x
         })

present_nominal_vars_all <- intersect(nominal_vars, names(data_all))
nominal_tbl <- data_all %>%
  select(Proband, all_of(present_nominal_vars_all)) %>%
  mutate(across(all_of(present_nominal_vars_all), minus99_to_na_chr)) %>%
  mutate(across(all_of(present_nominal_vars_all), ~ droplevels(factor(.x))))

# Main symptom scales
scores_tbls <- list(
  HAMD = make_hamd_from_existing(data_all),
  HAMA = make_sum_from_cols(data_all, items_by_scale$HAMA),
  SANS = make_sum_from_cols(data_all, items_by_scale$SANS),
  SAPS = make_sum_from_cols(data_all, items_by_scale$SAPS),
  YMRS = make_sum_from_cols(data_all, items_by_scale$YMRS)
)

if ("GAFscore" %in% names(data_all)) {
  scores_tbls$GAF <- data_all %>%
    select(Proband, GAFscore) %>%
    mutate(GAFscore = safe_as_numeric_flexible(GAFscore),
           n_items_present = ifelse(is.finite(GAFscore) & GAFscore != -99, 1L, 0L),
           Sum = ifelse(is.finite(GAFscore) & GAFscore != -99, GAFscore, NA_real_)) %>%
    select(Proband, n_items_present, Sum)
  attr(scores_tbls$GAF, "n_items") <- 1
  attr(scores_tbls$GAF, "min_items_required") <- 1
}

scores_tbls <- scores_tbls[!vapply(scores_tbls, is.null, logical(1))]

present_enigma_vars_all <- intersect(enigma_vars, names(data_all))
enigma_tbl <- data_all %>%
  select(Proband, all_of(present_enigma_vars_all)) %>%
  mutate(across(all_of(present_enigma_vars_all), safe_as_numeric_flexible))

# -----------------------------
# 4) Run per cohort
# -----------------------------
run_one_cohort <- function(cohort_name, assign_path) {
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing(assign_path)
  asg <- read_csv(assign_path, show_col_types = FALSE)
  
  id_col_asg <- detect_id_col(asg)
  if (is.null(id_col_asg)) stop("No ID column detected in assignment file: ", assign_path)
  if (id_col_asg != "Proband") asg <- asg %>% rename(Proband = all_of(id_col_asg))
  
  if (!(stage_col %in% names(asg))) stop("ML_Stage missing in assignment file: ", assign_path)
  if (!(subtype_col %in% names(asg))) stop("ML_Subtype missing in assignment file: ", assign_path)
  
  asg2 <- asg %>%
    mutate(Proband = norm_id(Proband),
           ML_Stage = safe_as_numeric_flexible(.data[[stage_col]]),
           ML_Subtype = safe_as_numeric_flexible(.data[[subtype_col]])) %>%
    derive_subtyp_from_ml(k = K_SUBTYPES) %>%
    select(Proband, Subtyp, ML_Stage) %>%
    filter(!is.na(Proband), is.finite(ML_Stage), !is.na(Subtyp))
  
  out_dir <- file.path(out_root, cohort_name)
  out_plots_dir <- file.path(out_dir, "plots")
  out_data_dir  <- file.path(out_dir, "data")
  out_qc_dir    <- file.path(out_dir, "qc")
  out_stats_dir <- file.path(out_dir, "stats")
  
  for (p in c(out_plots_dir, out_data_dir, out_qc_dir, out_stats_dir)) {
    if (!dir.exists(p)) dir.create(p, recursive = TRUE)
  }
  
  desc_stage <- bind_rows(
    desc_paper(asg2$ML_Stage) %>% mutate(group = "overall"),
    asg2 %>% group_by(Subtyp) %>% summarise(desc_paper(ML_Stage), .groups = "drop") %>%
      mutate(group = paste0("subtype_", as.character(Subtyp))) %>% select(-Subtyp)
  ) %>% mutate(cohort = cohort_name) %>% select(cohort, group, everything())
  
  safe_write_csv(desc_stage, file.path(out_stats_dir, paste0("desc_stage_", cohort_name, ".csv")), na = "")
  
  qc_rows <- list(); desc_rows <- list()
  cont_main_rows <- list(); cont_int_rows <- list(); cont_follow_rows <- list()
  nominal_main_rows <- list(); nominal_follow_rows <- list()
  plot_data_list <- list()
  
  # ---------------------------------
  # 4a) Continuous raw variables
  # ---------------------------------
  for (v in present_metric_vars_all) {
    domain_v <- domain_of_variable(v)
    correction_family_v <- correction_family_of_variable(v)
    
    dfv <- asg2 %>%
      inner_join(metric_tbl %>% select(Proband, all_of(v)), by = "Proband") %>%
      rename(Value = all_of(v)) %>%
      mutate(Value = safe_as_numeric_flexible(Value), ML_Stage = safe_as_numeric_flexible(ML_Stage),
             Subtyp = factor(Subtyp, levels = 1:K_SUBTYPES)) %>%
      filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtyp))
    
    out_csv <- file.path(out_data_dir, paste0("data_stage_vs_", v, ".csv"))
    safe_write_csv(dfv %>% mutate(variable = v, domain = domain_v, correction_family = correction_family_v), out_csv)
    
    out_png <- file.path(out_plots_dir, paste0("plot_scatter_stage_vs_", v, ".png"))
    make_scatter_plot(dfv, cohort_name, v, out_png, facet = FALSE)
    if (isTRUE(PLOT_SAVE_FACETS)) {
      out_png_facet <- file.path(out_plots_dir, paste0("plot_scatter_stage_vs_", v, "_FACET.png"))
      make_scatter_plot(dfv, cohort_name, v, out_png_facet, facet = TRUE)
    }
    plot_data_list[[v]] <- dfv
    
    qc_rows[[length(qc_rows) + 1]] <- tibble(
      cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = v,
      type = "continuous", n_asg = nrow(asg2), n_used = nrow(dfv), out_png = out_png, out_csv = out_csv
    )
    
    desc_rows[[length(desc_rows) + 1]] <- bind_rows(
      desc_paper(dfv$Value) %>% mutate(group = "overall"),
      dfv %>% group_by(Subtyp) %>% summarise(desc_paper(Value), .groups = "drop") %>%
        mutate(group = paste0("subtype_", as.character(Subtyp))) %>% select(-Subtyp)
    ) %>% mutate(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = v) %>%
      select(cohort, domain, correction_family, variable, group, everything())
    
    dmod <- prepare_continuous_model_data(dfv)
    
    fit_main <- lm(Value_std ~ ML_Stage_std, data = dmod)
    cont_main_rows[[length(cont_main_rows) + 1]] <- extract_lm_table_with_ci(
      fit_main, cohort_name, v, domain_v, "main_model", correction_family_name = correction_family_v
    ) %>% initialize_p_columns() %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "primary_stage_term", "other_terms"))
    
    fit_int <- lm(Value_std ~ ML_Stage_std * Subtyp, data = dmod)
    cont_int_rows[[length(cont_int_rows) + 1]] <- extract_lm_table_with_ci(
      fit_int, cohort_name, v, domain_v, "interaction_model", correction_family_name = correction_family_v
    ) %>% initialize_p_columns() %>%
      mutate(term_group = case_when(
        grepl("^ML_Stage_std:Subtyp", term) ~ "interaction_terms",
        term == "ML_Stage_std" ~ "reference_stage_term",
        grepl("^Subtyp", term) ~ "subtype_main_effect_terms",
        TRUE ~ "other_terms"
      ))
    
    cont_follow_rows[[length(cont_follow_rows) + 1]] <- dmod %>%
      group_by(Subtyp) %>%
      group_modify(function(dd, key) {
        nsub <- nrow(dd)
        if (nsub < MIN_N_MODEL) {
          return(tibble(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v,
                        variable = v, model = paste0("followup_subtype_", as.character(key$Subtyp)), n = nsub,
                        term = "ML_Stage_std", beta_std = NA_real_, std_error = NA_real_, t_value = NA_real_,
                        p_value = NA_real_, CI_lower = NA_real_, CI_upper = NA_real_,
                        r_squared = NA_real_, adj_r_squared = NA_real_))
        }
        fit_sub <- lm(Value_std ~ ML_Stage_std, data = dd)
        extract_lm_table_with_ci(fit_sub, cohort_name = cohort_name, variable_name = v, domain_name = domain_v,
                                 model_name = paste0("followup_subtype_", as.character(key$Subtyp)),
                                 correction_family_name = correction_family_v)
      }) %>% ungroup() %>% initialize_p_columns() %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "followup_stage_term", "other_terms"))
  }
  
  # ---------------------------------
  # 4b) Main symptom scales
  # ---------------------------------
  for (sc in names(scores_tbls)) {
    domain_v <- "Psychopathology"
    correction_family_v <- correction_family_of_variable(sc)
    
    dfv <- asg2 %>%
      inner_join(scores_tbls[[sc]], by = "Proband") %>%
      mutate(Value = safe_as_numeric_flexible(Sum)) %>%
      filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtyp)) %>%
      select(Proband, Subtyp, ML_Stage, Value)
    
    out_csv <- file.path(out_data_dir, paste0("data_stage_vs_", sc, ".csv"))
    safe_write_csv(dfv %>% mutate(variable = sc, domain = domain_v, correction_family = correction_family_v), out_csv)
    
    out_png <- file.path(out_plots_dir, paste0("plot_scatter_stage_vs_", sc, ".png"))
    make_scatter_plot(dfv, cohort_name, sc, out_png, facet = FALSE)
    if (isTRUE(PLOT_SAVE_FACETS)) {
      out_png_facet <- file.path(out_plots_dir, paste0("plot_scatter_stage_vs_", sc, "_FACET.png"))
      make_scatter_plot(dfv, cohort_name, sc, out_png_facet, facet = TRUE)
    }
    plot_data_list[[sc]] <- dfv
    
    qc_rows[[length(qc_rows) + 1]] <- tibble(
      cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = sc,
      type = "continuous", n_asg = nrow(asg2), n_used = nrow(dfv), out_png = out_png, out_csv = out_csv
    )
    
    desc_rows[[length(desc_rows) + 1]] <- bind_rows(
      desc_paper(dfv$Value) %>% mutate(group = "overall"),
      dfv %>% group_by(Subtyp) %>% summarise(desc_paper(Value), .groups = "drop") %>%
        mutate(group = paste0("subtype_", as.character(Subtyp))) %>% select(-Subtyp)
    ) %>% mutate(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = sc) %>%
      select(cohort, domain, correction_family, variable, group, everything())
    
    dmod <- prepare_continuous_model_data(dfv)
    
    fit_main <- lm(Value_std ~ ML_Stage_std, data = dmod)
    cont_main_rows[[length(cont_main_rows) + 1]] <- extract_lm_table_with_ci(
      fit_main, cohort_name, sc, domain_v, "main_model", correction_family_name = correction_family_v
    ) %>% initialize_p_columns() %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "primary_stage_term", "other_terms"))
    
    fit_int <- lm(Value_std ~ ML_Stage_std * Subtyp, data = dmod)
    cont_int_rows[[length(cont_int_rows) + 1]] <- extract_lm_table_with_ci(
      fit_int, cohort_name, sc, domain_v, "interaction_model", correction_family_name = correction_family_v
    ) %>% initialize_p_columns() %>%
      mutate(term_group = case_when(
        grepl("^ML_Stage_std:Subtyp", term) ~ "interaction_terms",
        term == "ML_Stage_std" ~ "reference_stage_term",
        grepl("^Subtyp", term) ~ "subtype_main_effect_terms",
        TRUE ~ "other_terms"
      ))
    
    cont_follow_rows[[length(cont_follow_rows) + 1]] <- dmod %>%
      group_by(Subtyp) %>%
      group_modify(function(dd, key) {
        nsub <- nrow(dd)
        if (nsub < MIN_N_MODEL) {
          return(tibble(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v,
                        variable = sc, model = paste0("followup_subtype_", as.character(key$Subtyp)), n = nsub,
                        term = "ML_Stage_std", beta_std = NA_real_, std_error = NA_real_, t_value = NA_real_,
                        p_value = NA_real_, CI_lower = NA_real_, CI_upper = NA_real_,
                        r_squared = NA_real_, adj_r_squared = NA_real_))
        }
        fit_sub <- lm(Value_std ~ ML_Stage_std, data = dd)
        extract_lm_table_with_ci(fit_sub, cohort_name = cohort_name, variable_name = sc, domain_name = domain_v,
                                 model_name = paste0("followup_subtype_", as.character(key$Subtyp)),
                                 correction_family_name = correction_family_v)
      }) %>% ungroup() %>% initialize_p_columns() %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "followup_stage_term", "other_terms"))
  }
  
  # ---------------------------------
  # 4c) ENIGMA subscales
  # ---------------------------------
  for (v in present_enigma_vars_all) {
    domain_v <- "Psychopathology"
    correction_family_v <- correction_family_of_variable(v)
    
    dfv <- asg2 %>%
      inner_join(enigma_tbl %>% select(Proband, all_of(v)), by = "Proband") %>%
      rename(Value = all_of(v)) %>%
      mutate(Value = safe_as_numeric_flexible(Value)) %>%
      filter(is.finite(ML_Stage), is.finite(Value), !is.na(Subtyp))
    
    out_csv <- file.path(out_data_dir, paste0("data_stage_vs_", v, ".csv"))
    safe_write_csv(dfv %>% mutate(variable = v, domain = domain_v, correction_family = correction_family_v), out_csv)
    
    out_png <- file.path(out_plots_dir, paste0("plot_scatter_stage_vs_", v, ".png"))
    make_scatter_plot(dfv, cohort_name, v, out_png, facet = FALSE)
    if (isTRUE(PLOT_SAVE_FACETS)) {
      out_png_facet <- file.path(out_plots_dir, paste0("plot_scatter_stage_vs_", v, "_FACET.png"))
      make_scatter_plot(dfv, cohort_name, v, out_png_facet, facet = TRUE)
    }
    plot_data_list[[v]] <- dfv
    
    qc_rows[[length(qc_rows) + 1]] <- tibble(
      cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = v,
      type = "continuous", n_asg = nrow(asg2), n_used = nrow(dfv), out_png = out_png, out_csv = out_csv
    )
    
    desc_rows[[length(desc_rows) + 1]] <- bind_rows(
      desc_paper(dfv$Value) %>% mutate(group = "overall"),
      dfv %>% group_by(Subtyp) %>% summarise(desc_paper(Value), .groups = "drop") %>%
        mutate(group = paste0("subtype_", as.character(Subtyp))) %>% select(-Subtyp)
    ) %>% mutate(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = v) %>%
      select(cohort, domain, correction_family, variable, group, everything())
    
    dmod <- prepare_continuous_model_data(dfv)
    
    fit_main <- lm(Value_std ~ ML_Stage_std, data = dmod)
    cont_main_rows[[length(cont_main_rows) + 1]] <- extract_lm_table_with_ci(
      fit_main, cohort_name, v, domain_v, "main_model", correction_family_name = correction_family_v
    ) %>% initialize_p_columns() %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "primary_stage_term", "other_terms"))
    
    fit_int <- lm(Value_std ~ ML_Stage_std * Subtyp, data = dmod)
    cont_int_rows[[length(cont_int_rows) + 1]] <- extract_lm_table_with_ci(
      fit_int, cohort_name, v, domain_v, "interaction_model", correction_family_name = correction_family_v
    ) %>% initialize_p_columns() %>%
      mutate(term_group = case_when(
        grepl("^ML_Stage_std:Subtyp", term) ~ "interaction_terms",
        term == "ML_Stage_std" ~ "reference_stage_term",
        grepl("^Subtyp", term) ~ "subtype_main_effect_terms",
        TRUE ~ "other_terms"
      ))
    
    cont_follow_rows[[length(cont_follow_rows) + 1]] <- dmod %>%
      group_by(Subtyp) %>%
      group_modify(function(dd, key) {
        nsub <- nrow(dd)
        if (nsub < MIN_N_MODEL) {
          return(tibble(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v,
                        variable = v, model = paste0("followup_subtype_", as.character(key$Subtyp)), n = nsub,
                        term = "ML_Stage_std", beta_std = NA_real_, std_error = NA_real_, t_value = NA_real_,
                        p_value = NA_real_, CI_lower = NA_real_, CI_upper = NA_real_,
                        r_squared = NA_real_, adj_r_squared = NA_real_))
        }
        fit_sub <- lm(Value_std ~ ML_Stage_std, data = dd)
        extract_lm_table_with_ci(fit_sub, cohort_name = cohort_name, variable_name = v, domain_name = domain_v,
                                 model_name = paste0("followup_subtype_", as.character(key$Subtyp)),
                                 correction_family_name = correction_family_v)
      }) %>% ungroup() %>% initialize_p_columns() %>%
      mutate(term_group = ifelse(term == "ML_Stage_std", "followup_stage_term", "other_terms"))
  }
  
  # ---------------------------------
  # 4d) Nominal variables
  # ---------------------------------
  for (v in present_nominal_vars_all) {
    domain_v <- "Early"
    correction_family_v <- correction_family_of_variable(v)
    
    dfn <- asg2 %>%
      inner_join(nominal_tbl %>% select(Proband, all_of(v)), by = "Proband") %>%
      rename(Category = all_of(v)) %>%
      mutate(Category = droplevels(factor(Category)), ML_Stage = safe_as_numeric_flexible(ML_Stage),
             Subtyp = factor(Subtyp, levels = 1:K_SUBTYPES)) %>%
      filter(is.finite(ML_Stage), !is.na(Category), !is.na(Subtyp))
    
    out_csv <- file.path(out_data_dir, paste0("data_stage_vs_nominal_", v, ".csv"))
    safe_write_csv(dfn %>% mutate(variable = v, domain = domain_v, correction_family = correction_family_v), out_csv)
    
    qc_rows[[length(qc_rows) + 1]] <- tibble(
      cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = v,
      type = "nominal", n_asg = nrow(asg2), n_used = nrow(dfn), out_csv = out_csv
    )
    
    desc_rows[[length(desc_rows) + 1]] <- dfn %>%
      count(Subtyp, Category, name = "n") %>%
      group_by(Subtyp) %>% mutate(pct_within_subtype = 100 * n / sum(n)) %>% ungroup() %>%
      mutate(cohort = cohort_name, domain = domain_v, correction_family = correction_family_v, variable = v) %>%
      select(cohort, domain, correction_family, variable, Subtyp, Category, n, pct_within_subtype)
    
    grp01 <- recode_binary_nominal(dfn$Category)
    valid_binary <- sum(is.finite(grp01)) == nrow(dfn) && length(unique(grp01[is.finite(grp01)])) == 2
    
    if (isTRUE(valid_binary)) {
      nominal_main_rows[[length(nominal_main_rows) + 1]] <- run_nominal_kruskal(
        df = dfn, cohort_name = cohort_name, variable_name = v, domain_name = domain_v,
        correction_family_name = correction_family_v, model_name = "main_model"
      )
      nominal_follow_rows[[length(nominal_follow_rows) + 1]] <- bind_rows(
        lapply(levels(dfn$Subtyp), function(s) {
          ds <- dfn %>% filter(Subtyp == s)
          run_nominal_kruskal(df = ds, cohort_name = cohort_name, variable_name = v, domain_name = domain_v,
                              correction_family_name = correction_family_v, model_name = paste0("followup_subtype_", s))
        })
      )
    }
  }
  
  # ---------------------------------
  # 5) Bind tables
  # ---------------------------------
  qc_all <- bind_rows(qc_rows)
  desc_all <- bind_rows(desc_rows)
  cont_main_all <- bind_rows(cont_main_rows)
  cont_int_all <- bind_rows(cont_int_rows)
  cont_follow_all <- bind_rows(cont_follow_rows)
  nominal_main_all <- bind_rows(nominal_main_rows)
  nominal_follow_all <- bind_rows(nominal_follow_rows)
  
  # ---------------------------------
  # 6) Family-specific BH correction
  # ---------------------------------
  if (nrow(cont_main_all) > 0) {
    cont_main_all <- apply_bh_subset_domain(cont_main_all, subset_idx = cont_main_all$term == "ML_Stage_std",
                                            family_label_prefix = paste0(cohort_name, "_continuous_main"))
  }
  if (nrow(cont_int_all) > 0) {
    cont_int_all <- apply_bh_subset_domain(cont_int_all, subset_idx = grepl("^ML_Stage_std:Subtyp", cont_int_all$term),
                                           family_label_prefix = paste0(cohort_name, "_continuous_interaction"))
  }
  if (nrow(cont_follow_all) > 0) {
    cont_follow_all <- apply_bh_subset_domain(cont_follow_all, subset_idx = cont_follow_all$term == "ML_Stage_std",
                                              family_label_prefix = paste0(cohort_name, "_continuous_followup"))
  }
  if (nrow(nominal_main_all) > 0) {
    nominal_main_all <- nominal_main_all %>%
      mutate(family_BH = paste0(cohort_name, "_nominal_main_", correction_family), q_BH = NA_real_, p_for_table = p_value)
    for (fam in unique(nominal_main_all$correction_family)) {
      idx <- nominal_main_all$correction_family == fam & is.finite(nominal_main_all$p_value)
      if (sum(idx) > 0) {
        nominal_main_all$q_BH[idx] <- p.adjust(nominal_main_all$p_value[idx], method = "BH")
        nominal_main_all$p_for_table[idx] <- nominal_main_all$q_BH[idx]
      }
    }
  }
  if (nrow(nominal_follow_all) > 0) {
    nominal_follow_all <- nominal_follow_all %>%
      mutate(family_BH = paste0(cohort_name, "_nominal_followup_", correction_family), q_BH = NA_real_, p_for_table = p_value)
    for (fam in unique(nominal_follow_all$correction_family)) {
      idx <- nominal_follow_all$correction_family == fam & is.finite(nominal_follow_all$p_value)
      if (sum(idx) > 0) {
        nominal_follow_all$q_BH[idx] <- p.adjust(nominal_follow_all$p_value[idx], method = "BH")
        nominal_follow_all$p_for_table[idx] <- nominal_follow_all$q_BH[idx]
      }
    }
  }
  
  # ---------------------------------
  # 7) Write full tables
  # ---------------------------------
  safe_write_csv(qc_all, file.path(out_qc_dir, paste0("qc_summary_", cohort_name, ".csv")), na = "")
  safe_write_csv(desc_all, file.path(out_stats_dir, paste0("desc_all_variables_", cohort_name, ".csv")), na = "")
  safe_write_csv(cont_main_all, file.path(out_stats_dir, paste0("continuous_main_model_", cohort_name, ".csv")), na = "")
  safe_write_csv(cont_int_all, file.path(out_stats_dir, paste0("continuous_interaction_model_", cohort_name, ".csv")), na = "")
  safe_write_csv(cont_follow_all, file.path(out_stats_dir, paste0("continuous_followup_models_", cohort_name, ".csv")), na = "")
  safe_write_csv(nominal_main_all, file.path(out_stats_dir, paste0("nominal_main_tests_", cohort_name, ".csv")), na = "")
  safe_write_csv(nominal_follow_all, file.path(out_stats_dir, paste0("nominal_followup_tests_", cohort_name, ".csv")), na = "")
  
  # ---------------------------------
  # 8) Compact tables
  # ---------------------------------
  continuous_table_main <- cont_main_all %>%
    filter(term == "ML_Stage_std") %>%
    transmute(Domain = domain, Correction_family = correction_family,
              Variable = ifelse(variable %in% names(var_labels), var_labels[variable], variable),
              Model = "Main model", Term = "ML stage",
              beta_std = force_text(fmt_num(beta_std, 4)),
              CI_95 = force_text(paste0("[", fmt_num(CI_lower, 4), ", ", fmt_num(CI_upper, 4), "]")),
              p = force_text(fmt_p_sci(p_value, 2)),
              `p_adj (BH)` = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
              R2_adj = force_text(fmt_num(adj_r_squared, 3)))
  
  continuous_table_int <- cont_int_all %>%
    filter(term %in% c("ML_Stage_std", "ML_Stage_std:Subtyp2", "ML_Stage_std:Subtyp3")) %>%
    transmute(Domain = domain, Correction_family = correction_family,
              Variable = ifelse(variable %in% names(var_labels), var_labels[variable], variable),
              Model = "Interaction model",
              Term = case_when(
                term == "ML_Stage_std" ~ "ML stage (Subtype 1 reference slope)",
                term == "ML_Stage_std:Subtyp2" ~ "Subtype 2 vs Subtype 1: slope difference",
                term == "ML_Stage_std:Subtyp3" ~ "Subtype 3 vs Subtype 1: slope difference",
                TRUE ~ term
              ),
              beta_std = force_text(fmt_num(beta_std, 4)),
              CI_95 = force_text(paste0("[", fmt_num(CI_lower, 4), ", ", fmt_num(CI_upper, 4), "]")),
              p = force_text(fmt_p_sci(p_value, 2)),
              `p_adj (BH)` = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
              R2_adj = force_text(fmt_num(adj_r_squared, 3)))
  
  continuous_table_follow <- cont_follow_all %>%
    filter(term == "ML_Stage_std") %>%
    transmute(Domain = domain, Correction_family = correction_family,
              Variable = ifelse(variable %in% names(var_labels), var_labels[variable], variable),
              Model = "Follow-up within subtype",
              Term = case_when(
                model == "followup_subtype_1" ~ "Subtype 1 slope",
                model == "followup_subtype_2" ~ "Subtype 2 slope",
                model == "followup_subtype_3" ~ "Subtype 3 slope",
                TRUE ~ model
              ),
              beta_std = force_text(fmt_num(beta_std, 4)),
              CI_95 = force_text(paste0("[", fmt_num(CI_lower, 4), ", ", fmt_num(CI_upper, 4), "]")),
              p = force_text(fmt_p_sci(p_value, 2)),
              `p_adj (BH)` = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
              R2_adj = force_text(fmt_num(adj_r_squared, 3)))
  
  nominal_table_main <- nominal_main_all %>%
    transmute(Domain = domain, Correction_family = correction_family,
              Variable = ifelse(variable %in% names(var_labels), var_labels[variable], variable),
              Model = "Main model", Term = paste0(group2, " vs ", group1),
              H = force_text(fmt_num(H, 3)), df = force_text(fmt_num(df, 0)),
              p = force_text(fmt_p_sci(p_value, 2)),
              `p_adj (BH)` = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
              epsilon2 = force_text(fmt_num(epsilon2, 3)), cliffs_delta = force_text(fmt_num(cliffs_delta, 3)))
  
  nominal_table_follow <- nominal_follow_all %>%
    transmute(Domain = domain, Correction_family = correction_family,
              Variable = ifelse(variable %in% names(var_labels), var_labels[variable], variable),
              Model = "Follow-up within subtype",
              Term = case_when(
                model == "followup_subtype_1" ~ paste0("Subtype 1: ", group2, " vs ", group1),
                model == "followup_subtype_2" ~ paste0("Subtype 2: ", group2, " vs ", group1),
                model == "followup_subtype_3" ~ paste0("Subtype 3: ", group2, " vs ", group1),
                TRUE ~ model
              ),
              H = force_text(fmt_num(H, 3)), df = force_text(fmt_num(df, 0)),
              p = force_text(fmt_p_sci(p_value, 2)),
              `p_adj (BH)` = force_text(ifelse(is.na(q_BH), "", fmt_p_sci(q_BH, 2))),
              epsilon2 = force_text(fmt_num(epsilon2, 3)), cliffs_delta = force_text(fmt_num(cliffs_delta, 3)))
  
  safe_write_tsv(continuous_table_main, file.path(out_stats_dir, paste0("TABLE_continuous_main_", cohort_name, ".tsv")), na = "")
  safe_write_tsv(continuous_table_int, file.path(out_stats_dir, paste0("TABLE_continuous_interaction_", cohort_name, ".tsv")), na = "")
  safe_write_tsv(continuous_table_follow, file.path(out_stats_dir, paste0("TABLE_continuous_followup_", cohort_name, ".tsv")), na = "")
  safe_write_tsv(nominal_table_main, file.path(out_stats_dir, paste0("TABLE_nominal_main_", cohort_name, ".tsv")), na = "")
  safe_write_tsv(nominal_table_follow, file.path(out_stats_dir, paste0("TABLE_nominal_followup_", cohort_name, ".tsv")), na = "")
  
  # ---------------------------------
  # 9) Combined figures
  # ---------------------------------
  make_combined_panel(plot_data_list, c("HAMD", "HAMA", "SANS", "SAPS", "YMRS", "GAF"), cohort_name,
                      file.path(out_plots_dir, paste0("figure_combined_MLStage_vs_total_scales_", cohort_name, ".png")),
                      "ML Stage vs symptom scale scores")
  make_combined_panel(plot_data_list, enigma_vars, cohort_name,
                      file.path(out_plots_dir, paste0("figure_combined_MLStage_vs_subscales_", cohort_name, ".png")),
                      "ML Stage vs subscale scores")
  
  invisible(list(qc = qc_all, desc = desc_all, continuous_main = cont_main_all, continuous_interaction = cont_int_all,
                 continuous_followup = cont_follow_all, nominal_main = nominal_main_all, nominal_followup = nominal_follow_all))
}

# -----------------------------
# 5) Run all cohorts
# -----------------------------
res_list <- purrr::imap(cohorts, ~ run_one_cohort(cohort_name = .y, assign_path = .x))

qc_all_cohorts <- purrr::map_dfr(res_list, "qc") %>% arrange(match(cohort, names(cohorts)))
safe_write_csv(qc_all_cohorts, file.path(out_root, "qc_all_cohorts.csv"))

desc_all_cohorts <- purrr::map_dfr(res_list, "desc") %>% arrange(match(cohort, names(cohorts)), domain, correction_family, variable)
safe_write_csv(desc_all_cohorts, file.path(out_root, "desc_all_cohorts.csv"))

cont_main_all_cohorts <- purrr::map_dfr(res_list, "continuous_main") %>% arrange(match(cohort, names(cohorts)), domain, correction_family, variable)
safe_write_csv(cont_main_all_cohorts, file.path(out_root, "continuous_main_all_cohorts.csv"))

cont_int_all_cohorts <- purrr::map_dfr(res_list, "continuous_interaction") %>% arrange(match(cohort, names(cohorts)), domain, correction_family, variable)
safe_write_csv(cont_int_all_cohorts, file.path(out_root, "continuous_interaction_all_cohorts.csv"))

cont_follow_all_cohorts <- purrr::map_dfr(res_list, "continuous_followup") %>% arrange(match(cohort, names(cohorts)), domain, correction_family, variable)
safe_write_csv(cont_follow_all_cohorts, file.path(out_root, "continuous_followup_all_cohorts.csv"))

nom_main_all_cohorts <- purrr::map_dfr(res_list, "nominal_main") %>% arrange(match(cohort, names(cohorts)), domain, correction_family, variable)
safe_write_csv(nom_main_all_cohorts, file.path(out_root, "nominal_main_all_cohorts.csv"))

nom_follow_all_cohorts <- purrr::map_dfr(res_list, "nominal_followup") %>% arrange(match(cohort, names(cohorts)), domain, correction_family, variable)
safe_write_csv(nom_follow_all_cohorts, file.path(out_root, "nominal_followup_all_cohorts.csv"))

message("[OK] ALL DONE. Output root: ", out_root)
