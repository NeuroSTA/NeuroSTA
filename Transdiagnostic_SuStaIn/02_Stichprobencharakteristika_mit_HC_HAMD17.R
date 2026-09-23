# ================================================================
# Table 1: demographic and clinical characteristics including HC
# ================================================================
# The analysis compares HC, MDD, ANX, BD, and SSD. HAM-D uses the
# prespecified HAMD_Sum17 variable. SAPS, SANS, HAM-A, and YMRS are
# calculated from the item whitelists below.
#
# Group coding:
#   1 = HC, 2 = MDD, 3 = BD, 4/5 = SSD, 7/8 = ANX
# ================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(rstatix)
  library(DescTools)  # Cramer's V
})

# -----------------------------
# 1) Settings and input
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
set.seed(20260916L) # Reproducible Monte Carlo chi-square p-values.
setwd(base_dir)

data_all_path <- getOption("SUSTAIN_CLINICAL_FILE", Sys.getenv("SUSTAIN_CLINICAL_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")))

output_dir <- file.path(base_dir, "Output Scripte", "table1_with_HC")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Item whitelists for the calculated sum scores. HAMD_Sum17 is read directly.
items_by_scale <- list(
  SAPS = c("SAPS1","SAPS2","SAPS3","SAPS4","SAPS5","SAPS6","SAPS8","SAPS9","SAPS10","SAPS11",
           "SAPS12","SAPS13","SAPS14","SAPS15","SAPS16","SAPS17","SAPS18","SAPS19","SAPS21",
           "SAPS22","SAPS23","SAPS24","SAPS26","SAPS27","SAPS28","SAPS29","SAPS30","SAPS31",
           "SAPS32","SAPS33","SAPS35"),
  SANS = c("SANS1","SANS2","SANS3","SANS4","SANS5","SANS6","SANS8","SANS9","SANS10","SANS11",
           "SANS13","SANS14","SANS15","SANS17","SANS18","SANS19","SANS20","SANS22","SANS23"),
  HAMA = paste0("HAMA", 1:14),
  YMRS = paste0("YMRS", 1:11)
)
scale_labels <- c(SAPS = "SAPS Total Score", SANS = "SANS Total Score",
                   HAMD = "HAM-D Total Score", HAMA = "HAM-A Total Score",
                   YMRS = "YMRS Total Score")

# -----------------------------
# 2) Helpers
# -----------------------------
stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

norm_id <- function(x) {
  tolower(gsub("_", "-", trimws(as.character(x))))
}

safe_as_numeric_flexible <- function(x) {
  if (is.numeric(x)) return(x)
  xx <- trimws(as.character(x))
  xx <- gsub(",", ".", xx, fixed = TRUE)
  suppressWarnings(as.numeric(xx))
}

normalize_gender <- function(x) {
  x_low <- tolower(trimws(as.character(x)))
  out <- dplyr::case_when(
    x_low %in% c("1", "m", "male", "maennlich", "männlich") ~ "Maennlich",
    x_low %in% c("2", "w", "female", "weiblich")            ~ "Weiblich",
    TRUE ~ NA_character_
  )
  factor(out, levels = c("Maennlich", "Weiblich"))
}

epsilon2_kw <- function(kw_stat, n, k) {
  (as.numeric(kw_stat) - k + 1) / (n - k)
}

fmt_mean_sd <- function(x) {
  x <- x[is.finite(x)]
  sprintf("%.2f (%.2f)", mean(x), sd(x))
}

minus99_to_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[x == -99] <- NA_real_
  x
}

# Summenscore aus Item-Spalten, NA-sicher (rowSums gibt sonst faelschlich
# 0 statt NA zurueck, wenn ALLE Items einer Person fehlen)
make_sum <- function(df, cols) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) stop("Required scale items missing: ", paste(missing, collapse = ", "))
  present <- cols
  df %>%
    dplyr::mutate(across(all_of(present), minus99_to_na)) %>%
    dplyr::mutate(.sum = rowSums(across(all_of(present)), na.rm = TRUE),
                  .n_present = rowSums(!is.na(across(all_of(present))))) %>%
    dplyr::mutate(.sum = ifelse(.n_present == 0, NA_real_, .sum)) %>%
    dplyr::pull(.sum)
}

p_to_signif <- function(p) {
  dplyr::case_when(
    is.na(p) ~ NA_character_,
    p > 0.05 ~ "ns",
    p <= 0.05 & p > 0.01 ~ "*",
    p <= 0.01 & p > 0.001 ~ "**",
    p <= 0.001 & p > 0.0001 ~ "***",
    p <= 0.0001 ~ "****"
  )
}

# -----------------------------
# 3) Load + prepare data
# -----------------------------
stop_if_missing(data_all_path)
data_all <- read_csv(data_all_path, show_col_types = FALSE)

if (!"Proband" %in% names(data_all)) {
  if ("Name"  %in% names(data_all)) data_all <- data_all %>% rename(Proband = Name)
  if ("names" %in% names(data_all)) data_all <- data_all %>% rename(Proband = names)
}
if (!"Proband" %in% names(data_all)) stop("Fehler: Proband-Spalte fehlt in data_all.")

required_cols <- c("Group", "Alter", "Geschlecht", "TIV", "Bildungsjahre")
missing_cols <- setdiff(required_cols, names(data_all))
if (length(missing_cols) > 0) {
  stop("Fehler: folgende Spalten fehlen in data_all: ", paste(missing_cols, collapse = ", "))
}

data_all <- data_all %>%
  mutate(
    Proband    = norm_id(Proband),
    Group      = suppressWarnings(as.integer(as.numeric(trimws(as.character(Group))))),
    Alter         = safe_as_numeric_flexible(Alter),
    TIV           = safe_as_numeric_flexible(TIV),
    Bildungsjahre = safe_as_numeric_flexible(Bildungsjahre),
    Geschlecht    = normalize_gender(Geschlecht)
  ) %>%
  # -99 is the documented missing-value code.
  mutate(across(c(Alter, TIV, Bildungsjahre), ~ ifelse(.x == -99, NA_real_, .x)))

# -----------------------------
# 4) Diagnosis mapping INKLUSIVE HC
# -----------------------------
data_grp <- data_all %>%
  mutate(
    Diagnosis = case_when(
      Group == 1          ~ "HC",
      Group == 2          ~ "MDD",
      Group == 3          ~ "BD",
      Group %in% c(4, 5)  ~ "SSD",
      Group %in% c(7, 8)  ~ "ANX",
      TRUE                ~ NA_character_
    ),
    Diagnosis = factor(Diagnosis, levels = c("HC", "MDD", "ANX", "BD", "SSD"))
  ) %>%
  filter(!is.na(Diagnosis))

if (nrow(data_grp) == 0) stop("Nach Diagnosis-Mapping keine Zeilen mehr uebrig - Group-Codierung pruefen.")

# -----------------------------
# 4b) Clinical sum scores
# -----------------------------
for (sc in names(items_by_scale)) {
  data_grp[[paste0(sc, "_final")]] <- make_sum(data_grp, items_by_scale[[sc]])
}

if (!"HAMD_Sum17" %in% names(data_grp)) {
  stop("Required HAMD_Sum17 column is missing; HAM-D analysis cannot run.")
}
data_grp[["HAMD_final"]] <- minus99_to_na(data_grp[["HAMD_Sum17"]])

n_by_group <- data_grp %>% count(Diagnosis, name = "n_total")
message(">>> N pro Gruppe (inkl. HC):")
print(n_by_group)
write_csv(n_by_group, file.path(output_dir, "n_by_group.csv"))

# ================================================================
# 5) Descriptive statistics for all five groups
# ================================================================
desc_age <- data_grp %>%
  group_by(Diagnosis) %>%
  summarise(
    n = sum(is.finite(Alter)),
    mean = mean(Alter, na.rm = TRUE),
    sd = sd(Alter, na.rm = TRUE),
    median = median(Alter, na.rm = TRUE),
    iqr = IQR(Alter, na.rm = TRUE),
    formatted = fmt_mean_sd(Alter),
    .groups = "drop"
  )

desc_tiv <- data_grp %>%
  group_by(Diagnosis) %>%
  summarise(
    n = sum(is.finite(TIV)),
    mean = mean(TIV, na.rm = TRUE),
    sd = sd(TIV, na.rm = TRUE),
    median = median(TIV, na.rm = TRUE),
    iqr = IQR(TIV, na.rm = TRUE),
    formatted = fmt_mean_sd(TIV),
    .groups = "drop"
  )

desc_edu <- data_grp %>%
  group_by(Diagnosis) %>%
  summarise(
    n = sum(is.finite(Bildungsjahre)),
    mean = mean(Bildungsjahre, na.rm = TRUE),
    sd = sd(Bildungsjahre, na.rm = TRUE),
    median = median(Bildungsjahre, na.rm = TRUE),
    iqr = IQR(Bildungsjahre, na.rm = TRUE),
    formatted = fmt_mean_sd(Bildungsjahre),
    .groups = "drop"
  )

desc_sex <- data_grp %>%
  filter(!is.na(Geschlecht)) %>%
  count(Diagnosis, Geschlecht, name = "n") %>%
  tidyr::complete(Diagnosis, Geschlecht, fill = list(n = 0)) %>%
  arrange(Diagnosis, Geschlecht)

desc_scales <- purrr::map_dfr(names(scale_labels), function(sc) {
  data_grp %>%
    group_by(Diagnosis) %>%
    summarise(
      n = sum(is.finite(.data[[paste0(sc, "_final")]])),
      mean = mean(.data[[paste0(sc, "_final")]], na.rm = TRUE),
      sd = sd(.data[[paste0(sc, "_final")]], na.rm = TRUE),
      median = median(.data[[paste0(sc, "_final")]], na.rm = TRUE),
      iqr = IQR(.data[[paste0(sc, "_final")]], na.rm = TRUE),
      formatted = fmt_mean_sd(.data[[paste0(sc, "_final")]]),
      .groups = "drop"
    ) %>%
    mutate(Scale = sc, .before = 1)
})

message("\n>>> Deskriptive Statistik Alter (alle 5 Gruppen):")
print(desc_age)
message("\n>>> Deskriptive Statistik TIV (alle 5 Gruppen):")
print(desc_tiv)
message("\n>>> Deskriptive Statistik Bildungsjahre (alle 5 Gruppen):")
print(desc_edu)
message("\n>>> Geschlecht, n pro Gruppe (alle 5 Gruppen):")
print(desc_sex)
message("\n>>> Deskriptive Statistik SAPS/SANS/HAMD/HAMA/YMRS (alle 5 Gruppen):")
print(desc_scales)

write_csv(desc_age, file.path(output_dir, "descriptives_age_incl_HC.csv"))
write_csv(desc_tiv, file.path(output_dir, "descriptives_tiv_incl_HC.csv"))
write_csv(desc_edu, file.path(output_dir, "descriptives_education_incl_HC.csv"))
write_csv(desc_sex, file.path(output_dir, "descriptives_sex_incl_HC.csv"))
write_csv(desc_scales, file.path(output_dir, "descriptives_clinical_scales_incl_HC.csv"))

# ================================================================
# 6) Inference for the five-group comparison (HC + MDD + ANX + BD + SSD)
# ================================================================

## --- Age ---
kw_age <- kruskal.test(Alter ~ Diagnosis, data = data_grp)
n_age <- sum(is.finite(data_grp$Alter))
k_age <- nlevels(droplevels(data_grp$Diagnosis[is.finite(data_grp$Alter)]))
eps2_age <- epsilon2_kw(kw_age$statistic, n_age, k_age)

dunn_age <- rstatix::dunn_test(data_grp, Alter ~ Diagnosis, p.adjust.method = "BH") %>%
  rename(p_adj = p.adj) %>%
  mutate(signif = p_to_signif(p_adj))

## --- TIV ---
kw_tiv <- kruskal.test(TIV ~ Diagnosis, data = data_grp)
n_tiv <- sum(is.finite(data_grp$TIV))
k_tiv <- nlevels(droplevels(data_grp$Diagnosis[is.finite(data_grp$TIV)]))
eps2_tiv <- epsilon2_kw(kw_tiv$statistic, n_tiv, k_tiv)

dunn_tiv <- rstatix::dunn_test(data_grp, TIV ~ Diagnosis, p.adjust.method = "BH") %>%
  rename(p_adj = p.adj) %>%
  mutate(signif = p_to_signif(p_adj))

## --- Education (Bildungsjahre) ---
kw_edu <- kruskal.test(Bildungsjahre ~ Diagnosis, data = data_grp)
n_edu <- sum(is.finite(data_grp$Bildungsjahre))
k_edu <- nlevels(droplevels(data_grp$Diagnosis[is.finite(data_grp$Bildungsjahre)]))
eps2_edu <- epsilon2_kw(kw_edu$statistic, n_edu, k_edu)

dunn_edu <- rstatix::dunn_test(data_grp, Bildungsjahre ~ Diagnosis, p.adjust.method = "BH") %>%
  rename(p_adj = p.adj) %>%
  mutate(signif = p_to_signif(p_adj))

## --- SAPS, SANS, HAMD, HAMA, YMRS ---
kw_scales <- list()
eps2_scales <- list()
dunn_scales <- list()

for (sc in names(scale_labels)) {
  varname <- paste0(sc, "_final")
  kw_fit <- kruskal.test(data_grp[[varname]] ~ data_grp$Diagnosis)
  n_sc <- sum(is.finite(data_grp[[varname]]))
  k_sc <- nlevels(droplevels(data_grp$Diagnosis[is.finite(data_grp[[varname]])]))
  kw_scales[[sc]] <- kw_fit
  eps2_scales[[sc]] <- epsilon2_kw(kw_fit$statistic, n_sc, k_sc)

  dunn_scales[[sc]] <- data_grp %>%
    dplyr::transmute(Diagnosis, .value = .data[[varname]]) %>%
    rstatix::dunn_test(.value ~ Diagnosis, p.adjust.method = "BH") %>%
    rename(p_adj = p.adj) %>%
    mutate(signif = p_to_signif(p_adj), Scale = sc, .before = 1)
}

## --- Sex ---
sex_tab_5grp <- table(data_grp$Diagnosis, data_grp$Geschlecht, useNA = "no")
chi_sex <- chisq.test(sex_tab_5grp, simulate.p.value = TRUE, B = 10000)
cramers_v_sex <- suppressWarnings(DescTools::CramerV(sex_tab_5grp))

# Posthoc pairwise Sex (BH), inkl. aller Paare mit HC
subtypes_sex <- levels(droplevels(data_grp$Diagnosis))
comparisons_sex <- combn(subtypes_sex, 2, simplify = FALSE)

posthoc_sex_list <- lapply(comparisons_sex, function(cmp) {
  sub_data <- data_grp %>% filter(Diagnosis %in% cmp, !is.na(Geschlecht))
  tab <- table(droplevels(factor(sub_data$Diagnosis, levels = cmp)), sub_data$Geschlecht)
  test <- chisq.test(tab, simulate.p.value = TRUE, B = 10000)
  eff <- suppressWarnings(DescTools::CramerV(tab))
  tibble(
    Group1 = as.character(cmp[1]),
    Group2 = as.character(cmp[2]),
    statistic = unname(test$statistic),
    p.value = as.numeric(test$p.value),
    cramers_v = as.numeric(eff)
  )
})

posthoc_sex <- bind_rows(posthoc_sex_list) %>%
  mutate(p_adj = p.adjust(p.value, method = "BH"),
         signif = p_to_signif(p_adj))

# -----------------------------
# 8) Zusammenfassung + Speichern
# -----------------------------
summary_omnibus_scales <- tibble(
  variable = names(scale_labels),
  test = "Kruskal-Wallis",
  statistic = sapply(kw_scales, function(x) unname(x$statistic)),
  df = sapply(kw_scales, function(x) unname(x$parameter)),
  p.value = sapply(kw_scales, function(x) x$p.value),
  effect_size = unlist(eps2_scales),
  effect_size_type = "epsilon2"
)

summary_omnibus <- bind_rows(
  tibble(
    variable = c("Age", "TIV", "Sex", "Education"),
    test = c("Kruskal-Wallis", "Kruskal-Wallis", "Chi-squared (Monte Carlo)", "Kruskal-Wallis"),
    statistic = c(unname(kw_age$statistic), unname(kw_tiv$statistic), unname(chi_sex$statistic), unname(kw_edu$statistic)),
    df = c(unname(kw_age$parameter), unname(kw_tiv$parameter), unname(chi_sex$parameter), unname(kw_edu$parameter)),
    p.value = c(kw_age$p.value, kw_tiv$p.value, chi_sex$p.value, kw_edu$p.value),
    effect_size = c(eps2_age, eps2_tiv, as.numeric(cramers_v_sex), eps2_edu),
    effect_size_type = c("epsilon2", "epsilon2", "Cramer's V", "epsilon2")
  ),
  summary_omnibus_scales
)

message("\n>>> 5-Gruppen-Vergleich (inkl. HC) - Omnibus-Tests fuer Table 1:")
print(summary_omnibus)

write_csv(summary_omnibus, file.path(output_dir, "omnibus_tests_incl_HC.csv"))
write_csv(dunn_age, file.path(output_dir, "posthoc_age_dunn_BH_incl_HC.csv"))
write_csv(dunn_tiv, file.path(output_dir, "posthoc_tiv_dunn_BH_incl_HC.csv"))
write_csv(dunn_edu, file.path(output_dir, "posthoc_education_dunn_BH_incl_HC.csv"))
write_csv(posthoc_sex, file.path(output_dir, "posthoc_sex_BH_incl_HC.csv"))
write_csv(bind_rows(dunn_scales), file.path(output_dir, "posthoc_clinical_scales_dunn_BH_incl_HC.csv"))

# -----------------------------
# 9) Fertig zusammengebaute Zeilen fuer Table 1 (zum Copy-Paste)
# -----------------------------
table1_hc_row_age <- desc_age %>%
  filter(Diagnosis == "HC") %>%
  transmute(Variable = "Age, Years", HC = formatted)

table1_hc_row_tiv <- desc_tiv %>%
  filter(Diagnosis == "HC") %>%
  transmute(Variable = "TIV, cm3", HC = formatted)

table1_hc_row_edu <- desc_edu %>%
  filter(Diagnosis == "HC") %>%
  transmute(Variable = "Years of education, Years", HC = formatted)

table1_hc_row_sex <- desc_sex %>%
  filter(Diagnosis == "HC") %>%
  tidyr::pivot_wider(names_from = Geschlecht, values_from = n) %>%
  transmute(Variable = "Sex, n", Female = Weiblich, Male = Maennlich)

table1_hc_row_scales <- desc_scales %>%
  filter(Diagnosis == "HC") %>%
  transmute(Variable = scale_labels[Scale], HC = formatted)

message("\n>>> Fertige HC-Zeilen fuer Table 1 (Copy-Paste-Vorlage):")
print(table1_hc_row_age)
print(table1_hc_row_sex)
print(table1_hc_row_tiv)
print(table1_hc_row_edu)
print(table1_hc_row_scales)

write_csv(table1_hc_row_age, file.path(output_dir, "Table1_HC_row_age.csv"))
write_csv(table1_hc_row_sex, file.path(output_dir, "Table1_HC_row_sex.csv"))
write_csv(table1_hc_row_tiv, file.path(output_dir, "Table1_HC_row_tiv.csv"))
write_csv(table1_hc_row_edu, file.path(output_dir, "Table1_HC_row_education.csv"))
write_csv(table1_hc_row_scales, file.path(output_dir, "Table1_HC_row_clinical_scales.csv"))

message("\n[OK] Alle Ergebnisse gespeichert unter: ", output_dir)
