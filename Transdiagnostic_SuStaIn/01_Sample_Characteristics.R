# ============================================================================== 
# Gemeinsame Stichprobencharakteristika: HC und alle Patientengruppen
# Replaces the former separate patient and healthy-control analyses.
#
# Auswertungsregeln:
#   1. Gemeinsame Tabelle mit HC, MDD, BD, SZ und Angst.
#   2. HAMD wird ausschliesslich aus HAMD_Sum17 gelesen. Kein Item-Fallback.
#   3. Akut, sobald mindestens ein Status-Trigger positiv ist.
#   4. Remittiert nur bei vollstaendigen Triggerdaten ohne positiven Trigger.
#   5. Omnibus-p-Werte bleiben unkorrigiert.
#   6. Post-hoc-p-Werte werden je Variable nach Benjamini-Hochberg korrigiert.
#   7. Alter, Geschlecht, Bildung, TIV und Symptomskalen: Vergleich aller 5 Gruppen.
#   8. Erkrankungsbeginn, Erkrankungsdauer, Komorbiditaet und Status:
#      Vergleich nur der 4 Patientengruppen, da diese Variablen fuer HC nicht
#      sinnvoll definiert sind.
# ============================================================================== 

rm(list = ls())

# ------------------------------------------------------------------------------
# 1) Pfade
# ------------------------------------------------------------------------------
base_dir <- getOption(
  "SUSTAIN_PROJECT_DIR",
  Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd())
)

data_path <- getOption(
  "SUSTAIN_CLINICAL_STATUS_FILE",
  Sys.getenv(
    "SUSTAIN_CLINICAL_STATUS_FILE",
    unset = file.path(
      base_dir,
      "private_data",
      "Datenbank_Update_DataFreeze_bereinigt_withGlobalRatings.csv"
    )
  )
)

output_dir <- file.path(
  base_dir,
  "Output Scripte",
  "Sample_Characteristics"
)

output_xlsx <- file.path(
  output_dir,
  "Sample_Characteristics.xlsx"
)

output_csv <- file.path(
  output_dir,
  "Sample_Characteristics.csv"
)

if (!file.exists(data_path)) stop("Eingabedatei nicht gefunden: ", data_path)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ------------------------------------------------------------------------------
# 2) Pakete
# ------------------------------------------------------------------------------
required_packages <- c(
  "dplyr", "readr", "tidyr", "purrr", "rstatix", "DescTools", "openxlsx"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop("Fehlende R-Pakete: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(rstatix)
  library(DescTools)
  library(openxlsx)
})

set.seed(20260916L)

# ------------------------------------------------------------------------------
# 3) Definitionen
# ------------------------------------------------------------------------------
alpha <- 0.05
group_levels <- c("HC", "MDD", "BD", "SZ", "Angst")
patient_levels <- c("MDD", "BD", "SZ", "Angst")

sans_items <- c(
  "SANS1", "SANS2", "SANS3", "SANS4", "SANS5", "SANS6", "SANS8",
  "SANS9", "SANS10", "SANS11", "SANS13", "SANS14", "SANS15",
  "SANS17", "SANS18", "SANS19", "SANS20", "SANS22", "SANS23"
)

saps_items <- c(
  "SAPS1", "SAPS2", "SAPS3", "SAPS4", "SAPS5", "SAPS6", "SAPS8",
  "SAPS9", "SAPS10", "SAPS11", "SAPS12", "SAPS13", "SAPS14",
  "SAPS15", "SAPS16", "SAPS17", "SAPS18", "SAPS19", "SAPS21",
  "SAPS22", "SAPS23", "SAPS24", "SAPS26", "SAPS27", "SAPS28",
  "SAPS29", "SAPS30", "SAPS31", "SAPS32", "SAPS33", "SAPS35"
)

hama_items <- paste0("HAMA", 1:14)
ymrs_items <- paste0("YMRS", 1:11)

status_global_items <- c(
  "SANS7", "SANS12", "SANS16", "SANS21",
  "SAPS7", "SAPS20", "SAPS25", "SAPS34"
)

required_columns <- c(
  "Group", "Alter", "Geschlecht", "Bildungsjahre", "TIV", "Komorbid",
  "AgeOfOnset", "HAMA_Sum", "HAMD_Sum17", "YMRS_Sum",
  sans_items, saps_items, hama_items, ymrs_items, status_global_items
)

# ------------------------------------------------------------------------------
# 4) Hilfsfunktionen
# ------------------------------------------------------------------------------
as_numeric_clean <- function(x) {
  out <- suppressWarnings(as.numeric(gsub(",", ".", trimws(as.character(x)), fixed = TRUE)))
  out[out %in% c(-99, -2)] <- NA_real_
  out
}

row_sum_observed <- function(data, variables) {
  values <- dplyr::select(data, dplyr::all_of(variables))
  n_observed <- rowSums(!is.na(values))
  result <- rowSums(values, na.rm = TRUE)
  result[n_observed == 0] <- NA_real_
  as.numeric(result)
}

format_p <- function(p, prefix = "p") {
  if (is.na(p)) return(paste0(prefix, " = n.a."))
  if (p < 0.001) return(paste0(prefix, " < 0.001"))
  paste0(prefix, " = ", format(round(p, 3), nsmall = 3))
}

format_continuous <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return("n.a.")
  paste0(
    "N=", length(x),
    "; M=", format(round(mean(x), 2), nsmall = 2),
    " (SD=", format(round(stats::sd(x), 2), nsmall = 2), ")",
    "; Median=", format(round(stats::median(x), 2), nsmall = 2),
    " [IQR=", format(round(stats::IQR(x), 2), nsmall = 2), "]"
  )
}

format_categorical <- function(x, levels_to_show = NULL) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0) return("n.a.")
  if (is.null(levels_to_show)) levels_to_show <- sort(unique(x))
  counts <- table(factor(x, levels = levels_to_show))
  percentages <- 100 * counts / sum(counts)
  paste0(
    names(counts), ": n=", as.integer(counts),
    " (", format(round(percentages, 2), nsmall = 2), "%)",
    collapse = "; "
  )
}

format_categorical_total_denominator <- function(x, levels_to_show) {
  denominator <- length(x)
  if (denominator == 0) return("n.a.")
  observed <- as.character(x)
  observed <- observed[!is.na(observed) & nzchar(observed)]
  counts <- table(factor(observed, levels = levels_to_show))
  percentages <- 100 * counts / denominator
  paste0(
    names(counts), ": n=", as.integer(counts),
    " (", format(round(percentages, 2), nsmall = 2), "%)",
    collapse = "; "
  )
}

posthoc_summary <- function(posthoc_df) {
  if (nrow(posthoc_df) == 0) return("")
  significant <- posthoc_df %>% filter(!is.na(p_adj), p_adj < alpha)
  if (nrow(significant) == 0) return("")
  paste0(
    significant$group1, " vs ", significant$group2,
    " (", vapply(significant$p_adj, format_p, character(1), prefix = "q"), ")",
    collapse = "; "
  )
}

status_flag_gt <- function(x, threshold) {
  ifelse(is.na(x), NA_real_, ifelse(x > threshold, 1, 0))
}

status_flag_ge <- function(x, threshold) {
  ifelse(is.na(x), NA_real_, ifelse(x >= threshold, 1, 0))
}

# ------------------------------------------------------------------------------
# 5) Daten einlesen und vorbereiten
# ------------------------------------------------------------------------------
data_all <- suppressWarnings(readr::read_csv(data_path, show_col_types = FALSE))

missing_columns <- setdiff(required_columns, names(data_all))
if (length(missing_columns) > 0) {
  stop("Fehlende Pflichtspalten: ", paste(missing_columns, collapse = ", "))
}

numeric_columns <- unique(c(
  "Group", "Alter", "Geschlecht", "Bildungsjahre", "TIV", "Komorbid",
  "AgeOfOnset", "HAMA_Sum", "HAMD_Sum17", "YMRS_Sum",
  sans_items, saps_items, hama_items, ymrs_items, status_global_items
))

data_all <- data_all %>%
  mutate(across(all_of(numeric_columns), as_numeric_clean)) %>%
  mutate(
    Gruppe = case_when(
      Group == 1 ~ "HC",
      Group == 2 ~ "MDD",
      Group == 3 ~ "BD",
      Group %in% c(4, 5) ~ "SZ",
      Group %in% c(7, 8) ~ "Angst",
      TRUE ~ NA_character_
    ),
    Gruppe = factor(Gruppe, levels = group_levels),
    Geschlecht_label = case_when(
      Geschlecht == 1 ~ "Male",
      Geschlecht == 2 ~ "Female",
      TRUE ~ NA_character_
    ),
    Komorbid_label = case_when(
      Komorbid == 0 ~ "No",
      Komorbid == 1 ~ "Yes",
      TRUE ~ NA_character_
    ),
    DurationOfIllness = ifelse(
      !is.na(Alter) & !is.na(AgeOfOnset) & Alter >= AgeOfOnset,
      Alter - AgeOfOnset,
      NA_real_
    ),
    Sum_SANS = row_sum_observed(., sans_items),
    Sum_SAPS = row_sum_observed(., saps_items),
    Sum_HAMA = row_sum_observed(., hama_items),
    Sum_YMRS = row_sum_observed(., ymrs_items),
    Sum_HAMD17 = HAMD_Sum17
  ) %>%
  filter(!is.na(Gruppe))

# ------------------------------------------------------------------------------
# 6) Akut versus remittiert, nur Patienten
# ------------------------------------------------------------------------------
status_flags <- cbind(
  status_flag_gt(data_all$SANS7, 2),
  status_flag_gt(data_all$SANS12, 2),
  status_flag_gt(data_all$SANS16, 2),
  status_flag_gt(data_all$SANS21, 2),
  status_flag_gt(data_all$SAPS7, 2),
  status_flag_gt(data_all$SAPS20, 2),
  status_flag_gt(data_all$SAPS25, 2),
  status_flag_gt(data_all$SAPS34, 2),
  status_flag_gt(data_all$HAMA_Sum, 19),
  status_flag_gt(data_all$HAMD_Sum17, 6),
  status_flag_ge(data_all$YMRS_Sum, 4)
)

any_positive <- apply(status_flags, 1, function(z) any(z == 1, na.rm = TRUE))
all_observed <- apply(status_flags, 1, function(z) all(is.finite(z)))

data_all <- data_all %>%
  mutate(
    Status = case_when(
      Gruppe == "HC" ~ NA_character_,
      any_positive ~ "acute",
      all_observed ~ "remittiert",
      TRUE ~ NA_character_
    ),
    Status = factor(Status, levels = c("remittiert", "acute"))
  )

# ------------------------------------------------------------------------------
# 7) Variablenplan
# ------------------------------------------------------------------------------
continuous_plan <- tibble::tribble(
  ~variable,            ~label,                       ~scope,
  "Alter",              "Age, years",                "all_5_groups",
  "AgeOfOnset",         "Age at onset, years",       "patients_only",
  "DurationOfIllness",  "Duration of illness, years","patients_only",
  "Bildungsjahre",      "Years of education",        "all_5_groups",
  "TIV",                "TIV",                       "all_5_groups",
  "Sum_HAMD17",         "HAM-D 17 total score",      "all_5_groups",
  "Sum_SANS",           "SANS total score",          "all_5_groups",
  "Sum_SAPS",           "SAPS total score",          "all_5_groups",
  "Sum_HAMA",           "HAM-A total score",         "all_5_groups",
  "Sum_YMRS",           "YMRS total score",          "all_5_groups"
)

categorical_plan <- tibble::tribble(
  ~variable,             ~label,                   ~scope,          ~levels_text,
  "Geschlecht_label",    "Sex",                    "all_5_groups", "Female|Male",
  "Komorbid_label",      "Comorbidity",            "patients_only", "No|Yes",
  "Status",              "Clinical status",        "patients_only", "remittiert|acute"
)

# ------------------------------------------------------------------------------
# 8) Stetige Variablen
# ------------------------------------------------------------------------------
continuous_main <- list()
continuous_omnibus <- list()
continuous_posthoc <- list()

for (i in seq_len(nrow(continuous_plan))) {
  variable <- continuous_plan$variable[i]
  label <- continuous_plan$label[i]
  scope <- continuous_plan$scope[i]
  comparison_groups <- if (scope == "all_5_groups") group_levels else patient_levels

  analysis_data <- data_all %>%
    filter(as.character(Gruppe) %in% comparison_groups) %>%
    transmute(Gruppe = droplevels(Gruppe), value = .data[[variable]]) %>%
    filter(is.finite(value))

  if (nrow(analysis_data) == 0 || n_distinct(analysis_data$Gruppe) < 2) {
    omnibus_p <- NA_real_
    statistic <- NA_real_
    df <- NA_real_
    posthoc <- tibble::tibble()
  } else {
    omnibus <- kruskal.test(value ~ Gruppe, data = analysis_data)
    omnibus_p <- as.numeric(omnibus$p.value)
    statistic <- as.numeric(omnibus$statistic)
    df <- as.numeric(omnibus$parameter)

    posthoc <- if (omnibus_p < alpha) {
      rstatix::dunn_test(analysis_data, value ~ Gruppe, p.adjust.method = "BH") %>%
        transmute(
          Variable = label,
          Scope = scope,
          group1,
          group2,
          n1,
          n2,
          statistic,
          p_raw = p,
          p_adj = p.adj,
          significant_BH = p.adj < alpha
        )
    } else {
      tibble::tibble()
    }
  }

  descriptions <- vapply(group_levels, function(group_name) {
    if (scope == "patients_only" && group_name == "HC") return("n.a.")
    format_continuous(data_all[[variable]][as.character(data_all$Gruppe) == group_name])
  }, character(1))

  continuous_main[[variable]] <- tibble::tibble(
    Variable = label,
    HC = descriptions["HC"],
    MDD = descriptions["MDD"],
    BD = descriptions["BD"],
    SZ = descriptions["SZ"],
    Angst = descriptions["Angst"],
    Comparison_scope = ifelse(scope == "all_5_groups", "HC + 4 patient groups", "4 patient groups"),
    Omnibus_raw = paste0(
      format_p(omnibus_p),
      " (H = ", ifelse(is.na(statistic), "n.a.", format(round(statistic, 2), nsmall = 2)),
      "; Kruskal-Wallis)"
    ),
    Significant_posthoc_BH = posthoc_summary(posthoc)
  )

  continuous_omnibus[[variable]] <- tibble::tibble(
    Variable = label,
    Variable_type = "continuous",
    Comparison_scope = ifelse(scope == "all_5_groups", "HC + 4 patient groups", "4 patient groups"),
    Test = "Kruskal-Wallis",
    Statistic = statistic,
    df = df,
    p_raw = omnibus_p,
    Significant_raw = !is.na(omnibus_p) && omnibus_p < alpha
  )

  if (nrow(posthoc) > 0) continuous_posthoc[[variable]] <- posthoc
}

# ------------------------------------------------------------------------------
# 9) Kategoriale Variablen
# ------------------------------------------------------------------------------
categorical_main <- list()
categorical_omnibus <- list()
categorical_posthoc <- list()

for (i in seq_len(nrow(categorical_plan))) {
  variable <- categorical_plan$variable[i]
  label <- categorical_plan$label[i]
  scope <- categorical_plan$scope[i]
  levels_to_show <- strsplit(categorical_plan$levels_text[i], "\\|", fixed = FALSE)[[1]]
  comparison_groups <- if (scope == "all_5_groups") group_levels else patient_levels
  skip_inference <- identical(variable, "Komorbid_label")

  analysis_data <- data_all %>%
    filter(as.character(Gruppe) %in% comparison_groups) %>%
    transmute(Gruppe = factor(as.character(Gruppe), levels = comparison_groups), value = as.character(.data[[variable]])) %>%
    filter(!is.na(value), nzchar(value))

  contingency <- table(analysis_data$value, analysis_data$Gruppe)

  if (skip_inference) {
    omnibus_p <- NA_real_
    statistic <- NA_real_
    posthoc <- tibble::tibble()
  } else if (nrow(contingency) < 2 || ncol(contingency) < 2) {
    omnibus_p <- NA_real_
    statistic <- NA_real_
    posthoc <- tibble::tibble()
  } else {
    omnibus <- suppressWarnings(chisq.test(contingency, simulate.p.value = TRUE, B = 20000))
    omnibus_p <- as.numeric(omnibus$p.value)
    statistic <- as.numeric(omnibus$statistic)

    posthoc_rows <- list()
    if (omnibus_p < alpha) {
      group_pairs <- combn(comparison_groups, 2, simplify = FALSE)
      for (pair in group_pairs) {
        pair_data <- analysis_data %>% filter(as.character(Gruppe) %in% pair)
        pair_table <- table(pair_data$value, factor(pair_data$Gruppe, levels = pair))
        p_pair <- if (nrow(pair_table) < 2 || ncol(pair_table) < 2) {
          NA_real_
        } else {
          as.numeric(fisher.test(pair_table)$p.value)
        }
        posthoc_rows[[paste(pair, collapse = "_")]] <- tibble::tibble(
          Variable = label,
          Scope = scope,
          group1 = pair[1],
          group2 = pair[2],
          Test = "Fisher exact",
          p_raw = p_pair
        )
      }
    }

    posthoc <- bind_rows(posthoc_rows)
    if (nrow(posthoc) > 0) {
      posthoc <- posthoc %>%
        mutate(
          p_adj = p.adjust(p_raw, method = "BH"),
          significant_BH = p_adj < alpha
        )
    }
  }

  descriptions <- vapply(group_levels, function(group_name) {
    if (scope == "patients_only" && group_name == "HC") return("n.a.")
    group_values <- data_all[[variable]][as.character(data_all$Gruppe) == group_name]
    if (skip_inference) {
      format_categorical_total_denominator(group_values, levels_to_show)
    } else {
      format_categorical(group_values, levels_to_show)
    }
  }, character(1))

  categorical_main[[variable]] <- tibble::tibble(
    Variable = label,
    HC = descriptions["HC"],
    MDD = descriptions["MDD"],
    BD = descriptions["BD"],
    SZ = descriptions["SZ"],
    Angst = descriptions["Angst"],
    Comparison_scope = ifelse(
      skip_inference,
      "Not tested by design",
      ifelse(scope == "all_5_groups", "HC + 4 patient groups", "4 patient groups")
    ),
    Omnibus_raw = ifelse(
      skip_inference,
      "n.a. (not tested by design)",
      paste0(
        format_p(omnibus_p),
        " (Chi-square = ", ifelse(is.na(statistic), "n.a.", format(round(statistic, 2), nsmall = 2)),
        "; Monte Carlo B=20000)"
      )
    ),
    Significant_posthoc_BH = posthoc_summary(posthoc)
  )

  categorical_omnibus[[variable]] <- tibble::tibble(
    Variable = label,
    Variable_type = "categorical",
    Comparison_scope = ifelse(
      skip_inference,
      "Not tested by design",
      ifelse(scope == "all_5_groups", "HC + 4 patient groups", "4 patient groups")
    ),
    Test = ifelse(skip_inference, "Not tested by design", "Chi-square, Monte Carlo B=20000"),
    Statistic = statistic,
    df = NA_real_,
    p_raw = omnibus_p,
    Significant_raw = !is.na(omnibus_p) && omnibus_p < alpha
  )

  if (nrow(posthoc) > 0) categorical_posthoc[[variable]] <- posthoc
}

# ------------------------------------------------------------------------------
# 10) Gemeinsame Tabelle und QC
# ------------------------------------------------------------------------------
main_table <- bind_rows(c(continuous_main, categorical_main))
omnibus_table <- bind_rows(c(continuous_omnibus, categorical_omnibus))
posthoc_continuous_table <- bind_rows(unname(continuous_posthoc))
posthoc_categorical_table <- bind_rows(unname(categorical_posthoc))

missingness_table <- bind_rows(
  map_dfr(continuous_plan$variable, function(variable) {
    data_all %>%
      group_by(Gruppe) %>%
      summarise(
        N_total = n(),
        Missing_N = sum(is.na(.data[[variable]])),
        Missing_percent = 100 * Missing_N / N_total,
        .groups = "drop"
      ) %>%
      mutate(Variable = variable, .before = 1)
  }),
  map_dfr(categorical_plan$variable, function(variable) {
    data_all %>%
      group_by(Gruppe) %>%
      summarise(
        N_total = n(),
        Missing_N = sum(is.na(.data[[variable]])),
        Missing_percent = 100 * Missing_N / N_total,
        .groups = "drop"
      ) %>%
      mutate(Variable = variable, .before = 1)
  })
)

status_counts <- data_all %>%
  filter(as.character(Gruppe) %in% patient_levels) %>%
  mutate(Status_display = ifelse(is.na(Status), "nicht bestimmbar", as.character(Status))) %>%
  count(Gruppe, Status_display, name = "N") %>%
  group_by(Gruppe) %>%
  mutate(Percent_total_group = 100 * N / sum(N)) %>%
  ungroup()

method_table <- tibble::tribble(
  ~Bereich, ~Festlegung,
  "Stichprobe", "HC, MDD, BD, SZ und Angst in einer gemeinsamen Tabelle.",
  "HAMD", "Ausschliesslich HAMD_Sum17. Keine Itemberechnung und kein Fallback.",
  "Akut", "Mindestens einer der elf Status-Trigger ist positiv.",
  "Remittiert", "Alle elf Status-Trigger liegen vor und keiner ist positiv.",
  "Nicht bestimmbar", "Kein positiver Trigger, aber mindestens ein fehlender Trigger. Keine Umkodierung zu remittiert.",
  "Omnibus", "Unkorrigierte p-Werte bei alpha = 0.05.",
  "Post-hoc", "Dunn fuer stetige und Fisher fuer kategoriale Variablen. BH-Korrektur getrennt je Variable.",
  "Vergleich mit HC", "Alter, Geschlecht, Bildung, TIV und Symptomskalen werden ueber alle fuenf Gruppen verglichen.",
  "Patientenvergleich", "Erkrankungsbeginn, Erkrankungsdauer, Komorbiditaet und Status werden nur ueber MDD, BD, SZ und Angst verglichen."
)

readr::write_csv(main_table, output_csv)
readr::write_csv(omnibus_table, file.path(output_dir, "Omnibus_Tests_raw_p.csv"))
readr::write_csv(posthoc_continuous_table, file.path(output_dir, "Posthoc_Dunn_BH.csv"))
readr::write_csv(posthoc_categorical_table, file.path(output_dir, "Posthoc_Fisher_BH.csv"))
readr::write_csv(missingness_table, file.path(output_dir, "Missingness_QC.csv"))
readr::write_csv(status_counts, file.path(output_dir, "Status_QC.csv"))

# ------------------------------------------------------------------------------
# 11) Excel-Datei direkt aus diesem R-Skript
# ------------------------------------------------------------------------------
workbook <- openxlsx::createWorkbook()

sheet_data <- list(
  "Gemeinsame Tabelle" = main_table,
  "Omnibus" = omnibus_table,
  "Posthoc stetig" = posthoc_continuous_table,
  "Posthoc kategorial" = posthoc_categorical_table,
  "Missingness QC" = missingness_table,
  "Status QC" = status_counts,
  "Methodik" = method_table
)

title_style <- openxlsx::createStyle(
  fontName = "Arial", fontSize = 15, textDecoration = "bold",
  fontColour = "#1F2937"
)

header_style <- openxlsx::createStyle(
  fontName = "Arial", fontSize = 10, textDecoration = "bold",
  fontColour = "#FFFFFF", fgFill = "#1F4E78",
  halign = "center", valign = "center", wrapText = TRUE,
  border = "Bottom", borderColour = "#FFFFFF"
)

body_style <- openxlsx::createStyle(
  fontName = "Arial", fontSize = 10, valign = "center"
)

wrap_style <- openxlsx::createStyle(
  fontName = "Arial", fontSize = 10, valign = "center", wrapText = TRUE
)

percent_style <- openxlsx::createStyle(numFmt = "0.00")
p_style <- openxlsx::createStyle(numFmt = "0.000E+00")

for (sheet_name in names(sheet_data)) {
  table_data <- sheet_data[[sheet_name]]
  openxlsx::addWorksheet(workbook, sheet_name, gridLines = FALSE)
  openxlsx::writeData(workbook, sheet_name, sheet_name, startRow = 2, startCol = 1)
  openxlsx::addStyle(workbook, sheet_name, title_style, rows = 2, cols = 1)
  openxlsx::writeData(
    workbook, sheet_name, table_data,
    startRow = 5, startCol = 1,
    headerStyle = header_style,
    withFilter = TRUE
  )
  if (nrow(table_data) > 0) {
    openxlsx::addStyle(
      workbook, sheet_name, body_style,
      rows = 6:(5 + nrow(table_data)),
      cols = seq_len(ncol(table_data)),
      gridExpand = TRUE
    )
  }
  openxlsx::freezePane(workbook, sheet_name, firstActiveRow = 6, firstActiveCol = 2)
  openxlsx::setColWidths(workbook, sheet_name, cols = seq_len(max(1, ncol(table_data))), widths = "auto")
}

# Haupttabelle lesbar formatieren.
openxlsx::addStyle(
  workbook, "Gemeinsame Tabelle", wrap_style,
  rows = 6:(5 + nrow(main_table)), cols = 1:ncol(main_table), gridExpand = TRUE
)
openxlsx::setColWidths(workbook, "Gemeinsame Tabelle", cols = 1, widths = 28)
openxlsx::setColWidths(workbook, "Gemeinsame Tabelle", cols = 2:6, widths = 42)
openxlsx::setColWidths(workbook, "Gemeinsame Tabelle", cols = 7, widths = 24)
openxlsx::setColWidths(workbook, "Gemeinsame Tabelle", cols = 8, widths = 34)
openxlsx::setColWidths(workbook, "Gemeinsame Tabelle", cols = 9, widths = 65)
openxlsx::setRowHeights(workbook, "Gemeinsame Tabelle", rows = 6:(5 + nrow(main_table)), heights = 58)

# Numerische Formate.
if (nrow(omnibus_table) > 0) {
  p_col <- match("p_raw", names(omnibus_table))
  openxlsx::addStyle(
    workbook, "Omnibus", p_style,
    rows = 6:(5 + nrow(omnibus_table)), cols = p_col, gridExpand = TRUE
  )
}

if (nrow(posthoc_continuous_table) > 0) {
  p_cols <- match(c("p_raw", "p_adj"), names(posthoc_continuous_table))
  openxlsx::addStyle(
    workbook, "Posthoc stetig", p_style,
    rows = 6:(5 + nrow(posthoc_continuous_table)), cols = p_cols, gridExpand = TRUE
  )
}

if (nrow(posthoc_categorical_table) > 0) {
  p_cols <- match(c("p_raw", "p_adj"), names(posthoc_categorical_table))
  openxlsx::addStyle(
    workbook, "Posthoc kategorial", p_style,
    rows = 6:(5 + nrow(posthoc_categorical_table)), cols = p_cols, gridExpand = TRUE
  )
}

if (nrow(missingness_table) > 0) {
  percent_col <- match("Missing_percent", names(missingness_table))
  openxlsx::addStyle(
    workbook, "Missingness QC", percent_style,
    rows = 6:(5 + nrow(missingness_table)), cols = percent_col, gridExpand = TRUE
  )
}

openxlsx::saveWorkbook(workbook, output_xlsx, overwrite = TRUE)

# ------------------------------------------------------------------------------
# 12) Abschlusskontrollen
# ------------------------------------------------------------------------------
expected_counts <- c(HC = 1062L, MDD = 764L, BD = 164L, SZ = 155L, Angst = 233L)
observed_counts <- table(factor(data_all$Gruppe, levels = group_levels))

if (!identical(as.integer(observed_counts), as.integer(expected_counts))) {
  warning(
    "Gruppengroessen weichen von der aktuellen Referenz ab. Beobachtet: ",
    paste(names(observed_counts), as.integer(observed_counts), sep = "=", collapse = ", ")
  )
}

if (any(is.na(data_all$Sum_HAMD17))) {
  warning("HAMD_Sum17 enthaelt fehlende Werte. Es wurde kein Fallback verwendet.")
}

message("\n[OK] Gemeinsames Skript erfolgreich ausgefuehrt.")
message("[OK] Gemeinsame Excel-Tabelle: ", output_xlsx)
message("[OK] Gemeinsame CSV-Tabelle:   ", output_csv)
message("[OK] HAMD-Quelle: ausschliesslich HAMD_Sum17, ohne Fallback.")
message("[OK] Omnibus: raw p. Post-hoc: BH-korrigiert je Variable.")
