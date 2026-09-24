#!/usr/bin/env Rscript

# Reconstructs all SuStaIn input tables from controlled-access source data.
# No participant-level data are distributed with this repository.

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(readr)
  library(stringr)
  library(tibble)
})

options(contrasts = c("contr.treatment", "contr.poly"))

project_dir <- normalizePath(
  getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd())),
  mustWork = FALSE
)
roi_file <- getOption(
  "SUSTAIN_T1_ROI_FILE",
  Sys.getenv("SUSTAIN_T1_ROI_FILE", unset = file.path(project_dir, "private_data", "merged_data_avg_only.csv"))
)
covariate_file <- getOption(
  "SUSTAIN_COVARIATE_FILE",
  Sys.getenv("SUSTAIN_COVARIATE_FILE", unset = file.path(project_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv"))
)
out_dir <- file.path(project_dir, "HPC", "input")

required_covariates <- c(
  "Alter", "Geschlecht", "TIV", "Dummy_BC_MR_pre", "Dummy_BC_MR_post"
)
group_col <- "Group"
medication_col <- "Sum_MED"
expected_roi_count <- 62L
id_candidates_roi <- c(
  "ID_clean", "names", "Name", "ID", "Proband", "Subject", "subject",
  "subject_id", "participant_id", "PatID", "Patient", "Teilnehmer"
)
id_candidates_covariates <- c(
  "Proband", "ID_clean", "names", "Name", "ID", "Subject", "subject",
  "subject_id", "participant_id", "PatID", "Patient", "Teilnehmer"
)

for (path in c(roi_file, covariate_file)) {
  if (!file.exists(path)) stop("Required controlled-access input not found: ", path)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_character_table <- function(path) {
  delimiters <- c(",", ";", "\t", "|")
  candidates <- lapply(delimiters, function(delimiter) {
    tryCatch(
      suppressWarnings(read_delim(
        path,
        delim = delimiter,
        col_types = cols(.default = col_character()),
        locale = locale(encoding = "UTF-8"),
        show_col_types = FALSE,
        progress = FALSE
      )),
      error = function(e) NULL
    )
  })
  widths <- vapply(candidates, function(x) if (is.null(x)) 0L else ncol(x), integer(1))
  if (max(widths) < 2L) stop("Could not determine delimiter for: ", path)
  candidates[[which.max(widths)]]
}

safe_numeric <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  value <- str_squish(as.character(x))
  value[value %in% c("", "NA", "NaN", "-99", "-99.0", "-99.00", "-99,0", "-99,00")] <- NA_character_
  comma_decimal <- sum(grepl("\\d,\\d", value), na.rm = TRUE) >
    sum(grepl("\\d\\.\\d", value), na.rm = TRUE)
  if (comma_decimal) {
    value <- gsub("\\.", "", value)
    value <- gsub(",", ".", value, fixed = TRUE)
  } else {
    value <- gsub(",", "", value, fixed = TRUE)
  }
  suppressWarnings(as.numeric(gsub(" ", "", value, fixed = TRUE)))
}

normalize_id <- function(x) {
  raw <- str_squish(as.character(x))
  normalized <- str_extract(raw, "\\d+")
  normalized <- ifelse(is.na(normalized), NA_character_, as.character(as.integer(normalized)))
  normalized
}

find_id_column <- function(data, candidates, label) {
  exact <- candidates[candidates %in% names(data)]
  if (length(exact)) return(exact[[1]])
  matched <- names(data)[tolower(names(data)) %in% tolower(candidates)]
  if (length(matched)) return(matched[[1]])
  stop("No supported ID column found in ", label, ".")
}

assert_unique_ids <- function(data, source_name) {
  if (anyNA(data$ID_norm)) stop("Missing or non-numeric IDs after normalization in ", source_name, ".")
  duplicates <- unique(data$ID_norm[duplicated(data$ID_norm)])
  if (length(duplicates)) {
    stop(
      "ID normalization produced duplicate IDs in ", source_name, ": ",
      paste(head(duplicates, 10L), collapse = ", ")
    )
  }
}

coerce_sex <- function(x, levels_reference = NULL) {
  value <- str_squish(as.character(x))
  value[value %in% c("", "NA", "NaN")] <- NA_character_
  value <- sub("\\.0+$", "", value)
  if (is.null(levels_reference)) factor(value) else factor(value, levels = levels_reference)
}

roi_raw <- read_character_table(roi_file)
covariate_raw <- read_character_table(covariate_file)
id_roi <- find_id_column(roi_raw, id_candidates_roi, "ROI input")
id_covariate <- find_id_column(covariate_raw, id_candidates_covariates, "covariate input")

roi_data <- roi_raw %>%
  rename(ID_raw = all_of(id_roi)) %>%
  mutate(ID_norm = normalize_id(ID_raw))
covariate_data <- covariate_raw %>%
  rename(ID_covariate_raw = all_of(id_covariate)) %>%
  mutate(ID_norm = normalize_id(ID_covariate_raw))

assert_unique_ids(roi_data, "ROI input")
assert_unique_ids(covariate_data, "covariate input")

required_columns <- c(group_col, required_covariates, medication_col)
missing_covariates <- setdiff(required_columns, names(covariate_data))
if (length(missing_covariates)) {
  stop("Required covariate columns are missing: ", paste(missing_covariates, collapse = ", "))
}

roi_columns <- names(roi_data)[grepl("_avg$", names(roi_data))]
if (length(roi_columns) != expected_roi_count) {
  stop("Expected exactly ", expected_roi_count, " ROI columns ending in '_avg'; found ", length(roi_columns), ".")
}

unmatched_roi_ids <- anti_join(
  roi_data %>% select(ID_norm), covariate_data %>% select(ID_norm), by = "ID_norm"
)
if (nrow(unmatched_roi_ids)) {
  stop(nrow(unmatched_roi_ids), " ROI records have no matching covariate record.")
}

covariate_small <- covariate_data %>%
  select(ID_norm, all_of(required_columns))
for (column in setdiff(required_columns, "Geschlecht")) {
  covariate_small[[column]] <- safe_numeric(covariate_small[[column]])
}
covariate_small$Geschlecht <- coerce_sex(covariate_small$Geschlecht)

roi_numeric <- roi_data %>% select(ID_raw, ID_norm, all_of(roi_columns))
for (column in roi_columns) roi_numeric[[column]] <- safe_numeric(roi_numeric[[column]])

analysis_data <- inner_join(roi_numeric, covariate_small, by = "ID_norm")
if (!nrow(analysis_data)) stop("Joining ROI and covariate data produced zero rows.")
if (anyDuplicated(analysis_data$ID_norm)) stop("The merged analysis table contains duplicate IDs.")
if (anyNA(analysis_data[[group_col]])) stop("Group contains missing or non-numeric values.")

patient_mask <- analysis_data[[group_col]] != 1
if (anyNA(analysis_data[[medication_col]][patient_mask])) {
  stop("Sum_MED is missing for one or more patients; mixed corrected and uncorrected residuals are not allowed.")
}

fit_one_roi <- function(roi) {
  hc <- analysis_data %>%
    filter(.data[[group_col]] == 1) %>%
    select(all_of(c(roi, required_covariates))) %>%
    filter(if_all(everything(), ~ !is.na(.x)))
  if (nrow(hc) < 30L) stop("Fewer than 30 complete healthy controls for ROI: ", roi)

  step1 <- lm(reformulate(required_covariates, response = roi), data = hc)
  prediction1 <- predict(step1, newdata = analysis_data)
  residual1 <- analysis_data[[roi]] - prediction1

  eligible_patients <- patient_mask & is.finite(residual1)
  medication <- analysis_data[[medication_col]]
  if (sum(eligible_patients) < 30L || n_distinct(medication[eligible_patients]) < 2L) {
    stop("Medication regression cannot be fitted for ROI: ", roi)
  }
  step2_data <- tibble(
    residual1 = residual1[eligible_patients],
    Sum_MED = medication[eligible_patients]
  )
  step2 <- lm(residual1 ~ Sum_MED, data = step2_data)
  residual_final <- residual1
  residual_final[eligible_patients] <- residual1[eligible_patients] -
    predict(step2, newdata = step2_data)

  hc_residual <- residual_final[analysis_data[[group_col]] == 1]
  hc_mean <- mean(hc_residual, na.rm = TRUE)
  hc_sd <- sd(hc_residual, na.rm = TRUE)
  if (!is.finite(hc_sd) || hc_sd < 1e-8) stop("Invalid healthy-control SD for ROI: ", roi)

  list(
    z = -((residual_final - hc_mean) / hc_sd),
    qc = tibble(
      ROI = roi,
      n_hc_step1 = nrow(hc),
      n_patients_step2 = sum(eligible_patients),
      hc_residual_mean = hc_mean,
      hc_residual_sd = hc_sd,
      medication_intercept = unname(coef(step2)[[1]]),
      medication_slope = unname(coef(step2)[[2]])
    )
  )
}

fits <- map(roi_columns, fit_one_roi)
z_matrix <- do.call(cbind, map(fits, "z"))
colnames(z_matrix) <- paste0(roi_columns, "_resid_final")

all_groups <- bind_cols(
  tibble(
    !!id_roi := analysis_data$ID_raw,
    Group = analysis_data[[group_col]]
  ),
  as_tibble(z_matrix)
)
if (anyNA(all_groups)) stop("The reconstructed z-score table contains missing values.")

patients <- all_groups %>% filter(Group != 1)
if (!nrow(patients)) stop("No patients found after excluding Group 1.")

write_csv(all_groups, file.path(out_dir, "residuals_z_scored_negated_paper.csv"))
write_csv(patients, file.path(out_dir, "residuals_z_scored_corrected_allGroups.csv"))
write_csv(bind_rows(map(fits, "qc")), file.path(out_dir, "qc_roi_fit_log_T1_2step.csv"))

diagnosis_groups <- list(MDD = 2, BD = 3, SZ = c(4, 5), Angst = 8)
diagnosis_counts <- imap_dfr(diagnosis_groups, function(codes, diagnosis) {
  subset <- patients %>% filter(Group %in% codes)
  if (!nrow(subset)) stop("No observations found for diagnosis: ", diagnosis)
  write_csv(subset, file.path(out_dir, paste0("residuals_z_scored_mainflip_", diagnosis, ".csv")))
  tibble(diagnosis = diagnosis, group_codes = paste(codes, collapse = ","), n = nrow(subset))
})

z_columns <- setdiff(names(patients), c(id_roi, group_col))
absolute_z <- abs(as.matrix(patients[z_columns]))
zmax <- tibble(
  ROI = z_columns,
  q90 = apply(absolute_z, 2, quantile, probs = 0.90, type = 7, names = FALSE),
  q95 = apply(absolute_z, 2, quantile, probs = 0.95, type = 7, names = FALSE),
  q99 = apply(absolute_z, 2, quantile, probs = 0.99, type = 7, names = FALSE),
  max_abs = apply(absolute_z, 2, max),
  mean_abs = colMeans(absolute_z),
  sd_abs = apply(absolute_z, 2, sd)
) %>% mutate(Zmax = ceiling(q95))
write_csv(zmax, file.path(out_dir, "Zmax_global_patients_only.csv"))
write_csv(diagnosis_counts, file.path(out_dir, "qc_diagnosis_subset_counts.csv"))

message("Created all SuStaIn input tables in: ", normalizePath(out_dir, mustWork = FALSE))
message("Patients: ", nrow(patients), "; ROIs: ", length(z_columns))
