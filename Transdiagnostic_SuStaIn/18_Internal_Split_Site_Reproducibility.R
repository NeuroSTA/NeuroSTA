#!/usr/bin/env Rscript
# ============================================================
# 18. Internal split-site reproducibility
# ============================================================
# Evaluate the frozen transdiagnostic T1 SuStaIn model across Marburg
# (site 1) and Muenster (site 2) using the existing subtype and stage assignments.
# Reconstruct the model input and check residual site offsets in healthy controls.
# Compare subtype frequencies, adjusted stages, and regional GMV profiles.
# Use Spearman correlations and permute site labels within subtype, within
# subtype and diagnosis, and within subtype, diagnosis, and exact stage.

options(stringsAsFactors = FALSE, warn = 1)

project_dir <- Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd())
cfg <- list(
  raw_roi_file = Sys.getenv("SUSTAIN_T1_ROI_FILE", unset = file.path(project_dir, "private_data", "merged_data_avg_only.csv")),
  covariate_file = Sys.getenv("SUSTAIN_COVARIATE_FILE", unset = file.path(project_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt.csv")),
  sustain_input_file = file.path(project_dir, "HPC", "input", "residuals_z_scored_corrected_allGroups.csv"),
  assignment_file = file.path(project_dir, "Output_sustainrun", "SuStaIn_assignments", "main_subject_subtype_assignment_3subtypes.csv"),
  control_group = 1,
  expected_roi_count = 62L,
  expected_subtypes = 3L,
  n_permutations = 10000L,
  hc_crossfit_folds = 10L,
  seed = 20261110L,
  exact_tolerance = 1e-8
)

out_dir <- file.path(project_dir, "Output_sustainrun", "internal_split_site_reproducibility")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

required_packages <- c("readr", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) stop("Missing R packages: ", paste(missing_packages, collapse = ", "))

for (path in unlist(cfg[c("raw_roi_file", "covariate_file", "sustain_input_file", "assignment_file")])) {
  if (!file.exists(path)) stop("Input file not found: ", path)
}

detect_delimiter <- function(path) {
  first <- readLines(path, n = 1L, warn = FALSE, encoding = "UTF-8")
  candidates <- c(",", ";", "\t", "|")
  counts <- vapply(candidates, function(x) {
    lengths(regmatches(first, gregexpr(x, first, fixed = TRUE)))
  }, integer(1))
  candidates[which.max(counts)]
}

read_table_auto <- function(path) {
  sep <- detect_delimiter(path)
  dec <- if (identical(sep, ";")) "," else "."
  out <- suppressWarnings(readr::read_delim(
    path,
    delim = sep,
    col_types = readr::cols(.default = readr::col_character()),
    locale = readr::locale(decimal_mark = dec, encoding = "UTF-8"),
    na = c("", "NA", "NaN", "-99", "-99.0", "-99,0"),
    name_repair = "minimal",
    trim_ws = TRUE,
    show_col_types = FALSE,
    progress = FALSE
  ))
  as.data.frame(out, check.names = FALSE)
}

safe_numeric <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  y <- trimws(as.character(x))
  y[y %in% c("", "NA", "NaN", "-99", "-99.0", "-99.00", "-99,0", "-99,00")] <- NA_character_
  comma_decimal <- sum(grepl("[0-9],[0-9]", y), na.rm = TRUE) >
    sum(grepl("[0-9]\\.[0-9]", y), na.rm = TRUE)
  if (comma_decimal) {
    y <- gsub(".", "", y, fixed = TRUE)
    y <- gsub(",", ".", y, fixed = TRUE)
  } else {
    y <- gsub(",", "", y, fixed = TRUE)
  }
  suppressWarnings(as.numeric(gsub(" ", "", y, fixed = TRUE)))
}

normalize_id <- function(x) {
  y <- trimws(as.character(x))
  m <- regexpr("[0-9]+", y)
  out <- rep(NA_character_, length(y))
  ok <- !is.na(y) & m > 0
  out[ok] <- as.character(as.integer(regmatches(y, m)[ok]))
  out
}

find_column <- function(data, candidates, label) {
  exact <- candidates[candidates %in% names(data)]
  if (length(exact)) return(exact[1])
  idx <- match(tolower(candidates), tolower(names(data)), nomatch = 0L)
  idx <- idx[idx > 0L]
  if (length(idx)) return(names(data)[idx[1]])
  stop("Missing column for ", label, ". Candidate names: ", paste(candidates, collapse = ", "))
}

write_csv_plain <- function(x, filename) {
  utils::write.csv(x, file.path(out_dir, filename), row.names = FALSE, na = "")
}

checks <- data.frame(
  check = character(), pass = logical(), value = character(), criterion = character()
)
add_check <- function(check, pass, value, criterion) {
  checks <<- rbind(checks, data.frame(
    check = check,
    pass = isTRUE(pass),
    value = as.character(value),
    criterion = as.character(criterion)
  ))
}

hc3_table <- function(model, model_name) {
  X <- stats::model.matrix(model)
  e <- stats::residuals(model)
  h <- stats::hatvalues(model)
  bread <- solve(crossprod(X))
  adjusted <- e / pmax(1 - h, .Machine$double.eps)
  meat <- crossprod(X * adjusted)
  vc <- bread %*% meat %*% bread
  estimate <- stats::coef(model)
  se <- sqrt(diag(vc))
  statistic <- estimate / se
  p <- 2 * stats::pt(abs(statistic), df = stats::df.residual(model), lower.tail = FALSE)
  data.frame(
    model = model_name,
    term = names(estimate),
    estimate = unname(estimate),
    std_error_HC3 = unname(se),
    statistic = unname(statistic),
    df = stats::df.residual(model),
    p_value = unname(p),
    conf_low_95 = unname(estimate - stats::qt(0.975, stats::df.residual(model)) * se),
    conf_high_95 = unname(estimate + stats::qt(0.975, stats::df.residual(model)) * se),
    r_squared = summary(model)$r.squared,
    adjusted_r_squared = summary(model)$adj.r.squared
  )
}

cat("Reading raw GMV, covariates, SuStaIn input, and assignments...\n")
raw <- read_table_auto(cfg$raw_roi_file)
covar <- read_table_auto(cfg$covariate_file)
sustain_input <- read_table_auto(cfg$sustain_input_file)
assignment <- read_table_auto(cfg$assignment_file)

id_candidates <- c("Proband", "names", "Name", "ID_clean", "ID", "Subject", "participant_id")
id_raw <- find_column(raw, id_candidates, "raw GMV ID")
id_cov <- find_column(covar, id_candidates, "covariate ID")
id_input <- find_column(sustain_input, id_candidates, "SuStaIn input ID")
id_assignment <- find_column(assignment, id_candidates, "assignment ID")

raw$ID_norm <- normalize_id(raw[[id_raw]])
covar$ID_norm <- normalize_id(covar[[id_cov]])
sustain_input$ID_norm <- normalize_id(sustain_input[[id_input]])
assignment$ID_norm <- normalize_id(assignment[[id_assignment]])

for (object_name in c("raw", "covar", "sustain_input", "assignment")) {
  object <- get(object_name)
  if (anyNA(object$ID_norm)) stop(object_name, ": participant ID could not be normalized.")
  if (anyDuplicated(object$ID_norm)) stop(object_name, ": duplicate participant ID.")
}

raw_roi_cols <- grep("_avg$", names(raw), value = TRUE)
input_roi_cols <- grep("_resid_final$", names(sustain_input), value = TRUE)
if (length(raw_roi_cols) != cfg$expected_roi_count) {
  stop("Unexpected number of raw GMV ROIs: ", length(raw_roi_cols))
}
if (length(input_roi_cols) != cfg$expected_roi_count) {
  stop("Unexpected number of SuStaIn ROIs: ", length(input_roi_cols))
}
input_to_raw <- setNames(sub("_resid_final$", "", input_roi_cols), input_roi_cols)
if (!setequal(unname(input_to_raw), raw_roi_cols)) stop("Raw GMV and SuStaIn ROI names do not match.")

required_covariates <- c(
  "Group", "Site", "Alter", "Geschlecht", "TIV",
  "Dummy_BC_MR_pre", "Dummy_BC_MR_post", "Sum_MED"
)
missing_covariates <- setdiff(required_covariates, names(covar))
if (length(missing_covariates)) {
  stop("Missing covariate columns: ", paste(missing_covariates, collapse = ", "))
}

raw_small <- raw[, c("ID_norm", unname(input_to_raw)), drop = FALSE]
names(raw_small)[-1] <- input_roi_cols
cov_small <- covar[, c("ID_norm", required_covariates), drop = FALSE]
for (nm in setdiff(required_covariates, "Geschlecht")) cov_small[[nm]] <- safe_numeric(cov_small[[nm]])
cov_small$Geschlecht <- factor(trimws(as.character(cov_small$Geschlecht)))
for (nm in input_roi_cols) raw_small[[nm]] <- safe_numeric(raw_small[[nm]])

idx_cov <- match(raw_small$ID_norm, cov_small$ID_norm)
if (anyNA(idx_cov)) stop("Some raw GMV participant IDs are missing from the covariate file.")
analysis <- cbind(raw_small, cov_small[idx_cov, setdiff(names(cov_small), "ID_norm"), drop = FALSE])

if (!setequal(unique(analysis$Site), c(1, 2))) stop("Expected both site codes 1 and 2 and no other site codes.")
if (anyNA(analysis[, required_covariates])) stop("At least one required covariate has missing values.")

# The scanner indicators determine site: either Marburg indicator identifies
# site 1, and two zero indicators identify site 2.
site_from_scanner <- ifelse(
  analysis$Dummy_BC_MR_pre == 1 | analysis$Dummy_BC_MR_post == 1,
  1,
  2
)
site_scanner_exact <- identical(as.numeric(analysis$Site), as.numeric(site_from_scanner))
add_check(
  "Site is determined by the scanner indicators",
  site_scanner_exact,
  paste0(sum(analysis$Site == site_from_scanner), "/", nrow(analysis)),
  paste0(nrow(analysis), "/", nrow(analysis))
)
if (!site_scanner_exact) stop("Site codes and scanner indicators are inconsistent.")

cat("Reconstructing the 62 HC-standardized GMV z-scores...\n")
natural_gmv_z <- matrix(
  NA_real_,
  nrow = nrow(analysis),
  ncol = length(input_roi_cols),
  dimnames = list(NULL, input_roi_cols)
)
patient_mask <- analysis$Group != cfg$control_group

for (j in seq_along(input_roi_cols)) {
  roi <- input_roi_cols[j]
  hc_mask <- analysis$Group == cfg$control_group & complete.cases(
    analysis[, c(roi, "Alter", "Geschlecht", "TIV", "Dummy_BC_MR_pre", "Dummy_BC_MR_post"), drop = FALSE]
  )
  hc <- analysis[hc_mask, , drop = FALSE]
  step1 <- stats::lm(
    stats::reformulate(
      c("Alter", "Geschlecht", "TIV", "Dummy_BC_MR_pre", "Dummy_BC_MR_post"),
      response = roi
    ),
    data = hc
  )
  residual1 <- analysis[[roi]] - stats::predict(step1, newdata = analysis)

  eligible <- patient_mask & is.finite(residual1) & is.finite(analysis$Sum_MED)
  medication_model <- stats::lm(residual1 ~ Sum_MED, data = data.frame(
    residual1 = residual1[eligible],
    Sum_MED = analysis$Sum_MED[eligible]
  ))
  residual_final <- residual1
  residual_final[eligible] <- residual1[eligible] - stats::predict(
    medication_model,
    newdata = data.frame(Sum_MED = analysis$Sum_MED[eligible])
  )

  hc_residual <- residual_final[analysis$Group == cfg$control_group]
  hc_mean <- mean(hc_residual)
  hc_sd <- stats::sd(hc_residual)
  if (!is.finite(hc_sd) || hc_sd < 1e-8) stop("Invalid healthy-control standard deviation for ", roi)
  natural_gmv_z[, j] <- (residual_final - hc_mean) / hc_sd
}

# Compare the reconstructed values with the sign-inverted patient input used for modeling.
idx_input <- match(sustain_input$ID_norm, analysis$ID_norm)
if (anyNA(idx_input)) stop("SuStaIn input contains unknown participant IDs.")
saved_input <- as.matrix(data.frame(
  lapply(sustain_input[, input_roi_cols, drop = FALSE], safe_numeric),
  check.names = FALSE
))
reconstructed_sustain <- -natural_gmv_z[idx_input, input_roi_cols, drop = FALSE]
max_input_error <- max(abs(saved_input - reconstructed_sustain))
add_check(
  "Reconstruction matches the SuStaIn input",
  max_input_error <= cfg$exact_tolerance,
  format(max_input_error, scientific = TRUE),
  paste0("maximum absolute error <= ", cfg$exact_tolerance)
)
if (max_input_error > cfg$exact_tolerance) stop("Reconstructed values differ from the saved model input.")

# Check site offsets in healthy controls using non-inverted GMV z-scores.
hc_rows <- which(analysis$Group == cfg$control_group)
hc_site_rows <- split(hc_rows, analysis$Site[hc_rows])
hc_roi_results <- do.call(rbind, lapply(names(hc_site_rows), function(site_code) {
  rows <- hc_site_rows[[site_code]]
  data.frame(
    Site = as.integer(site_code),
    Site_label = ifelse(site_code == "1", "Marburg", "Muenster"),
    ROI = input_roi_cols,
    n_HC = length(rows),
    mean_natural_GMV_z = colMeans(natural_gmv_z[rows, , drop = FALSE]),
    sd_natural_GMV_z = apply(natural_gmv_z[rows, , drop = FALSE], 2, stats::sd)
  )
}))
hc_summary <- do.call(rbind, lapply(split(hc_roi_results, hc_roi_results$Site), function(d) {
  data.frame(
    Site = d$Site[1],
    Site_label = d$Site_label[1],
    n_HC = d$n_HC[1],
    mean_across_ROI_means = mean(d$mean_natural_GMV_z),
    mean_absolute_ROI_mean = mean(abs(d$mean_natural_GMV_z)),
    root_mean_square_ROI_mean = sqrt(mean(d$mean_natural_GMV_z^2)),
    max_absolute_ROI_mean = max(abs(d$mean_natural_GMV_z))
  )
}))
write_csv_plain(hc_roi_results, "01_HC_site_offset_by_ROI.csv")
write_csv_plain(hc_summary, "02_HC_site_offset_summary.csv")
add_check(
  "Site-specific healthy-control means are numerically close to zero",
  max(abs(hc_roi_results$mean_natural_GMV_z)) <= cfg$exact_tolerance,
  format(max(abs(hc_roi_results$mean_natural_GMV_z)), scientific = TRUE),
  paste0("maximum absolute ROI mean <= ", cfg$exact_tolerance)
)

# Scanner indicators constrain the in-sample healthy-control means to zero.
# For the cross-fitted check, fit nuisance models on nine folds and apply them
# to the held-out fold. Balance folds within the three scanner categories.
set.seed(cfg$seed)
scanner_category <- ifelse(
  analysis$Dummy_BC_MR_pre == 1,
  "Marburg_pre",
  ifelse(analysis$Dummy_BC_MR_post == 1, "Marburg_post", "Muenster")
)
hc_fold <- rep(NA_integer_, nrow(analysis))
for (category in unique(scanner_category[hc_rows])) {
  rows <- hc_rows[scanner_category[hc_rows] == category]
  hc_fold[rows] <- sample(rep(seq_len(cfg$hc_crossfit_folds), length.out = length(rows)))
}
hc_crossfit_z <- matrix(
  NA_real_, nrow = length(hc_rows), ncol = length(input_roi_cols),
  dimnames = list(analysis$ID_norm[hc_rows], input_roi_cols)
)
for (j in seq_along(input_roi_cols)) {
  roi <- input_roi_cols[j]
  for (fold in seq_len(cfg$hc_crossfit_folds)) {
    test_rows <- hc_rows[hc_fold[hc_rows] == fold]
    train_rows <- hc_rows[hc_fold[hc_rows] != fold]
    crossfit_model <- stats::lm(
      stats::reformulate(
        c("Alter", "Geschlecht", "TIV", "Dummy_BC_MR_pre", "Dummy_BC_MR_post"),
        response = roi
      ),
      data = analysis[train_rows, , drop = FALSE]
    )
    train_residual <- analysis[[roi]][train_rows] - stats::predict(
      crossfit_model, newdata = analysis[train_rows, , drop = FALSE]
    )
    test_residual <- analysis[[roi]][test_rows] - stats::predict(
      crossfit_model, newdata = analysis[test_rows, , drop = FALSE]
    )
    test_z <- (test_residual - mean(train_residual)) / stats::sd(train_residual)
    hc_crossfit_z[match(analysis$ID_norm[test_rows], rownames(hc_crossfit_z)), j] <- test_z
  }
}
if (anyNA(hc_crossfit_z) || any(!is.finite(hc_crossfit_z))) stop("Healthy-control cross-fitting produced missing or non-finite values.")
hc_crossfit_site <- analysis$Site[match(rownames(hc_crossfit_z), analysis$ID_norm)]
hc_crossfit_roi <- do.call(rbind, lapply(c(1, 2), function(site) {
  rows <- which(hc_crossfit_site == site)
  data.frame(
    Site = site,
    Site_label = ifelse(site == 1, "Marburg", "Muenster"),
    ROI = input_roi_cols,
    n_HC = length(rows),
    mean_crossfit_natural_GMV_z = colMeans(hc_crossfit_z[rows, , drop = FALSE]),
    sd_crossfit_natural_GMV_z = apply(hc_crossfit_z[rows, , drop = FALSE], 2, stats::sd)
  )
}))
hc_crossfit_summary <- do.call(rbind, lapply(split(hc_crossfit_roi, hc_crossfit_roi$Site), function(d) {
  data.frame(
    Site = d$Site[1],
    Site_label = d$Site_label[1],
    n_HC = d$n_HC[1],
    mean_across_ROI_means = mean(d$mean_crossfit_natural_GMV_z),
    mean_absolute_ROI_mean = mean(abs(d$mean_crossfit_natural_GMV_z)),
    root_mean_square_ROI_mean = sqrt(mean(d$mean_crossfit_natural_GMV_z^2)),
    max_absolute_ROI_mean = max(abs(d$mean_crossfit_natural_GMV_z))
  )
}))
write_csv_plain(hc_crossfit_roi, "02b_HC_site_offset_crossfit_by_ROI.csv")
write_csv_plain(hc_crossfit_summary, "02c_HC_site_offset_crossfit_summary.csv")
add_check(
  "Healthy-control cross-fitting is complete",
  all(is.finite(hc_crossfit_z)),
  paste0(sum(is.finite(hc_crossfit_z)), "/", length(hc_crossfit_z)),
  paste0(length(hc_crossfit_z), "/", length(hc_crossfit_z))
)

# Join assignments, covariates, and reconstructed non-inverted GMV z-scores.
idx_assignment <- match(assignment$ID_norm, analysis$ID_norm)
if (anyNA(idx_assignment)) stop("The assignment file contains unknown participant IDs.")
patient_data <- data.frame(
  ID_norm = assignment$ID_norm,
  Group = safe_numeric(assignment$Group),
  ML_Subtype = safe_numeric(assignment$ML_Subtype),
  ML_Stage = safe_numeric(assignment$ML_Stage),
  Prob_ML_Subtype = safe_numeric(assignment$Prob_ML_Subtype),
  Site = analysis$Site[idx_assignment],
  Dummy_BC_MR_pre = analysis$Dummy_BC_MR_pre[idx_assignment],
  Dummy_BC_MR_post = analysis$Dummy_BC_MR_post[idx_assignment],
  Alter = analysis$Alter[idx_assignment],
  Geschlecht = as.character(analysis$Geschlecht[idx_assignment]),
  stringsAsFactors = FALSE
)
patient_z <- natural_gmv_z[idx_assignment, input_roi_cols, drop = FALSE]

if (!identical(patient_data$Group, analysis$Group[idx_assignment])) stop("Group codes differ between the assignment and covariate files.")
if (!setequal(unique(patient_data$ML_Subtype), seq_len(cfg$expected_subtypes))) stop("Unexpected subtype labels in the assignment file.")

diagnosis_labels <- c(`2` = "MDD", `3` = "BD", `4` = "SSD", `5` = "SSD", `8` = "ANX")
patient_data$Diagnosis <- unname(diagnosis_labels[as.character(patient_data$Group)])
if (anyNA(patient_data$Diagnosis)) stop("At least one patient has an unrecognized diagnosis code.")
patient_data$Site_label <- factor(patient_data$Site, levels = c(1, 2), labels = c("Marburg", "Muenster"))
patient_data$Diagnosis_f <- factor(patient_data$Diagnosis, levels = c("MDD", "BD", "SSD", "ANX"))
patient_data$Geschlecht_f <- factor(patient_data$Geschlecht)
patient_data$Subtype_f <- factor(patient_data$ML_Subtype)
patient_data$Scanner_f <- factor(
  ifelse(
    patient_data$Dummy_BC_MR_pre == 1,
    "Marburg_pre",
    ifelse(patient_data$Dummy_BC_MR_post == 1, "Marburg_post", "Muenster")
  ),
  levels = c("Muenster", "Marburg_pre", "Marburg_post")
)

write_csv_plain(patient_data, "03_patient_analysis_data.csv")

# Subtype composition by site.
composition <- as.data.frame(with(patient_data, table(Site_label, ML_Subtype)))
names(composition) <- c("Site", "ML_Subtype", "n")
composition$site_total <- ave(composition$n, composition$Site, FUN = sum)
composition$percent_within_site <- 100 * composition$n / composition$site_total
write_csv_plain(composition, "04_subtype_composition_by_site.csv")

composition_table <- with(patient_data, table(Site_label, ML_Subtype))
chi <- suppressWarnings(stats::chisq.test(composition_table, correct = FALSE))
fisher <- stats::fisher.test(composition_table)
cramers_v <- sqrt(unname(chi$statistic) / (sum(composition_table) * min(dim(composition_table) - 1L)))
composition_tests <- data.frame(
  test = c("Pearson chi-square", "Fisher exact"),
  statistic = c(unname(chi$statistic), NA_real_),
  df = c(unname(chi$parameter), NA_real_),
  p_value = c(chi$p.value, fisher$p.value),
  minimum_expected_count = c(min(chi$expected), NA_real_),
  cramers_v = c(cramers_v, NA_real_)
)
write_csv_plain(composition_tests, "05_subtype_by_site_tests.csv")

# Report the diagnostic composition at each site.
diagnosis_composition <- as.data.frame(with(patient_data, table(Site_label, Diagnosis_f)))
names(diagnosis_composition) <- c("Site", "Diagnosis", "n")
write_csv_plain(diagnosis_composition, "06_diagnosis_composition_by_site.csv")
diagnosis_table <- with(patient_data, table(Site_label, Diagnosis_f))
diagnosis_chi <- suppressWarnings(stats::chisq.test(diagnosis_table, correct = FALSE))
diagnosis_cramers_v <- sqrt(
  unname(diagnosis_chi$statistic) /
    (sum(diagnosis_table) * min(dim(diagnosis_table) - 1L))
)
diagnosis_site_test <- data.frame(
  test = "Pearson chi-square",
  statistic = unname(diagnosis_chi$statistic),
  df = unname(diagnosis_chi$parameter),
  p_value = diagnosis_chi$p.value,
  minimum_expected_count = min(diagnosis_chi$expected),
  cramers_v = diagnosis_cramers_v
)
write_csv_plain(diagnosis_site_test, "06b_diagnosis_by_site_test.csv")

# Stage models adjusted for diagnosis, age, and sex, with sensitivity models.
model_requested <- stats::lm(
  ML_Stage ~ Site_label + Diagnosis_f + Alter + Geschlecht_f,
  data = patient_data
)
model_subtype_adjusted <- stats::lm(
  ML_Stage ~ Site_label + Diagnosis_f + Alter + Geschlecht_f + Subtype_f,
  data = patient_data
)
model_scanner_adjusted <- stats::lm(
  ML_Stage ~ Scanner_f + Diagnosis_f + Alter + Geschlecht_f + Subtype_f,
  data = patient_data
)
stage_models <- rbind(
  hc3_table(model_requested, "requested: site + diagnosis + age + sex"),
  hc3_table(model_subtype_adjusted, "sensitivity: requested + subtype"),
  hc3_table(model_scanner_adjusted, "sensitivity: three scanner groups + diagnosis + age + sex + subtype")
)
write_csv_plain(stage_models, "07_stage_site_models_HC3.csv")

stage_descriptives <- do.call(rbind, lapply(
  split(patient_data, interaction(patient_data$Site_label, patient_data$ML_Subtype, drop = TRUE)),
  function(d) data.frame(
    Site = as.character(d$Site_label[1]),
    ML_Subtype = d$ML_Subtype[1],
    n = nrow(d),
    mean_stage = mean(d$ML_Stage),
    sd_stage = stats::sd(d$ML_Stage),
    median_stage = stats::median(d$ML_Stage),
    q1_stage = unname(stats::quantile(d$ML_Stage, 0.25)),
    q3_stage = unname(stats::quantile(d$ML_Stage, 0.75))
  )
))
write_csv_plain(stage_descriptives, "08_stage_descriptives_site_subtype.csv")

# Regional GMV profiles by site and assigned subtype.
profile_rows <- list()
counter <- 1L
for (subtype in seq_len(cfg$expected_subtypes)) {
  for (site in c(1, 2)) {
    rows <- which(patient_data$ML_Subtype == subtype & patient_data$Site == site)
    if (length(rows) < 20L) stop("Too few participants for subtype ", subtype, " at site ", site)
    profile_rows[[counter]] <- data.frame(
      ML_Subtype = subtype,
      Site = site,
      Site_label = ifelse(site == 1, "Marburg", "Muenster"),
      ROI = input_roi_cols,
      n = length(rows),
      mean_natural_GMV_z = colMeans(patient_z[rows, , drop = FALSE])
    )
    counter <- counter + 1L
  }
}
profiles <- do.call(rbind, profile_rows)
write_csv_plain(profiles, "09_site_subtype_ROI_profiles.csv")

profile_vector <- function(subtype, site) {
  d <- profiles[profiles$ML_Subtype == subtype & profiles$Site == site, ]
  d <- d[match(input_roi_cols, d$ROI), ]
  if (anyNA(d$mean_natural_GMV_z)) stop("Incomplete regional GMV profile.")
  d$mean_natural_GMV_z
}

same_subtype_similarity <- do.call(rbind, lapply(seq_len(cfg$expected_subtypes), function(subtype) {
  mr <- profile_vector(subtype, 1)
  ms <- profile_vector(subtype, 2)
  data.frame(
    ML_Subtype = subtype,
    n_Marburg = sum(patient_data$ML_Subtype == subtype & patient_data$Site == 1),
    n_Muenster = sum(patient_data$ML_Subtype == subtype & patient_data$Site == 2),
    Spearman = stats::cor(mr, ms, method = "spearman")
  )
}))

permute_similarity <- function(X, site, diagnosis, stage, n_permutations, scheme) {
  n <- nrow(X)
  n_site1 <- sum(site == 1)
  total <- colSums(X)
  spearman <- numeric(n_permutations)
  for (b in seq_len(n_permutations)) {
    if (identical(scheme, "within_subtype_and_diagnosis")) {
      permuted_site <- site
      for (diagnosis_level in unique(diagnosis)) {
        idx <- which(diagnosis == diagnosis_level)
        permuted_site[idx] <- site[idx[sample.int(length(idx))]]
      }
      idx1 <- which(permuted_site == 1)
    } else if (identical(scheme, "within_subtype_diagnosis_and_exact_stage")) {
      permuted_site <- site
      strata <- interaction(diagnosis, stage, drop = TRUE)
      for (stratum in levels(strata)) {
        idx <- which(strata == stratum)
        permuted_site[idx] <- site[idx[sample.int(length(idx))]]
      }
      idx1 <- which(permuted_site == 1)
    } else {
      idx1 <- sample.int(n, n_site1, replace = FALSE)
    }
    sum1 <- colSums(X[idx1, , drop = FALSE])
    mean1 <- sum1 / length(idx1)
    mean2 <- (total - sum1) / (n - length(idx1))
    spearman[b] <- suppressWarnings(stats::cor(mean1, mean2, method = "spearman"))
  }
  data.frame(iteration = seq_len(n_permutations), Spearman = spearman)
}

set.seed(cfg$seed)
cat("Permuting site labels...\n")
permutation_rows <- list()
permutation_summary_rows <- list()
counter <- 1L
for (subtype in seq_len(cfg$expected_subtypes)) {
  rows <- which(patient_data$ML_Subtype == subtype)
  X <- patient_z[rows, , drop = FALSE]
  site <- patient_data$Site[rows]
  diagnosis <- patient_data$Diagnosis[rows]
  stage <- patient_data$ML_Stage[rows]
  observed <- same_subtype_similarity[same_subtype_similarity$ML_Subtype == subtype, ]

  for (scheme in c(
    "within_subtype",
    "within_subtype_and_diagnosis",
    "within_subtype_diagnosis_and_exact_stage"
  )) {
    perm <- permute_similarity(
      X = X,
      site = site,
      diagnosis = diagnosis,
      stage = stage,
      n_permutations = cfg$n_permutations,
      scheme = scheme
    )
    perm$ML_Subtype <- subtype
    perm$scheme <- scheme
    permutation_rows[[counter]] <- perm

    obs <- observed$Spearman
    values <- perm$Spearman
    permutation_summary_rows[[counter]] <- data.frame(
      ML_Subtype = subtype,
      scheme = scheme,
      metric = "Spearman",
      observed = obs,
      permutation_mean = mean(values),
      permutation_sd = stats::sd(values),
      permutation_q025 = unname(stats::quantile(values, 0.025)),
      permutation_median = stats::median(values),
      permutation_q975 = unname(stats::quantile(values, 0.975)),
      percentile_of_observed = mean(values <= obs),
      p_lower_observed_unusually_low = (1 + sum(values <= obs)) / (length(values) + 1)
    )
    counter <- counter + 1L
  }
}
permutation_values <- do.call(rbind, permutation_rows)
permutation_summary <- do.call(rbind, permutation_summary_rows)
permutation_summary$p_lower_BH_within_scheme <- ave(
  permutation_summary$p_lower_observed_unusually_low,
  permutation_summary$scheme,
  FUN = function(p) stats::p.adjust(p, method = "BH")
)
write_csv_plain(same_subtype_similarity, "10_same_subtype_site_similarity.csv")
write_csv_plain(permutation_summary, "11_site_label_permutation_summary.csv")
write_csv_plain(permutation_values, "12_site_label_permutation_values.csv")

# Cross-site Spearman correlations for all subtype pairs.
similarity_long <- do.call(rbind, lapply(seq_len(cfg$expected_subtypes), function(mr_subtype) {
  do.call(rbind, lapply(seq_len(cfg$expected_subtypes), function(ms_subtype) {
    mr <- profile_vector(mr_subtype, 1)
    ms <- profile_vector(ms_subtype, 2)
    data.frame(
      Marburg_Subtype = mr_subtype,
      Muenster_Subtype = ms_subtype,
      same_subtype = mr_subtype == ms_subtype,
      Spearman = stats::cor(mr, ms, method = "spearman")
    )
  }))
}))
write_csv_plain(similarity_long, "13_all_subtype_cross_site_similarity.csv")

dominance <- do.call(rbind, lapply(seq_len(cfg$expected_subtypes), function(subtype) {
  row <- similarity_long[similarity_long$Marburg_Subtype == subtype, ]
  diagonal <- row[row$Muenster_Subtype == subtype, "Spearman"]
  alternatives <- row[row$Muenster_Subtype != subtype, "Spearman"]
  data.frame(
    metric = "Spearman",
    Marburg_Subtype = subtype,
    same_subtype_similarity = diagonal,
    maximum_different_subtype_similarity = max(alternatives),
    margin_same_minus_best_different = diagonal - max(alternatives),
    same_subtype_is_row_maximum = diagonal > max(alternatives)
  )
}))
write_csv_plain(dominance, "14_same_vs_different_subtype_dominance.csv")

# Exact test over the six possible permutations of Muenster subtype labels.
all_permutations <- function(x) {
  if (length(x) == 1L) return(matrix(x, nrow = 1L))
  do.call(rbind, lapply(seq_along(x), function(i) {
    cbind(x[i], all_permutations(x[-i]))
  }))
}
label_permutations <- all_permutations(seq_len(cfg$expected_subtypes))
M <- matrix(NA_real_, cfg$expected_subtypes, cfg$expected_subtypes)
for (i in seq_len(cfg$expected_subtypes)) {
  for (j in seq_len(cfg$expected_subtypes)) {
    M[i, j] <- similarity_long[
      similarity_long$Marburg_Subtype == i & similarity_long$Muenster_Subtype == j,
      "Spearman"
    ]
  }
}
scores <- apply(label_permutations, 1L, function(permutation) mean(M[cbind(seq_len(nrow(M)), permutation)]))
observed <- mean(diag(M))
global_label_test <- data.frame(
  metric = "Spearman",
  observed_mean_diagonal = observed,
  mean_off_diagonal = mean(M[row(M) != col(M)]),
  observed_rank_among_six_labelings = rank(-scores, ties.method = "min")[1],
  exact_p_upper_six_labelings = mean(scores >= observed),
  note = "Only 3! = 6 labelings; minimum possible exact p is 1/6. Interpret descriptively."
)
write_csv_plain(global_label_test, "15_global_subtype_label_alignment_test.csv")

# Figures.
profile_plot_data <- profiles
profile_plot_data$ROI_index <- match(profile_plot_data$ROI, input_roi_cols)
p_profiles <- ggplot2::ggplot(
  profile_plot_data,
  ggplot2::aes(x = ROI_index, y = mean_natural_GMV_z, colour = Site_label)
) +
  ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey50") +
  ggplot2::geom_line(linewidth = 0.55) +
  ggplot2::facet_wrap(~ML_Subtype, ncol = 1, scales = "free_y", labeller = ggplot2::label_both) +
  ggplot2::scale_colour_manual(values = c(Marburg = "#D55E00", Muenster = "#0072B2")) +
  ggplot2::labs(
    x = "ROI index in model input order",
    y = "Mean natural-direction GMV z",
    colour = "Site",
    title = "Site-split GMV profiles within fixed TDM subtypes"
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), legend.position = "bottom")
ggplot2::ggsave(
  file.path(out_dir, "Figure_1_site_split_profiles.png"),
  p_profiles, width = 10, height = 9, units = "in", dpi = 300, bg = "white"
)

heatmap_data <- similarity_long
heatmap_data$label <- sprintf("%.2f", heatmap_data$Spearman)
p_heatmap <- ggplot2::ggplot(
  heatmap_data,
  ggplot2::aes(x = factor(Muenster_Subtype), y = factor(Marburg_Subtype), fill = Spearman)
) +
  ggplot2::geom_tile(colour = "white", linewidth = 0.8) +
  ggplot2::geom_text(ggplot2::aes(label = label), size = 4) +
  ggplot2::scale_fill_gradient2(
    low = "#B2182B", mid = "white", high = "#2166AC", midpoint = 0,
    limits = c(-1, 1)
  ) +
  ggplot2::labs(
    x = "Muenster subtype",
    y = "Marburg subtype",
    fill = "Spearman rho",
    title = "Cross-site subtype-profile similarity"
  ) +
  ggplot2::coord_equal() +
  ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(panel.grid = ggplot2::element_blank(), legend.position = "bottom")
ggplot2::ggsave(
  file.path(out_dir, "Figure_2_cross_site_similarity_matrices.png"),
  p_heatmap, width = 9, height = 4.7, units = "in", dpi = 300, bg = "white"
)

p_hc <- ggplot2::ggplot(
  hc_crossfit_roi,
  ggplot2::aes(x = Site_label, y = mean_crossfit_natural_GMV_z, colour = Site_label)
) +
  ggplot2::geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey30") +
  ggplot2::geom_jitter(width = 0.08, height = 0, alpha = 0.7, size = 1.7) +
  ggplot2::scale_colour_manual(values = c(Marburg = "#D55E00", Muenster = "#0072B2")) +
  ggplot2::labs(
    x = NULL,
    y = "Cross-fitted HC mean natural-direction GMV z per ROI",
    title = "Cross-fitted healthy-control site offset check"
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(legend.position = "none", panel.grid.minor = ggplot2::element_blank())
ggplot2::ggsave(
  file.path(out_dir, "Figure_3_HC_site_offset_check.png"),
  p_hc, width = 6.5, height = 4.8, units = "in", dpi = 300, bg = "white"
)

write_csv_plain(checks, "00_QC_checks.csv")
failed <- checks$check[!checks$pass]
site_terms <- stage_models[grepl("^Site_label|^Scanner_f", stage_models$term), ]
summary_lines <- c(
  "T1 internal split-site reproducibility analysis",
  "",
  "Design: frozen TDM K=3 assignments; no SuStaIn refit.",
  "Site 1 = Marburg; Site 2 = Muenster.",
  "This is an internal site-split reproducibility analysis, not external validation.",
  "",
  paste0("Patients: ", nrow(patient_data), "; HC: ", length(hc_rows)),
  paste0("Permutations per subtype and scheme: ", cfg$n_permutations),
  paste0("HC cross-fitting folds: ", cfg$hc_crossfit_folds),
  paste0("Subtype x site chi-square p: ", format(chi$p.value, digits = 5)),
  paste0("Subtype x site Fisher p: ", format(fisher$p.value, digits = 5)),
  paste0("Cramer's V: ", format(cramers_v, digits = 4)),
  paste0("Diagnosis x site chi-square p: ", format(diagnosis_chi$p.value, digits = 5)),
  paste0("Diagnosis x site Cramer's V: ", format(diagnosis_cramers_v, digits = 4)),
  "",
  "Stage site and scanner terms with HC3 standard errors:",
  capture.output(print(site_terms, row.names = FALSE)),
  "",
  "Same-subtype cross-site similarity:",
  capture.output(print(same_subtype_similarity, row.names = FALSE)),
  "",
  "Same-subtype versus different-subtype dominance:",
  capture.output(print(dominance, row.names = FALSE)),
  "",
  paste0("QC status: ", if (length(failed)) "FAILED" else "ALL IMPLEMENTED CHECKS PASSED"),
  if (length(failed)) paste0("Failed: ", paste(failed, collapse = "; ")) else "Failed: none",
  "",
  "Interpretation limits:",
  "Site and scanner are not separable here: Marburg contains pre/post scanner categories and Muenster is the reference scanner category.",
  "The original HC model already adjusted for these scanner categories, so this analysis checks residual site structure after that adjustment.",
  "The in-sample HC means are exactly zero by construction; the cross-fitted HC output is the informative offset check.",
  "A high observed profile correlation is descriptive support for site reproducibility, not proof of equivalence or external generalizability.",
  "The lower-tail permutation p asks whether the actual site split is unusually less similar than random splits of the same subtype.",
  "Lower-tail Spearman p-values are BH-adjusted across the three subtypes within each permutation scheme.",
  "The diagnosis-stratified permutation addresses diagnostic case mix.",
  "The exact-stage-and-diagnosis-stratified permutation additionally holds the stage distribution fixed and is the most specific site-pattern sensitivity check."
)
writeLines(summary_lines, file.path(out_dir, "16_README_results.txt"), useBytes = TRUE)

cat("Results written to: ", out_dir, "\n", sep = "")
cat("QC: ", sum(checks$pass), "/", nrow(checks), " checks passed.\n", sep = "")
if (length(failed)) {
  cat("Failed checks: ", paste(failed, collapse = "; "), "\n", sep = "")
  quit(status = 2L)
}
