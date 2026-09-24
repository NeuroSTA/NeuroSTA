#!/usr/bin/env Rscript

## ============================================================
## CVIC analysis for all cohorts
## - Reads cvic_matrix.npy and loglike_matrix.npy
## - Computes mean CVIC across folds
## - Identifies:
##   1) best K by minimum mean CVIC
##   2) knee K by maximum distance to line on normalized CVIC curve
##   3) maximum curvature K on normalized CVIC curve
##   4) simplest K within 0.5% of the minimum mean CVIC
## - Exports summary CSVs and elbow PDF
##
## PATHS:
##   base_dir = <project>/HPC
##   out_dir  = <project>/Output_sustainrun_reanalysis
## ============================================================

suppressPackageStartupMessages({
  library(RcppCNPy)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})

## ------------------------------------------------------------
## 1) Base and output directories
## ------------------------------------------------------------

project_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
base_dir <- file.path(project_dir, "HPC")
out_dir  <- file.path(project_dir, "Output_sustainrun_reanalysis")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("BASE_DIR:\n", base_dir, "\n")
cat("OUT_DIR :\n", out_dir, "\n\n")

## ------------------------------------------------------------
## 2) Paths to cross-validation results per cohort
## ------------------------------------------------------------

cohort_dirs <- list(
  main  = file.path(base_dir, "SuStaIn_CV_3fold", "results"),
  MDD   = file.path(base_dir, "SuStaIn_diag", "MDD", "folds", "results"),
  BD    = file.path(base_dir, "SuStaIn_diag", "BD", "folds", "results"),
  SZ    = file.path(base_dir, "SuStaIn_diag", "SZ", "folds", "results"),
  Angst = file.path(base_dir, "SuStaIn_diag", "Angst", "folds", "results")
)

## ------------------------------------------------------------
## 3) Helper functions
## ------------------------------------------------------------

normalize_minmax <- function(x) {
  x <- as.numeric(x)
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2])) {
    return(rep(NA_real_, length(x)))
  }
  if (diff(rng) == 0) {
    return(rep(0, length(x)))
  }
  (x - rng[1]) / diff(rng)
}

point_line_distance <- function(x, y) {
  ## Computes perpendicular distance of each point (x,y)
  ## to the line between the first and last point.
  ##
  ## Works on normalized coordinates.
  
  x <- as.numeric(x)
  y <- as.numeric(y)
  
  x1 <- x[1]
  y1 <- y[1]
  x2 <- x[length(x)]
  y2 <- y[length(y)]
  
  denom <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
  if (!is.finite(denom) || denom == 0) {
    return(rep(NA_real_, length(x)))
  }
  
  abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
}

discrete_curvature <- function(x, y) {
  ## Computes discrete curvature for ordered points.
  ## Endpoints receive NA because curvature needs a left and right neighbor.
  
  x <- as.numeric(x)
  y <- as.numeric(y)
  n <- length(x)
  
  out <- rep(NA_real_, n)
  
  if (n < 3) {
    return(out)
  }
  
  for (i in 2:(n - 1)) {
    x1 <- x[i - 1]
    y1 <- y[i - 1]
    x2 <- x[i]
    y2 <- y[i]
    x3 <- x[i + 1]
    y3 <- y[i + 1]
    
    a <- sqrt((x2 - x1)^2 + (y2 - y1)^2)
    b <- sqrt((x3 - x2)^2 + (y3 - y2)^2)
    c <- sqrt((x3 - x1)^2 + (y3 - y1)^2)
    
    if (!is.finite(a) || !is.finite(b) || !is.finite(c) || a == 0 || b == 0 || c == 0) {
      out[i] <- NA_real_
      next
    }
    
    ## Triangle area from determinant
    area2 <- abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    
    ## Curvature = 4 * area / (a*b*c)
    ## Since area2 = 2*area, curvature = 2*area2 / (a*b*c)
    out[i] <- 2 * area2 / (a * b * c)
  }
  
  out
}

get_knee_summary <- function(df_one_cohort, tolerance_pct = 0.5) {
  ## Expects columns:
  ## cohort, K, mean_CVIC
  
  df <- df_one_cohort %>%
    arrange(K) %>%
    mutate(
      K_norm = normalize_minmax(K),
      CVIC_norm_raw = normalize_minmax(mean_CVIC),
      
      ## CVIC lower is better. For plotting and knee detection,
      ## use improvement scale where higher values indicate better fit.
      ## This transforms a decreasing CVIC curve into an increasing fit curve.
      Fit_norm = 1 - CVIC_norm_raw,
      
      knee_distance = point_line_distance(K_norm, Fit_norm),
      curvature = discrete_curvature(K_norm, Fit_norm)
    )
  
  best_cvic <- min(df$mean_CVIC, na.rm = TRUE)
  
  df <- df %>%
    mutate(
      delta_CVICmin = mean_CVIC - best_cvic,
      pct_worse_vs_best = (mean_CVIC - best_cvic) / best_cvic * 100,
      within_0_5pct_best = pct_worse_vs_best <= tolerance_pct
    )
  
  best_row <- df %>%
    filter(mean_CVIC == min(mean_CVIC, na.rm = TRUE)) %>%
    slice(1)
  
  knee_row <- df %>%
    filter(!is.na(knee_distance)) %>%
    filter(knee_distance == max(knee_distance, na.rm = TRUE)) %>%
    slice(1)
  
  curvature_row <- df %>%
    filter(!is.na(curvature)) %>%
    filter(curvature == max(curvature, na.rm = TRUE)) %>%
    slice(1)
  
  simplest_within_tolerance_row <- df %>%
    filter(within_0_5pct_best) %>%
    arrange(K) %>%
    slice(1)
  
  tibble(
    cohort = unique(df$cohort),
    K_best_CVIC = best_row$K,
    best_mean_CVIC = best_row$mean_CVIC,
    
    K_knee_distance = knee_row$K,
    knee_distance = knee_row$knee_distance,
    mean_CVIC_at_knee_distance = knee_row$mean_CVIC,
    pct_worse_at_knee_distance = knee_row$pct_worse_vs_best,
    
    K_max_curvature = curvature_row$K,
    max_curvature = curvature_row$curvature,
    mean_CVIC_at_max_curvature = curvature_row$mean_CVIC,
    pct_worse_at_max_curvature = curvature_row$pct_worse_vs_best,
    
    K_simplest_within_0_5pct = simplest_within_tolerance_row$K,
    mean_CVIC_simplest_within_0_5pct = simplest_within_tolerance_row$mean_CVIC,
    pct_worse_simplest_within_0_5pct = simplest_within_tolerance_row$pct_worse_vs_best
  )
}

## ------------------------------------------------------------
## 4) Load CVIC and log-likelihood per cohort
## ------------------------------------------------------------

load_cvic_for_cohort <- function(cohort_name, path_results) {
  cat("Processing cohort:", cohort_name, "\n")
  
  file_cvic <- file.path(path_results, "cvic_matrix.npy")
  file_ll   <- file.path(path_results, "loglike_matrix.npy")
  
  if (!file.exists(file_cvic) || !file.exists(file_ll)) {
    stop(sprintf(
      "cvic_matrix.npy or loglike_matrix.npy missing for cohort: %s in %s",
      cohort_name,
      path_results
    ))
  }
  
  cvic <- npyLoad(file_cvic)
  logl <- npyLoad(file_ll)
  
  ## Force vector into matrix
  if (is.null(dim(cvic))) cvic <- matrix(cvic, nrow = 1)
  if (is.null(dim(logl))) logl <- matrix(logl, nrow = 1)
  
  dim_c <- dim(cvic)
  dim_l <- dim(logl)
  
  cat("  CVIC dimension:", paste(dim_c, collapse = "x"),
      " | LogLik dimension:", paste(dim_l, collapse = "x"), "\n")
  
  if (dim_c[2] != dim_l[2]) {
    stop(sprintf(
      "Column mismatch: %s CVIC %s vs LogLik %s",
      cohort_name,
      paste(dim_c, collapse = "x"),
      paste(dim_l, collapse = "x")
    ))
  }
  
  K <- dim_c[2]
  
  ## Handle fold dimensions
  if (dim_c[1] == dim_l[1]) {
    n_folds   <- dim_c[1]
    mean_cvic <- colMeans(cvic, na.rm = TRUE)
    sd_cvic   <- apply(cvic, 2, sd, na.rm = TRUE)
    mean_ll   <- colMeans(logl, na.rm = TRUE)
    sd_ll     <- apply(logl, 2, sd, na.rm = TRUE)
    
  } else if (dim_c[1] == 1 && dim_l[1] > 1) {
    n_folds   <- dim_l[1]
    mean_cvic <- as.numeric(cvic[1, ])
    sd_cvic   <- rep(NA_real_, K)
    mean_ll   <- colMeans(logl, na.rm = TRUE)
    sd_ll     <- apply(logl, 2, sd, na.rm = TRUE)
    cat("  Note: CVIC is already stored as 1 x K.\n")
    
  } else if (dim_l[1] == 1 && dim_c[1] > 1) {
    n_folds   <- dim_c[1]
    mean_cvic <- colMeans(cvic, na.rm = TRUE)
    sd_cvic   <- apply(cvic, 2, sd, na.rm = TRUE)
    mean_ll   <- as.numeric(logl[1, ])
    sd_ll     <- rep(NA_real_, K)
    cat("  Note: LogLik is already stored as 1 x K.\n")
    
  } else {
    stop(sprintf(
      "Unhandled dimension case for %s: CVIC %s, LogLik %s",
      cohort_name,
      paste(dim_c, collapse = "x"),
      paste(dim_l, collapse = "x")
    ))
  }
  
  ## Stepwise change in CVIC from K-1 to K
  delta_step <- c(NA_real_, diff(mean_cvic))
  
  ## Best model by minimum CVIC
  best_cvic <- min(mean_cvic, na.rm = TRUE)
  
  ## Absolute and relative distance to best model
  delta_best <- mean_cvic - best_cvic
  
  ## Percentage improvement relative to previous K
  prev_cvic <- dplyr::lag(mean_cvic)
  impr_step_pct <- ifelse(
    is.na(prev_cvic),
    NA_real_,
    (prev_cvic - mean_cvic) / prev_cvic * 100
  )
  
  pct_worse_vs_best <- (mean_cvic - best_cvic) / best_cvic * 100
  pct_of_best <- 100 * mean_cvic / best_cvic
  
  out <- tibble(
    cohort            = cohort_name,
    K                 = seq_len(K),
    mean_CVIC         = mean_cvic,
    sd_CVIC           = sd_cvic,
    mean_LogLik       = mean_ll,
    sd_LogLik         = sd_ll,
    delta_CVIC        = delta_step,
    delta_CVICmin     = delta_best,
    impr_step_pct     = impr_step_pct,
    pct_worse_vs_best = pct_worse_vs_best,
    pct_of_best       = pct_of_best,
    n_folds           = n_folds
  )
  
  ## Add normalized fit, knee distance, and curvature per cohort
  out <- out %>%
    arrange(K) %>%
    mutate(
      K_norm = normalize_minmax(K),
      CVIC_norm_raw = normalize_minmax(mean_CVIC),
      Fit_norm = 1 - CVIC_norm_raw,
      knee_distance = point_line_distance(K_norm, Fit_norm),
      curvature = discrete_curvature(K_norm, Fit_norm)
    )
  
  out
}

## ------------------------------------------------------------
## 5) Process all cohorts
## ------------------------------------------------------------

all_list <- list()

for (nm in names(cohort_dirs)) {
  df_c <- load_cvic_for_cohort(nm, cohort_dirs[[nm]])
  
  if (!is.null(df_c)) {
    all_list[[nm]] <- df_c
  }
}

if (length(all_list) == 0) {
  stop("No cohort could be loaded successfully. Please check paths and files.")
}

all_df <- bind_rows(all_list)

## ------------------------------------------------------------
## 6) Best K, knee K, curvature K, and parsimony K
## ------------------------------------------------------------

best_K_df <- all_df %>%
  group_by(cohort) %>%
  filter(mean_CVIC == min(mean_CVIC, na.rm = TRUE)) %>%
  slice(1) %>%
  ungroup() %>%
  arrange(cohort) %>%
  select(
    cohort,
    K_best_CVIC = K,
    best_mean_CVIC = mean_CVIC,
    n_folds
  )

knee_summary_df <- all_df %>%
  group_by(cohort) %>%
  group_split() %>%
  lapply(get_knee_summary, tolerance_pct = 0.5) %>%
  bind_rows() %>%
  arrange(cohort)

## Main model-selection summary
model_selection_df <- knee_summary_df %>%
  left_join(best_K_df, by = c("cohort", "K_best_CVIC", "best_mean_CVIC")) %>%
  mutate(
    final_K_suggested_by_knee_and_tolerance = case_when(
      K_knee_distance == K_simplest_within_0_5pct ~ K_knee_distance,
      TRUE ~ K_simplest_within_0_5pct
    ),
    note = case_when(
      K_knee_distance == K_simplest_within_0_5pct ~
        "Knee-distance solution equals simplest model within 0.5 percent of best CVIC.",
      TRUE ~
        "Knee-distance solution differs from simplest model within 0.5 percent of best CVIC; final choice should be documented using the prespecified parsimony and subtype-size criteria."
    )
  )

## Add flags to detailed table
all_df <- all_df %>%
  left_join(
    model_selection_df %>%
      select(
        cohort,
        K_best_CVIC,
        K_knee_distance,
        K_max_curvature,
        K_simplest_within_0_5pct
      ),
    by = "cohort"
  ) %>%
  mutate(
    is_best_CVIC = K == K_best_CVIC,
    is_knee_distance = K == K_knee_distance,
    is_max_curvature = K == K_max_curvature,
    is_simplest_within_0_5pct = K == K_simplest_within_0_5pct
  )

## ------------------------------------------------------------
## 7) Export CSV files
## ------------------------------------------------------------

out_summary <- file.path(out_dir, "CVIC_summary_all_cohorts_with_knee.csv")
readr::write_csv(all_df, out_summary)
cat("\nDetailed CVIC table saved to:\n", out_summary, "\n")

out_best <- file.path(out_dir, "CVIC_bestK_per_cohort.csv")
readr::write_csv(best_K_df, out_best)
cat("Best-K table saved to:\n", out_best, "\n")

out_knee <- file.path(out_dir, "CVIC_knee_detection_summary_all_cohorts.csv")
readr::write_csv(knee_summary_df, out_knee)
cat("Knee-detection summary saved to:\n", out_knee, "\n")

out_selection <- file.path(out_dir, "CVIC_model_selection_summary_all_cohorts.csv")
readr::write_csv(model_selection_df, out_selection)
cat("Model-selection summary saved to:\n", out_selection, "\n")

## ------------------------------------------------------------
## 8) Elbow plot with best CVIC, knee-distance, and max-curvature
## ------------------------------------------------------------

plot_markers_df <- all_df %>%
  filter(is_best_CVIC | is_knee_distance | is_max_curvature | is_simplest_within_0_5pct) %>%
  mutate(
    marker = case_when(
      is_best_CVIC ~ "Minimum CVIC",
      is_knee_distance ~ "Knee distance",
      is_max_curvature ~ "Maximum curvature",
      is_simplest_within_0_5pct ~ "Simplest within 0.5 percent",
      TRUE ~ "Other"
    )
  )

p_elbow <- ggplot(all_df, aes(x = K, y = mean_CVIC, group = cohort)) +
  geom_line() +
  geom_point() +
  geom_point(
    data = all_df %>% filter(is_best_CVIC),
    aes(x = K, y = mean_CVIC),
    size = 3,
    shape = 16
  ) +
  geom_point(
    data = all_df %>% filter(is_knee_distance),
    aes(x = K, y = mean_CVIC),
    size = 3,
    shape = 17
  ) +
  geom_point(
    data = all_df %>% filter(is_max_curvature),
    aes(x = K, y = mean_CVIC),
    size = 3,
    shape = 15
  ) +
  geom_point(
    data = all_df %>% filter(is_simplest_within_0_5pct),
    aes(x = K, y = mean_CVIC),
    size = 3,
    shape = 18
  ) +
  facet_wrap(~ cohort, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(all_df$K))) +
  labs(
    title = "CVIC elbow plot per cohort",
    x = "Number of subtypes K",
    y = "Mean CVIC across folds",
    caption = "Symbols mark minimum CVIC, knee-distance solution, maximum-curvature solution, and simplest model within 0.5 percent of minimum CVIC."
  ) +
  theme_bw()

out_pdf <- file.path(out_dir, "CVIC_elbow_allCohorts_with_knee.pdf")
ggsave(out_pdf, p_elbow, width = 10, height = 6)
cat("\nElbow plot saved to:\n", out_pdf, "\n")

## ------------------------------------------------------------
## 9) Normalized fit plot for knee detection
## ------------------------------------------------------------

p_knee <- ggplot(all_df, aes(x = K_norm, y = Fit_norm, group = cohort)) +
  geom_line() +
  geom_point() +
  geom_point(
    data = all_df %>% filter(is_knee_distance),
    aes(x = K_norm, y = Fit_norm),
    size = 3,
    shape = 17
  ) +
  facet_wrap(~ cohort) +
  labs(
    title = "Normalized CVIC fit curve with knee-distance solution",
    x = "Normalized K",
    y = "Normalized fit (1 - normalized CVIC)",
    caption = "The knee was defined as the point with maximum perpendicular distance from the line connecting the first and last points of the normalized fit curve."
  ) +
  theme_bw()

out_knee_pdf <- file.path(out_dir, "CVIC_knee_detection_normalized_fit_allCohorts.pdf")
ggsave(out_knee_pdf, p_knee, width = 10, height = 6)
cat("Knee-detection plot saved to:\n", out_knee_pdf, "\n")

cat("\nFinished CVIC knee-detection analysis.\n")
