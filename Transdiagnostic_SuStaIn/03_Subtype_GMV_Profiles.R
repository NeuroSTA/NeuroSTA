# ============================================================
# GMV SuStaIn Subtype Analysis - K subtypes per cohort
# Cohorts: TDM, Angst, MDD, BD, SZ
#
# IMPORTANT:
# - The GMV input is negated. We undo this negation EXACTLY ONCE globally (gmv_z * -1).
# - We do NOT flip again inside each cohort (prevents any accidental sign inconsistencies).
#
# Input GMV (already negated): HPC/input/residuals_z_scored_negated_paper.csv
# Input assignments: SuStaIn_assignments/*_Ksubtypes.csv
# Multiple testing: Benjamini-Hochberg (BH/FDR)
# Output: CSV + PNG + PDF only
# Notes:
# - Subtype levels 1-K are enforced as factor levels even if one is empty.
# - We do NOT stop on empty subtypes; we write counts and continue.
# - "Group" is never used for plotting or stats.
#
# PATH HANDLING (master_runner compatible):
# - No setwd() required.
# - base_dir comes from option SUSTAIN_PROJECT_DIR (fallback default below).
# - out_root comes from option SUSTAIN_OUTPUT_DIR (fallback default below).
# ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(rstatix)
  library(moments)
  library(broom)
  library(car)
  library(pheatmap)
  library(ggplot2)
})

# -----------------------------
# 0) Settings (master_runner compatible)
# -----------------------------
base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))

# Per-script output directory (preferred)
K <- 3L
out_root <- getOption(
  "SUSTAIN_OUTPUT_DIR",
  file.path(base_dir, "Output_master", paste0("gmv_subtypes_", K, "subtypes"))
)
if (!dir.exists(out_root)) dir.create(out_root, recursive = TRUE)

gmv_z_path <- file.path(base_dir, "HPC", "input", "residuals_z_scored_corrected_allGroups.csv")

alpha_fdr <- 0.05

# If TRUE: flip sign of all ROI columns ONCE globally (undo negation)
flip_sign_undo_negation <- TRUE

# Cohort -> assignment file (K subtypes)
cohorts <- list(
  TDM  = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", paste0("main_subject_subtype_assignment_", K, "subtypes.csv")),
  Angst = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", paste0("Angst_subject_subtype_assignment_", K, "subtypes.csv")),
  MDD   = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", paste0("MDD_subject_subtype_assignment_", K, "subtypes.csv")),
  BD    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", paste0("BD_subject_subtype_assignment_", K, "subtypes.csv")),
  SZ    = file.path(base_dir, "Output_sustainrun", "SuStaIn_assignments", paste0("SZ_subject_subtype_assignment_", K, "subtypes.csv"))
)

# -----------------------------
# 1) Helpers
# -----------------------------
stop_if_missing_file <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
}

as_proband_chr <- function(x) trimws(as.character(x))

safe_shapiro_p <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 3) return(NA_real_)
  out <- tryCatch(stats::shapiro.test(x)$p.value, error = function(e) NA_real_)
  as.numeric(out)
}

derive_subtyp_from_ml <- function(df) {
  if (!("ML_Subtype" %in% names(df))) stop("ML_Subtype column not found.")
  if (all(is.na(df$ML_Subtype))) stop("ML_Subtype is all NA.")
  
  ml_min <- suppressWarnings(min(df$ML_Subtype, na.rm = TRUE))
  ml_max <- suppressWarnings(max(df$ML_Subtype, na.rm = TRUE))
  
  if (is.infinite(ml_min) || is.infinite(ml_max)) stop("ML_Subtype min/max not finite.")
  
  if (ml_min == 0) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype) + 1L)
  } else if (ml_min == 1) {
    df <- df %>% mutate(Subtyp = as.integer(ML_Subtype))
  } else {
    stop(
      "Unexpected ML_Subtype coding. min=", ml_min, " max=", ml_max,
      " (expected to start at 0 or 1)."
    )
  }
  df
}

dunn_with_r <- function(df, y, g) {
  tmp <- df %>%
    select(.y = {{y}}, .g = {{g}}) %>%
    filter(!is.na(.y), !is.na(.g))
  
  if (nrow(tmp) == 0) return(tibble::tibble())
  
  tab <- table(tmp$.g)
  if (sum(tab >= 2) < 2) return(tibble::tibble())
  
  ## 1) Always compute RAW p-values first
  dunn_raw <- rstatix::dunn_test(tmp, .y ~ .g, p.adjust.method = "none")
  
  ## 2) Apply BH explicitly (robust across rstatix versions)
  ##    rstatix uses column name "p" for raw p-values
  dunn_raw <- dunn_raw %>%
    mutate(
      p.adj = stats::p.adjust(p, method = "BH")
    )
  
  ## 3) Add effect size r (z / sqrt(N))
  dunn_raw %>%
    rowwise() %>%
    mutate(
      n1 = sum(tmp$.g == group1),
      n2 = sum(tmp$.g == group2),
      N  = n1 + n2,
      r  = ifelse(N > 0, statistic / sqrt(N), NA_real_)
    ) %>%
    ungroup()
}

# Short ROI labels (vectorized):
short_roi <- function(x) {
  x2 <- gsub("(_avg_resid_final|_avg_resid|_resid_final|_final)$", "", x, ignore.case = TRUE)
  x2 <- sub("_.*$", "", x2)
  
  is_allcaps <- grepl("^[A-Z0-9.]+$", x2) & nchar(x2) > 1
  if (any(is_allcaps)) {
    first <- substr(x2[is_allcaps], 1, 1)
    rest  <- substr(x2[is_allcaps], 2, nchar(x2[is_allcaps]))
    x2[is_allcaps] <- paste0(toupper(first), tolower(rest))
  }
  x2
}

# Breaks centered at 0 while covering true min/max
make_centered_breaks <- function(vmin, vmax, n = 255L) {
  if (!is.finite(vmin) || !is.finite(vmax) || vmin >= vmax) {
    return(seq(-1, 1, length.out = n))
  }
  
  if (vmin < 0 && vmax > 0) {
    n_neg <- floor((n - 1L) / 2L)
    n_pos <- (n - 1L) - n_neg
    bneg <- seq(vmin, 0, length.out = n_neg + 1L)
    bpos <- seq(0, vmax, length.out = n_pos + 1L)
    breaks <- c(bneg, bpos[-1])
  } else {
    breaks <- seq(vmin, vmax, length.out = n)
  }
  
  breaks <- sort(as.numeric(breaks))
  if (any(diff(breaks) <= 0)) {
    breaks <- seq(vmin, vmax, length.out = n)
  }
  breaks
}

# Minimal legend ticks to avoid stacked labels
make_legend_ticks <- function(vmin, vmax) {
  if (!is.finite(vmin) || !is.finite(vmax) || vmin >= vmax) {
    br <- c(-1, 0, 1)
    return(list(breaks = br, labels = sprintf("%.3f", br)))
  }
  
  if (vmin < 0 && vmax > 0) {
    br <- c(vmin, 0, vmax)
  } else {
    br <- c(vmin, (vmin + vmax) / 2, vmax)
  }
  
  tol <- max(1e-10, 1e-6 * max(abs(c(vmin, vmax))))
  br <- sort(br)
  keep <- c(TRUE, diff(br) > tol)
  br <- br[keep]
  
  list(breaks = br, labels = sprintf("%.3f", br))
}

# Robust CSV read: try comma-delimited first, then csv2 (semicolon) if needed
read_any_csv <- function(path) {
  x <- tryCatch(
    readr::read_csv(path, show_col_types = FALSE),
    error = function(e) NULL
  )
  if (!is.null(x) && ncol(x) > 1) return(x)
  
  x2 <- tryCatch(
    readr::read_csv2(path, show_col_types = FALSE),
    error = function(e) NULL
  )
  if (!is.null(x2) && ncol(x2) > 1) return(x2)
  
  stop("Could not read CSV with read_csv or read_csv2: ", path)
}

# -----------------------------
# 2) Load GMV Z once (and flip ONCE globally)
# -----------------------------
stop_if_missing_file(gmv_z_path)

gmv_z <- read_any_csv(gmv_z_path)

if (!"Proband" %in% names(gmv_z)) {
  if ("names" %in% names(gmv_z)) gmv_z <- gmv_z %>% rename(Proband = names)
  if ("Name"  %in% names(gmv_z)) gmv_z <- gmv_z %>% rename(Proband = Name)
}
if (!"Proband" %in% names(gmv_z)) stop("Could not identify subject id column in GMV file.")

gmv_z <- gmv_z %>% mutate(Proband = as_proband_chr(Proband))

exclude_cols_gmv <- c("Proband", "Group")
unexpected_gmv_cols <- setdiff(names(gmv_z), c(exclude_cols_gmv, grep("_avg_resid_final$", names(gmv_z), value = TRUE)))
if (length(unexpected_gmv_cols) > 0L) stop("Unexpected columns in GMV input: ", paste(unexpected_gmv_cols, collapse = ", "))
roi_cols_gmv <- grep("_avg_resid_final$", names(gmv_z), value = TRUE)
if (length(roi_cols_gmv) != 62L || !all(vapply(gmv_z[roi_cols_gmv], is.numeric, logical(1))))
  stop("Expected exactly 62 numeric ROI columns with suffix _avg_resid_final.")

if (isTRUE(flip_sign_undo_negation)) {
  gmv_z[roi_cols_gmv] <- lapply(gmv_z[roi_cols_gmv], function(x) -1 * x)
  message("Global GMV flip applied once to gmv_z (undo negation).")
} else {
  message("Global GMV flip disabled (flip_sign_undo_negation = FALSE).")
}

# -----------------------------
# 3) Run one cohort
# -----------------------------
run_one_cohort <- function(cohort_name, sustain_csv) {
  
  message("======================================")
  message("COHORT: ", cohort_name)
  message("======================================")
  
  stop_if_missing_file(sustain_csv)
  
  out_dir <- file.path(out_root, cohort_name)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  sustain <- read_any_csv(sustain_csv)
  
  if (!"Proband" %in% names(sustain)) {
    if ("Name"  %in% names(sustain)) sustain <- sustain %>% rename(Proband = Name)
    if ("names" %in% names(sustain)) sustain <- sustain %>% rename(Proband = names)
  }
  if (!"Proband" %in% names(sustain)) stop("Could not identify subject id column in sustain file: ", sustain_csv)
  
  sustain <- sustain %>% mutate(Proband = as_proband_chr(Proband))
  sustain <- derive_subtyp_from_ml(sustain)
  
  sustain <- sustain %>%
    filter(Subtyp %in% 1:K) %>%
    mutate(Subtyp = factor(Subtyp, levels = 1:K))
  
  subtype_counts <- sustain %>%
    count(Subtyp, name = "n") %>%
    tidyr::complete(Subtyp = factor(1:K, levels = 1:K), fill = list(n = 0)) %>%
    arrange(Subtyp)
  readr::write_csv(subtype_counts, file.path(out_dir, "subtype_counts.csv"))
  
  merged <- inner_join(
    gmv_z,
    sustain %>% select(Proband, Subtyp),
    by = "Proband"
  )
  if (nrow(merged) == 0) stop("Join produced 0 rows for cohort: ", cohort_name)
  
  merged <- merged %>% mutate(Subtyp = factor(Subtyp, levels = 1:K))
  
  subtype_counts_join <- merged %>%
    count(Subtyp, name = "n") %>%
    tidyr::complete(Subtyp = factor(1:K, levels = 1:K), fill = list(n = 0)) %>%
    arrange(Subtyp)
  readr::write_csv(subtype_counts_join, file.path(out_dir, "subtype_counts_after_join.csv"))
  
  missing_levels <- subtype_counts_join %>% filter(n == 0) %>% pull(Subtyp) %>% as.character()
  if (length(missing_levels) > 0) {
    stop(
      "Cohort ", cohort_name,
      ": empty subtypes after join (n=0): ", paste(missing_levels, collapse = ",")
    )
  }
  
  missing_roi_cols <- setdiff(roi_cols_gmv, names(merged))
  if (length(missing_roi_cols) > 0L) stop("ROI columns lost after join: ", paste(missing_roi_cols, collapse = ", "))
  roi_cols <- roi_cols_gmv
  
  merged <- merged %>%
    mutate(GMV_volume = rowMeans(as.matrix(select(., all_of(roi_cols))), na.rm = TRUE))
  
  # -----------------------------
  # A) Diagnostics
  # -----------------------------
  normality <- merged %>%
    group_by(Subtyp) %>%
    summarise(
      n = sum(!is.na(GMV_volume)),
      shapiro_p = safe_shapiro_p(GMV_volume),
      skewness  = moments::skewness(GMV_volume, na.rm = TRUE),
      kurtosis  = moments::kurtosis(GMV_volume, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::complete(Subtyp = factor(1:K, levels = 1:K))
  readr::write_csv(normality, file.path(out_dir, "qc_normality_per_subtyp.csv"))
  
  levene_p <- tryCatch({
    lv <- car::leveneTest(GMV_volume ~ Subtyp, data = merged)
    as.numeric(lv$`Pr(>F)`[1])
  }, error = function(e) NA_real_)
  readr::write_csv(tibble::tibble(levene_p = levene_p), file.path(out_dir, "qc_levene.csv"))
  
  all_normal <- all(normality$shapiro_p > 0.05, na.rm = TRUE)
  levene_ok  <- !is.na(levene_p) && (levene_p > 0.05)
  
  # -----------------------------
  # B) Global test GMV_volume + posthoc (NONPARAMETRIC ONLY)
  # -----------------------------
  decision <- "Kruskal_Dunn_BH"
  writeLines(decision, con = file.path(out_dir, "decision_global_test.txt"))
  
  global_test <- NA_character_
  global_stat <- NA_real_
  global_df1  <- NA_real_
  global_df2  <- NA_real_
  global_p    <- NA_real_
  global_eff  <- NA_real_
  global_eff_name <- NA_character_
  
  ## Global Kruskal-Wallis
  kw <- kruskal.test(GMV_volume ~ Subtyp, data = merged)
  
  global_test <- "Kruskal"
  global_stat <- as.numeric(kw$statistic)
  global_df1  <- as.numeric(kw$parameter)
  global_df2  <- NA_real_
  global_p    <- as.numeric(kw$p.value)
  
  kw_tbl <- tibble::tibble(
    statistic = global_stat,
    df = global_df1,
    p_value = global_p
  )
  readr::write_csv(kw_tbl, file.path(out_dir, "global_kruskal.csv"))
  
  ## Effect size: epsilon-squared (nonparametric analogue)
  n <- nrow(merged)
  k <- nlevels(merged$Subtyp)
  epsilon2 <- (global_stat - k + 1) / (n - k)
  
  global_eff <- as.numeric(epsilon2)
  global_eff_name <- "epsilon2"
  readr::write_csv(tibble::tibble(epsilon2 = epsilon2), file.path(out_dir, "global_kruskal_epsilon2.csv"))
  
  ## Post-hoc: Dunn test with BH adjustment + r effect size per pair
  dunn <- dunn_with_r(merged, GMV_volume, Subtyp)
  if (nrow(dunn) > 0) {
    readr::write_csv(dunn, file.path(out_dir, "posthoc_dunn_BH_with_r.csv"))
  } else {
    readr::write_csv(
      tibble::tibble(note = "No valid Dunn posthoc (insufficient group sizes)."),
      file.path(out_dir, "posthoc_dunn_BH_with_r.csv")
    )
  }
  # -----------------------------
  # C) Descriptives GMV_volume
  # -----------------------------
  desc <- merged %>%
    group_by(Subtyp) %>%
    summarise(
      n = sum(!is.na(GMV_volume)),
      mean = mean(GMV_volume, na.rm = TRUE),
      sd = sd(GMV_volume, na.rm = TRUE),
      median = median(GMV_volume, na.rm = TRUE),
      iqr = IQR(GMV_volume, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::complete(
      Subtyp = factor(1:K, levels = 1:K),
      fill = list(n = 0, mean = NA_real_, sd = NA_real_, median = NA_real_, iqr = NA_real_)
    ) %>%
    arrange(Subtyp)
  readr::write_csv(desc, file.path(out_dir, "descriptives_GMV_volume.csv"))
  
  # -----------------------------
  # D) ROI-level global Kruskal + BH + effect size + significance
  # -----------------------------
  p_to_stars <- function(p) {
    dplyr::case_when(
      is.na(p)    ~ NA_character_,
      p <= 1e-4   ~ "****",
      p <= 1e-3   ~ "***",
      p <= 1e-2   ~ "**",
      p <= 5e-2   ~ "*",
      TRUE        ~ "ns"
    )
  }
  
  roi_global <- purrr::map_dfr(roi_cols, function(roi) {
    x <- merged[[roi]]
    g <- merged$Subtyp
    
    tab <- tapply(!is.na(x), g, sum)
    if (sum(tab >= 2, na.rm = TRUE) < 2) {
      return(tibble::tibble(
        ROI = roi,
        H = NA_real_,
        df = NA_real_,
        p = NA_real_,
        epsilon2 = NA_real_,
        N = sum(!is.na(x))
      ))
    }
    
    tmp <- tibble::tibble(x = x, g = g) %>%
      dplyr::filter(!is.na(x), !is.na(g))
    
    kt <- kruskal.test(x ~ g, data = tmp)
    
    n_roi <- nrow(tmp)
    k_roi <- dplyr::n_distinct(tmp$g)
    H_roi <- as.numeric(kt$statistic)
    
    epsilon2_roi <- if (!is.finite(H_roi) || !is.finite(n_roi) || !is.finite(k_roi) || n_roi <= k_roi) {
      NA_real_
    } else {
      (H_roi - k_roi + 1) / (n_roi - k_roi)
    }
    
    tibble::tibble(
      ROI = roi,
      H = H_roi,
      df = as.numeric(kt$parameter),
      p = as.numeric(kt$p.value),
      epsilon2 = as.numeric(epsilon2_roi),
      N = n_roi
    )
  }) %>%
    mutate(
      q_BH = p.adjust(p, method = "BH"),
      q_BH_signif = p_to_stars(q_BH)
    )
  
  readr::write_csv(roi_global, file.path(out_dir, "roi_global_kruskal_BH.csv"))
  
  n_sig_rois <- sum(!is.na(roi_global$q_BH) & roi_global$q_BH < alpha_fdr)
  ## ============================================================
  ## PATCH: Make p.adj + significance stars refer to GLOBAL BH
  ##        across all ROI-by-pair Dunn tests (not within-ROI).
  ##        - Keep raw p in column "p"
  ##        - Overwrite/define "p.adj" = BH(p) across ALL rows
  ##        - Recompute "p.adj.signif" from the GLOBAL p.adj
  ##        - Drop q_BH_all (optional), because p.adj is now global
  ## ============================================================
  
  ## Helper: significance stars for adjusted p-values (paper-style)
  p_to_stars <- function(p) {
    dplyr::case_when(
      is.na(p)        ~ NA_character_,
      p <= 0.0001     ~ "****",
      p <= 0.001      ~ "***",
      p <= 0.01       ~ "**",
      p <= 0.05       ~ "*",
      TRUE            ~ "ns"
    )
  }
  
  ## --- inside run_one_cohort(), replace ONLY the block E) ROI-level posthoc Dunn ---
  ## -----------------------------
  ## E) ROI-level posthoc Dunn (GLOBAL BH across all ROI-by-pair tests)
  ## -----------------------------
  sig_rois <- roi_global %>% filter(!is.na(q_BH) & q_BH < alpha_fdr) %>% pull(ROI)
  
  roi_posthoc <- purrr::map_dfr(sig_rois, function(roi) {
    tmp <- merged %>% select(Subtyp, value = all_of(roi)) %>% filter(!is.na(value))
    if (nrow(tmp) == 0) return(tibble::tibble())
    
    tab <- table(tmp$Subtyp)
    if (sum(tab >= 2) < 2) return(tibble::tibble())
    
    rstatix::dunn_test(tmp, value ~ Subtyp, p.adjust.method = "none") %>%
      mutate(
        ROI = roi,
        n1  = as.integer(tab[as.character(group1)]),
        n2  = as.integer(tab[as.character(group2)]),
        N   = n1 + n2,
        r   = ifelse(N > 0, statistic / sqrt(N), NA_real_)
      )
  })
  
  if (nrow(roi_posthoc) > 0) {
    
    ## 1) GLOBAL BH across ALL ROI-by-pair tests (all rows)
    roi_posthoc <- roi_posthoc %>%
      mutate(
        p.adj = stats::p.adjust(p, method = "BH")
      )
    
    ## 2) Make the stars correspond to GLOBAL p.adj (overwrite any existing column)
    ##    rstatix creates p.adj.signif based on its own p.adj; we overwrite to be safe.
    roi_posthoc <- roi_posthoc %>%
      mutate(
        p.adj.signif = p_to_stars(p.adj)
      )
    
    ## 3) Optional: drop q_BH_all to avoid duplicate/confusing columns
    if ("q_BH_all" %in% names(roi_posthoc)) {
      roi_posthoc <- roi_posthoc %>% select(-q_BH_all)
    }
    
    ## 4) Sort by GLOBAL adjusted p
    roi_posthoc <- roi_posthoc %>%
      arrange(p.adj, ROI)
    
    readr::write_csv(
      roi_posthoc,
      file.path(out_dir, "roi_posthoc_dunn_BH_across_all_pairs.csv")
    )
    
  } else {
    readr::write_csv(
      tibble::tibble(note = "No ROI posthoc tests (no significant ROI globals after BH, or insufficient data)."),
      file.path(out_dir, "roi_posthoc_dunn_BH_across_all_pairs.csv")
    )
  }
  
  n_roi_posthoc_rows <- nrow(roi_posthoc)

  
  # -----------------------------
  # F) Heatmap: ROI means per subtype
  # -----------------------------
  roi_means <- merged %>%
    group_by(Subtyp) %>%
    summarise(across(all_of(roi_cols), function(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
    tidyr::complete(Subtyp = factor(1:K, levels = 1:K)) %>%
    arrange(Subtyp)
  
  readr::write_csv(roi_means, file.path(out_dir, "roi_means_per_subtyp.csv"))
  
  mat <- as.matrix(roi_means %>% select(-Subtyp))
  rownames(mat) <- paste0("Subtyp ", as.character(roi_means$Subtyp))
  
  keep_rows <- apply(mat, 1, function(r) any(is.finite(r)))
  mat_plot <- mat[keep_rows, , drop = FALSE]
  
  colnames(mat_plot) <- short_roi(colnames(mat_plot))
  
  vals <- as.numeric(mat_plot)
  vals <- vals[is.finite(vals)]
  
  if (length(vals) == 0) {
    hm_min <- NA_real_; hm_max <- NA_real_
    q05 <- NA_real_; q50 <- NA_real_; q95 <- NA_real_
  } else {
    hm_min <- min(vals)
    hm_max <- max(vals)
    q05 <- as.numeric(quantile(vals, 0.05, na.rm = TRUE))
    q50 <- as.numeric(quantile(vals, 0.50, na.rm = TRUE))
    q95 <- as.numeric(quantile(vals, 0.95, na.rm = TRUE))
  }
  
  message(sprintf(
    "Heatmap value range (%s): min=%.3f | q05=%.3f | median=%.3f | q95=%.3f | max=%.3f",
    cohort_name, hm_min, q05, q50, q95, hm_max
  ))
  
  n_breaks <- 255L
  hm_breaks <- make_centered_breaks(hm_min, hm_max, n = n_breaks)
  
  hm_cols <- grDevices::colorRampPalette(c("#B2182B", "#F0F0F0", "#2166AC"))(length(hm_breaks) - 1)
  leg <- make_legend_ticks(hm_min, hm_max)
  
  grDevices::pdf(file.path(out_dir, "heatmap_roi_means.pdf"), width = 14, height = 5)
  
  if (nrow(mat_plot) >= 1 && ncol(mat_plot) >= 1) {
    pheatmap::pheatmap(
      mat_plot,
      cluster_rows = FALSE,
      cluster_cols = FALSE,
      color = hm_cols,
      breaks = hm_breaks,
      legend_breaks = leg$breaks,
      legend_labels = leg$labels,
      na_col = "grey92",
      TDM = paste0("GMV profiles across SuStaIn subtypes - ", cohort_name),
      border_color = NA,
      fontsize = 9,
      fontsize_row = 11,
      fontsize_col = 8,
      angle_col = 45
    )
  } else {
    graphics::plot.new()
    graphics::title(TDM = paste0("Heatmap not available (no data after filtering) - ", cohort_name))
  }
  
  grDevices::dev.off()
  
  
  
  # -----------------------------
  # F2) Add-on: ROIs with strongest "loss" per subtype
  #     Definition: delta_to_overall = mean(Subtype) - mean(Overall cohort)
  #     -> most negative deltas = strongest loss vs cohort baseline
  # -----------------------------
  top_n_loss <- 20L  # adjust if you want 5/15/20
  
  # Overall ROI means across the cohort (all subjects pooled)
  roi_overall <- merged %>%
    summarise(across(all_of(roi_cols), ~mean(.x, na.rm = TRUE)))
  
  # Subtype ROI means (already computed as roi_means)
  # Convert to long format and compute delta to overall mean
  roi_means_long <- roi_means %>%
    pivot_longer(
      cols = all_of(roi_cols),
      names_to = "ROI",
      values_to = "mean_subtype"
    ) %>%
    mutate(Subtyp = factor(Subtyp, levels = 1:K))
  
  roi_overall_long <- roi_overall %>%
    pivot_longer(
      cols = all_of(roi_cols),
      names_to = "ROI",
      values_to = "mean_overall"
    )
  
  roi_loss_table <- roi_means_long %>%
    left_join(roi_overall_long, by = "ROI") %>%
    mutate(
      delta_to_overall = mean_subtype - mean_overall,   # negative = loss
      abs_delta = abs(delta_to_overall),
      ROI_short = short_roi(ROI)
    )
  
  # 1) Top-N strongest losses per subtype (most negative deltas)
  top_losses_per_subtype <- roi_loss_table %>%
    group_by(Subtyp) %>%
    arrange(delta_to_overall, .by_group = TRUE) %>%  # ascending -> most negative first
    slice_head(n = top_n_loss) %>%
    ungroup()
  
  readr::write_csv(
    top_losses_per_subtype,
    file.path(out_dir, paste0("top_ROI_losses_per_subtype_top", top_n_loss, ".csv"))
  )
  
  # 2) Optional: also write strongest increases per subtype (largest positive deltas)
  top_gains_per_subtype <- roi_loss_table %>%
    group_by(Subtyp) %>%
    arrange(desc(delta_to_overall), .by_group = TRUE) %>%  # descending -> most positive first
    slice_head(n = top_n_loss) %>%
    ungroup()
  
  readr::write_csv(
    top_gains_per_subtype,
    file.path(out_dir, paste0("top_ROI_gains_per_subtype_top", top_n_loss, ".csv"))
  )
  
  # 3) Optional: subtype "signature" table (wide) for quick inspection (loss-only)
  top_losses_wide <- top_losses_per_subtype %>%
    group_by(Subtyp) %>%
    mutate(rank_loss = row_number()) %>%
    ungroup() %>%
    select(Subtyp, rank_loss, ROI, ROI_short, mean_subtype, mean_overall, delta_to_overall) %>%
    tidyr::pivot_wider(
      names_from = rank_loss,
      values_from = c(ROI_short, delta_to_overall),
      names_glue = "{.value}_rank{rank_loss}"
    )
  
  readr::write_csv(
    top_losses_wide,
    file.path(out_dir, paste0("top_ROI_losses_per_subtype_wide_top", top_n_loss, ".csv"))
  )
  # -----------------------------
  # G) Boxplot GMV_volume
  # -----------------------------
  p <- ggplot2::ggplot(merged, ggplot2::aes(x = Subtyp, y = GMV_volume, fill = Subtyp)) +
    ggplot2::geom_boxplot(outlier.shape = 16) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::labs(
      title = paste0("GMV volume by subtype - ", cohort_name),
      x = "Subtype (1-3)",
      y = "GMV (mean across ROIs)"
    )
  ggplot2::ggsave(file.path(out_dir, "boxplot_GMV_volume.png"), p, width = 8, height = 6, dpi = 300)
  
  # -----------------------------
  # H) Build cohort summary row for the final end table
  # -----------------------------
  st_counts <- subtype_counts_join %>%
    mutate(Subtyp = as.character(Subtyp)) %>%
    select(Subtyp, n) %>%
    tidyr::pivot_wider(names_from = Subtyp, values_from = n, names_prefix = "n_subtyp_") %>%
    mutate(across(everything(), ~as.integer(.)))
  
  # ensure n_subtyp_1..n_subtyp_K exist
  expected <- paste0("n_subtyp_", 1:K)
  for (nm in expected) {
    if (!nm %in% names(st_counts)) st_counts[[nm]] <- 0L
  }
  st_counts <- st_counts %>% select(all_of(expected))
  
  min_shapiro <- suppressWarnings(min(normality$shapiro_p, na.rm = TRUE))
  if (is.infinite(min_shapiro)) min_shapiro <- NA_real_
  
  summary_row <- tibble::tibble(
    cohort = cohort_name,
    n_total = nrow(merged),
    n_roi = length(roi_cols),
    n_empty_subtypes = sum(subtype_counts_join$n == 0),
    levene_p = levene_p,
    min_shapiro_p = min_shapiro,
    global_test = global_test,
    global_stat = global_stat,
    global_df1 = global_df1,
    global_df2 = global_df2,
    global_p = global_p,
    effect_name = global_eff_name,
    effect_value = global_eff,
    n_sig_rois_BH = n_sig_rois,
    n_roi_posthoc_rows = n_roi_posthoc_rows,
    heatmap_min = hm_min,
    heatmap_q05 = q05,
    heatmap_median = q50,
    heatmap_q95 = q95,
    heatmap_max = hm_max
  ) %>%
    bind_cols(st_counts)
  
  readr::write_csv(summary_row, file.path(out_dir, "summary_cohort.csv"))
  
  message("[OK] Finished cohort: ", cohort_name)
  
  list(summary = summary_row, subtype_counts_after_join = subtype_counts_join)
}

# -----------------------------
# K) Run all cohorts + write final end tables
# -----------------------------
results <- purrr::imap(cohorts, ~run_one_cohort(cohort_name = .y, sustain_csv = .x))

summary_all <- purrr::map_dfr(results, "summary") %>%
  arrange(match(cohort, names(cohorts)))

subtype_counts_all <- purrr::map_dfr(names(results), function(nm) {
  results[[nm]]$subtype_counts_after_join %>% mutate(cohort = nm)
}) %>%
  relocate(cohort, .before = Subtyp)

readr::write_csv(summary_all, file.path(out_root, "summary_all_cohorts.csv"))
readr::write_csv(subtype_counts_all, file.path(out_root, "subtype_counts_after_join_all_cohorts_long.csv"))

subtype_counts_wide <- subtype_counts_all %>%
  mutate(Subtyp = as.character(Subtyp)) %>%
  tidyr::pivot_wider(names_from = Subtyp, values_from = n, names_prefix = "n_subtyp_") %>%
  arrange(match(cohort, names(cohorts)))

readr::write_csv(subtype_counts_wide, file.path(out_root, "subtype_counts_after_join_all_cohorts_wide.csv"))

message("[OK] ALL COHORTS DONE. Results in: ", out_root)
message("[OK] End tables written: summary_all_cohorts.csv and subtype_counts_after_join_all_cohorts_*.csv")

cat("GMV PATH:", gmv_z_path, "\n")
print(file.info(gmv_z_path)[, c("size", "mtime")])

cat("GMV dim:", nrow(gmv_z), "x", ncol(gmv_z), "\n")

# ID + ROI detection wie im Script
exclude_cols_gmv <- c("Proband", "Group")
roi_cols_gmv <- setdiff(names(gmv_z), exclude_cols_gmv)
roi_cols_gmv <- roi_cols_gmv[sapply(gmv_z[roi_cols_gmv], is.numeric)]

cat("N numeric cols used for GMV_volume:", length(roi_cols_gmv), "\n")
cat("First 20 numeric cols:\n")
print(head(roi_cols_gmv, 20))

# Fingerprint: ein ROI exemplarisch
roi0 <- roi_cols_gmv[1]
cat("ROI example:", roi0, "\n")
print(summary(gmv_z[[roi0]]))
