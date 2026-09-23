options(stringsAsFactors = FALSE)

base_dir <- getOption("SUSTAIN_PROJECT_DIR", Sys.getenv("SUSTAIN_PROJECT_DIR", unset = getwd()))
data_file <- getOption("SUSTAIN_CLINICAL_STATUS_FILE", Sys.getenv("SUSTAIN_CLINICAL_STATUS_FILE", unset = file.path(base_dir, "private_data", "Datenbank_Update_DataFreeze_bereinigt_withGlobalRatings.csv")))
assignment_file <- file.path(base_dir, "Output_sustainrun/SuStaIn_assignments/main_subject_subtype_assignment_3subtypes.csv")
output_dir <- file.path(base_dir, "Output Scripte/QC_Akut_Remittiert_Sensitivitaet")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

trigger_names <- c(
  "SANS7", "SANS12", "SANS16", "SANS21",
  "SAPS7", "SAPS20", "SAPS25", "SAPS34",
  "HAMA_Sum", "HAMD_Sum17", "YMRS_Sum"
)

dat <- read.csv(data_file, check.names = FALSE)
stopifnot(all(c("Proband", "Group", trigger_names) %in% names(dat)))
dat <- dat[dat$Group != 1, ]
dat$Proband <- gsub("_", "-", tolower(trimws(as.character(dat$Proband))), fixed = TRUE)
dat$Diagnosis <- c(`2` = "MDD", `3` = "BD", `4` = "SSD", `5` = "SSD", `8` = "ANX")[as.character(dat$Group)]

triggers <- as.data.frame(lapply(dat[trigger_names], function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[x %in% c(-99, -2)] <- NA_real_
  x
}))

positive <- data.frame(
  SANS7 = triggers$SANS7 > 2,
  SANS12 = triggers$SANS12 > 2,
  SANS16 = triggers$SANS16 > 2,
  SANS21 = triggers$SANS21 > 2,
  SAPS7 = triggers$SAPS7 > 2,
  SAPS20 = triggers$SAPS20 > 2,
  SAPS25 = triggers$SAPS25 > 2,
  SAPS34 = triggers$SAPS34 > 2,
  HAMA_Sum = triggers$HAMA_Sum > 19,
  HAMD_Sum17 = triggers$HAMD_Sum17 > 6,
  YMRS_Sum = triggers$YMRS_Sum >= 4
)

any_positive <- rowSums(positive, na.rm = TRUE) > 0
all_observed <- rowSums(!is.na(triggers)) == length(trigger_names)
dat$Status_sensitivity <- ifelse(any_positive, "acute", "remitted")
dat$Would_be_indeterminate <- !any_positive & !all_observed

diagnosis_missing <- as.data.frame(table(dat$Diagnosis[dat$Would_be_indeterminate]))
names(diagnosis_missing) <- c("Diagnosis", "n")
diagnosis_missing <- diagnosis_missing[diagnosis_missing$n > 0, ]
write.csv(diagnosis_missing, file.path(output_dir, "Indeterminate_nach_Diagnose.csv"), row.names = FALSE)

assignment <- read.csv(assignment_file, check.names = FALSE)
assignment$Proband <- gsub("_", "-", tolower(trimws(as.character(assignment$Proband))), fixed = TRUE)
assignment$ML_Subtype <- as.integer(assignment$ML_Subtype)
if (min(assignment$ML_Subtype, na.rm = TRUE) == 0) assignment$ML_Subtype <- assignment$ML_Subtype + 1L

merged <- merge(
  assignment[c("Proband", "ML_Subtype")],
  dat[c("Proband", "Status_sensitivity", "Would_be_indeterminate")],
  by = "Proband",
  all.x = TRUE,
  sort = FALSE
)
stopifnot(!anyNA(merged$Status_sensitivity))

tab <- table(
  factor(merged$ML_Subtype, levels = 1:3),
  factor(merged$Status_sensitivity, levels = c("acute", "remitted"))
)
chi <- chisq.test(tab, correct = FALSE)
cramers_v <- sqrt(unname(chi$statistic) / sum(tab))

pairs <- list(c(1, 2), c(1, 3), c(2, 3))
pair_results <- do.call(rbind, lapply(pairs, function(pair) {
  test <- fisher.test(tab[pair, , drop = FALSE])
  data.frame(
    comparison = paste0("Subtype ", pair[1], " vs Subtype ", pair[2]),
    odds_ratio = unname(test$estimate),
    p_value = test$p.value
  )
}))
pair_results$q_BH <- p.adjust(pair_results$p_value, method = "BH")

counts <- data.frame(
  ML_Subtype = 1:3,
  acute = as.integer(tab[, "acute"]),
  remitted = as.integer(tab[, "remitted"]),
  reassigned_to_remitted = as.integer(table(factor(
    merged$ML_Subtype[merged$Would_be_indeterminate], levels = 1:3
  )))
)
counts$n <- counts$acute + counts$remitted
counts$acute_percent <- 100 * counts$acute / counts$n

write.csv(counts, file.path(output_dir, "Status_nach_Subtyp_alle_Unbestimmten_remittiert.csv"), row.names = FALSE)
write.csv(pair_results, file.path(output_dir, "Paarvergleiche_alle_Unbestimmten_remittiert.csv"), row.names = FALSE)

report <- c(
  "# Sensitivitätsanalyse: alle unbestimmten Fälle als remittiert",
  "",
  "## Diagnoseprüfung",
  "",
  sprintf("Unter den %d nach der strengen Regel unbestimmten Fällen befinden sich %d SSD-Patienten. Die vollständige Verteilung lautet: MDD %d, ANX %d, BD %d und SSD %d. Das Fehlen der SANS/SAPS-Werte kann daher nicht allgemein mit dem Fehlen psychotischer oder negativer Symptome gleichgesetzt werden.",
          sum(dat$Would_be_indeterminate), sum(dat$Would_be_indeterminate & dat$Diagnosis == "SSD", na.rm = TRUE),
          sum(dat$Would_be_indeterminate & dat$Diagnosis == "MDD", na.rm = TRUE),
          sum(dat$Would_be_indeterminate & dat$Diagnosis == "ANX", na.rm = TRUE),
          sum(dat$Would_be_indeterminate & dat$Diagnosis == "BD", na.rm = TRUE),
          sum(dat$Would_be_indeterminate & dat$Diagnosis == "SSD", na.rm = TRUE)),
  "",
  "## Rein rechnerische Sensitivitätsannahme",
  "",
  sprintf("Für diese Analyse wurden alle %d unbestimmten Fälle als remittiert behandelt. Dies ergibt insgesamt %d acute und %d remitted.",
          sum(merged$Would_be_indeterminate), sum(counts$acute), sum(counts$remitted)),
  "",
  "| Subtyp | Acute | Remitted | Davon unbestimmt und als remitted gesetzt | Acute % |",
  "|---|---:|---:|---:|---:|",
  sprintf("| 1 | %d | %d | %d | %.1f |", counts$acute[1], counts$remitted[1], counts$reassigned_to_remitted[1], counts$acute_percent[1]),
  sprintf("| 2 | %d | %d | %d | %.1f |", counts$acute[2], counts$remitted[2], counts$reassigned_to_remitted[2], counts$acute_percent[2]),
  sprintf("| 3 | %d | %d | %d | %.1f |", counts$acute[3], counts$remitted[3], counts$reassigned_to_remitted[3], counts$acute_percent[3]),
  "",
  sprintf("Der globale Zusammenhang ist %s: χ²(2) = %.3f, p = %.4f, Cramér's V = %.3f.",
          if (chi$p.value < .05) "signifikant" else "nicht signifikant", unname(chi$statistic), chi$p.value, cramers_v),
  "",
  sprintf("Die BH-korrigierten Paarvergleiche sind %s: Subtyp 1 versus 2 q = %.3f, Subtyp 1 versus 3 q = %.3f und Subtyp 2 versus 3 q = %.3f.",
          if (all(pair_results$q_BH >= .05)) "alle nicht signifikant" else "teilweise signifikant",
          pair_results$q_BH[1], pair_results$q_BH[2], pair_results$q_BH[3]),
  "",
  "## Interpretation",
  "",
  sprintf("Diese Variante ist eine Sensitivitätsanalyse und keine belegte Hauptklassifikation. Fehlende SANS/SAPS-Ratings dürfen nur dann als Symptomfreiheit behandelt werden, wenn das Erhebungsprotokoll ausdrücklich bestätigt, dass eine Nichterhebung regelhaft das Nichtvorliegen dieser Symptome bedeutete. Die Anwesenheit von %d SSD-Patienten unter den unbestimmten Fällen spricht gegen eine pauschale Gleichsetzung von fehlend und symptomfrei.",
          sum(dat$Would_be_indeterminate & dat$Diagnosis == "SSD", na.rm = TRUE))
)
writeLines(report, file.path(output_dir, "Sensitivitaetsanalyse_Interpretation.md"), useBytes = TRUE)

print(tab)
print(chi)
print(cramers_v)
print(pair_results)
print(diagnosis_missing)
