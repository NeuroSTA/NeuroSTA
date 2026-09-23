# Reproduction code for transdiagnostic SuStaIn analyses

This repository contains the analysis code for data-driven subtype and disease-stage inference in a transdiagnostic psychiatric sample. Participant-level data are not included. Qualified researchers must obtain the controlled-access data separately and place them in the local directory structure described below.

## Scope

The code covers:

1. healthy-control normalization and medication adjustment of 62 regional GMV measures;
2. deterministic construction of the transdiagnostic and diagnosis-specific SuStaIn inputs;
3. transdiagnostic and diagnosis-specific SuStaIn main runs and three-fold cross-validation;
4. model selection, subtype and stage analyses, figures, and quality-control outputs.

The diagnosis-specific SuStaIn input filenames contain `mainflip` to match the model configuration. No additional sign transformation is performed. These files are diagnosis-specific row subsets of the common patient z-score table.

## Software

The analyses were performed with R 4.3.1 and Python 3.9.23. Direct R package versions are listed in `r-requirements.txt`. Recorded Python dependency versions are listed in `requirements.txt`; tqdm is listed without an exact version because its version was not recorded. The SuStaIn implementation uses pySuStaIn commit `45a39ab`.

## Tested operating systems and hardware

The SuStaIn main models and three-fold cross-validation were run on a SLURM-managed x86_64 high-performance computing cluster using Rocky Linux 8.10. The submitted jobs used CPU resources only; no GPU was requested or required. The R preprocessing and downstream analysis code was additionally checked with R 4.3.1 on macOS 26.1 using Apple silicon (`aarch64`).

The SLURM resource requests used for the reported model runs were:

| Analysis | CPU cores | Requested memory | Wall-time limit |
|---|---:|---:|---:|
| Transdiagnostic main model | 26 | 120 GB | 10 days |
| Transdiagnostic cross-validation | 26 | 120 GB | 10 days |
| MDD main model | 10 | 32 GB | 10 days |
| MDD cross-validation | 10 | 16 GB | 10 days |
| BD main model and cross-validation | 8 | 16 GB | 6 and 10 days |
| SSD main model | 8 | 16 GB | 10 days |
| SSD cross-validation | 10 | 16 GB | 10 days |
| ANX main model and cross-validation | 8 | 16 GB | 10 days |

These values are the tested resource allocations, not experimentally established minimum requirements. The downstream R analyses require substantially fewer resources than the SuStaIn estimation jobs.

## Installation time

Dependency installation was not timed during the original analysis. On a current workstation or compute node, allow approximately 30 to 60 minutes to create the Python environment and install the listed Python and R packages. This is an estimate rather than a benchmark and depends on network speed, availability of binary R packages, and whether packages must be compiled from source.

## Local configuration

Run all commands from the repository root. The default project root is the current working directory. Paths can be changed without editing any script:

```bash
export SUSTAIN_PROJECT_DIR=/path/to/local/project
export SUSTAIN_T1_ROI_FILE=/path/to/merged_data_avg_only.csv
export SUSTAIN_COVARIATE_FILE=/path/to/Datenbank_Update_DataFreeze_bereinigt.csv
```

Additional optional variables are documented in `config.example.env`.

## Controlled-access inputs

By default, the preparation script expects:

```text
private_data/
  merged_data_avg_only.csv
  Datenbank_Update_DataFreeze_bereinigt.csv
  Datenbank_Update_DataFreeze_bereinigt_withGlobalRatings.csv
```

The ROI file must contain one participant identifier and exactly 62 columns ending in `_avg`. The covariate file must contain a participant identifier plus `Group`, `Alter`, `Geschlecht`, `TIV`, `Dummy_BC_MR_pre`, `Dummy_BC_MR_post`, and `Sum_MED`. The third controlled-access file contains the SANS and SAPS global-rating variables required for clinical-status classification. Its provenance must be supplied with the controlled-access data.

Group coding used for derived diagnosis files is:

| Output label | Group code |
|---|---:|
| MDD | 2 |
| BD | 3 |
| SSD | 4 and 5 |
| ANX | 8 |

## Reconstructing SuStaIn inputs

```bash
Rscript 00_prepare_sustain_inputs.R
```

The script fits the covariate model only in healthy controls, applies it to all participants, estimates the medication association in patients, standardizes residuals using the healthy-control distribution, and reverses the sign so that higher z-scores represent greater atrophy.

It creates the patient and full-sample z-score tables, all four diagnosis-specific subsets, `Zmax_global_patients_only.csv`, and QC summaries under `HPC/input/`. For each ROI, `Zmax` is:

```r
ceiling(quantile(abs(z), 0.95, type = 7))
```

The script stops on missing required covariates, ID collisions, an unexpected ROI count, unmatched ROI records, missing patient medication values, or missing output values.

## Running SuStaIn

Create the Python environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The principal scripts are located in `HPC_Scripts/`. Local paths are derived from the environment variables above. The SLURM files contain resource requests from the original computation, but no personal account paths or email addresses. Cluster module and partition names may require local adaptation.

### Observed model runtimes

The following elapsed times were calculated from the start and completion timestamps in the original successful SLURM logs. They are representative only for the tested cluster and dataset. Runtime will vary with processor speed, cluster load, software configuration, and sample size.

| Cohort | Main model | Three-fold cross-validation |
|---|---:|---:|
| Transdiagnostic | 5 days 22 hours | 7 days 23 hours |
| MDD | 6 days 14 hours | 4 days 10 hours |
| BD | 1 day 10 hours | 2 days 21 hours |
| SSD | 1 day 14 hours | 1 day 18 hours |
| ANX | 2 days 8 hours | 3 days 18 hours |

Independent cohort jobs can be submitted in parallel when cluster resources permit. The transdiagnostic run required the largest tested allocation. Preprocessing, assignment export, model-selection summaries, and most downstream R analyses are short relative to model estimation, but their elapsed times were not formally recorded.

After the model runs, export participant assignments with:

```bash
python export_sustain_assignments.py
```

The exporter requires exact subject-count alignment and never truncates mismatched arrays. It retains the original maximum-likelihood subtype and stage labels and writes three distinctly named stage probabilities: the joint subtype-stage probability, the stage probability conditional on the selected subtype, and the marginal stage probability. Downstream analyses use the files under `Output_sustainrun/SuStaIn_assignments/`.

Create the model-selection, subtype-size, and K = 3 progression outputs with:

```bash
Rscript 03_Modellauswahl_CVIC.R
python compute_subtype_counts.py
python make_progression_TDM_K3.py
```

## Downstream analysis order

The numbered R scripts follow the manuscript analysis sequence. Script `09_Subtyp_Klinischer_Status.R` must run before script `07_Subtyp_Hauptsymptomskalen_HAMD17.R`, because script 07 incorporates the status omnibus p-value into the prespecified seven-test correction family.

`13_Stage_GMV.R` runs the reported `paper_z` analysis by default. Set `options(SUSTAIN_RUN_RAW_GMV = TRUE)` only to run the additional raw-GMV branch.

## Data protection

The `.gitignore` excludes controlled-access inputs, participant-level CSV files, SuStaIn pickle objects, generated results, logs, and local environments. Before publishing a release, inspect staged files with `git status` and confirm that no data or local paths are present.
