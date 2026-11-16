#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pySuStaIn.ZscoreSustain import ZscoreSustain
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import HPC_DIR, INPUT_DIR

# ==========================================================
# Paths
# ==========================================================
DATA_FILE = str(INPUT_DIR / "residuals_z_scored_mainflip_Angst.csv")
ZMAX_FILE = str(INPUT_DIR / "Zmax_global_patients_only.csv")

OUTPUT_MAIN = str(HPC_DIR / "SuStaIn_diag" / "Angst" / "main")
OUTPUT_LOG  = f"{OUTPUT_MAIN}/logs"
OUTPUT_RES  = f"{OUTPUT_MAIN}/results"

os.makedirs(OUTPUT_MAIN, exist_ok=True)
os.makedirs(OUTPUT_LOG, exist_ok=True)
os.makedirs(OUTPUT_RES, exist_ok=True)

print("\n=== Starting the SuStaIn main run: ANX ===")
print("Input:", DATA_FILE)
print("Zmax:", ZMAX_FILE)

# ==========================================================
# Load the input data
# ==========================================================
df = pd.read_csv(DATA_FILE)

print("\nInput file columns:")
print(df.columns.tolist())

# Check the input column order
expected_first_cols = ["Proband", "Group"]
if not all(col in df.columns[:2] for col in expected_first_cols):
    print("\nWARNING: The first two columns are not ['Proband', 'Group'].")
    print("Found:", df.columns[:5].tolist())

# ==========================================================
# Extract ROI data from column index 2 onward
#   Column 0: Proband
#   Column 1: Group
#   Columns 2 onward: ROIs
# ==========================================================
data = df.iloc[:, 2:].to_numpy(dtype=float)
biomarkers = df.columns[2:].tolist()
N = len(biomarkers)

if list(df.columns[:2]) != ["Proband", "Group"]:
    raise ValueError("The first two columns must be exactly Proband and Group.")
if N != 62:
    raise ValueError(f"Expected 62 ROI columns, found {N}.")
if df["Proband"].astype(str).duplicated().any():
    raise ValueError("Duplicate participant IDs found in the input table.")
if not np.isfinite(data).all():
    raise ValueError("ROI input contains missing or non-finite values.")

print("\nROI matrix shape:", data.shape)
print("N Biomarker =", N)

# ==========================================================
# Load Zmax and align it by ROI name
#   R output: one row per ROI
#   Columns: ROI, q90, q95, q99, max_abs, mean_abs, sd_abs, Zmax
# ==========================================================
zdf = pd.read_csv(ZMAX_FILE)

print("\nZmax file columns:")
print(zdf.columns.tolist())

if "ROI" not in zdf.columns or "Zmax" not in zdf.columns:
    raise ValueError("The Zmax file must contain 'ROI' and 'Zmax' columns.")

# Strip whitespace from ROI names
zdf["ROI"] = zdf["ROI"].astype(str).str.strip()
zdf["Zmax"] = pd.to_numeric(zdf["Zmax"], errors="coerce")
zdf = zdf.set_index("ROI")

# Check that Zmax includes all biomarkers
missing_in_zmax = [roi for roi in biomarkers if roi not in zdf.index]
extra_in_zmax   = [roi for roi in zdf.index if roi not in biomarkers]

if missing_in_zmax:
    raise ValueError(
        f"Zmax is missing for the following ROIs: {missing_in_zmax}"
    )

if extra_in_zmax:
    print("\nThe Zmax file contains ROIs outside this dataset:")
    print(extra_in_zmax)

# Align Zmax values with the biomarker order
Z_max = zdf.loc[biomarkers, "Zmax"].to_numpy(dtype=float)

if zdf.index.duplicated().any() or not np.isfinite(Z_max).all() or np.any(Z_max <= 0):
    raise ValueError("Zmax contains duplicate ROIs, non-finite values, or non-positive values.")

if len(Z_max) != N:
    raise ValueError(f"Found {len(Z_max)} Zmax values, expected {N}.")

print("\nFirst 10 Zmax values (ROI -> Zmax):")
for roi, z in list(zip(biomarkers, Z_max))[:10]:
    print(f"  {roi}: {z}")

# ==========================================================
# Z_vals: one threshold (z=1) for all biomarkers
# ==========================================================
Z_vals = np.ones((N, 1), dtype=float)

print("\nZ_max and Z_vals loaded and aligned.")
print("Z_vals Shape:", Z_vals.shape)

# ==========================================================
# SuStaIn parameters
# ==========================================================
N_startpoints = 25
N_S_max = 6              # maximum number of subtypes (1 through 6)
N_iterations_MCMC = int(1e5)

dataset_name = "SuStaIn_mainRun_Angst"

print("\nSuStaIn parameters:")
print(f"  N_startpoints       = {N_startpoints}")
print(f"  N_S_max             = {N_S_max}")
print(f"  N_iterations_MCMC   = {N_iterations_MCMC}")
print(f"  dataset_name        = {dataset_name}")

# ==========================================================
# Initialize the SuStaIn model
# ==========================================================
model = ZscoreSustain(
    data=data,
    Z_vals=Z_vals,
    Z_max=Z_max,
    biomarker_labels=biomarkers,
    N_startpoints=N_startpoints,
    N_S_max=N_S_max,
    N_iterations_MCMC=N_iterations_MCMC,
    output_folder=OUTPUT_RES,
    dataset_name=dataset_name,
    use_parallel_startpoints=True,
    seed=42
)

print("\nModel initialized. Starting MCMC.")

# ==========================================================
# Run SuStaIn
# ==========================================================
(
    samples_sequence,
    samples_f,
    ml_subtype,
    prob_ml_subtype,
    ml_stage,
    prob_ml_stage,
    prob_subtype_stage
) = model.run_sustain_algorithm()

print("\n=== SuStaIn main run completed (ANX) ===")
print("Results saved to:", OUTPUT_RES)

# Canonical participant assignments are exported after all model files exist by
# export_participant_assignments.py, which aligns participants and defines
# every posterior probability unambiguously.
