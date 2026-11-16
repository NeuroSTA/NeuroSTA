#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pySuStaIn.ZscoreSustain import ZscoreSustain
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import HPC_DIR, INPUT_DIR

# ==========================================================
# Paths
# ==========================================================
DATA_FILE = str(INPUT_DIR / "residuals_z_scored_mainflip_SZ.csv")
ZMAX_FILE = str(INPUT_DIR / "Zmax_global_patients_only.csv")

BASE_OUT = str(HPC_DIR / "SuStaIn_diag" / "SZ" / "folds")
OUT_LOG  = f"{BASE_OUT}/logs"
OUT_RES  = f"{BASE_OUT}/results"

os.makedirs(BASE_OUT, exist_ok=True)
os.makedirs(OUT_LOG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

print("\n=== STARTING THREE-FOLD CROSS-VALIDATION (SZ) ===")
print("Input:", DATA_FILE)
print("Zmax:", ZMAX_FILE)
print("BASE_OUT:", BASE_OUT)

# ==========================================================
# Load the input data
# ==========================================================
df = pd.read_csv(DATA_FILE)

print("\nInput file columns:")
print(df.columns.tolist())

# ROI data from column index 2 onward
X = df.iloc[:, 2:].to_numpy(dtype=float)
y = df["Group"].astype(int).to_numpy()

biomarkers = df.columns[2:].tolist()
N = len(biomarkers)

if list(df.columns[:2]) != ["Proband", "Group"] or N != 62:
    raise ValueError("Expected Proband, Group, followed by exactly 62 ROI columns.")
if df["Proband"].astype(str).duplicated().any() or not np.isfinite(X).all():
    raise ValueError("Input contains duplicate participant IDs or non-finite ROI values.")

print(f"\nData shape = {X.shape} (Subjects x Biomarker = {X.shape[0]} x {N})")
classes, counts = np.unique(y, return_counts=True)
print("Group counts for StratifiedKFold:")
for c, n in zip(classes, counts):
    print(f"  Group {c}: n = {n}")

# ==========================================================
# Load and align the Zmax values
# ==========================================================
zdf = pd.read_csv(ZMAX_FILE)
zdf["ROI"] = zdf["ROI"].astype(str).str.strip()
zdf = zdf.set_index("ROI")

missing = [roi for roi in biomarkers if roi not in zdf.index]
if missing:
    raise ValueError("Zmax is missing for the following ROIs: " + str(missing))

Z_max = zdf.loc[biomarkers, "Zmax"].to_numpy(dtype=float)

if zdf.index.duplicated().any() or not np.isfinite(Z_max).all() or np.any(Z_max <= 0):
    raise ValueError("Zmax contains duplicate ROIs, non-finite values, or non-positive values.")

# ==========================================================
# Z_vals: one threshold for all biomarkers
# ==========================================================
Z_vals = np.ones((N, 1), dtype=float)

print("\nZ_vals Shape:", Z_vals.shape)
print("First five Z_max values:", Z_max[:5])

# ==========================================================
# SuStaIn parameters
# ==========================================================
N_S_MAX = 6
N_START = 25
N_ITER  = int(1e5)
SEED    = 42

print("\nSuStaIn parameters:")
print(f" N_S_max       = {N_S_MAX}")
print(f" N_startpoints = {N_START}")
print(f" N_MCMC        = {N_ITER}")
print(f" SEED          = {SEED}")

# ==========================================================
# CV-Splits
# ==========================================================
print("\nCreating three stratified folds.")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
test_idxs = [test for _, test in cv.split(X, y)]
test_idxs = np.array(test_idxs, dtype=object)

print(" Fold Sizes:", [len(idx) for idx in test_idxs])

# ==========================================================
# Initialize the SuStaIn model
# ==========================================================
model = ZscoreSustain(
    data=X,
    Z_vals=Z_vals,
    Z_max=Z_max,
    biomarker_labels=biomarkers,
    N_startpoints=N_START,
    N_S_max=N_S_MAX,
    N_iterations_MCMC=N_ITER,
    output_folder=OUT_RES,
    dataset_name="SuStaIn_CV_3fold_SZ",
    use_parallel_startpoints=True,
    seed=SEED
)

# ==========================================================
# Run cross-validation
# ==========================================================
print("\n=== STARTING SuStaIn CROSS-VALIDATION (SZ) ===")
CVIC, loglike = model.cross_validate_sustain_model(test_idxs)

print("\nCVIC-Shape:", CVIC.shape)
print("LogLike-Shape:", loglike.shape)

np.save(f"{OUT_RES}/cvic_matrix.npy", CVIC)
np.save(f"{OUT_RES}/loglike_matrix.npy", loglike)

# ==========================================================
# Summarize CVIC and log-likelihood for summary.csv
# ==========================================================
# CVIC is a vector with one value per K
cvic_vec = np.array(CVIC, dtype=float).reshape(-1)

if cvic_vec.shape[0] != N_S_MAX:
    print("WARNING: The number of CVIC values differs from N_S_MAX.")
    print("cvic_vec.shape[0] =", cvic_vec.shape[0], "N_S_MAX =", N_S_MAX)

# Average log-likelihood across folds (axis 0 of the n_folds by K array)
mean_ll = np.nanmean(loglike, axis=0)

summary = pd.DataFrame({
    "K": np.arange(1, N_S_MAX + 1),
    "CVIC": cvic_vec,
    "LogLikelihood": mean_ll
})
summary.to_csv(f"{OUT_RES}/summary.csv", index=False)

print("\n=== CROSS-VALIDATION SUMMARY (SZ) ===")
print(summary)

print("\nThree-fold cross-validation (SZ) completed.")
print("Results saved to:", OUT_RES)
