#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from pySuStaIn.ZscoreSustain import ZscoreSustain
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import HPC_DIR, INPUT_DIR

# ==========================================================
# Paths
# ==========================================================
DATA_FILE = str(INPUT_DIR / "residuals_z_scored_mainflip_MDD.csv")
ZMAX_FILE = str(INPUT_DIR / "Zmax_global_patients_only.csv")

BASE_OUT = str(HPC_DIR / "SuStaIn_diag" / "MDD" / "folds")
OUT_LOG  = f"{BASE_OUT}/logs"
OUT_RES  = f"{BASE_OUT}/results"

os.makedirs(BASE_OUT, exist_ok=True)
os.makedirs(OUT_LOG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

print("\n=== STARTING THREE-FOLD CROSS-VALIDATION (MDD) ===")
print("Input:", DATA_FILE)
print("Zmax:", ZMAX_FILE)
print("BASE_OUT:", BASE_OUT)

# ==========================================================
# Read input and Zmax tables with their required columns
# ==========================================================
def read_csv_robust(path, required_cols=None):
    """
    Tries common separators; ensures header is not parsed as a single combined column.
    """
    if required_cols is None:
        required_cols = []

    seps = [";", ",", "\t"]
    last_err = None

    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            # "everything in one column" case
            if df.shape[1] == 1 and any(s in str(df.columns[0]) for s in [",", ";", "\t"]):
                continue

            # Strip BOM + whitespace
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

            if required_cols and not all(c in df.columns for c in required_cols):
                continue

            return df
        except Exception as e:
            last_err = e

    # Auto-sniff fallback
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if required_cols and not all(c in df.columns for c in required_cols):
            raise ValueError("Required columns missing after auto-sniff.")
        return df
    except Exception as e:
        last_err = e

    raise ValueError(f"Could not parse file: {path}. Required={required_cols}. Last error: {last_err}")

# ==========================================================
# Load data with Proband and Group columns
# ==========================================================
df = read_csv_robust(DATA_FILE, required_cols=["Proband", "Group"])

print("\nInput file columns:")
print(df.columns.tolist())

# ROI data from column index 2 onward
X = df.iloc[:, 2:].to_numpy(dtype=float)
y = df["Group"].astype(int).to_numpy()

# Sanity: NaN/Inf check
if not np.isfinite(X).all():
    n_bad = int(np.sum(~np.isfinite(X)))
    raise ValueError(f"The ROI matrix contains {n_bad} non-finite values.")

biomarkers = df.columns[2:].tolist()
N = len(biomarkers)

if list(df.columns[:2]) != ["Proband", "Group"] or N != 62:
    raise ValueError("Expected Proband, Group, followed by exactly 62 ROI columns.")
if df["Proband"].astype(str).duplicated().any() or not np.isfinite(X).all():
    raise ValueError("Input contains duplicate participant IDs or non-finite ROI values.")

print(f"\nData shape = {X.shape} (Subjects x Biomarker = {X.shape[0]} x {N})")
classes, counts = np.unique(y, return_counts=True)
print("Group counts:")
for c, n in zip(classes, counts):
    print(f"  Group {c}: n = {n}")

# ==========================================================
# Load the Zmax table with ROI and Zmax columns
# ==========================================================
zdf = read_csv_robust(ZMAX_FILE, required_cols=["ROI", "Zmax"])
zdf["ROI"] = zdf["ROI"].astype(str).str.strip()
zdf["Zmax"] = pd.to_numeric(zdf["Zmax"], errors="coerce")
zdf = zdf.dropna(subset=["Zmax"]).set_index("ROI")

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
N_S_MAX  = 6
N_START  = 25
N_ITER   = int(1e5)
SEED     = 42
N_SPLITS = 3

print("\nSuStaIn parameters:")
print(f" N_S_max       = {N_S_MAX}")
print(f" N_startpoints = {N_START}")
print(f" N_MCMC        = {N_ITER}")
print(f" SEED          = {SEED}")
print(f" N_SPLITS      = {N_SPLITS}")

# ==========================================================
# Use KFold when the MDD input contains a single group
# ==========================================================
min_count = int(np.min(counts)) if len(counts) > 0 else 0
use_stratified = (len(classes) >= 2) and (min_count >= N_SPLITS)

if use_stratified:
    print("\nCreating three stratified folds with StratifiedKFold.")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    test_idxs = [test for _, test in cv.split(X, y)]
else:
    print("\nCreating three folds with KFold.")
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    test_idxs = [test for _, test in cv.split(X)]

test_idxs = np.array(test_idxs, dtype=object)
print("Fold Sizes:", [len(idx) for idx in test_idxs])

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
    dataset_name="SuStaIn_CV_3fold_MDD",
    use_parallel_startpoints=True,
    seed=SEED
)

# ==========================================================
# Run cross-validation
# ==========================================================
print("\n=== STARTING SuStaIn CROSS-VALIDATION (MDD) ===")
CVIC, loglike = model.cross_validate_sustain_model(test_idxs)

# Summarize CVIC and log-likelihood from one- or two-dimensional arrays
CVIC_arr = np.array(CVIC, dtype=float)
LL_arr   = np.array(loglike, dtype=float)

print("\nCVIC array shape:", CVIC_arr.shape)
print("LogLike array shape:", LL_arr.shape)

np.save(f"{OUT_RES}/cvic_matrix.npy", CVIC_arr)
np.save(f"{OUT_RES}/loglike_matrix.npy", LL_arr)

# Mean values for the summary
# - CVIC: average a folds-by-K array across folds or use the vector directly
if CVIC_arr.ndim == 2:
    mean_cvic = np.nanmean(CVIC_arr, axis=0)
else:
    mean_cvic = CVIC_arr.reshape(-1)

# - Log-likelihood: average a folds-by-K array across folds or use the vector directly
if LL_arr.ndim == 2:
    mean_ll = np.nanmean(LL_arr, axis=0)
else:
    mean_ll = LL_arr.reshape(-1)

# Trim or pad to N_S_MAX if needed
mean_cvic = np.asarray(mean_cvic, dtype=float).reshape(-1)
mean_ll   = np.asarray(mean_ll, dtype=float).reshape(-1)

if mean_cvic.shape[0] < N_S_MAX:
    mean_cvic = np.pad(mean_cvic, (0, N_S_MAX - mean_cvic.shape[0]), constant_values=np.nan)
if mean_ll.shape[0] < N_S_MAX:
    mean_ll = np.pad(mean_ll, (0, N_S_MAX - mean_ll.shape[0]), constant_values=np.nan)

summary = pd.DataFrame({
    "K": np.arange(1, N_S_MAX + 1),
    "CVIC": mean_cvic[:N_S_MAX],
    "LogLikelihood": mean_ll[:N_S_MAX]
})
summary.to_csv(f"{OUT_RES}/summary.csv", index=False)

print("\n=== CROSS-VALIDATION SUMMARY (MDD) ===")
print(summary)

print("\nThree-fold cross-validation (MDD) completed.")
print("Results saved to:", OUT_RES)
