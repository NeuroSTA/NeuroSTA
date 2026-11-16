#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd
from project_paths import HPC_DIR, OUTPUT_DIR

# ==========================================================
# Settings
# ==========================================================

BASE_DIR = str(HPC_DIR)
OUT_DIR = str(OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Model range: K=2 through 6 (subtype1 through subtype5)
K_MIN = 2
K_MAX = 6

COHORT_CONFIG = {
    "main": {
        "pickle_dir": os.path.join(BASE_DIR, "SuStaIn_mainRun", "results", "pickle_files"),
        "file_prefix": "SuStaIn_mainRun_patientsOnly_subtype"
    },
    "MDD": {
        "pickle_dir": os.path.join(BASE_DIR, "SuStaIn_diag", "MDD", "main", "results", "pickle_files"),
        "file_prefix": "SuStaIn_mainRun_MDD_subtype"
    },
    "BD": {
        "pickle_dir": os.path.join(BASE_DIR, "SuStaIn_diag", "BD", "main", "results", "pickle_files"),
        "file_prefix": "SuStaIn_mainRun_BD_subtype"
    },
    "SZ": {
        "pickle_dir": os.path.join(BASE_DIR, "SuStaIn_diag", "SZ", "main", "results", "pickle_files"),
        "file_prefix": "SuStaIn_mainRun_SZ_subtype"
    },
    "Angst": {
        "pickle_dir": os.path.join(BASE_DIR, "SuStaIn_diag", "Angst", "main", "results", "pickle_files"),
        "file_prefix": "SuStaIn_mainRun_Angst_subtype"
    },
}


def load_ml_subtype(pickle_path):
    """Load ml_subtype from a pySuStaIn pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(
            f"Unexpected pickle object type ({type(obj)}). "
            f"Expected a dictionary containing 'ml_subtype'."
        )

    if "ml_subtype" not in obj:
        raise KeyError(f"Pickle file is missing 'ml_subtype': {pickle_path}")

    return np.asarray(obj["ml_subtype"])


def compute_counts_for_cohort_and_K(cohort_name, cfg, K_model):
    """
    Compute subtype counts and percentages for a cohort and model size.
    Detect whether subtype labels start at zero or one.
    """
    pickle_dir = cfg["pickle_dir"]
    prefix = cfg["file_prefix"]

    file_index = K_model - 1  # subtype0 -> K=1, subtype5 -> K=6
    pickle_file = f"{prefix}{file_index}.pickle"
    pickle_path = os.path.join(pickle_dir, pickle_file)

    print(f"\n=== Cohort {cohort_name} | K={K_model}: loading {pickle_path} ===")

    ml_subtype = load_ml_subtype(pickle_path).astype(int)

    unique_labels = np.unique(ml_subtype)
    print(f"  Raw labels: {unique_labels}")

    n_total = int(ml_subtype.shape[0])

    if n_total == 0:
        raise ValueError("ml_subtype is empty.")

    # Convert zero-based pySuStaIn labels to one-based labels
    if unique_labels.min() == 0:
        label_offset = 1
    else:
        label_offset = 0

    # Report models with fewer observed labels than model subtypes
    if len(unique_labels) < K_model:
        print(
            f"  Model K={K_model} has {len(unique_labels)} observed labels."
        )

    rows = []
    for lab in unique_labels:
        subtype_display = int(lab + label_offset)
        n_s = int(np.sum(ml_subtype == lab))
        pct_s = 100.0 * n_s / n_total

        rows.append(
            {
                "cohort": cohort_name,
                "K_model": int(K_model),
                "K_eff": int(len(unique_labels)),
                "N_total": int(n_total),
                "subtype": int(subtype_display),
                "N_subtype": int(n_s),
                "pct_subtype": float(pct_s),
            }
        )

    return pd.DataFrame(rows)


def main():
    any_export = False

    for K_model in range(K_MIN, K_MAX + 1):
        all_tables = []

        print("\n" + "=" * 70)
        print(f"STARTING EXPORT: K={K_model} (subtype{K_model-1}.pickle)")
        print("=" * 70)

        for cohort_name, cfg in COHORT_CONFIG.items():
            try:
                df_cohort = compute_counts_for_cohort_and_K(cohort_name, cfg, K_model)
                all_tables.append(df_cohort)
            except FileNotFoundError as e:
                print(f"  [SKIP] {cohort_name} | K={K_model}: {e}")
                continue
            except Exception as e:
                print(f"  [ERROR] {cohort_name} | K={K_model}: {e}")
                continue

        if not all_tables:
            print(f"[WARN] K={K_model}: No cohort could be summarized. No file was written.")
            continue

        result_df = pd.concat(all_tables, ignore_index=True)

        out_path = os.path.join(OUT_DIR, f"Subtype_counts_K{K_model}.csv")
        result_df.to_csv(out_path, index=False)
        any_export = True

        print("\n=== Subtype count summary ===")
        print(result_df)
        print(f"\nSaved to: {out_path}")

    if not any_export:
        print("\n[WARN] No files were exported. Check the paths and pickle files.")


if __name__ == "__main__":
    main()
