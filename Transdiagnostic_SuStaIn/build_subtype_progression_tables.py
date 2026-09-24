#!/usr/bin/env python3
"""
Progression tables + paper-style plots for SuStaIn main-run pickles (K=3) across cohorts.

Cohorts processed:
- TDM (transdiagnostic main run, patientsOnly)
- MDD
- BD
- ANX (Angst)
- SSD (SZ folder, labeled as SSD)

Enforces K=3 by always loading subtype2.pickle (k_idx=2) and verifying K==3 in pickle.

Outputs:
OUTPUT_ROOT/
  TDM/  progression_table_subtype1..3.csv, plot_progression_TDM_subtype1..3.png, progression_table_all_subtypes.csv, qc_summary.txt
  MDD/  ...
  BD/   ...
  ANX/  ...
  SSD/  ...

Also writes:
OUTPUT_ROOT/qc_master_summary.csv
OUTPUT_ROOT/qc_master_log.txt
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import HPC_DIR, OUTPUT_DIR


# ============================================================
# 1) PATHS
# ============================================================
BASE_DIR = str(HPC_DIR)

OUTPUT_ROOT = str(OUTPUT_DIR / "progression_all_K3")
MAX_SAMPLES = 100000  # posterior subsampling (speed)

K_EXPECTED = 3
PICKLE_SUBTYPE_INDEX = 2  # K=3 -> subtype2.pickle


# ------------------------------------------------------------
# Cohort config
# ------------------------------------------------------------
COHORTS = {
    "TDM": {
        "label": "TDM",
        "pickle_path": os.path.join(
            BASE_DIR,
            "SuStaIn_mainRun",
            "results",
            "pickle_files",
            "SuStaIn_mainRun_patientsOnly_subtype2.pickle",
        ),
        "input_csv": os.path.join(
            BASE_DIR,
            "input",
            "residuals_z_scored_corrected_allGroups.csv",
        ),
    },
    "MDD": {
        "label": "MDD",
        "pickle_path": os.path.join(
            BASE_DIR,
            "SuStaIn_diag",
            "MDD",
            "main",
            "results",
            "pickle_files",
            "SuStaIn_mainRun_MDD_subtype2.pickle",
        ),
        "input_csv": os.path.join(
            BASE_DIR,
            "input",
            "residuals_z_scored_mainflip_MDD.csv",
        ),
    },
    "BD": {
        "label": "BD",
        "pickle_path": os.path.join(
            BASE_DIR,
            "SuStaIn_diag",
            "BD",
            "main",
            "results",
            "pickle_files",
            "SuStaIn_mainRun_BD_subtype2.pickle",
        ),
        "input_csv": os.path.join(
            BASE_DIR,
            "input",
            "residuals_z_scored_mainflip_BD.csv",
        ),
    },
    "ANX": {  # folder is Angst
        "label": "ANX",
        "pickle_path": os.path.join(
            BASE_DIR,
            "SuStaIn_diag",
            "Angst",
            "main",
            "results",
            "pickle_files",
            "SuStaIn_mainRun_Angst_subtype2.pickle",
        ),
        "input_csv": os.path.join(
            BASE_DIR,
            "input",
            "residuals_z_scored_mainflip_Angst.csv",
        ),
    },
    "SSD": {  # folder is SZ, but we label as SSD
        "label": "SSD",
        "pickle_path": os.path.join(
            BASE_DIR,
            "SuStaIn_diag",
            "SZ",
            "main",
            "results",
            "pickle_files",
            "SuStaIn_mainRun_SZ_subtype2.pickle",
        ),
        "input_csv": os.path.join(
            BASE_DIR,
            "input",
            "residuals_z_scored_mainflip_SZ.csv",
        ),
    },
}


# ============================================================
# 2) UTIL
# ============================================================
def stop_if_missing(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_roi_columns_from_input_csv(path: str) -> list[str]:
    """
    Detect ROI columns as everything after 'Group' in the header.
    Tries comma and semicolon separators.
    """
    last_err = None
    for sep in [",", ";"]:
        try:
            df0 = pd.read_csv(path, sep=sep, engine="python", nrows=0)
            cols = list(df0.columns)
            cols_low = [c.lower() for c in cols]
            if "group" not in cols_low:
                continue
            group_idx = cols_low.index("group")
            roi_cols = cols[group_idx + 1 :]
            if len(roi_cols) == 0:
                raise ValueError("No ROI columns found after Group.")
            return roi_cols
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not parse header / find ROI columns in {path}. Last error: {last_err}"
    )


def invert_permutation(seq):
    """
    seq: permutation of 0..E-1 (length E)
    returns inv where inv[roi_index] = position (0..E-1)
    """
    seq = np.asarray(seq).astype(int)
    E = seq.shape[0]
    inv = np.empty(E, dtype=int)
    inv[seq] = np.arange(E, dtype=int)
    return inv


def positions_from_samples(samples_seq_k):
    """
    samples_seq_k: shape (E, S), each column is a permutation of ROI indices
    Returns posterior distribution of positions for each ROI:
      mean_pos, lo, hi  (each length E; positions 0..E-1)
    """
    E, S = samples_seq_k.shape
    pos_samples = np.empty((E, S), dtype=np.int32)
    for s in range(S):
        perm = samples_seq_k[:, s].astype(int)
        inv = np.empty(E, dtype=np.int32)
        inv[perm] = np.arange(E, dtype=np.int32)
        pos_samples[:, s] = inv

    mean_pos = np.mean(pos_samples, axis=1)
    lo = np.quantile(pos_samples, 0.025, axis=1)
    hi = np.quantile(pos_samples, 0.975, axis=1)
    return mean_pos, lo, hi


def plot_progression_paper(df_sub, out_png, title):
    """
    Paper-style progression plot:
    - y-axis: ROI ordered by ML_stage (early top)
    - x-axis: stage index (1..E)
    - show posterior mean and 95% CI as horizontal errorbars
    - show ML_stage as separate marker
    """
    dfp = df_sub.sort_values("ML_stage").reset_index(drop=True)

    y = np.arange(len(dfp))
    roi_labels = dfp["ROI"].tolist()

    ml = dfp["ML_stage"].to_numpy(dtype=float)
    mu = dfp["post_mean_stage"].to_numpy(dtype=float)
    lo = dfp["post_lo_stage"].to_numpy(dtype=float)
    hi = dfp["post_hi_stage"].to_numpy(dtype=float)

    xerr = np.vstack([mu - lo, hi - mu])

    plt.figure(figsize=(10.5, 12.5))

    plt.errorbar(
        mu, y, xerr=xerr,
        fmt="o", markersize=3.6,
        elinewidth=0.8, capsize=2.5, alpha=0.85,
        label="Posterior mean (95% CI)"
    )

    plt.scatter(
        ml, y, s=14, alpha=0.65,
        label="ML order (point estimate)"
    )

    plt.yticks(y, roi_labels)
    plt.gca().invert_yaxis()

    plt.xlabel("Stage (1..E), smaller = earlier")
    plt.ylabel("ROI (sorted by ML order)")
    plt.title(title)

    plt.grid(axis="x", alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# 3) COHORT PROCESSOR
# ============================================================
def process_cohort(cohort_key: str, cfg: dict) -> dict:
    label = cfg["label"]
    pickle_path = cfg["pickle_path"]
    input_csv = cfg["input_csv"]

    out_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    qc = {
        "cohort": label,
        "pickle_path": pickle_path,
        "input_csv": input_csv,
        "status": "OK",
        "K": None,
        "E": None,
        "S_total": None,
        "S_used": None,
        "error": "",
        "out_dir": out_dir,
    }

    try:
        stop_if_missing(pickle_path)
        stop_if_missing(input_csv)

        data = load_pickle(pickle_path)

        if "ml_sequence_EM" not in data:
            raise KeyError("ml_sequence_EM missing in pickle.")
        if "samples_sequence" not in data:
            raise KeyError("samples_sequence missing in pickle.")

        ml_seq = np.asarray(data["ml_sequence_EM"]).astype(int)         # (K, E)
        samp_seq = np.asarray(data["samples_sequence"]).astype(int)     # (K, E, S)

        if ml_seq.ndim != 2:
            raise RuntimeError(f"ml_sequence_EM not 2D: shape={ml_seq.shape}")
        if samp_seq.ndim != 3:
            raise RuntimeError(f"samples_sequence not 3D: shape={samp_seq.shape}")

        K, E = ml_seq.shape
        if samp_seq.shape[0] != K or samp_seq.shape[1] != E:
            raise RuntimeError(
                f"samples_sequence shape mismatch: {samp_seq.shape} vs ml_sequence_EM {ml_seq.shape}"
            )

        qc["K"] = int(K)
        qc["E"] = int(E)
        qc["S_total"] = int(samp_seq.shape[2])

        if K != K_EXPECTED:
            raise RuntimeError(
                f"Expected K={K_EXPECTED} but pickle has K={K}. "
                f"Use *_subtype{PICKLE_SUBTYPE_INDEX}.pickle for K=3."
            )

        roi_cols = read_roi_columns_from_input_csv(input_csv)
        if len(roi_cols) != E:
            raise RuntimeError(
                f"ROI columns in input_csv ({len(roi_cols)}) != E ({E}). "
                "Wrong CSV for this pickle."
            )

        # subsample posterior samples
        S_total = samp_seq.shape[2]
        S_use = min(MAX_SAMPLES, S_total)
        qc["S_used"] = int(S_use)

        if S_use < S_total:
            idx = np.linspace(0, S_total - 1, S_use).astype(int)
            samp_seq_use = samp_seq[:, :, idx]
        else:
            samp_seq_use = samp_seq

        all_rows = []

        for k in range(K):
            inv = invert_permutation(ml_seq[k, :])      # inv[roi] = position 0..E-1
            ml_pos = inv + 1                             # 1..E
            ml_stage = ml_pos                            # stage index

            mean_pos, lo, hi = positions_from_samples(samp_seq_use[k, :, :])
            post_mean_stage = mean_pos + 1.0
            post_lo_stage = lo + 1.0
            post_hi_stage = hi + 1.0

            dfk = pd.DataFrame({
                "Subtype": (k + 1),
                "ROI_index_0based": np.arange(E, dtype=int),
                "ROI": roi_cols,
                "ML_position_1based": ml_pos.astype(int),
                "ML_stage": ml_stage.astype(int),
                "post_mean_stage": post_mean_stage,
                "post_lo_stage": post_lo_stage,
                "post_hi_stage": post_hi_stage,
            }).sort_values("ML_stage")

            out_csv = os.path.join(out_dir, f"progression_table_subtype{k+1}.csv")
            dfk.to_csv(out_csv, index=False)
            all_rows.append(dfk)

            out_png = os.path.join(out_dir, f"plot_progression_{label}_subtype{k+1}.png")
            plot_progression_paper(
                dfk,
                out_png,
                title=f"SuStaIn progression ({label}, K=3, Subtype {k+1})"
            )

        df_all = pd.concat(all_rows, axis=0, ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, "progression_table_all_subtypes.csv"), index=False)

        with open(os.path.join(out_dir, "qc_summary.txt"), "w") as f:
            f.write(f"COHORT: {label}\n")
            f.write(f"PICKLE: {pickle_path}\n")
            f.write(f"INPUT_CSV: {input_csv}\n")
            f.write(f"K={K}, E={E}, S_total={S_total}, posterior_samples_used={samp_seq_use.shape[2]}\n")

    except Exception as e:
        qc["status"] = "FAIL"
        qc["error"] = str(e)
        # write cohort error file
        with open(os.path.join(out_dir, "qc_error.txt"), "w") as f:
            f.write(f"COHORT: {label}\n")
            f.write(f"ERROR: {str(e)}\n")

    return qc


# ============================================================
# 4) MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    qc_rows = []
    qc_log_path = os.path.join(OUTPUT_ROOT, "qc_master_log.txt")

    with open(qc_log_path, "w") as log:
        log.write(f"OUTPUT_ROOT: {OUTPUT_ROOT}\n")
        log.write(f"MAX_SAMPLES: {MAX_SAMPLES}\n")
        log.write(f"K_EXPECTED: {K_EXPECTED}\n")
        log.write("\n")

        for cohort_key, cfg in COHORTS.items():
            qc = process_cohort(cohort_key, cfg)
            qc_rows.append(qc)

            log.write(f"[{qc['cohort']}] status={qc['status']} K={qc['K']} E={qc['E']} S_used={qc['S_used']}\n")
            if qc["status"] != "OK":
                log.write(f"  ERROR: {qc['error']}\n")

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(os.path.join(OUTPUT_ROOT, "qc_master_summary.csv"), index=False)

    print("OK")
    print("Wrote:", OUTPUT_ROOT)
    print("QC:", os.path.join(OUTPUT_ROOT, "qc_master_summary.csv"))


if __name__ == "__main__":
    main()
