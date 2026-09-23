#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd
from project_paths import HPC_DIR, OUTPUT_DIR

# ==========================================================
# Einstellungen
# ==========================================================

BASE_DIR = str(HPC_DIR)
OUT_DIR = str(OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Laufbereich: K=2..6 (entspricht subtype1..subtype5)
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
    """Laedt ml_subtype aus einem pySuStaIn-Pickle."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle nicht gefunden: {pickle_path}")

    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(
            f"Unerwarteter Pickle-Typ ({type(obj)}). "
            f"Erwarte ein Dict mit Key 'ml_subtype'."
        )

    if "ml_subtype" not in obj:
        raise KeyError(f"'ml_subtype' fehlt im Pickle: {pickle_path}")

    return np.asarray(obj["ml_subtype"])


def compute_counts_for_cohort_and_K(cohort_name, cfg, K_model):
    """
    Berechnet n und % pro Subtyp fuer eine Kohorte und ein gegebenes K_model
    (nur fuer Auswahl der richtigen Datei). Labels werden automatisch als 0- oder 1-basiert erkannt.
    """
    pickle_dir = cfg["pickle_dir"]
    prefix = cfg["file_prefix"]

    file_index = K_model - 1  # subtype0 -> K=1, subtype5 -> K=6
    pickle_file = f"{prefix}{file_index}.pickle"
    pickle_path = os.path.join(pickle_dir, pickle_file)

    print(f"\n=== Kohorte {cohort_name} | K={K_model}: lade {pickle_path} ===")

    ml_subtype = load_ml_subtype(pickle_path).astype(int)

    unique_labels = np.unique(ml_subtype)
    print(f"  Labels (roh): {unique_labels}")

    n_total = int(ml_subtype.shape[0])

    if n_total == 0:
        raise ValueError("ml_subtype ist leer (n=0).")

    # 0-basiert (typisch pySuStaIn) -> Ausgabe 1..K
    if unique_labels.min() == 0:
        label_offset = 1
    else:
        label_offset = 0

    # Hinweis falls weniger Labels belegt als K_model
    if len(unique_labels) < K_model:
        print(
            f"  [Hinweis] Datei formal K={K_model}, aber nur {len(unique_labels)} Labels belegt."
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
        print(f"STARTE EXPORT: K={K_model} (subtype{K_model-1}.pickle)")
        print("=" * 70)

        for cohort_name, cfg in COHORT_CONFIG.items():
            try:
                df_cohort = compute_counts_for_cohort_and_K(cohort_name, cfg, K_model)
                all_tables.append(df_cohort)
            except FileNotFoundError as e:
                print(f"  [SKIP] {cohort_name} | K={K_model}: {e}")
                continue
            except Exception as e:
                print(f"  [FEHLER] {cohort_name} | K={K_model}: {e}")
                continue

        if not all_tables:
            print(f"[WARN] K={K_model}: keine Kohorte erfolgreich. Keine Datei geschrieben.")
            continue

        result_df = pd.concat(all_tables, ignore_index=True)

        out_path = os.path.join(OUT_DIR, f"Subtype_counts_K{K_model}.csv")
        result_df.to_csv(out_path, index=False)
        any_export = True

        print("\n=== Zusammenfassung Subtyp-Verteilungen ===")
        print(result_df)
        print(f"\nGespeichert: {out_path}")

    if not any_export:
        print("\n[WARN] Es wurde keine Datei exportiert. Pfade/Pickles pruefen.")


if __name__ == "__main__":
    main()
