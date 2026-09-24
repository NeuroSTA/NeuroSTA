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
# Pfade
# ==========================================================
DATA_FILE = str(INPUT_DIR / "residuals_z_scored_mainflip_MDD.csv")
ZMAX_FILE = str(INPUT_DIR / "Zmax_global_patients_only.csv")

OUTPUT_MAIN = str(HPC_DIR / "SuStaIn_diag" / "MDD" / "main")
OUTPUT_LOG  = f"{OUTPUT_MAIN}/logs"
OUTPUT_RES  = f"{OUTPUT_MAIN}/results"

os.makedirs(OUTPUT_MAIN, exist_ok=True)
os.makedirs(OUTPUT_LOG, exist_ok=True)
os.makedirs(OUTPUT_RES, exist_ok=True)

print("\n=== Starte SuStaIn Hauptlauf: MDD ===")
print("Input:", DATA_FILE)
print("Zmax:", ZMAX_FILE)

# ==========================================================
# Daten laden
# ==========================================================
df = pd.read_csv(DATA_FILE)

print("\nSpalten erkannt im Daten-File:")
print(df.columns.tolist())

# Minimaler Sanity-Check
expected_first_cols = ["Proband", "Group"]
if not all(col in df.columns[:2] for col in expected_first_cols):
    print("\nWARNUNG: Erste Spalten sind nicht ['Proband', 'Group'].")
    print("Gefunden:", df.columns[:5].tolist())

# ==========================================================
# ROI-Daten extrahieren (ab Spalte 2: ROIs)
#   Spalte 0: Proband
#   Spalte 1: Group
#   Spalte 2+: ROIs
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

print("\nROI-Daten Shape:", data.shape)
print("N Biomarker =", N)

# ==========================================================
# Zmax laden und an ROI-Namen ausrichten
#   R-Output: eine Zeile pro ROI
#   Spalten: ROI, q90, q95, q99, max_abs, mean_abs, sd_abs, Zmax
# ==========================================================
zdf = pd.read_csv(ZMAX_FILE)

print("\nSpalten erkannt im Zmax-File:")
print(zdf.columns.tolist())

if "ROI" not in zdf.columns or "Zmax" not in zdf.columns:
    raise ValueError("FEHLER: Zmax-Datei muss mindestens die Spalten 'ROI' und 'Zmax' enthalten!")

# ROI-Namen trimmen (Vorsicht bei Leerzeichen)
zdf["ROI"] = zdf["ROI"].astype(str).str.strip()
zdf["Zmax"] = pd.to_numeric(zdf["Zmax"], errors="coerce")
zdf = zdf.set_index("ROI")

# Prüfen, ob alle Biomarker im Zmax-File vorhanden sind
missing_in_zmax = [roi for roi in biomarkers if roi not in zdf.index]
extra_in_zmax   = [roi for roi in zdf.index if roi not in biomarkers]

if missing_in_zmax:
    raise ValueError(
        f"FEHLER: Für folgende ROIs aus den Daten fehlt ein Zmax-Eintrag: {missing_in_zmax}"
    )

if extra_in_zmax:
    print("\nHinweis: Zmax-Datei enthält zusätzliche ROIs, die nicht im MDD-Datensatz vorkommen:")
    print(extra_in_zmax)

# Zmax in der exakt richtigen Reihenfolge ziehen
Z_max = zdf.loc[biomarkers, "Zmax"].to_numpy(dtype=float)

if zdf.index.duplicated().any() or not np.isfinite(Z_max).all() or np.any(Z_max <= 0):
    raise ValueError("Zmax contains duplicate ROIs, non-finite values, or non-positive values.")

if len(Z_max) != N:
    raise ValueError(f"FEHLER: Zmax hat {len(Z_max)} Werte, erwartet wurden {N}!")

print("\nErste 10 Zmax-Werte (ROI -> Zmax):")
for roi, z in list(zip(biomarkers, Z_max))[:10]:
    print(f"  {roi}: {z}")

# ==========================================================
# Z_vals: nur eine Schwelle (z=1) für alle Biomarker
# ==========================================================
Z_vals = np.ones((N, 1), dtype=float)

print("\nZ_max und Z_vals geladen und ausgerichtet.")
print("Z_vals Shape:", Z_vals.shape)

# ==========================================================
# SuStaIn Parameter
# ==========================================================
N_startpoints = 25
N_S_max = 6              # max. Subtypen (1..6)
N_iterations_MCMC = int(1e5)

dataset_name = "SuStaIn_mainRun_MDD"

print("\nSuStaIn-Parameter:")
print(f"  N_startpoints       = {N_startpoints}")
print(f"  N_S_max             = {N_S_max}")
print(f"  N_iterations_MCMC   = {N_iterations_MCMC}")
print(f"  dataset_name        = {dataset_name}")

# ==========================================================
# SuStaIn Model initialisieren
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

print("\nInitialisierung erfolgreich. Starte MCMC...")

# ==========================================================
# SuStaIn ausführen
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

print("\n=== SuStaIn Hauptlauf MDD abgeschlossen ===")
print("Ergebnisse gespeichert unter:", OUTPUT_RES)

# Canonical participant assignments are exported after all model files exist by
# export_participant_assignments.py, which aligns participants and defines
# every posterior probability unambiguously.
