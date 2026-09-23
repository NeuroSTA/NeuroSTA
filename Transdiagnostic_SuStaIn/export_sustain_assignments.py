#!/usr/bin/env python3
"""Export aligned SuStaIn assignments with unambiguous probabilities."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import HPC_DIR, INPUT_DIR, OUTPUT_DIR


CONTROL_GROUP = 1
MAX_K = 6
OUTPUT_ASSIGNMENTS = OUTPUT_DIR / "SuStaIn_assignments"

COHORTS = {
    "main": {
        "input": INPUT_DIR / "residuals_z_scored_corrected_allGroups.csv",
        "pickle_dir": HPC_DIR / "SuStaIn_mainRun/results/pickle_files",
        "prefix": "SuStaIn_mainRun_patientsOnly",
        "patients_only": True,
    },
    "MDD": {
        "input": INPUT_DIR / "residuals_z_scored_mainflip_MDD.csv",
        "pickle_dir": HPC_DIR / "SuStaIn_diag/MDD/main/results/pickle_files",
        "prefix": "SuStaIn_mainRun_MDD",
        "patients_only": False,
    },
    "BD": {
        "input": INPUT_DIR / "residuals_z_scored_mainflip_BD.csv",
        "pickle_dir": HPC_DIR / "SuStaIn_diag/BD/main/results/pickle_files",
        "prefix": "SuStaIn_mainRun_BD",
        "patients_only": False,
    },
    "SZ": {
        "input": INPUT_DIR / "residuals_z_scored_mainflip_SZ.csv",
        "pickle_dir": HPC_DIR / "SuStaIn_diag/SZ/main/results/pickle_files",
        "prefix": "SuStaIn_mainRun_SZ",
        "patients_only": False,
    },
    "Angst": {
        "input": INPUT_DIR / "residuals_z_scored_mainflip_Angst.csv",
        "pickle_dir": HPC_DIR / "SuStaIn_diag/Angst/main/results/pickle_files",
        "prefix": "SuStaIn_mainRun_Angst",
        "patients_only": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_subject_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required subject table not found: {path}")
    attempts = ((",", "."), (";", ","), (";", "."))
    for separator, decimal in attempts:
        try:
            frame = pd.read_csv(path, sep=separator, decimal=decimal, engine="python")
        except Exception:
            continue
        frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
        if frame.shape[1] < 2:
            continue
        if "Proband" not in frame.columns:
            for alternative in ("names", "Name"):
                if alternative in frame.columns:
                    frame = frame.rename(columns={alternative: "Proband"})
                    break
        if {"Proband", "Group"}.issubset(frame.columns):
            break
    else:
        raise ValueError(f"Could not read Proband and Group from: {path}")

    frame["Proband"] = frame["Proband"].astype(str).str.strip()
    frame["Group"] = pd.to_numeric(frame["Group"], errors="raise").astype(int)
    if frame["Proband"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs in: {path}")
    return frame


def align_subjects(frame: pd.DataFrame, expected_n: int, patients_only: bool) -> pd.DataFrame:
    if len(frame) == expected_n:
        return frame.copy()
    if patients_only:
        patients = frame.loc[frame["Group"] != CONTROL_GROUP].copy()
        if len(patients) == expected_n:
            return patients
    raise ValueError(
        f"Unsafe subject alignment: table contains {len(frame)} rows, model contains "
        f"{expected_n} rows, patients_only={patients_only}."
    )


def load_model(path: Path) -> dict:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, dict):
        raise TypeError(f"Expected a dictionary in model file: {path}")
    required = {
        "ml_subtype",
        "ml_stage",
        "prob_ml_subtype",
        "prob_subtype",
        "prob_subtype_stage",
    }
    missing = required.difference(model)
    if missing:
        raise KeyError(f"Missing model arrays in {path}: {sorted(missing)}")
    return model


def posterior_probabilities(model: dict, n_subjects: int, k: int) -> dict[str, np.ndarray]:
    subtype = np.asarray(model["ml_subtype"]).reshape(-1).astype(int)
    stage = np.asarray(model["ml_stage"]).reshape(-1).astype(int)
    prob_subtype = np.asarray(model["prob_subtype"], dtype=float)
    prob_subtype_stage_raw = np.asarray(model["prob_subtype_stage"], dtype=float)

    if subtype.shape[0] != n_subjects or stage.shape[0] != n_subjects:
        raise ValueError("ML assignment length does not match the subject table.")
    if prob_subtype.shape != (n_subjects, k):
        raise ValueError(f"Unexpected prob_subtype shape: {prob_subtype.shape}; expected {(n_subjects, k)}")
    if prob_subtype_stage_raw.ndim != 3 or prob_subtype_stage_raw.shape[0] != n_subjects:
        raise ValueError(
            "Unexpected prob_subtype_stage shape: "
            f"{prob_subtype_stage_raw.shape}; expected a three-dimensional subject array."
        )
    if prob_subtype_stage_raw.shape[1] == k:
        prob_subtype_stage = prob_subtype_stage_raw
    elif prob_subtype_stage_raw.shape[2] == k:
        prob_subtype_stage = np.transpose(prob_subtype_stage_raw, (0, 2, 1))
    else:
        raise ValueError(
            "Cannot identify the subtype axis in prob_subtype_stage with shape "
            f"{prob_subtype_stage_raw.shape} for K={k}."
        )
    if np.any((subtype < 0) | (subtype >= k)):
        raise ValueError("ML subtype indices fall outside the fitted model.")
    if np.any((stage < 0) | (stage >= prob_subtype_stage.shape[2])):
        raise ValueError("ML stage indices fall outside prob_subtype_stage.")

    joint_total = prob_subtype_stage.sum(axis=(1, 2))
    if not np.allclose(joint_total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("prob_subtype_stage is not normalized for every participant.")

    rows = np.arange(n_subjects)
    joint = prob_subtype_stage[rows, subtype, stage]
    selected_subtype_probability = prob_subtype[rows, subtype]
    joint_subtype_probability = prob_subtype_stage.sum(axis=2)[rows, subtype]
    conditional = np.divide(
        joint,
        joint_subtype_probability,
        out=np.full(n_subjects, np.nan),
        where=joint_subtype_probability > 0,
    )
    marginal_stage = prob_subtype_stage.sum(axis=1)[rows, stage]

    recorded_subtype_probability = np.asarray(model["prob_ml_subtype"]).reshape(-1)
    if not np.allclose(
        recorded_subtype_probability, selected_subtype_probability, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("prob_ml_subtype is inconsistent with prob_subtype at the ML subtype.")

    return {
        "subtype": subtype,
        "stage": stage,
        "prob_subtype": prob_subtype,
        "prob_ml_subtype": selected_subtype_probability,
        "prob_joint": joint,
        "prob_stage_given_subtype": conditional,
        "prob_stage_marginal": marginal_stage,
    }


def export_cohort(name: str, config: dict) -> list[dict]:
    subject_table = read_subject_table(config["input"])
    records = []
    for k in range(1, MAX_K + 1):
        model_path = config["pickle_dir"] / f"{config['prefix']}_subtype{k - 1}.pickle"
        if not model_path.is_file():
            continue
        model = load_model(model_path)
        n_model = np.asarray(model["ml_subtype"]).size
        aligned = align_subjects(subject_table, n_model, config["patients_only"])
        posterior = posterior_probabilities(model, len(aligned), k)

        output = pd.DataFrame(
            {
                "Proband": aligned["Proband"].to_numpy(),
                "Group": aligned["Group"].to_numpy(),
                "ML_Subtype": posterior["subtype"] + 1,
                "Prob_ML_Subtype": posterior["prob_ml_subtype"],
                "ML_Stage": posterior["stage"],
                "Prob_ML_SubtypeStage_Joint": posterior["prob_joint"],
                "Prob_ML_Stage_Given_Subtype": posterior["prob_stage_given_subtype"],
                "Prob_ML_Stage_Marginal": posterior["prob_stage_marginal"],
            }
        )
        for index in range(k):
            output[f"Prob_S{index + 1}"] = posterior["prob_subtype"][:, index]

        output_path = OUTPUT_ASSIGNMENTS / f"{name}_subject_subtype_assignment_{k}subtypes.csv"
        output.to_csv(output_path, index=False)
        counts = output["ML_Subtype"].value_counts().sort_index()
        records.append(
            {
                "cohort": name,
                "K": k,
                "n": len(output),
                "input_sha256": sha256_file(config["input"]),
                "model_sha256": sha256_file(model_path),
                "subtype_counts": ";".join(f"{index}:{int(count)}" for index, count in counts.items()),
                "output": str(output_path),
            }
        )
    if not records:
        raise FileNotFoundError(f"No fitted model files found for cohort {name}: {config['pickle_dir']}")
    return records


def main() -> None:
    OUTPUT_ASSIGNMENTS.mkdir(parents=True, exist_ok=True)
    all_records = []
    for cohort, config in COHORTS.items():
        all_records.extend(export_cohort(cohort, config))
    pd.DataFrame(all_records).to_csv(OUTPUT_DIR / "QA_assignment_export.csv", index=False)
    print(f"Assignments written to {OUTPUT_ASSIGNMENTS}")


if __name__ == "__main__":
    main()
