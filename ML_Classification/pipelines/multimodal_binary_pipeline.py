# Project2/pipelines/multimodal_binary_pipeline.py
"""
PhotonAI pipeline for binary classification across modality-defined feature sets.

Targets
-------
- Binary label already present (e.g., 'patient' ∈ {0,1}), OR
- Thresholded numeric target (e.g., 'GAFscore' with --threshold 80 → 1 if <=80, else 0).

Modalities
----------
- Feature columns imported from Project2/src/features_spaps.py via `make_modalities`.
- Experiments differ ONLY by feature set; model space and preprocessing are identical.

Design
------
- No imputation; strict listwise deletion per experiment on (features + target).
- Optional matched-cohort: identical participants across all experiments for fair comparison.
- Preprocessing: RobustScaler → VarianceThreshold(threshold ∈ {0.0, 0.005, 0.01, 0.02, 0.05}) → PCA(n_components ∈ {0.5..0.9}).
- Classifier switch: GradientBoosting / RandomForest / LogisticRegression / SVC / AdaBoost.
- class_weight='balanced' for RF/LR/SVC. Model selection metric: balanced_accuracy.
- Nested CV folds adapt to minority class count to prevent split errors.

Outputs
-------
- PhotonAI artifacts: ClassBIN/<run_name>_<context>/
- Summary manifest: ClassBIN/run_manifest.json
- Attrition summary: ClassBIN/run_attrition_summary.csv

Run examples (from Project2/)
-----------------------------
# Use existing binary target 'patient'
python pipelines/multimodal_binary_pipeline.py --data Data_SPAPS.xlsx --target-col patient

# Threshold GAFscore at 80 → 1 if <=80 (non-functional), else 0
python pipelines/multimodal_binary_pipeline.py --data Data_SPAPS.xlsx --target-col GAFscore --threshold 80 --matched
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import photonai as _ph
import sklearn
from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import Categorical
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------------- Constants ------------------------------------
PROJECT_ROOT: str = "ClassBIN"
RANDOM_STATE: int = 42

# ------------------------------- Import features registry ---------------------
def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "src",  # Project2/src
        Path.cwd() / "src",          # if CWD is Project2/
    ]
    for p in candidates:
        if p.exists():
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)

_ensure_src_on_path()
try:
    from features_spaps import make_modalities  # central registry of modality columns
except Exception as e:
    raise ImportError(
        "Could not import 'make_modalities' from Project2/src/features_spaps.py. "
        f"Original error: {type(e).__name__}: {e}"
    )

# ------------------------------- Utilities ------------------------------------
def intersect_existing(df: pd.DataFrame, desired: Sequence[str]) -> Tuple[List[str], List[str]]:
    seen = set(); present, missing = [], []
    for c in desired:
        if c in seen:
            continue
        seen.add(c)
        (present if c in df.columns else missing).append(c)
    return present, missing

def adaptive_cv_splits(y: pd.Series, desired_outer: int = 5, desired_inner: int = 3) -> Tuple[int, int]:
    vc = y.value_counts(dropna=False)
    if vc.empty:
        return 2, 2
    min_class = int(vc.min())
    outer = max(2, min(desired_outer, min_class))
    inner = max(2, min(desired_inner, max(min_class - 1, 2)))
    return outer, inner

def build_binary_target(series: pd.Series, threshold: Optional[float]) -> pd.Series:
    if threshold is None:
        # Expect {0,1} (allow strings '0'/'1'); coerce safely
        vals = pd.unique(series)
        # Map non-integer labels to ints deterministically if needed
        if set(pd.to_numeric(pd.Series(vals), errors="coerce").dropna().astype(int)) <= {0, 1}:
            y = pd.to_numeric(series, errors="coerce").astype("Int64")
        else:
            # Fallback: two-class mapping by sorted label name (documented behavior)
            uniq = sorted(map(str, map(lambda x: "" if pd.isna(x) else x, vals)))
            if len([u for u in uniq if u != ""]) != 2:
                raise ValueError("Target is not binary and no --threshold provided.")
            mapping = {uniq[0]: 0, uniq[1]: 1}
            y = series.astype(str).map(mapping).astype("Int64")
    else:
        # Thresholded numeric: 1 if <= thr else 0
        num = pd.to_numeric(series, errors="coerce")
        if num.dropna().empty:
            raise ValueError("All target values are NaN after numeric coercion for thresholding.")
        y = (num <= float(threshold)).astype("Int64")
    if y.isna().any():
        # We'll drop NaNs together with features per experiment; keep Int64 dtype for now
        pass
    return y

def union_ordered(*seqs: Sequence[str]) -> List[str]:
    seen: set = set(); out: List[str] = []
    for s in seqs:
        for c in s:
            if c not in seen:
                out.append(c); seen.add(c)
    return out

# ------------------------------- Pipeline -------------------------------------
def make_hyperpipe(name: str, outer_splits: int, inner_splits: int) -> Hyperpipe:
    pipe = Hyperpipe(
        name=name,
        project_folder=str(Path(PROJECT_ROOT) / name),
        optimizer="grid_search",
        metrics=["accuracy", "balanced_accuracy", "specificity", "sensitivity", "f1_score", "auc"],
        best_config_metric="balanced_accuracy",
        outer_cv=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE),
        inner_cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE),
    )
    pipe += PipelineElement("RobustScaler")
    pipe += PipelineElement("VarianceThreshold",
                            hyperparameters={"threshold": Categorical([0.0, 0.005, 0.01, 0.02, 0.05])})
    pipe += PipelineElement("PCA",
                            hyperparameters={"n_components": Categorical([0.5, 0.6, 0.7, 0.8, 0.9])},
                            random_state=RANDOM_STATE)

    clf_switch = Switch("ClassifierSwitch")
    clf_switch += PipelineElement("GradientBoostingClassifier",
                                  hyperparameters={"n_estimators": Categorical([50, 100, 200])},
                                  random_state=RANDOM_STATE)
    clf_switch += PipelineElement("RandomForestClassifier",
                                  hyperparameters={
                                      "max_features": Categorical(["sqrt", "log2"]),
                                      "min_samples_leaf": Categorical([0.01, 0.05, 0.1]),
                                      "n_estimators": Categorical([200, 500]),
                                  },
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE,
                                  n_jobs=-1)
    clf_switch += PipelineElement("LogisticRegression",
                                  hyperparameters={"C": Categorical([0.01, 0.1, 1, 10]),
                                                   "penalty": Categorical(["l2"])},
                                  solver="saga",
                                  max_iter=5000,
                                  n_jobs=-1,
                                  random_state=RANDOM_STATE,
                                  class_weight="balanced")
    clf_switch += PipelineElement("SVC",
                                  hyperparameters={"C": Categorical([1e-3, 1, 100]),
                                                   "gamma": Categorical(["scale", 0.01, 0.1]),
                                                   "kernel": Categorical(["linear", "rbf"])},
                                  class_weight="balanced",
                                  max_iter=100000,
                                  random_state=RANDOM_STATE)
    clf_switch += PipelineElement("AdaBoostClassifier",
                                  hyperparameters={"n_estimators": Categorical([100, 200]),
                                                   "learning_rate": Categorical([0.3, 0.6, 1.0])},
                                  random_state=RANDOM_STATE)
    pipe += clf_switch
    return pipe

# ------------------------------- Orchestration --------------------------------
@dataclass
class RunSummary:
    name: str
    context: str           # "main" or "matched"
    n_features: int
    n_missing: int
    n_rows: int
    dropped_rows: int
    positive_prevalence: float
    outer_splits: int
    inner_splits: int
    project_dir: str
    target_col: str
    threshold: Optional[float]

def build_experiments(df: pd.DataFrame) -> Dict[str, Sequence[str]]:
    mods = make_modalities(df)
    by_name: Dict[str, Tuple[str, ...]] = {m.name: m.columns for m in mods}
    acoustic = by_name.get("speech_acoustic", tuple())
    transcript = by_name.get("speech_text", tuple())
    neuro = by_name.get("neurocog", tuple())
    psycho = by_name.get("psychopathology", tuple())
    smri = by_name.get("smri", tuple())

    speech_all = union_ordered(acoustic, transcript)
    neuro_only = list(neuro)
    psycho_only = list(psycho)
    smri_only = list(smri)

    speech_plus_neuro = union_ordered(speech_all, neuro_only)
    speech_plus_psycho = union_ordered(speech_all, psycho_only)
    speech_plus_mri = union_ordered(speech_all, smri_only)
    multimodal = union_ordered(speech_all, neuro_only, psycho_only, smri_only)

    exps: Dict[str, Sequence[str]] = {
        "acoustic_only": list(acoustic),
        "transcript_only": list(transcript),
        "speech_all": speech_all,
        "neuropsych_only": neuro_only,
        "psychopathology_only": psycho_only,
        "sMRI_only": smri_only,
        "speech_plus_neuro": speech_plus_neuro,
        "speech_plus_psychopathology": speech_plus_psycho,
        "speech_plus_MRI": speech_plus_mri,
        "multimodal": multimodal,
    }
    return {k: v for k, v in exps.items() if len(v) > 0}

def run_experiment(
    df: pd.DataFrame,
    run_name: str,
    feature_list: Sequence[str],
    y_all: pd.Series,
    target_col: str,
    threshold: Optional[float],
    context: str,
) -> RunSummary:
    present, missing = intersect_existing(df, feature_list)
    if not present:
        warnings.warn(f"[WARN] No features found for '{run_name}' [{context}]. Skipping.")
        return RunSummary(run_name, context, 0, len(missing), 0, 0, float("nan"), 0, 0,
                          str(Path(PROJECT_ROOT) / f"{run_name}_{context}"), target_col, threshold)

    # Strict per-experiment listwise deletion (features + target)
    before = len(df)
    drop_cols = list(present) + [target_col]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df_run = df.dropna(subset=drop_cols)
    dropped = before - len(df_run)
    if df_run.empty:
        warnings.warn(f"[WARN] No rows left after dropna for '{run_name}' [{context}]. Skipping.")
        return RunSummary(run_name, context, len(present), len(missing), 0, dropped, float("nan"), 0, 0,
                          str(Path(PROJECT_ROOT) / f"{run_name}_{context}"), target_col, threshold)

    y_run = y_all.loc[df_run.index].astype("int32")
    X = df_run[present].to_numpy()
    pos_prev = float((y_run == 1).mean()) if len(y_run) else float("nan")

    outer, inner = adaptive_cv_splits(y_run)
    pipe = make_hyperpipe(f"{run_name}_{context}", outer_splits=outer, inner_splits=inner)

    print(f"[RUN] {run_name} [{context}]: rows={len(df_run)}, dropped={dropped}, "
          f"features={len(present)}, missing_features={len(missing)}, pos_prev={pos_prev:.3f}")
    pipe.fit(X, y_run.to_numpy())

    return RunSummary(run_name, context, len(present), len(missing), len(df_run), dropped,
                      pos_prev, outer, inner, str(Path(PROJECT_ROOT) / f"{run_name}_{context}"),
                      target_col, threshold)

def run_block(df: pd.DataFrame, experiments: Dict[str, Sequence[str]],
              y: pd.Series, target_col: str, threshold: Optional[float], context: str) -> List[RunSummary]:
    summaries: List[RunSummary] = []
    for name, feats in experiments.items():
        print(f"\n===== RUN: {name} [{context}] =====")
        summaries.append(run_experiment(df, name, feats, y, target_col, threshold, context))
    return summaries

# ------------------------------- Entry point ----------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="PhotonAI multimodal binary pipeline.")
    parser.add_argument("--data", type=str, default="Data_SPAPS.xlsx", help="Path to data (xlsx/csv).")
    parser.add_argument("--target-col", type=str, required=True, help="Target column name (binary or numeric for thresholding).")
    parser.add_argument("--threshold", type=float, default=None, help="If set, binarize target: 1 if target <= threshold else 0.")
    parser.add_argument("--matched", action="store_true", help="Also run matched-cohort pass with identical participants across experiments.")
    args = parser.parse_args()

    print(f"[INFO] pandas={pd.__version__} sklearn={sklearn.__version__} photonai={_ph.__version__}")

    # Load
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")
    if data_path.suffix.lower() in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(data_path, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel '{data_path.name}'. Install 'openpyxl' and close the file. "
                               f"Original error: {e}")
    else:
        df = pd.read_csv(data_path)

    if args.target_col not in df.columns:
        raise KeyError(f"Target column '{args.target_col}' not found in {data_path.name}.")

    # Build target (supports both direct binary and thresholded numeric)
    y = build_binary_target(df[args.target_col], args.threshold)
    base_prev = float((y == 1).mean(skipna=True))
    thr_msg = f" (<= {args.threshold:.1f} → 1)" if args.threshold is not None else ""
    print(f"[TARGET] {args.target_col}{thr_msg}; prevalence_positive={base_prev:.3f}")

    # Experiments from registry present in df
    experiments = build_experiments(df)
    if not experiments:
        raise RuntimeError("No non-empty experiments assembled from available modalities.")

    # Main block (max-N per experiment)
    summaries_main = run_block(df, experiments, y, args.target_col, args.threshold, context="main")

    # Matched-cohort block (optional)
    summaries_matched: List[RunSummary] = []
    if args.matched:
        union_features: List[str] = []
        for cols in experiments.values():
            for c in cols:
                if c not in union_features:
                    union_features.append(c)
        needed = [c for c in union_features if c in df.columns] + [args.target_col]
        mask = df[needed].notna().all(axis=1)
        df_matched = df.loc[mask].copy()
        print(f"[MATCHED] kept={int(mask.sum())}/{len(df)} ({mask.mean()*100:.1f}%) rows; "
              f"union_features={len(union_features)}")
        y_matched = build_binary_target(df_matched[args.target_col], args.threshold)
        summaries_matched = run_block(df_matched, experiments, y_matched, args.target_col, args.threshold, context="matched")

    # Manifests
    out_dir = Path(PROJECT_ROOT); out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = summaries_main + summaries_matched
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in all_summaries], f, indent=2)
    print(f"[DONE] Wrote manifest to {manifest_path}")

    # Attrition table (clean, concise)
    attr_cols = ["context", "name", "n_rows", "dropped_rows", "n_features", "n_missing",
                 "outer_splits", "inner_splits", "positive_prevalence", "target_col", "threshold", "project_dir"]
    attr_df = pd.DataFrame([asdict(s) for s in all_summaries], columns=attr_cols)
    attr_out = out_dir / "run_attrition_summary.csv"
    attr_df.to_csv(attr_out, index=False)
    print(f"[DONE] Wrote attrition summary to {attr_out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        raise
