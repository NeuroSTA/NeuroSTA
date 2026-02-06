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
- Optional imputation; otherwise strict listwise deletion per experiment on (features + target).
- Optional matched-cohort: identical participants across all experiments for fair comparison.
- Preprocessing: [SimpleImputer] → RobustScaler → VarianceThreshold → PCA.
- Classifier switch: GradientBoosting / RandomForest / LogisticRegression / SVC / AdaBoost.
- class_weight='balanced' for RF/LR/SVC. Model selection metric: balanced_accuracy.
- Nested CV folds adapt to minority class count to prevent split errors.

Outputs
-------
- Artifacts: ClassBIN/<analysis_key>/<target_key>/<run_name>_<context>/
- Manifest: ClassBIN/<analysis_key>/<target_key>/run_manifest.json
- Attrition summary: ClassBIN/<analysis_key>/<target_key>/run_attrition_summary.csv
- Metadata: ClassBIN/<analysis_key>/<target_key>/run_metadata.json
"""

from __future__ import annotations
import argparse, json, sys, warnings
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
from sklearn.impute import SimpleImputer
from photonai.base import Hyperpipe, PipelineElement, Switch, Stack, Branch, DataFilter


warnings.filterwarnings("ignore", category=ConvergenceWarning)
RANDOM_STATE: int = 42

# ------------------------------- Import features registry ---------------------
def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for p in [here.parent.parent / "src", Path.cwd() / "src"]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
_ensure_src_on_path()
from features import make_modalities

# ------------------------------- Utilities ------------------------------------
def intersect_existing(df: pd.DataFrame, desired: Sequence[str]) -> Tuple[List[str], List[str]]:
    seen = set(); present, missing = [], []
    for c in desired:
        if c not in seen:
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
        vals = pd.unique(series)
        if set(pd.to_numeric(pd.Series(vals), errors="coerce").dropna().astype(int)) <= {0, 1}:
            y = pd.to_numeric(series, errors="coerce").astype("Int64")
        else:
            uniq = sorted(map(str, map(lambda x: "" if pd.isna(x) else x, vals)))
            if len([u for u in uniq if u != ""]) != 2:
                raise ValueError("Target is not binary and no --threshold provided.")
            mapping = {uniq[0]: 0, uniq[1]: 1}
            y = series.astype(str).map(mapping).astype("Int64")
    else:
        num = pd.to_numeric(series, errors="coerce")
        y = (num <= float(threshold)).astype("Int64")
    return y

def union_ordered(*seqs: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for s in seqs:
        for c in s:
            if c not in seen:
                out.append(c); seen.add(c)
    return out

def slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")

def make_target_key(target_col: str, threshold: Optional[float]) -> str:
    base = slugify(target_col)
    if threshold is None:
        return f"{base}_native"
    thr = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
    return f"{base}_thr{thr}"

def make_analysis_key(data_path: Path, analysis_tag: Optional[str]) -> str:
    return slugify(analysis_tag) if analysis_tag else slugify(data_path.stem)

# ------------------------------- Pipeline -------------------------------------
def make_hyperpipe(name: str, project_root: Path, outer_splits: int, inner_splits: int,
                   impute: bool = False) -> Hyperpipe:
    pipe = Hyperpipe(
        name=name,
        project_folder=str(project_root / name),
        optimizer="grid_search",
        metrics=["accuracy", "balanced_accuracy", "specificity", "sensitivity", "f1_score", "auc"],
        best_config_metric="balanced_accuracy",
        outer_cv=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE),
        inner_cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE),
    )

    # Optional imputation step
    if impute:
        pipe += PipelineElement("SimpleImputer",
                                hyperparameters={"strategy": Categorical(["mean", "median"])})
    else:
        print("[INFO] No imputation applied (strict listwise deletion).")

    pipe += PipelineElement("RobustScaler")
    pipe += PipelineElement("VarianceThreshold",
                            hyperparameters={"threshold": Categorical([0.0, 0.005, 0.01, 0.02, 0.05])})
    pipe += PipelineElement("PCA",
                            hyperparameters={"n_components": Categorical([None, 0.5, 0.6, 0.7, 0.8, 0.9])},
                            random_state=RANDOM_STATE)

    clf_switch = Switch("ClassifierSwitch")
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
                                  hyperparameters={"C": Categorical([0.01, 0.1, 1, 10])},
                                  solver="saga",
                                  max_iter=5000,
                                  n_jobs=-1,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE)
    clf_switch += PipelineElement("SVC",
                                  hyperparameters={"C": Categorical([1e-3, 1, 100]),
                                                   "gamma": Categorical(["scale", 0.01, 0.1]),
                                                   "kernel": Categorical(["linear", "rbf"])},
                                  class_weight="balanced",
                                  max_iter=100000,
                                  random_state=RANDOM_STATE)
    pipe += clf_switch
    return pipe

# ------------------------------- Orchestration --------------------------------
@dataclass
class RunSummary:
    name: str; context: str; n_features: int; n_missing: int
    n_rows: int; dropped_rows: int; positive_prevalence: float
    outer_splits: int; inner_splits: int
    project_dir: str; target_col: str; threshold: Optional[float]
    target_key: str; analysis_key: str; imputation: bool

def build_experiments(df: pd.DataFrame) -> Dict[str, Sequence[str]]:
    """
    Define modality-based experiments.

    Modalities (from make_modalities):
      - speech_acoustic
      - speech_text
      - neurocog
      - smri
      - network

    Design:
      - Unimodal baselines per modality
      - Speech-only integration (acoustic + transcript)
      - Multimodal integrations:
          * speech + neurocog + sMRI + network
    """
    mods = make_modalities(df)
    by_name = {m.name: m.columns for m in mods}

    acoustic = by_name.get("speech_acoustic", ())
    transcript = by_name.get("speech_text", ())
    neuro = by_name.get("neurocog", ())
    smri = by_name.get("smri", ())
    network = by_name.get("network", ())

    # Speech representation: PCA learn a *joint* low-dimensional representation over acoustic + text
    speech_all = union_ordered(acoustic, transcript)

    exps = {
        # Unimodal baselines
        "acoustic_only": list(acoustic),
        "transcript_only": list(transcript),
        "neurocog_only": list(neuro),
        "sMRI_only": list(smri),
        "network": list(network),

        # Within-speech fusion (acoustic + transcript)
        "speech_all": speech_all,

        # joint representation over speech (acoustic + text), neurocognition, and MRI
        "multimodal": union_ordered(speech_all, neuro, smri, network),
    }

    # Drop experiments that have no features in the current dataframe
    return {k: v for k, v in exps.items() if len(v) > 0}


def run_experiment(df, run_name, features, y_all, target_col, threshold,
                   context, project_root, target_key, analysis_key, impute):
    present, missing = intersect_existing(df, features)
    if not present:
        warnings.warn(f"[WARN] No features for '{run_name}'. Skipping.")
        return RunSummary(run_name, context, 0, len(missing), 0, 0, np.nan, 0, 0,
                          str(project_root / f"{run_name}_{context}"), target_col, threshold, target_key, analysis_key, impute)

    # Only drop rows if not imputing
    df_run = df if impute else df.dropna(subset=list(present) + [target_col])
    if df_run.empty:
        warnings.warn(f"[WARN] No rows left for '{run_name}'.")
        return RunSummary(run_name, context, len(present), len(missing), 0, 0, np.nan, 0, 0,
                          str(project_root / f"{run_name}_{context}"), target_col, threshold, target_key, analysis_key, impute)

    y_run = y_all.loc[df_run.index].astype(int)
    X = df_run[present].to_numpy()
    pos_prev = float((y_run == 1).mean()) if len(y_run) else np.nan

    outer, inner = adaptive_cv_splits(y_run)
    pipe = make_hyperpipe(f"{run_name}_{context}", project_root, outer, inner, impute)
    print(f"[RUN] {run_name} [{context}] rows={len(df_run)}, features={len(present)}, impute={impute}, pos_prev={pos_prev:.3f}")
    pipe.fit(X, y_run.to_numpy())

    return RunSummary(run_name, context, len(present), len(missing), len(df_run), len(df)-len(df_run),
                      pos_prev, outer, inner, str(project_root / f"{run_name}_{context}"),
                      target_col, threshold, target_key, analysis_key, impute)


def run_stacking_experiment(df: pd.DataFrame,
                            y_all: pd.Series,
                            target_col: str,
                            threshold: Optional[float],
                            project_root: Path,
                            target_key: str,
                            analysis_key: str,
                            impute: bool) -> RunSummary:
    """
    Late-fusion / stacking multimodal experiment.

    Design:
      - One Branch per modality (speech_acoustic, speech_text, neurocog, smri)
      - Each Branch:
          DataFilter(indices for that modality) -> LogisticRegression (modality-specific base model)
      - Stack(use_probabilities=True) concatenates per-modality predicted probabilities.
      - Final Switch meta-classifier (RF / LR / SVC) operates on stacked predictions.

    Uses the same outer/inner CV structure as other experiments.
    """

    run_name = "stacked_multimodal"
    context = "stack"

    # Build modalities from the full dataframe (column sets only)
    mods = make_modalities(df)
    if len(mods) < 2:
        warnings.warn("[WARN] Less than two modalities present, skipping stacking experiment.")
        return RunSummary(run_name, context, 0, 0, 0, 0, np.nan, 0, 0,
                          str(project_root / f"{run_name}_{context}"),
                          target_col, threshold, target_key, analysis_key, impute)

    by_name = {m.name: list(m.columns) for m in mods}
    # Union of all modality feature columns
    all_feature_names = union_ordered(*by_name.values())

    present, missing = intersect_existing(df, all_feature_names)
    if not present:
        warnings.warn("[WARN] No features available for stacking experiment. Skipping.")
        return RunSummary(run_name, context, 0, len(missing), 0, 0, np.nan, 0, 0,
                          str(project_root / f"{run_name}_{context}"),
                          target_col, threshold, target_key, analysis_key, impute)

    # Row-wise handling (listwise deletion unless imputation is enabled)
    df_run = df if impute else df.dropna(subset=present + [target_col])
    if df_run.empty:
        warnings.warn("[WARN] No rows left after NA handling for stacking experiment.")
        return RunSummary(run_name, context, len(present), len(missing), 0, 0, np.nan, 0, 0,
                          str(project_root / f"{run_name}_{context}"),
                          target_col, threshold, target_key, analysis_key, impute)

    # Align y with df_run
    y_run = y_all.loc[df_run.index].astype(int)
    X = df_run[present].to_numpy()
    pos_prev = float((y_run == 1).mean()) if len(y_run) else np.nan

    # Map feature name -> column index (in X)
    feat_to_idx = {f: i for i, f in enumerate(present)}

    # Build modality branches (only for modalities that have at least one present feature)
    branches = []
    for mod in mods:
        mod_feats = [f for f in mod.columns if f in feat_to_idx]
        if not mod_feats:
            continue

        indices = [feat_to_idx[f] for f in mod_feats]

        branch = Branch(f"{mod.name}_branch")
        # Restrict to this modality's columns
        branch += DataFilter(indices=indices)
        # Simple, robust base learner per modality
        branch += PipelineElement(
            "LogisticRegression",
            hyperparameters={"C": Categorical([0.01, 0.1, 1, 10])},
            solver="saga",
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        branches.append(branch)

    if len(branches) < 2:
        warnings.warn("[WARN] Less than two non-empty modality branches, skipping stacking.")
        return RunSummary(run_name, context, len(present), len(missing), 0, 0, np.nan, 0, 0,
                          str(project_root / f"{run_name}_{context}"),
                          target_col, threshold, target_key, analysis_key, impute)

    # Adaptive nested CV as for other experiments
    outer, inner = adaptive_cv_splits(y_run)

    # Build Hyperpipe for stacking
    stack_pipe = Hyperpipe(
        name=f"{run_name}_{context}",
        project_folder=str(project_root / f"{run_name}_{context}"),
        optimizer="grid_search",
        metrics=["accuracy", "balanced_accuracy", "specificity", "sensitivity", "f1_score", "auc"],
        best_config_metric="balanced_accuracy",
        outer_cv=StratifiedKFold(n_splits=outer, shuffle=True, random_state=RANDOM_STATE),
        inner_cv=StratifiedKFold(n_splits=inner, shuffle=True, random_state=RANDOM_STATE),
    )

    # Optional imputation + global robust scaling before modality split
    if impute:
        stack_pipe += PipelineElement("SimpleImputer",
                                      hyperparameters={"strategy": Categorical(["mean"])})
    else:
        print("[INFO] Stacking experiment: no imputation applied (strict listwise deletion).")

    stack_pipe += PipelineElement("RobustScaler")

    # Parallel modality-specific models → stacked probabilities (late-fusion features)
    stack_pipe += Stack("ModalityStack", branches, use_probabilities=True)

    # Meta-classifier on stacked predictions
    meta_switch = Switch("MetaClassifierSwitch")
    meta_switch += PipelineElement(
        "RandomForestClassifier",
        hyperparameters={
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Categorical([0.01, 0.05]),
            "n_estimators": Categorical([200, 500]),
        },
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    meta_switch += PipelineElement(
        "LogisticRegression",
        hyperparameters={"C": Categorical([0.01, 0.1, 1, 10])},
        solver="saga",
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    meta_switch += PipelineElement(
        "SVC",
        hyperparameters={
            "C": Categorical([1e-3, 1, 100]),
            "gamma": Categorical(["scale", 0.01, 0.1]),
            "kernel": Categorical(["linear", "rbf"]),
        },
        class_weight="balanced",
        max_iter=100000,
        random_state=RANDOM_STATE,
    )
    stack_pipe += meta_switch

    print(f"[RUN] {run_name} [{context}] rows={len(df_run)}, features={len(present)}, "
          f"branches={len(branches)}, impute={impute}, pos_prev={pos_prev:.3f}")

    stack_pipe.fit(X, y_run.to_numpy())

    return RunSummary(
        run_name,
        context,
        len(present),
        len(missing),
        len(df_run),
        len(df) - len(df_run),
        pos_prev,
        outer,
        inner,
        str(project_root / f"{run_name}_{context}"),
        target_col,
        threshold,
        target_key,
        analysis_key,
        impute,
    )


# ------------------------------- Entry point ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="PhotonAI multimodal binary classification pipeline.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--analysis-tag", type=str, default=None)
    parser.add_argument("--impute", action="store_true", help="Enable mean imputation for missing values.")
    args = parser.parse_args()

    print(f"[INFO] pandas={pd.__version__}, sklearn={sklearn.__version__}, photonai={_ph.__version__}")

    path = Path(args.data)
    df = pd.read_excel(path, engine="openpyxl") if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    if args.target_col not in df.columns:
        raise KeyError(f"Target column '{args.target_col}' not found.")

    y = build_binary_target(df[args.target_col], args.threshold)
    analysis_key = make_analysis_key(path, args.analysis_tag)
    target_key = make_target_key(args.target_col, args.threshold)
    project_root = Path("ClassBIN") / analysis_key / target_key
    project_root.mkdir(parents=True, exist_ok=True)

    print(f"[TARGET] {args.target_col}, threshold={args.threshold}, impute={args.impute}, analysis={analysis_key}")

    experiments = build_experiments(df)
    summaries = [run_experiment(df, name, feats, y, args.target_col, args.threshold,
                                "main", project_root, target_key, analysis_key, args.impute)
                 for name, feats in experiments.items()]

    # Add late-fusion / stacking multimodal experiment on top of unimodal / early-fusion runs
    try:
        stack_summary = run_stacking_experiment(
            df=df,
            y_all=y,
            target_col=args.target_col,
            threshold=args.threshold,
            project_root=project_root,
            target_key=target_key,
            analysis_key=analysis_key,
            impute=args.impute,
        )
        summaries.append(stack_summary)
    except Exception as e:
        warnings.warn(f"[WARN] Stacking experiment failed with error: {e}")


    manifest = [asdict(s) for s in summaries]
    (project_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(manifest).to_csv(project_root / "run_attrition_summary.csv", index=False)

    meta = {
        "data_file": str(path),
        "analysis_key": analysis_key,
        "target_col": args.target_col,
        "threshold": args.threshold,
        "imputation": args.impute,
        "versions": {"pandas": pd.__version__, "sklearn": sklearn.__version__, "photonai": _ph.__version__}
    }
    json.dump(meta, open(project_root / "run_metadata.json", "w"), indent=2)
    print(f"[DONE] Results written to {project_root.resolve()}")

if __name__ == "__main__":
    main()
