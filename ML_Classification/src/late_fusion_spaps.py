"""
late-fusion pipeline
- Per-modality inner-CV with calibrated base learners; calibrated meta-learner; repeated outer CV.
- OOF predictions, 95% CIs, meta-level permutation importance; optional label permutation test.
- Version-safe CalibratedClassifierCV (estimator/base_estimator).

Usage:
  python -m src.late_fusion_spaps \
    --data Data_SPAPS.xlsx \
    --outdir runs \
    --run_name exp_10x5 \
    --outer_splits 5 --outer_repeats 10 --inner_splits 5 \
    --seed 42 --permutation_test --n_permutations 200 \
    --verbose 2
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import signal
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

# --- shared feature registry you maintain (EDIT THERE, not here)
from src.features_spaps import TARGET, CONFOUNDS, Modality, make_modalities


# ========================= Small utilities (why-only comments) =========================

def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def run_id_from_args(args: argparse.Namespace) -> str:
    h = hashlib.sha1(repr(sorted(vars(args).items())).encode()).hexdigest()[:8]
    return f"{args.run_name}_{h}"

def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def to_python_scalar(x):
    if isinstance(x, np.generic): return x.item()
    return x

def json_safe(obj):
    if isinstance(obj, BaseEstimator): return obj.__class__.__name__
    if isinstance(obj, (np.generic,)): return obj.item()
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    try:
        json.dumps(obj); return obj
    except TypeError:
        return str(obj)

def make_calibrator(base_estimator: BaseEstimator, method: str = "isotonic", cv: int = 3) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)

def _to_int_binary(y: pd.Series) -> np.ndarray:
    vals = pd.Series(y).astype(int).values
    if not set(np.unique(vals)).issubset({0, 1}): raise ValueError("Target must be binary 0/1.")
    return vals

def _ci95(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float); m = float(np.nanmean(arr))
    s = float(np.nanstd(arr, ddof=1)); n = int(np.sum(~np.isnan(arr)))
    half = 1.96 * s / math.sqrt(max(n, 1)) if n > 1 else float("nan")
    return m, half

def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")

def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    def safe_auc():
        try: return roc_auc_score(y_true, y_prob)
        except Exception: return float("nan")
    def safe_ap():
        try: return average_precision_score(y_true, y_prob)
        except Exception: return float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": safe_auc(),
        "pr_auc": safe_ap(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": _specificity(y_true, y_pred),
    }


# ========================= DataFrame-safe transformers =========================

class DataFrameColumnImputer(BaseEstimator, TransformerMixin):
    """Impute specified feature columns; return DataFrame (+missing flags)."""
    def __init__(self, feature_cols: Sequence[str], strategy: str = "median", add_indicator: bool = True):
        self.feature_cols = tuple(feature_cols); self.strategy = strategy; self.add_indicator = add_indicator
        self._imputer: Optional[SimpleImputer] = None; self._out_cols_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        self._check_cols(X)
        self._imputer = SimpleImputer(strategy=self.strategy, add_indicator=self.add_indicator)
        self._imputer.fit(X.loc[:, self.feature_cols])
        n_base = len(self.feature_cols)
        n_out = self._imputer.transform(X.loc[:, self.feature_cols]).shape[1]
        n_ind = n_out - n_base
        out_cols = list(self.feature_cols)
        if n_ind > 0: out_cols += [f"{c}__missing" for c in self.feature_cols][:n_ind]
        self._out_cols_ = out_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["_imputer", "_out_cols_"]); self._check_cols(X)
        X_out = X.copy()
        imputed = self._imputer.transform(X.loc[:, self.feature_cols])
        imputed_df = pd.DataFrame(imputed, columns=self._out_cols_, index=X.index)
        X_out = X_out.drop(columns=list(self.feature_cols), errors="ignore")
        return pd.concat([X_out, imputed_df], axis=1)

    def _check_cols(self, X: pd.DataFrame):
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing: raise ValueError(f"Imputer missing feature columns: {missing}")


class FeatureResidualizer(BaseEstimator, TransformerMixin):
    """Fold-safe linear residualization of features on confounds (with fold-wise confound imputation + standardization)."""
    def __init__(self, feature_cols: Sequence[str], confound_cols: Sequence[str], standardize: bool = True):
        self.feature_cols = tuple(feature_cols); self.confound_cols = tuple(confound_cols); self.standardize = bool(standardize)

    def fit(self, X: pd.DataFrame, y=None):
        self._check_cols(X)
        C = X.loc[:, self.confound_cols].copy()
        self._c_median_ = C.median(0, numeric_only=True); C = C.fillna(self._c_median_)
        if self.standardize:
            self._c_mean_ = C.mean(0); self._c_std_ = C.std(0).replace(0, 1.0); C = (C - self._c_mean_) / self._c_std_
        else:
            self._c_mean_ = None; self._c_std_ = None
        F = X.loc[:, self.feature_cols].to_numpy()
        if np.isnan(F).any(): raise ValueError("Residualizer received NaNs in features. Ensure imputation precedes residualization.")
        reg = LinearRegression().fit(C.to_numpy(), F)
        self._coef_ = reg.coef_; self._intercept_ = reg.intercept_
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_coef_", "_intercept_", "_c_median_"]); self._check_cols(X)
        C = X.loc[:, self.confound_cols].copy().fillna(self._c_median_)
        if self.standardize: C = (C - self._c_mean_) / self._c_std_
        F = X.loc[:, self.feature_cols].to_numpy()
        if np.isnan(F).any(): raise ValueError("Residualizer received NaNs in features at transform.")
        fitted = C.to_numpy() @ self._coef_.T + self._intercept_
        return F - fitted

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_cols, dtype=object)

    def _check_cols(self, X: pd.DataFrame) -> None:
        mf = [c for c in self.feature_cols if c not in X.columns]
        mc = [c for c in self.confound_cols if c not in X.columns]
        if mf or mc: raise ValueError(f"Missing columns. features={mf}, confounds={mc}")


# ========================= Model builder =========================

def build_modality_pipeline(mod: Modality, confounds: Sequence[str], random_state: int) -> Tuple[Pipeline, List[Dict]]:
    # Order matters: impute features -> residualize -> vth -> (scale/PCA) -> est
    impute_features = DataFrameColumnImputer(feature_cols=mod.columns, strategy="median", add_indicator=True)
    resid = FeatureResidualizer(feature_cols=mod.columns, confound_cols=confounds, standardize=True)
    vth = VarianceThreshold(threshold=0.0)
    scaler = RobustScaler(with_centering=True, with_scaling=True)
    pca = PCA(svd_solver="full")

    logit = LogisticRegression(
        solver="saga", penalty="elasticnet", l1_ratio=0.5, C=1.0,
        class_weight="balanced", max_iter=5000, random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=0.05,
        class_weight="balanced", n_jobs=-1, random_state=random_state,
    )

    pipe = Pipeline([
        ("impute_features", impute_features),
        ("resid", resid),
        ("vth", vth),
        ("scale", scaler),   # off for RF via grid
        ("pca", pca),        # None for RF
        ("est", logit),
    ])

    grid: List[Dict] = [
        {   # Linear path
            "scale__with_centering": [True],
            "scale__with_scaling": [True],
            "pca__n_components": [0.6, 0.75, 0.9],
            "est": [logit],
            "est__C": [0.01, 0.1, 1.0, 10.0],
            "est__l1_ratio": [0.2, 0.5, 0.8],
        },
        {   # Tree path
            "scale__with_centering": [False],
            "scale__with_scaling": [False],
            "pca__n_components": [None],
            "est": [rf],
            "est__max_features": ["sqrt", "log2"],
            "est__min_samples_leaf": [0.02, 0.05, 0.1],
        },
    ]
    return pipe, grid


# ========================= CV helpers =========================

def cv_prob_oof(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    prob = np.zeros_like(y, dtype=float)
    for tr, te in cv.split(X, y):
        est = clone(estimator); est.fit(X.iloc[tr], y[tr])
        prob[te] = est.predict_proba(X.iloc[te])[:, 1]
    return prob

def oof_meta_permutation_importance(
    meta_model: CalibratedClassifierCV, X_meta: np.ndarray, y: np.ndarray, n_repeats: int = 30, random_state: int = 42
) -> np.ndarray:
    rng = check_random_state(random_state)
    base = balanced_accuracy_score(y, meta_model.predict(X_meta))
    imps = np.zeros(X_meta.shape[1], dtype=float)
    for j in range(X_meta.shape[1]):
        scores, Xp = [], X_meta.copy()
        for _ in range(n_repeats):
            rng.shuffle(Xp[:, j])
            scores.append(base - balanced_accuracy_score(y, meta_model.predict(Xp)))
        imps[j] = float(np.mean(scores))
    return imps

def permutation_test_balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray, n_permutations: int, random_state: int) -> float:
    rng = check_random_state(random_state)
    y_pred = (y_prob >= 0.5).astype(int)
    obs = balanced_accuracy_score(y_true, y_pred)
    cnt = sum(balanced_accuracy_score(rng.permutation(y_true), y_pred) >= obs for _ in range(n_permutations))
    return (cnt + 1) / (n_permutations + 1)


# ========================= Core experiment =========================

def run_experiment(
    df: pd.DataFrame,
    modalities: Tuple[Modality, ...],
    out_root: Path,
    run_name: str,
    outer_splits: int,
    outer_repeats: int,
    inner_splits: int,
    seed: int,
    do_permutation_test: bool,
    n_permutations: int,
    calibration_method: str,
    verbose: int,
) -> None:
    run_dir = out_root / run_name
    mkdir(run_dir)
    models_dir = run_dir / "models"; mkdir(models_dir)
    logs_dir = run_dir / "logs"; mkdir(logs_dir)
    artifacts_dir = run_dir / "artifacts"; mkdir(artifacts_dir)

    # Logging sinks
    log_txt = logs_dir / "run.log"
    log_jsonl = logs_dir / "events.jsonl"
    best_params_jsonl = logs_dir / "best_params.jsonl"

    def log_event(event: str, payload: Dict):
        rec = {"ts": now_utc_iso(), "event": event, **{k: json_safe(v) for k, v in payload.items()}}
        with open(log_jsonl, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")

    def log_line(msg: str):
        if verbose >= 1: print(msg)
        with open(log_txt, "a", encoding="utf-8") as f: f.write(msg + "\n")

    # Env capture
    meta = {
        "started": now_utc_iso(),
        "python": sys.version.split()[0],
        "sklearn": sklearn_version,
        "argv": sys.argv,
        "seed": seed,
        "outer_splits": outer_splits,
        "outer_repeats": outer_repeats,
        "inner_splits": inner_splits,
        "calibration": calibration_method,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # SIGINT/TERM -> save partials and exit cleanly
    interrupted = {"flag": False}
    def _handle_signal(signum, frame):
        interrupted["flag"] = True
        log_line(f"[signal] received {signum}; finishing current fold then exiting.")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Basic validations
    missing_conf = [c for c in CONFOUNDS if c not in df.columns]
    if missing_conf: raise ValueError(f"Missing confounds: {missing_conf}")
    for m in modalities:
        miss = [c for c in m.columns if c not in df.columns]
        if miss: raise ValueError(f"Modality '{m.name}' missing columns: {miss}")

    # Target & CV
    y = _to_int_binary(df[TARGET]); y_series = pd.Series(y, index=df.index, name=TARGET)
    outer = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    rng = check_random_state(seed)

    # OOF stores
    oof_prob = np.zeros_like(y, dtype=float); oof_pred = np.zeros_like(y, dtype=int)
    meta_blocks_X: List[np.ndarray] = []; meta_blocks_y: List[np.ndarray] = []; fold_metrics: List[Dict[str, float]] = []

    log_line("=== run start ===")
    log_event("run_start", {"n_samples": len(df), "modalities": [m.name for m in modalities]})

    # Outer loop
    for fold_id, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
        X_train, X_test = df.iloc[tr], df.iloc[te]; y_tr, y_te = y[tr], y[te]
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=rng.randint(0, 1_000_000))

        t0 = time.time()
        log_line(f"[outer] fold {fold_id} start | train={len(tr)} test={len(te)}")
        log_event("outer_start", {"fold": fold_id, "n_train": int(len(tr)), "n_test": int(len(te))})

        meta_train_cols, meta_test_cols = [], []

        # Per-modality
        for mod in modalities:
            log_line(f"  [mod] {mod.name} tuning...")
            pipe, grid = build_modality_pipeline(mod, CONFOUNDS, seed)
            gs = GridSearchCV(
                pipe, param_grid=grid, scoring="balanced_accuracy",
                cv=inner, n_jobs=-1, refit=True, verbose=max(0, verbose - 1)
            )
            gs.fit(X_train, y_tr)

            # Persist CV results & best estimator per modality/fold
            cv_path = artifacts_dir / f"fold{fold_id:02d}_{mod.name}_cv_results.csv"
            pd.DataFrame(gs.cv_results_).to_csv(cv_path, index=False)
            best_est = clone(gs.best_estimator_)
            best_path = models_dir / f"fold{fold_id:02d}_{mod.name}_best.joblib"
            dump(best_est, best_path)

            # Log best params (JSONL)
            rec = {
                "fold": fold_id,
                "modality": mod.name,
                "best_score_balanced_accuracy": float(gs.best_score_),
                "best_params": json_safe(gs.best_params_),
                "model_path": str(best_path),
            }
            with open(best_params_jsonl, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
            log_event("modality_best", rec)

            # Calibrate base learner (train-only)
            calibrated = make_calibrator(best_est, method=calibration_method, cv=3)
            calibrated.fit(X_train, y_tr)

            # Compose meta features (no leakage)
            oof_train_prob = cv_prob_oof(best_est, X_train, y_tr, inner)
            test_prob = calibrated.predict_proba(X_test)[:, 1]
            meta_train_cols.append(oof_train_prob.reshape(-1, 1))
            meta_test_cols.append(test_prob.reshape(-1, 1))

        # Meta-learner (calibrated)
        meta_X_tr = np.hstack(meta_train_cols); meta_X_te = np.hstack(meta_test_cols)
        meta_base = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed)
        meta_clf = make_calibrator(meta_base, method=calibration_method, cv=3)
        meta_clf.fit(meta_X_tr, y_tr)
        dump(meta_clf, models_dir / f"fold{fold_id:02d}_meta.joblib")

        prob_te = meta_clf.predict_proba(meta_X_te)[:, 1]
        pred_te = (prob_te >= 0.5).astype(int)
        oof_prob[te] = prob_te; oof_pred[te] = pred_te
        meta_blocks_X.append(meta_X_te); meta_blocks_y.append(y_te)

        m = _compute_metrics(y_te, prob_te, pred_te); fold_metrics.append(m)
        dt = time.time() - t0
        log_line(f"[outer] fold {fold_id} done | BA={m['balanced_accuracy']:.3f} AUC={m['auc'] if not np.isnan(m['auc']) else float('nan'):.3f} in {dt:.1f}s")
        log_event("outer_done", {"fold": fold_id, "metrics": m, "seconds": round(dt, 2)})

        # Checkpoints
        pd.DataFrame({"y_true": y_series.values, "y_prob": oof_prob, "y_pred": oof_pred}, index=df.index)\
          .to_csv(run_dir / "oof_predictions_partial.csv")
        pd.DataFrame(fold_metrics).assign(fold_id=np.arange(1, len(fold_metrics) + 1))\
          .to_csv(run_dir / "fold_metrics_partial.csv", index=False)

        if interrupted["flag"]: break

    # Aggregate & save final artifacts
    metrics_df = pd.DataFrame(fold_metrics)
    means = {k: float(metrics_df[k].mean()) for k in metrics_df.columns}
    ci_half = {k: _ci95(metrics_df[k].tolist())[1] for k in metrics_df.columns}

    pd.DataFrame({"y_true": y_series.values, "y_prob": oof_prob, "y_pred": oof_pred}, index=df.index)\
      .to_csv(run_dir / "oof_predictions.csv")
    metrics_df.assign(fold_id=np.arange(1, len(metrics_df) + 1))\
      .to_csv(run_dir / "fold_metrics.csv", index=False)
    (run_dir / "summary_metrics.json").write_text(json.dumps({"means": means, "ci95_half_width": ci_half}, indent=2), encoding="utf-8")

    # Meta-level OOF permutation importance
    if len(meta_blocks_X) > 0:
        X_meta_all = np.vstack(meta_blocks_X); y_meta_all = np.concatenate(meta_blocks_y)
        meta_probe = make_calibrator(
            LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed),
            method=calibration_method, cv=3
        )
        meta_probe.fit(X_meta_all, y_meta_all)
        meta_imp = oof_meta_permutation_importance(meta_probe, X_meta_all, y_meta_all, n_repeats=30, random_state=seed)
        modality_names = [m.name for m in modalities]
        pd.DataFrame({"modality": modality_names, "oof_perm_importance": meta_imp})\
          .sort_values("oof_perm_importance", ascending=False)\
          .to_csv(run_dir / "meta_oof_permutation_importance.csv", index=False)

    # Optional label permutation test
    if do_permutation_test:
        pval = permutation_test_balanced_accuracy(y_true=y, y_prob=oof_prob, n_permutations=n_permutations, random_state=seed)
        (run_dir / "permutation_test.json").write_text(
            json.dumps({"metric": "balanced_accuracy", "n_permutations": n_permutations, "p_value": float(pval)}, indent=2),
            encoding="utf-8",
        )

    log_line("=== run end ===")
    log_event("run_end", {"ended": now_utc_iso(), "interrupted": interrupted["flag"], "means": means, "ci95": ci_half})


# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPAPS late-fusion (SoTA) with structured logging and robust artifacts.")
    p.add_argument("--data", default="Data_SPAPS.xlsx", help="Path to Data_SPAPS.(xlsx|csv).")
    p.add_argument("--outdir", default="runs", help="Root directory to write runs.")
    p.add_argument("--run_name", default="exp_10x5", help="Name for this run (a hash suffix is added).")
    p.add_argument("--outer_splits", type=int, default=5)
    p.add_argument("--outer_repeats", type=int, default=10)
    p.add_argument("--inner_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permutation_test", action="store_true")
    p.add_argument("--n_permutations", type=int, default=200)
    p.add_argument("--calibration", default="isotonic", choices=["isotonic", "sigmoid"])
    p.add_argument("--verbose", type=int, default=1, help="0=silent, 1=fold logs, 2+=GridSearch progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.outdir)
    run_name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{run_id_from_args(args)}"
    run_root = out_root / run_name
    mkdir(run_root)

    # Load data
    if args.data.lower().endswith((".xlsx", ".xls")):
        try:
            import openpyxl  # noqa: F401
        except Exception:
            print("Install openpyxl for Excel support: pip install openpyxl", file=sys.stderr)
            sys.exit(1)
        df = pd.read_excel(args.data, engine="openpyxl")
    else:
        df = pd.read_csv(args.data)

    # Ensure Geschlecht numeric if strings
    if "Geschlecht" in df.columns and df["Geschlecht"].dtype == object:
        df["Geschlecht"] = pd.Categorical(df["Geschlecht"]).codes

    # Build modalities from the shared registry
    modalities = make_modalities(df)
    if not modalities:
        raise RuntimeError("No modalities with available columns. Fill FEATURE_SETS in features_spaps.py.")

    # Write quick dataset snapshot (for reproducibility)
    snapshot = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": TARGET,
        "confounds": CONFOUNDS,
        "modalities": {m.name: len(m.columns) for m in modalities},
        "started": now_utc_iso(),
    }
    (run_root / "dataset_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # Run
    run_experiment(
        df=df,
        modalities=modalities,
        out_root=out_root,
        run_name=run_name,
        outer_splits=args.outer_splits,
        outer_repeats=args.outer_repeats,
        inner_splits=args.inner_splits,
        seed=args.seed,
        do_permutation_test=args.permutation_test,
        n_permutations=args.n_permutations,
        calibration_method=args.calibration,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

