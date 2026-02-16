from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

# ------------------------------- Import features registry ---------------------
def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for p in [here.parent.parent / "src", Path.cwd() / "src"]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

_ensure_src_on_path()
from features import make_modalities


def union_ordered(*seqs):
    seen, out = set(), []
    for s in seqs:
        for c in s:
            if c not in seen:
                out.append(c)
                seen.add(c)
    return out


def intersect_existing(df: pd.DataFrame, desired):
    seen = set()
    present, missing = [], []
    for c in desired:
        if c in seen:
            continue
        seen.add(c)
        (present if c in df.columns else missing).append(c)
    return present, missing


# ------------------------------- Load data ------------------------------------
df = pd.read_excel("Mastertable_SPAPS.xlsx", engine="openpyxl")

# group: 0=HC, 1=affective, 2=psychotic
df["group"] = pd.to_numeric(df["diagnose"], errors="coerce").map({1: 0, 2: 1, 3: 1, 4: 2, 5: 2})
df = df.dropna(subset=["group"]).copy()
df["group"] = df["group"].astype(int)

# ------------------------------- Build feature set from modalities -------------
mods = make_modalities(df)
by_name = {m.name: list(m.columns) for m in mods}

all_feature_columns = union_ordered(*by_name.values())

present_features, missing_features = intersect_existing(df, all_feature_columns)
if len(present_features) == 0:
    raise RuntimeError("No modality features found in dataframe. Check features.py registry vs column names in Excel.")

print(f"[INFO] Modalities present: {list(by_name.keys())}")
print(f"[INFO] Using features: {len(present_features)} present, {len(missing_features)} missing")

X = df[present_features].to_numpy()
y = df["group"].to_numpy()


# ------------------------------- Define pipeline -------------------------------
my_pipe = Hyperpipe(
    name="multi_dnn",
    optimizer="grid_search",
    metrics=["accuracy", "balanced_accuracy"],
    best_config_metric="balanced_accuracy",
    outer_cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    inner_cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    verbosity=1,
    allow_multidim_targets=True,
    project_folder="multiclass_dnn2",
)

# Preprocessing
my_pipe += PipelineElement(
    "SimpleImputer",
    hyperparameters={"strategy": Categorical(["mean", "median"])},
)
my_pipe += PipelineElement("RobustScaler")
my_pipe += PipelineElement(
    "VarianceThreshold",
    hyperparameters={"threshold": Categorical([0.005, 0.01, 0.02, 0.05])},
)

# Classifier
my_pipe += PipelineElement(
    "KerasDnnClassifier",
    hyperparameters={
        "hidden_layer_sizes": Categorical([[64, 32], [128, 64], [64, 32, 16]]),
        "dropout_rate": Categorical([0.3]),
    },
    activations="relu",
    nn_batch_size=16,
    epochs=50,
    multi_class=True,
    verbosity=0,
)

my_pipe.fit(X, y)
