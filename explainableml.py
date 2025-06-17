"""
Explainable Machine Learning on Tabular Data with LightGBM + SHAP
================================================================
A *from‑scratch* yet production‑ready script that walks you through the
*entire* life‑cycle of an explainable tabular model:

1. Environment & reproducibility setup
2. Flexible data ingestion (CSV or scikit‑learn toy set)
3. LightGBM hyper‑parameter tuning with stratified CV
4. Final training, metrics, and model serialization
5. Global & local explanations with SHAP (summary‑bar, beeswarm,
   dependence, waterfall, and force plots)

Every step is peppered with conversational, human‑friendly comments to make
the why just as clear as the how.
"""

# ------------------------------------------------------------------
# 0) Install dependencies (if running in a fresh environment)
# ------------------------------------------------------------------
# !pip install -q lightgbm shap pandas scikit-learn matplotlib tqdm joblib seaborn
# (The `-q` keeps the output tidy.)

# ------------------------------------------------------------------
# 1) Imports & global configuration
# ------------------------------------------------------------------
import os
import joblib
import warnings
import shap
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report)

import lightgbm as lgb

warnings.filterwarnings('ignore')  # Less clutter, more signal ✨
plt.rcParams['figure.dpi'] = 110   # Nicer plots in most notebooks

# -----------------------
# Reproducibility helper
# -----------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def set_lgb_seed(seed: int = 42):
    """LightGBM has its own RNG; keep it deterministic."""
    lgb.params._ConfigAliases._alias_dict['seed'] = 'seed'  # sometimes needed


set_lgb_seed(SEED)

# ------------------------------------------------------------------
# 2) Data ingestion: CSV or fallback to toy dataset
# ------------------------------------------------------------------

def load_data(csv_path: str | Path | None = None,
              target: str | None = None):
    """Returns (X, y, feature_names) regardless of source."""
    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        if target is None:
            raise ValueError("When you pass a CSV you must name the target column.")
        y = df[target]
        X = df.drop(columns=[target])
        print(f"Loaded {df.shape[0]:,} rows & {df.shape[1]-1:,} features from {csv_path}")
    else:
        print("No dataset supplied → falling back to scikit‑learn's breast‑cancer set.")
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        X = df.drop(columns=[data.target.name])
        y = df[data.target.name]
        target = data.target.name
        print(f"Dataset shape → {df.shape}")
    return X, y, X.columns.tolist(), target


# ------------------------------------------------------------------
# 3) Train/valid split + Stratified CV for hyper‑params
# ------------------------------------------------------------------

def tune_params(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """A *tiny* hand‑rolled grid search to keep things lightweight."""
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 30]
    }

    best_auc, best_params = 0, None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Cartesian product over the (small) grid
    from itertools import product

    for num_leaves, lr, min_child in tqdm(list(product(*param_grid.values())),
                                          desc="Grid search"):
        fold_aucs = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_boost_round': 9999,  # early stopping will cut it
                'early_stopping_rounds': 80,
                'verbose': -1,
                'seed': SEED,
                'num_leaves': num_leaves,
                'learning_rate': lr,
                'min_child_samples': min_child,
            }

            gbm = lgb.train(params, dtrain,
                            valid_sets=[dvalid],
                            verbose_eval=False)

            preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
            fold_aucs.append(roc_auc_score(y_val, preds))

        mean_auc = np.mean(fold_aucs)
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, {
                'num_leaves': num_leaves,
                'learning_rate': lr,
                'min_child_samples': min_child,
            }

    print(f"Best CV AUC = {best_auc:.4f} with params: {best_params}")
    return best_params


# ------------------------------------------------------------------
# 4) Final train/valid split & model fitting
# ------------------------------------------------------------------

def train_final_model(X: pd.DataFrame, y: pd.Series, params: dict,
                      test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    lgb_params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'seed': SEED,
        'verbose': -1,
        **params  # tuned hyper‑params
    }

    print("Training final LightGBM model ...")
    gbm = lgb.train(lgb_params, dtrain,
                    num_boost_round=5000,
                    valid_sets=[dtrain, dvalid],
                    valid_names=['train', 'valid'],
                    early_stopping_rounds=120,
                    verbose_eval=100)

    # Evaluation
    y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\n▁▁▁▁▁ Evaluation Metrics ▁▁▁▁▁")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC‑AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Persist model to disk so you can reuse without retraining
    model_path = Path("lightgbm_model.txt")
    gbm.save_model(model_path)
    print(f"Model saved to → {model_path.resolve()}")

    return gbm, (X_test, y_test)


# ------------------------------------------------------------------
# 5) SHAP explanations (global & local)
# ------------------------------------------------------------------

def explain_model(gbm: lgb.Booster, X_test: pd.DataFrame, feature_names: list[str]):
    """Generates several SHAP visualizations."""
    print("\nCalculating SHAP values ... this can take ~seconds‑ish.")
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_test)

    # -- Global 1: bar summary plot (mean(|shap|))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      show=False, feature_names=feature_names)
    plt.title("Global feature importance — mean |SHAP| over test set")
    plt.tight_layout()
    plt.show()

    # -- Global 2: beeswarm (distribution + directionality)
    shap.summary_plot(shap_values, X_test, show=False, feature_names=feature_names)
    plt.title("Beeswarm plot — SHAP value distribution per feature")
    plt.tight_layout()
    plt.show()

    # -- Local: pick a particularly uncertain sample (prob ≈ 0.5)
    abs_diffs = np.abs(gbm.predict(X_test) - 0.5)
    idx = int(np.argmin(abs_diffs))
    print(f"Local explanation for test sample #{idx} (model p≈0.5):")

    shap.plots._waterfall.waterfall_legacy(explainer.expected_value,
                                           shap_values[idx],
                                           feature_names=feature_names,
                                           max_display=14)
    plt.tight_layout()
    plt.show()

    # Optional HTML force plot (nice in notebooks)
    try:
        shap.initjs()
        display(shap.force_plot(explainer.expected_value,
                                shap_values[idx],
                                X_test.iloc[idx, :]))
    except Exception as e:
        print("(HTML force plot skipped — not in a notebook?)")
    return shap_values


# ------------------------------------------------------------------
# 6) Orchestrate the full pipeline from a single entry‑point
# ------------------------------------------------------------------

def main(csv_path: str | None = None, target: str | None = None):
    X, y, feature_names, target_name = load_data(csv_path, target)
    tuned_params = tune_params(X, y, n_splits=5)
    gbm, (X_test, y_test) = train_final_model(X, y, tuned_params)
    explain_model(gbm, X_test, feature_names)


# ------------------------------------------------------------------
# 7) Fire away if run as a script
# ------------------------------------------------------------------
if __name__ == "__main__":
    # If you want to hook up a custom CSV, just run:
    # python shap_lightgbm.py --csv_path yourfile.csv --target target_column
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM & explain with SHAP")
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to a CSV file (defaults to breast‑cancer toy set)')
    parser.add_argument('--target', type=str, default=None,
                        help='Name of the target column in your CSV')

    args = parser.parse_args()
    main(args.csv_path, args.target)
