"""Threshold tuning for the already-trained stacking ensemble.

No retraining. Loads the three base models and stacking_meta.pkl as-is,
generates ensemble probability scores, and sweeps decision thresholds.

Threshold selection happens on VALIDATION only, never on test, so the
selection process itself does not leak test information. Test is touched
exactly once at the end, to apply the selected threshold and report final
metrics, mirroring the test-set discipline in train_stacking_ensemble.py.

Outputs:
  src/models/artifacts/threshold_tuning.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"

RF_PATH = ARTIFACTS_DIR / "random_forest.pkl"
XGB_PATH = ARTIFACTS_DIR / "xgboost.pkl"
LGBM_PATH = ARTIFACTS_DIR / "lightgbm.pkl"
META_MODEL_PATH = ARTIFACTS_DIR / "stacking_meta.pkl"
OUTPUT_PATH = ARTIFACTS_DIR / "threshold_tuning.json"

BASE_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]
POSITIVE_LABEL = "phishing"

FPR_TARGETS = [0.005, 0.01, 0.02]
DEFAULT_THRESHOLD = 0.5


def load_split(name: str):
    """Return (X, y, source_dataset) for a split, filtered to training-eligible rows.

    Mirrors train_stacking_ensemble.load_split: per
    .claude/rules/training-data-integrity.md, anything derived from
    merged_raw.csv must be filtered to included_in_training == True.
    """
    meta = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    X = sparse.load_npz(FEATURES_DIR / f"{name}_final.npz").tocsr()
    labels = pd.read_csv(FEATURES_DIR / f"{name}_labels.csv")["label"]

    if len(meta) != X.shape[0] or len(labels) != X.shape[0]:
        raise ValueError(
            f"{name}: row mismatch meta={len(meta)} X={X.shape[0]} labels={len(labels)}"
        )
    if not labels.reset_index(drop=True).equals(meta["label"].reset_index(drop=True)):
        raise ValueError(f"{name}: {name}_labels.csv does not match {name}.csv label column")

    mask = meta["included_in_training"].to_numpy(dtype=bool)
    n_excluded = int((~mask).sum())
    if n_excluded:
        print(f"[{name}] dropping {n_excluded} rows with included_in_training == False")
    meta = meta.loc[mask].reset_index(drop=True)
    X = X[mask]
    y = (meta["label"].to_numpy() == POSITIVE_LABEL).astype(int)
    return X, y, meta["source_dataset"].to_numpy()


def ensemble_scores(base_models: dict, meta_clf, X) -> np.ndarray:
    probs = np.column_stack(
        [base_models[name].predict_proba(X)[:, 1] for name in BASE_MODEL_NAMES]
    )
    return meta_clf.predict_proba(probs)[:, 1]


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    precision = recall = f1 = None
    if n_pos > 0:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None, zero_division=0
        )
        precision, recall, f1 = float(p[0]), float(r[0]), float(f[0])

    fpr = float(fp / (fp + tn)) if n_neg > 0 else None

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_by_source(y_true, y_score, threshold, source) -> dict:
    by_source = {}
    for src in sorted(pd.unique(source)):
        m = source == src
        by_source[src] = metrics_at_threshold(y_true[m], y_score[m], threshold)
    return by_source


def build_threshold_sweep(y_val: np.ndarray, val_score: np.ndarray) -> list[dict]:
    coarse = np.round(np.arange(0.50, 0.99, 0.01), 4)
    fine = np.round(np.arange(0.990, 0.9991, 0.001), 4)
    boundary = np.array([0.999])
    thresholds = np.unique(np.concatenate([coarse, fine, boundary]))
    return [metrics_at_threshold(y_val, val_score, t) for t in thresholds]


def pick_threshold_for_fpr_target(sweep: list[dict], fpr_target: float) -> dict | None:
    """Among thresholds with FPR < fpr_target, return the one maximizing recall.

    Ties on recall are broken by the lowest qualifying threshold, since that
    sacrifices the least additional recall margin.
    """
    candidates = [
        m for m in sweep
        if m["false_positive_rate"] is not None and m["false_positive_rate"] < fpr_target
    ]
    if not candidates:
        return None
    best_recall = max(m["recall"] for m in candidates if m["recall"] is not None)
    tied = [m for m in candidates if m["recall"] == best_recall]
    return min(tied, key=lambda m: m["threshold"])


def fmt(value) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def print_metrics_row(label: str, m: dict) -> None:
    print(
        f"{label:<28} thr={m['threshold']:.4f}  prec={fmt(m['precision'])}  "
        f"recall={fmt(m['recall'])}  f1={fmt(m['f1'])}  fpr={fmt(m['false_positive_rate'])}"
    )


def main() -> None:
    print("=" * 78)
    print("STACKING ENSEMBLE - THRESHOLD TUNING (no retraining)")
    print("=" * 78)

    print("\nLoading trained base models and meta-classifier (not retrained) ...")
    base_models = {
        "random_forest": joblib.load(RF_PATH),
        "xgboost": joblib.load(XGB_PATH),
        "lightgbm": joblib.load(LGBM_PATH),
    }
    meta_clf = joblib.load(META_MODEL_PATH)

    print("Loading val and test splits ...")
    X_val, y_val, val_source = load_split("val")
    X_test, y_test, test_source = load_split("test")
    print(f"val:  {X_val.shape}  phishing={int(y_val.sum())}  legit={int((y_val == 0).sum())}")
    print(f"test: {X_test.shape}  phishing={int(y_test.sum())}  legit={int((y_test == 0).sum())}")

    print("\nGenerating ensemble probability scores on val ...")
    val_score = ensemble_scores(base_models, meta_clf, X_val)

    print("Sweeping thresholds 0.50 -> 0.999 on val ...")
    sweep = build_threshold_sweep(y_val, val_score)

    default_metrics = metrics_at_threshold(y_val, val_score, DEFAULT_THRESHOLD)
    print("\n" + "-" * 78)
    print("VALIDATION - reference point at default threshold 0.5")
    print("-" * 78)
    print_metrics_row("default (0.5)", default_metrics)

    print("\n" + "-" * 78)
    print("VALIDATION - thresholds hit at each FPR target (max recall subject to FPR < target)")
    print("-" * 78)
    fpr_target_results = {}
    for target in FPR_TARGETS:
        picked = pick_threshold_for_fpr_target(sweep, target)
        fpr_target_results[target] = picked
        if picked is None:
            print(f"FPR < {target:<6}: no threshold in sweep range achieves this")
        else:
            print_metrics_row(f"FPR < {target}", picked)

    selected = fpr_target_results.get(0.005)
    if selected is None:
        raise RuntimeError(
            "No threshold in the sweep range [0.5, 0.999] achieves validation FPR < 0.005. "
            "Widen the sweep or accept a higher FPR target."
        )
    selected_threshold = selected["threshold"]

    print("\n" + "-" * 78)
    print(f"SELECTED THRESHOLD (target FPR < 0.005): {selected_threshold}")
    print("-" * 78)
    print_metrics_row("selected (val)", selected)

    print("\nApplying selected threshold to TEST (first use of this threshold on test) ...")
    test_score = ensemble_scores(base_models, meta_clf, X_test)
    test_at_selected = metrics_at_threshold(y_test, test_score, selected_threshold)
    test_at_default = metrics_at_threshold(y_test, test_score, DEFAULT_THRESHOLD)
    n_pos_test = int((y_test == 1).sum())
    n_neg_test = int((y_test == 0).sum())
    test_auc = float(roc_auc_score(y_test, test_score)) if n_pos_test and n_neg_test else None

    print("\n" + "-" * 78)
    print("TEST SET - STACKED ENSEMBLE at selected threshold vs default 0.5")
    print("-" * 78)
    print_metrics_row("default (0.5)", test_at_default)
    print_metrics_row(f"selected ({selected_threshold})", test_at_selected)
    print(f"auc-roc (threshold-independent) : {fmt(test_auc)}")

    test_by_source_selected = evaluate_by_source(y_test, test_score, selected_threshold, test_source)
    test_by_source_default = evaluate_by_source(y_test, test_score, DEFAULT_THRESHOLD, test_source)

    print("\n" + "-" * 78)
    print(f"TEST SET - by source_dataset at selected threshold ({selected_threshold})")
    print("-" * 78)
    print(f"{'source':<22} {'n':>5} {'prec':>7} {'recall':>7} {'f1':>7} {'fpr':>7}   TN/FP/FN/TP")
    for src, sm in test_by_source_selected.items():
        n = sum(sm["confusion_matrix"].values())
        c = sm["confusion_matrix"]
        print(f"{src:<22} {n:>5} {fmt(sm['precision']):>7} {fmt(sm['recall']):>7} "
              f"{fmt(sm['f1']):>7} {fmt(sm['false_positive_rate']):>7}   "
              f"{c['tn']}/{c['fp']}/{c['fn']}/{c['tp']}")

    recall_drop = None
    if test_at_default["recall"] is not None and test_at_selected["recall"] is not None:
        recall_drop = test_at_default["recall"] - test_at_selected["recall"]
        n_missed_additional = int(round(recall_drop * n_pos_test))
        print("\n" + "-" * 78)
        print("RECALL COST OF THE STRICTER THRESHOLD")
        print("-" * 78)
        print(
            f"recall dropped from {test_at_default['recall']:.4f} to {test_at_selected['recall']:.4f} "
            f"({recall_drop * 100:.2f} percentage points), meaning ~{n_missed_additional} more "
            f"phishing emails out of {n_pos_test} in the test set would now be missed."
        )

    output = {
        "description": "Threshold tuning on the pre-trained stacking ensemble. No retraining performed.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_models_retrained": False,
        "meta_classifier_retrained": False,
        "selection_set": "validation",
        "validation": {
            "n_rows": int(X_val.shape[0]),
            "default_threshold_0_5": default_metrics,
            "threshold_sweep": sweep,
            "fpr_target_results": {
                str(target): result for target, result in fpr_target_results.items()
            },
            "selected_threshold": selected_threshold,
            "selected_threshold_target": "FPR < 0.005, max recall",
        },
        "test": {
            "n_rows": int(X_test.shape[0]),
            "n_phishing": n_pos_test,
            "n_legitimate": n_neg_test,
            "auc_roc": test_auc,
            "at_default_threshold_0_5": test_at_default,
            "at_selected_threshold": test_at_selected,
            "by_source_dataset_at_selected_threshold": test_by_source_selected,
            "by_source_dataset_at_default_threshold": test_by_source_default,
            "recall_drop_selected_vs_default": recall_drop,
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print("\n" + "=" * 78)
    print(f"saved threshold tuning report -> {OUTPUT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
