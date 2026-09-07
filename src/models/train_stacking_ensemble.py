"""Phase 5: build and evaluate the stacking ensemble.

Answers RQ1: does combining the three baselines (Random Forest, XGBoost,
LightGBM) outperform any of them individually?

Stacking procedure:
  1. Load the three already-trained base models. They are not retrained here.
  2. Generate base-model predicted probabilities on the validation set. Val is
     the correct, uncontaminated set to train the meta-classifier on, since
     none of the three base models ever trained on val.
  3. Train a Logistic Regression meta-classifier on those three probability
     columns, target = val labels.
  4. Generate base-model probabilities on the TEST set (first and only use of
     test in this project so far), feed them through the trained
     meta-classifier, and evaluate the resulting ensemble predictions.
  5. For a fair RQ1 comparison, also evaluate each of the three individual
     base models on the same TEST set (previous phases only evaluated them on
     val).

Evaluation covers precision / recall / F1 / false positive rate / AUC-ROC and
the confusion matrix, reported overall and broken down by source_dataset, for
all four models (RF, XGBoost, LightGBM, Stacked Ensemble), all on test.

Positive class = phishing. A false positive is a legitimate email flagged as
phishing, so FPR = FP / (FP + TN) over the legitimate rows.

Outputs:
  src/models/artifacts/stacking_meta.pkl
  src/models/artifacts/final_comparison.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"

RF_PATH = ARTIFACTS_DIR / "random_forest.pkl"
XGB_PATH = ARTIFACTS_DIR / "xgboost.pkl"
LGBM_PATH = ARTIFACTS_DIR / "lightgbm.pkl"
META_MODEL_PATH = ARTIFACTS_DIR / "stacking_meta.pkl"
COMPARISON_PATH = ARTIFACTS_DIR / "final_comparison.json"

BASE_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]

POSITIVE_LABEL = "phishing"
NEGATIVE_LABEL = "legitimate"

META_PARAMS = dict(
    random_state=42,
)


def load_split(name: str):
    """Return (X, y, source_dataset) for a split, filtered to training-eligible rows.

    Per .claude/rules/training-data-integrity.md, anything derived from
    merged_raw.csv must be filtered to included_in_training == True before use.
    The splits are already fully filtered upstream; this re-applies it defensively
    and keeps the feature matrix row-aligned to whatever survives.
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


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Metrics with positive = phishing. Fields that are undefined for the given
    label support (e.g. a single-class subset) are returned as None with a note.
    """
    n = int(len(y_true))
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    notes = []
    if n_pos == 0:
        notes.append("no phishing rows: precision/recall/F1/AUC-ROC undefined")
    if n_neg == 0:
        notes.append("no legitimate rows: FPR/specificity undefined; AUC-ROC undefined")

    precision = recall = f1 = None
    if n_pos > 0:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None, zero_division=0
        )
        precision, recall, f1 = float(p[0]), float(r[0]), float(f[0])

    fpr = float(fp / (fp + tn)) if n_neg > 0 else None
    accuracy = float((tp + tn) / n) if n else None

    auc = None
    if n_pos > 0 and n_neg > 0:
        auc = float(roc_auc_score(y_true, y_score))

    return {
        "n": n,
        "n_phishing": n_pos,
        "n_legitimate": n_neg,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "auc_roc": auc,
        "accuracy": accuracy,
        "confusion_matrix": {
            "labels": ["legitimate", "phishing"],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        },
        "notes": notes,
    }


def fmt(value) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def evaluate_by_source(y_true, y_pred, y_score, source) -> tuple[dict, dict]:
    overall = binary_metrics(y_true, y_pred, y_score)
    by_source = {}
    for src in sorted(pd.unique(source)):
        m = source == src
        by_source[src] = binary_metrics(y_true[m], y_pred[m], y_score[m])
    return overall, by_source


def print_report(title: str, overall: dict, by_source: dict) -> None:
    print("\n" + "-" * 78)
    print(f"{title} - overall (positive class = phishing)")
    print("-" * 78)
    print(f"precision           : {fmt(overall['precision'])}")
    print(f"recall              : {fmt(overall['recall'])}")
    print(f"f1                  : {fmt(overall['f1'])}")
    print(f"false positive rate : {fmt(overall['false_positive_rate'])}   (target < 0.005)")
    print(f"auc-roc             : {fmt(overall['auc_roc'])}")
    print(f"accuracy            : {fmt(overall['accuracy'])}")
    cm = overall["confusion_matrix"]
    print("confusion matrix (rows = actual, cols = predicted):")
    print(f"                 pred_legit  pred_phish")
    print(f"  actual_legit   {cm['tn']:>10}  {cm['fp']:>10}")
    print(f"  actual_phish   {cm['fn']:>10}  {cm['tp']:>10}")

    print(f"\n{title} - by source_dataset")
    print(f"{'source':<22} {'n':>5} {'prec':>7} {'recall':>7} {'f1':>7} "
          f"{'fpr':>7} {'auc':>7}   TN/FP/FN/TP")
    for src, sm in by_source.items():
        c = sm["confusion_matrix"]
        print(f"{src:<22} {sm['n']:>5} {fmt(sm['precision']):>7} {fmt(sm['recall']):>7} "
              f"{fmt(sm['f1']):>7} {fmt(sm['false_positive_rate']):>7} "
              f"{fmt(sm['auc_roc']):>7}   {c['tn']}/{c['fp']}/{c['fn']}/{c['tp']}")
        for note in sm["notes"]:
            print(f"{'':<22}   note: {note}")


def main() -> None:
    print("=" * 78)
    print("PHASE 5: STACKING ENSEMBLE (RQ1)")
    print("=" * 78)
    print("\nNOTE: this step touches the TEST set for the first time in this project.")
    print("Test is used only for the final evaluation below, never for training.")

    print("\nLoading trained base models (not retrained here) ...")
    base_models = {
        "random_forest": joblib.load(RF_PATH),
        "xgboost": joblib.load(XGB_PATH),
        "lightgbm": joblib.load(LGBM_PATH),
    }

    print("\nLoading splits ...")
    X_val, y_val, val_source = load_split("val")
    X_test, y_test, test_source = load_split("test")
    print(f"val:  {X_val.shape}  phishing={int(y_val.sum())}  legit={int((y_val == 0).sum())}")
    print(f"test: {X_test.shape}  phishing={int(y_test.sum())}  legit={int((y_test == 0).sum())}")

    # Step 1: base model probabilities on val -> meta-classifier training features.
    print("\nGenerating base-model probabilities on val (meta-classifier training data) ...")
    val_probs = {
        name: base_models[name].predict_proba(X_val)[:, 1] for name in BASE_MODEL_NAMES
    }
    X_meta_train = np.column_stack([val_probs[name] for name in BASE_MODEL_NAMES])

    # Step 2: train Logistic Regression meta-classifier on val base-model probs.
    print(f"Fitting LogisticRegression meta-classifier ({META_PARAMS}) on val base-model probabilities ...")
    meta_clf = LogisticRegression(**META_PARAMS)
    meta_clf.fit(X_meta_train, y_val)
    print("done.")
    coefs = {name: float(c) for name, c in zip(BASE_MODEL_NAMES, meta_clf.coef_[0])}
    print(f"meta-classifier coefficients: {coefs}")
    print(f"meta-classifier intercept   : {float(meta_clf.intercept_[0]):.5f}")

    # Step 3: base model probabilities on test, individually evaluated, and fed
    # through the meta-classifier for the ensemble evaluation.
    print("\nGenerating base-model probabilities on test ...")
    test_probs = {
        name: base_models[name].predict_proba(X_test)[:, 1] for name in BASE_MODEL_NAMES
    }
    X_meta_test = np.column_stack([test_probs[name] for name in BASE_MODEL_NAMES])
    ensemble_score = meta_clf.predict_proba(X_meta_test)[:, 1]
    ensemble_pred = meta_clf.predict(X_meta_test)

    results = {}

    display_names = {
        "random_forest": "RANDOM FOREST",
        "xgboost": "XGBOOST",
        "lightgbm": "LIGHTGBM",
    }
    for name in BASE_MODEL_NAMES:
        score = test_probs[name]
        pred = (score >= 0.5).astype(int)
        overall, by_source = evaluate_by_source(y_test, pred, score, test_source)
        print_report(f"TEST SET - {display_names[name]} (individual baseline)", overall, by_source)
        results[name] = {"validation_overall": None, "test_overall": overall, "test_by_source_dataset": by_source}

    overall_ens, by_source_ens = evaluate_by_source(y_test, ensemble_pred, ensemble_score, test_source)
    print_report("TEST SET - STACKED ENSEMBLE", overall_ens, by_source_ens)
    results["stacked_ensemble"] = {
        "test_overall": overall_ens,
        "test_by_source_dataset": by_source_ens,
    }

    # Explicit synthetic_healthcare comparison across all four models.
    print("\n" + "-" * 78)
    print("SYNTHETIC_HEALTHCARE SLICE - ALL FOUR MODELS ON TEST")
    print("-" * 78)
    print(f"{'model':<20} {'fpr':>7} {'recall':>7}")
    synth_summary = {}
    for name in BASE_MODEL_NAMES + ["stacked_ensemble"]:
        sm = results[name]["test_by_source_dataset"]["synthetic_healthcare"]
        synth_summary[name] = {"fpr": sm["false_positive_rate"], "recall": sm["recall"]}
        print(f"{name:<20} {fmt(sm['false_positive_rate']):>7} {fmt(sm['recall']):>7}")

    # RQ1 headline comparison table (overall test metrics, all four models).
    print("\n" + "-" * 78)
    print("RQ1 FINAL COMPARISON - ALL FOUR MODELS ON TEST (overall)")
    print("-" * 78)
    print(f"{'model':<20} {'prec':>7} {'recall':>7} {'f1':>7} {'fpr':>7} {'auc':>7}")
    headline = {}
    for name in BASE_MODEL_NAMES + ["stacked_ensemble"]:
        o = results[name]["test_overall"]
        headline[name] = {
            "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
            "false_positive_rate": o["false_positive_rate"], "auc_roc": o["auc_roc"],
        }
        print(f"{name:<20} {fmt(o['precision']):>7} {fmt(o['recall']):>7} {fmt(o['f1']):>7} "
              f"{fmt(o['false_positive_rate']):>7} {fmt(o['auc_roc']):>7}")

    # Does the ensemble beat the best individual baseline on test, and by how much?
    best_baseline_f1 = max(results[n]["test_overall"]["f1"] for n in BASE_MODEL_NAMES)
    best_baseline_name = max(BASE_MODEL_NAMES, key=lambda n: results[n]["test_overall"]["f1"])
    ensemble_f1 = overall_ens["f1"]
    f1_delta = ensemble_f1 - best_baseline_f1

    best_baseline_fpr = min(results[n]["test_overall"]["false_positive_rate"] for n in BASE_MODEL_NAMES)
    best_baseline_fpr_name = min(BASE_MODEL_NAMES, key=lambda n: results[n]["test_overall"]["false_positive_rate"])
    ensemble_fpr = overall_ens["false_positive_rate"]
    fpr_delta = ensemble_fpr - best_baseline_fpr

    best_baseline_auc = max(results[n]["test_overall"]["auc_roc"] for n in BASE_MODEL_NAMES)
    best_baseline_auc_name = max(BASE_MODEL_NAMES, key=lambda n: results[n]["test_overall"]["auc_roc"])
    ensemble_auc = overall_ens["auc_roc"]
    auc_delta = ensemble_auc - best_baseline_auc

    print("\n" + "-" * 78)
    print("RQ1 ANSWER")
    print("-" * 78)
    print(f"best individual baseline by F1  : {best_baseline_name} (F1={best_baseline_f1:.4f})")
    print(f"stacked ensemble F1             : {ensemble_f1:.4f}  (delta {f1_delta:+.4f})")
    print(f"best individual baseline by FPR : {best_baseline_fpr_name} (FPR={best_baseline_fpr:.4f})")
    print(f"stacked ensemble FPR            : {ensemble_fpr:.4f}  (delta {fpr_delta:+.4f})")
    print(f"best individual baseline by AUC : {best_baseline_auc_name} (AUC={best_baseline_auc:.4f})")
    print(f"stacked ensemble AUC            : {ensemble_auc:.4f}  (delta {auc_delta:+.4f})")

    rq1_answer = {
        "best_individual_baseline_by_f1": best_baseline_name,
        "best_individual_baseline_f1": best_baseline_f1,
        "ensemble_f1": ensemble_f1,
        "f1_delta_ensemble_minus_best_baseline": f1_delta,
        "ensemble_outperforms_best_baseline_on_f1": bool(f1_delta > 0),
        "best_individual_baseline_by_fpr": best_baseline_fpr_name,
        "best_individual_baseline_fpr": best_baseline_fpr,
        "ensemble_fpr": ensemble_fpr,
        "fpr_delta_ensemble_minus_best_baseline": fpr_delta,
        "ensemble_improves_fpr_vs_best_baseline": bool(fpr_delta < 0),
        "best_individual_baseline_by_auc": best_baseline_auc_name,
        "best_individual_baseline_auc": best_baseline_auc,
        "ensemble_auc": ensemble_auc,
        "auc_delta_ensemble_minus_best_baseline": auc_delta,
        "ensemble_improves_auc_vs_best_baseline": bool(auc_delta > 0),
    }

    metrics = {
        "phase": "5",
        "description": "Stacking ensemble (RQ1): RF + XGBoost + LightGBM base probabilities into a Logistic Regression meta-classifier trained on val, evaluated on test.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "positive_class": POSITIVE_LABEL,
        "test_set_touched": True,
        "test_set_touched_note": "First and only step in this project permitted to use test_final.npz / test_labels.csv, and only for this final evaluation.",
        "base_models_retrained": False,
        "meta_classifier": {
            "type": "LogisticRegression",
            "params": META_PARAMS,
            "trained_on": "val base-model probabilities (val never used in base model training)",
            "input_features": BASE_MODEL_NAMES,
            "coefficients": coefs,
            "intercept": float(meta_clf.intercept_[0]),
        },
        "data": {
            "val_rows": int(X_val.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "results_by_model": results,
        "synthetic_healthcare_summary": synth_summary,
        "headline_comparison_test_overall": headline,
        "rq1_answer": rq1_answer,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_clf, META_MODEL_PATH)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("\n" + "=" * 78)
    print(f"saved meta-classifier -> {META_MODEL_PATH}")
    print(f"saved comparison      -> {COMPARISON_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
