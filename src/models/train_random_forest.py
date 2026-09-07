"""Phase 4, Step 1: train and evaluate the first baseline model (Random Forest).

Loads the assembled final feature matrices (TF-IDF + scaled dense text/URL
features) for the train and validation splits, fits a Random Forest, and
evaluates on validation only. The test split is never loaded here.

Evaluation covers precision / recall / F1 / false positive rate / AUC-ROC and
the confusion matrix, reported overall and broken down by source_dataset so we
can see whether performance is being carried by a single-source confound (the
nazario = phishing-only / spamassassin = legitimate-only split) or by the URL
artifact resurfacing. Also reports the top 20 Random Forest feature importances
by name.

Positive class = phishing. A false positive is a legitimate email flagged as
phishing, so FPR = FP / (FP + TN) over the legitimate rows.

Outputs:
  src/models/artifacts/random_forest.pkl
  src/models/artifacts/random_forest.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = ARTIFACTS_DIR / "random_forest.pkl"
METRICS_PATH = ARTIFACTS_DIR / "random_forest.json"

# Dense feature column order must match src/features/assemble_final_features.py:
# final = hstack([tfidf, dense]) where dense = TEXT_DENSE_COLS + URL_DENSE_COLS.
TEXT_DENSE_COLS = [
    "urgency_score", "healthcare_term_count", "word_count",
    "avg_word_length", "exclamation_count", "capitalized_word_count",
]
URL_DENSE_COLS = [
    "url_count", "has_ip_literal", "has_url_shortener",
    "avg_domain_entropy", "suspicious_tld", "has_at_symbol",
]
DENSE_COLS = TEXT_DENSE_COLS + URL_DENSE_COLS

POSITIVE_LABEL = "phishing"
NEGATIVE_LABEL = "legitimate"

RF_PARAMS = dict(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
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


def feature_names() -> list[str]:
    vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_names = [f"tfidf:{tok}" for tok in vectorizer.get_feature_names_out()]
    return tfidf_names + list(DENSE_COLS)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Metrics with positive = phishing. Fields that are undefined for the given
    label support (e.g. a single-class subset) are returned as None with a note.
    """
    n = int(len(y_true))
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    # confusion_matrix with fixed label order [neg, pos] -> [[TN, FP], [FN, TP]]
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


def main() -> None:
    print("=" * 78)
    print("PHASE 4 STEP 1: RANDOM FOREST BASELINE")
    print("=" * 78)

    X_train, y_train, _ = load_split("train")
    X_val, y_val, val_source = load_split("val")
    print(f"train: {X_train.shape}  phishing={int(y_train.sum())}  legit={int((y_train == 0).sum())}")
    print(f"val:   {X_val.shape}  phishing={int(y_val.sum())}  legit={int((y_val == 0).sum())}")

    # Classes are already 50/50 from upstream sampling, so class_weight is left at
    # default (None). Setting class_weight='balanced' here would be a no-op on
    # balanced data and only adds noise to the comparison across baseline models.
    print(f"\nFitting RandomForestClassifier({RF_PARAMS}) ...")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print("done.")

    y_pred = clf.predict(X_val)
    y_score = clf.predict_proba(X_val)[:, 1]

    overall = binary_metrics(y_val, y_pred, y_score)

    print("\n" + "-" * 78)
    print("VALIDATION METRICS (overall, positive class = phishing)")
    print("-" * 78)
    print(f"precision           : {fmt(overall['precision'])}")
    print(f"recall              : {fmt(overall['recall'])}")
    print(f"f1                  : {fmt(overall['f1'])}")
    print(f"false positive rate : {fmt(overall['false_positive_rate'])}   (target < 0.005)")
    print(f"auc-roc             : {fmt(overall['auc_roc'])}")
    print(f"accuracy            : {fmt(overall['accuracy'])}")
    cm = overall["confusion_matrix"]
    print("\nconfusion matrix (rows = actual, cols = predicted):")
    print(f"                 pred_legit  pred_phish")
    print(f"  actual_legit   {cm['tn']:>10}  {cm['fp']:>10}")
    print(f"  actual_phish   {cm['fn']:>10}  {cm['tp']:>10}")

    print("\n" + "-" * 78)
    print("VALIDATION METRICS BY source_dataset")
    print("-" * 78)
    print(f"{'source':<22} {'n':>5} {'prec':>7} {'recall':>7} {'f1':>7} "
          f"{'fpr':>7} {'auc':>7}   TN/FP/FN/TP")
    by_source = {}
    for src in sorted(pd.unique(val_source)):
        m = val_source == src
        sm = binary_metrics(y_val[m], y_pred[m], y_score[m])
        by_source[src] = sm
        c = sm["confusion_matrix"]
        print(f"{src:<22} {sm['n']:>5} {fmt(sm['precision']):>7} {fmt(sm['recall']):>7} "
              f"{fmt(sm['f1']):>7} {fmt(sm['false_positive_rate']):>7} "
              f"{fmt(sm['auc_roc']):>7}   {c['tn']}/{c['fp']}/{c['fn']}/{c['tp']}")
        for note in sm["notes"]:
            print(f"{'':<22}   note: {note}")

    names = feature_names()
    if len(names) != X_train.shape[1]:
        raise ValueError(
            f"feature name count {len(names)} != feature matrix width {X_train.shape[1]}"
        )
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_features = [
        {"rank": i + 1, "feature": names[j], "index": int(j),
         "importance": float(importances[j])}
        for i, j in enumerate(top_idx)
    ]

    print("\n" + "-" * 78)
    print("TOP 20 FEATURES BY RANDOM FOREST IMPORTANCE")
    print("-" * 78)
    for f in top_features:
        print(f"  {f['rank']:>2}. {f['feature']:<32} {f['importance']:.5f}")

    dense_importance = float(importances[-len(DENSE_COLS):].sum())
    print(f"\nsum of importance over the 12 dense text/URL features: {dense_importance:.5f}")
    print("dense feature importances:")
    for k, col in enumerate(DENSE_COLS):
        j = X_train.shape[1] - len(DENSE_COLS) + k
        print(f"  {col:<24} {importances[j]:.5f}")

    metrics = {
        "model": "RandomForestClassifier",
        "phase": "4",
        "step": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "positive_class": POSITIVE_LABEL,
        "params": RF_PARAMS,
        "class_weight_note": (
            "left at default (None); train split is already 50/50 phishing/legit "
            "from upstream sampling, so class_weight='balanced' would be a no-op"
        ),
        "data": {
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "n_features": int(X_train.shape[1]),
            "tfidf_features": int(X_train.shape[1] - len(DENSE_COLS)),
            "dense_features": len(DENSE_COLS),
            "test_set_touched": False,
        },
        "validation_overall": overall,
        "validation_by_source_dataset": by_source,
        "top_20_features": top_features,
        "dense_feature_importance": {
            "sum_over_12_dense_features": dense_importance,
            "per_feature": {
                col: float(importances[X_train.shape[1] - len(DENSE_COLS) + k])
                for k, col in enumerate(DENSE_COLS)
            },
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("\n" + "=" * 78)
    print(f"saved model   -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
