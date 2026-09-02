"""Combine TF-IDF, text/lexical, and URL features into the final scaled
feature matrix per split, ready for model training.

Fits a StandardScaler on train's dense numeric features only (the 6
text/lexical + 6 URL columns) and applies it, unrefit, to val/test.
TF-IDF is deliberately never StandardScaled:
  - TF-IDF rows are already L2-normalized by TfidfVectorizer, so they're
    comparably scaled across documents without any further transform.
  - StandardScaler centers each column by subtracting its mean, which
    turns a sparse matrix dense -- for a 5,000-column TF-IDF matrix
    that's a ~10-100x memory blow-up for no benefit, which is exactly
    why sklearn's own StandardScaler documents with_mean=False for
    sparse input. We keep TF-IDF sparse and only scale the small dense
    block, then hstack the two back together.

Reads only the already-built per-split feature files; does not touch
train.csv/val.csv/test.csv or re-run text/URL feature extraction.

Outputs:
  src/models/artifacts/feature_scaler.pkl
  data/processed/features/{split}_final.npz    (sparse, TF-IDF + scaled dense)
  data/processed/features/{split}_labels.csv   (row-aligned label column)
"""

from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"

SPLITS = ["train", "val", "test"]

TEXT_DENSE_COLS = [
    "urgency_score", "healthcare_term_count", "word_count",
    "avg_word_length", "exclamation_count", "capitalized_word_count",
]
URL_DENSE_COLS = [
    "url_count", "has_ip_literal", "has_url_shortener",
    "avg_domain_entropy", "suspicious_tld", "has_at_symbol",
]
DENSE_COLS = TEXT_DENSE_COLS + URL_DENSE_COLS


def load_dense_features(name: str) -> pd.DataFrame:
    text_feats = pd.read_csv(FEATURES_DIR / f"{name}_text_features.csv")[TEXT_DENSE_COLS]
    url_feats = pd.read_csv(FEATURES_DIR / f"{name}_url_features.csv")[URL_DENSE_COLS]
    assert len(text_feats) == len(url_feats), (
        f"{name}: text_features rows ({len(text_feats)}) != url_features rows ({len(url_feats)})"
    )
    combined = pd.concat(
        [text_feats.reset_index(drop=True), url_feats.reset_index(drop=True)], axis=1
    )
    return combined.astype(float)


def load_labels(name: str) -> pd.Series:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")["label"]


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tfidf = {name: sparse.load_npz(FEATURES_DIR / f"{name}_tfidf.npz") for name in SPLITS}
    dense = {name: load_dense_features(name) for name in SPLITS}
    labels = {name: load_labels(name) for name in SPLITS}

    for name in SPLITS:
        assert tfidf[name].shape[0] == len(dense[name]) == len(labels[name]), (
            f"{name}: row count mismatch -- tfidf={tfidf[name].shape[0]}, "
            f"dense={len(dense[name])}, labels={len(labels[name])}"
        )

    scaler = StandardScaler()
    scaled_dense = {"train": scaler.fit_transform(dense["train"][DENSE_COLS])}
    joblib.dump(scaler, SCALER_PATH)
    print(f"Fitted StandardScaler on train dense features ({dense['train'].shape[1]} columns)")
    print(f"Saved scaler to {SCALER_PATH}")
    print()

    for name in ["val", "test"]:
        scaled_dense[name] = scaler.transform(dense[name][DENSE_COLS])

    final = {}
    for name in SPLITS:
        dense_sparse = sparse.csr_matrix(scaled_dense[name])
        final[name] = sparse.hstack([tfidf[name], dense_sparse], format="csr")
        final_path = FEATURES_DIR / f"{name}_final.npz"
        sparse.save_npz(final_path, final[name])

        labels_path = FEATURES_DIR / f"{name}_labels.csv"
        labels[name].to_csv(labels_path, index=False)

        print(f"[{name}] tfidf {tfidf[name].shape} + dense {scaled_dense[name].shape} "
              f"-> final {final[name].shape} -> {final_path}")
        print(f"[{name}] labels ({len(labels[name])}) -> {labels_path}")

    print()
    print(f"Final feature dimension per email: {tfidf['train'].shape[1]} (TF-IDF) + "
          f"{len(DENSE_COLS)} (dense text+URL) = {final['train'].shape[1]}")

    print()
    print("=" * 70)
    print("ROW COUNT CHECK")
    print("=" * 70)
    for name, expected in [("train", 10500), ("val", 2250), ("test", 2250)]:
        actual = final[name].shape[0]
        print(f"[{name}] final matrix rows: {actual} | labels rows: {len(labels[name])} "
              f"| expected: {expected} | match: {actual == len(labels[name]) == expected}")

    print()
    print("=" * 70)
    print("ALIGNMENT SPOT CHECK (train, rows 0, 1, 5000, 10499)")
    print("=" * 70)
    check_idx = [0, 1, 5000, 10499]
    dense_train = dense["train"].reset_index(drop=True)
    for i in check_idx:
        label_val = labels["train"].iloc[i]
        urgency = dense_train.loc[i, "urgency_score"]
        url_count = dense_train.loc[i, "url_count"]
        scaled_urgency = scaled_dense["train"][i, DENSE_COLS.index("urgency_score")]
        print(f"row {i}: label={label_val!r} | raw urgency_score={urgency:.4f} | "
              f"raw url_count={url_count:.0f} | scaled urgency_score={scaled_urgency:.4f}")


if __name__ == "__main__":
    main()
