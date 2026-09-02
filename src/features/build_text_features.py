"""Build text/lexical feature matrices for train/val/test.

Fits the TF-IDF vectorizer on train only, saves it as a reusable artifact,
and applies the same fitted vectorizer to val/test. Lexicon and structural
features (src/features/text_features.py) need no fitting and are computed
directly per split.

No header-based features, no URL/domain features, no model training here.

Outputs:
  src/models/artifacts/tfidf_vectorizer.pkl
  data/processed/features/{split}_tfidf.npz          (sparse TF-IDF matrix)
  data/processed/features/{split}_text_features.csv  (dense lexicon/structural
                                                        features, row-aligned
                                                        with data/processed/{split}.csv)
"""

from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from text_features import extract_row_features

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"

SPLITS = ["train", "val", "test"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    df["text"] = df["text"].fillna("")
    return df


def build_lexicon_structural_features(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(texts.apply(extract_row_features).tolist())


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {name: load_split(name) for name in SPLITS}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    train_tfidf = vectorizer.fit_transform(data["train"]["text"])
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Fitted TF-IDF on train ({train_tfidf.shape[0]} rows), "
          f"vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print()

    tfidf_by_split = {"train": train_tfidf}
    for name in ["val", "test"]:
        tfidf_by_split[name] = vectorizer.transform(data[name]["text"])

    lexicon_by_split = {
        name: build_lexicon_structural_features(data[name]["text"])
        for name in SPLITS
    }

    for name in SPLITS:
        tfidf_path = FEATURES_DIR / f"{name}_tfidf.npz"
        sparse.save_npz(tfidf_path, tfidf_by_split[name])

        csv_path = FEATURES_DIR / f"{name}_text_features.csv"
        lexicon_by_split[name].to_csv(csv_path, index=False)

        print(f"[{name}] rows: {len(data[name])} | "
              f"tfidf shape: {tfidf_by_split[name].shape} -> {tfidf_path} | "
              f"lexicon/structural shape: {lexicon_by_split[name].shape} -> {csv_path}")

    print()
    print("Feature count per email (TF-IDF dims + lexicon/structural dims): "
          f"{train_tfidf.shape[1]} + {lexicon_by_split['train'].shape[1]} "
          f"= {train_tfidf.shape[1] + lexicon_by_split['train'].shape[1]}")

    print()
    print("=" * 70)
    print("SANITY CHECK: urgency_score / healthcare_term_count by label (train)")
    print("=" * 70)
    check = lexicon_by_split["train"].copy()
    check["label"] = data["train"]["label"].values
    check["source_dataset"] = data["train"]["source_dataset"].values
    print(check.groupby("label")[["urgency_score", "healthcare_term_count"]].mean())
    print()
    print("By source_dataset x label:")
    print(check.groupby(["source_dataset", "label"])[["urgency_score", "healthcare_term_count"]].mean())


if __name__ == "__main__":
    main()
