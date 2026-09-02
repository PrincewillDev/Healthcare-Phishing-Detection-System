"""Build URL/domain feature matrices for train/val/test.

Rule-based, nothing to fit -- unlike TF-IDF there's no vectorizer artifact
to save. Reads only train.csv/val.csv/test.csv and writes one CSV per split.

domain_age is skipped entirely (see src/features/url_features.py docstring):
it would require a live or cached WHOIS lookup, which is out of scope per
the project's hard constraint against external calls in the data/feature
pipeline.

No text/lexical features, no header features, no model training here.

Outputs:
  data/processed/features/{split}_url_features.csv, row-aligned with
  data/processed/{split}.csv
"""

from pathlib import Path

import pandas as pd

from url_features import extract_row_url_features

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"

SPLITS = ["train", "val", "test"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    df["text"] = df["text"].fillna("")
    return df


def build_url_features(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(texts.apply(extract_row_url_features).tolist())


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    data = {name: load_split(name) for name in SPLITS}
    features = {name: build_url_features(data[name]["text"]) for name in SPLITS}

    for name in SPLITS:
        csv_path = FEATURES_DIR / f"{name}_url_features.csv"
        features[name].to_csv(csv_path, index=False)
        print(f"[{name}] rows: {len(data[name])} | "
              f"url feature shape: {features[name].shape} -> {csv_path}")

    print()
    print(f"Feature count added: {features['train'].shape[1]} "
          f"({', '.join(features['train'].columns)})")

    print()
    print("=" * 70)
    print("ZERO-URL EMAILS")
    print("=" * 70)
    for name in SPLITS:
        df = data[name].copy()
        df["url_count"] = features[name]["url_count"].values
        zero = df["url_count"] == 0
        print(f"[{name}] {zero.sum()} / {len(df)} emails have zero URLs "
              f"({zero.mean():.1%})")

    print()
    print("Zero-URL breakdown on train, by source_dataset x label:")
    train_df = data["train"].copy()
    train_df["url_count"] = features["train"]["url_count"].values
    train_df["has_url"] = train_df["url_count"] > 0
    breakdown = (
        train_df.groupby(["source_dataset", "label"])["has_url"]
        .agg(zero_urls=lambda s: (~s).sum(), total="size")
    )
    print(breakdown)

    print()
    print("=" * 70)
    print("SANITY CHECK: url_count / avg_domain_entropy by label (train)")
    print("=" * 70)
    check = features["train"].copy()
    check["label"] = data["train"]["label"].values
    check["source_dataset"] = data["train"]["source_dataset"].values
    print(check.groupby("label")[["url_count", "avg_domain_entropy"]].mean())
    print()
    print("By source_dataset x label:")
    print(check.groupby(["source_dataset", "label"])[["url_count", "avg_domain_entropy"]].mean())
    print()
    print("Within synthetic_healthcare only:")
    hc = check[check.source_dataset == "synthetic_healthcare"]
    print(hc.groupby("label")[["url_count", "avg_domain_entropy", "has_ip_literal",
                                "has_url_shortener", "suspicious_tld", "has_at_symbol"]].mean())


if __name__ == "__main__":
    main()
