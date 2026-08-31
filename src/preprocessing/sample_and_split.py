"""Sample the final 15,000-row training dataset from data/processed/cleaned.csv,
split it into train/val/test, and build a separate confound-safeguard slice.

Reads ONLY data/processed/cleaned.csv. Does not touch merged_raw.csv, any raw
source files, cleaned.csv itself, or data/synthetic/**.

Steps:
  1. Stratified sampling to exact per-source/per-label targets -> 15,000 rows
  2. 70/15/15 train/val/test split, stratified by (source_dataset, label)
  3. Separate 1,000-row confound-safeguard slice (kaggle_chakraborty +
     synthetic_healthcare only, 500/500 phishing/legitimate), drawn from the
     leftover rows in cleaned.csv after step 1, never overlapping with it
  4. Consolidated report: counts, breakdowns, overlap check
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CLEANED_PATH = PROCESSED_DIR / "cleaned.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
CONFOUND_PATH = PROCESSED_DIR / "confound_check.csv"

RANDOM_STATE = 42

SAMPLE_TARGETS = {
    ("kaggle_chakraborty", "phishing"): 5500,
    ("nazario", "phishing"): 1500,
    ("synthetic_healthcare", "phishing"): 500,
    ("kaggle_chakraborty", "legitimate"): 5500,
    ("spamassassin", "legitimate"): 1500,
    ("synthetic_healthcare", "legitimate"): 500,
}

CONFOUND_SOURCES = {"kaggle_chakraborty", "synthetic_healthcare"}
CONFOUND_TARGET_PER_LABEL = 500


def load_cleaned() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_PATH)
    assert (df["included_in_training"] == True).all(), (  # noqa: E712
        "cleaned.csv contains included_in_training=False rows"
    )
    return df


def sample_15000(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: stratified sampling to 15,000-row target")
    print("=" * 70)
    parts = []
    for (source, label), n in SAMPLE_TARGETS.items():
        group = df[(df["source_dataset"] == source) & (df["label"] == label)]
        available = len(group)
        sampled = group.sample(n=n, random_state=RANDOM_STATE)
        print(f"{source:22s} {label:11s} target={n:5d}  available={available:5d}  drawn={len(sampled):5d}")
        parts.append(sampled)
    sample = pd.concat(parts).sort_index()

    print()
    print(f"Total sampled: {len(sample)}")
    print("By source_dataset x label:")
    print(sample.groupby(["source_dataset", "label"]).size())
    print()
    return sample


def split_train_val_test(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("STEP 2: 70/15/15 train/val/test split (stratified by source_dataset x label)")
    print("=" * 70)
    strat_key = sample["source_dataset"] + "|" + sample["label"]

    train, temp = train_test_split(
        sample,
        test_size=0.30,
        stratify=strat_key,
        random_state=RANDOM_STATE,
    )
    temp_strat_key = temp["source_dataset"] + "|" + temp["label"]
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp_strat_key,
        random_state=RANDOM_STATE,
    )

    for name, split_df in (("train", train), ("val", val), ("test", test)):
        print(f"\n{name}: {len(split_df)} rows")
        print(split_df.groupby(["source_dataset", "label"]).size())
    print()
    return train, test, val  # noqa -- returned below in explicit order


def build_confound_check(df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 3: confound-safeguard validation slice (separate from train/val/test)")
    print("=" * 70)
    leftover = df.drop(index=sample.index)
    eligible = leftover[leftover["source_dataset"].isin(CONFOUND_SOURCES)]

    print("Leftover rows available for confound-check, by source x label:")
    print(eligible.groupby(["source_dataset", "label"]).size())
    print()

    parts = []
    for label in ("phishing", "legitimate"):
        pool = eligible[eligible["label"] == label]
        drawn = pool.sample(n=CONFOUND_TARGET_PER_LABEL, random_state=RANDOM_STATE)
        parts.append(drawn)
    confound = pd.concat(parts).sort_index()

    print(f"Confound-check total: {len(confound)}")
    print("By source_dataset x label:")
    print(confound.groupby(["source_dataset", "label"]).size())
    print()
    return confound


def main() -> None:
    df = load_cleaned()
    print(f"Loaded cleaned.csv: {len(df)} rows (all included_in_training=True)\n")

    sample = sample_15000(df)
    train, test, val = split_train_val_test(sample)
    confound = build_confound_check(df, sample)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    confound.to_csv(CONFOUND_PATH, index=False)

    print("=" * 70)
    print("STEP 4: consolidated report")
    print("=" * 70)
    total_used = len(sample) + len(confound)
    print(f"Total rows across all outputs: {len(sample)} (15k sample) + "
          f"{len(confound)} (confound-check) = {total_used}")

    overlap = set(sample.index) & set(confound.index)
    print(f"Row-index overlap between 15k sample and confound-check: {len(overlap)}")

    split_indices = [set(train.index), set(val.index), set(test.index)]
    split_overlap = (split_indices[0] & split_indices[1]) | \
                     (split_indices[0] & split_indices[2]) | \
                     (split_indices[1] & split_indices[2])
    print(f"Row-index overlap among train/val/test: {len(split_overlap)}")

    reconstructed = len(train) + len(val) + len(test)
    print(f"train + val + test = {reconstructed} (expect 15000)")

    all_incl = all(
        (d["included_in_training"] == True).all()  # noqa: E712
        for d in (train, val, test, confound)
    )
    print(f"All output rows have included_in_training=True: {all_incl}")

    print()
    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {VAL_PATH}")
    print(f"Saved: {TEST_PATH}")
    print(f"Saved: {CONFOUND_PATH}")


if __name__ == "__main__":
    main()
