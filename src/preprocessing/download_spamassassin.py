"""Download the SpamAssassin corpus (talby/spamassassin on Hugging Face) to data/raw/.

Idempotent: skips the download if data/raw/spamassassin.csv already exists.

Reads Hugging Face's auto-converted parquet directly via pandas + huggingface_hub's
hf:// filesystem. This executes no remote code (unlike the dataset's own loading
script, which downloads and unpacks arbitrary tarballs).
"""

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "spamassassin.csv"

# talby/spamassassin ships no parquet on its main branch, only a legacy loading
# script. Hugging Face auto-converts it to parquet on refs/convert/parquet under
# the "text" config (not "plain_text").
PARQUET_URI = "hf://datasets/talby/spamassassin@refs/convert/parquet/text/train/0000.parquet"


def main() -> None:
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        print(f"[spamassassin] {OUTPUT_PATH} already exists, skipping download.")
        print(f"[spamassassin] Existing row count: {len(df)}")
        return

    print(f"[spamassassin] Reading parquet from {PARQUET_URI}...")
    try:
        df = pd.read_parquet(PARQUET_URI)
    except Exception as exc:
        print(f"[spamassassin] Parquet read failed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[spamassassin] Saved {len(df)} rows to {OUTPUT_PATH}")

    print("[spamassassin] value_counts() for 'group' column:")
    print(df["group"].value_counts())


if __name__ == "__main__":
    main()
