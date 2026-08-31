"""Download the healthcare-context phishing dataset (Kaggle: subhajournal/phishingemails)
to data/raw/healthcare_phishing.csv.

Idempotent: skips the download if data/raw/healthcare_phishing.csv already exists.

Requires Kaggle API credentials. If none are configured, this script stops and reports
what's needed instead of attempting a workaround.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "healthcare_phishing.csv"
KAGGLE_DATASET = "subhajournal/phishingemails"

CREDENTIALS_MESSAGE = """
[healthcare] Kaggle API credentials are not configured.

kagglehub needs one of the following before it can download datasets:

  1. A kaggle.json API token file at:
       Windows: %USERPROFILE%\\.kaggle\\kaggle.json
       (or the location reported by kagglehub's own error above, if shown)
     Get one from https://www.kaggle.com/settings -> API -> "Create New Token".
     This downloads a kaggle.json containing your username and key.

  2. OR the environment variables KAGGLE_USERNAME and KAGGLE_KEY set to the
     values from that same kaggle.json.

Set up one of the above, then re-run:
    python src/preprocessing/download_healthcare.py
"""


def main() -> None:
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        print(f"[healthcare] {OUTPUT_PATH} already exists, skipping download.")
        print(f"[healthcare] Existing row count: {len(df)}")
        return

    import os

    has_env_creds = bool(os.environ.get("KAGGLE_USERNAME")) and bool(
        os.environ.get("KAGGLE_KEY")
    )
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_json_creds = kaggle_json.exists()

    if not has_env_creds and not has_json_creds:
        print(CREDENTIALS_MESSAGE)
        sys.exit(1)

    import kagglehub

    print(f"[healthcare] Downloading {KAGGLE_DATASET} via kagglehub...")
    try:
        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as exc:
        print(f"[healthcare] kagglehub download failed: {exc}")
        print(CREDENTIALS_MESSAGE)
        sys.exit(1)

    download_dir = Path(download_path)
    csv_files = list(download_dir.glob("*.csv"))
    if not csv_files:
        print(f"[healthcare] No CSV file found in downloaded dataset at {download_dir}")
        sys.exit(1)

    source_csv = csv_files[0]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_csv, OUTPUT_PATH)
    print(f"[healthcare] Saved dataset to {OUTPUT_PATH}")

    df = pd.read_csv(OUTPUT_PATH)
    print(f"[healthcare] Total row count: {len(df)}")

    label_col = None
    for candidate in ("Email Type", "label", "Label", "class", "Class", "type", "Type"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None:
        print(f"[healthcare] Class balance (column '{label_col}'):")
        print(df[label_col].value_counts())
    else:
        print(f"[healthcare] Could not identify a label column among: {list(df.columns)}")
        print("[healthcare] Inspect the CSV manually to determine the class balance.")


if __name__ == "__main__":
    main()
