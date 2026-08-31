"""Single entry point to fetch all raw datasets for the project.

Runs, in sequence:
  1. download_spamassassin.py -> data/raw/spamassassin.csv
  2. download_nazario.py      -> data/raw/nazario/ (or data/raw/nazario_fallback.csv)
  3. download_healthcare.py   -> data/raw/healthcare_phishing.csv

Each step is idempotent and skips re-downloading if its output already exists.
Prints a final summary of what was downloaded fresh vs. skipped, and the
resulting row/file counts for each dataset.

No step executes remote code: SpamAssassin reads a pre-converted parquet file
directly (no dataset loading script), Nazario runs only the wget binary, and
Healthcare downloads a data file via the Kaggle API client.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_healthcare  # noqa: E402
import download_nazario  # noqa: E402
import download_spamassassin  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def spamassassin_status() -> tuple[bool, str]:
    path = RAW_DIR / "spamassassin.csv"
    if not path.exists():
        return False, "not present"
    return True, f"{len(pd.read_csv(path))} rows"


def nazario_status() -> tuple[bool, str]:
    nazario_dir = RAW_DIR / "nazario"
    fallback_path = RAW_DIR / "nazario_fallback.csv"
    if nazario_dir.exists():
        count = sum(1 for p in nazario_dir.rglob("*") if p.is_file())
        if count > 0:
            return True, f"{count} files (primary mirror)"
    if fallback_path.exists():
        return True, f"1 file (fallback CSV, {len(pd.read_csv(fallback_path))} rows)"
    return False, "not present"


def healthcare_status() -> tuple[bool, str]:
    path = RAW_DIR / "healthcare_phishing.csv"
    if not path.exists():
        return False, "not present"
    return True, f"{len(pd.read_csv(path))} rows"


def summarize(name: str, was_present_before: bool, present_after: bool, detail: str) -> str:
    if not present_after:
        state = "MISSING"
    elif was_present_before:
        state = "skipped (already present)"
    else:
        state = "downloaded fresh"
    return f"  {name}: {state} - {detail}"


def main() -> None:
    before_spam, _ = spamassassin_status()
    before_naz, _ = nazario_status()
    before_health, _ = healthcare_status()

    print("=" * 60)
    print("Step 1/3: SpamAssassin corpus")
    print("=" * 60)
    try:
        download_spamassassin.main()
    except SystemExit:
        print("[pipeline] SpamAssassin step stopped (see message above).")

    print()
    print("=" * 60)
    print("Step 2/3: Nazario Phishing Corpus")
    print("=" * 60)
    try:
        download_nazario.main()
    except SystemExit:
        print("[pipeline] Nazario step stopped (see message above).")

    print()
    print("=" * 60)
    print("Step 3/3: Healthcare-context dataset (Kaggle)")
    print("=" * 60)
    try:
        download_healthcare.main()
    except SystemExit:
        print("[pipeline] Healthcare dataset step stopped (see message above).")

    present_spam, after_spam = spamassassin_status()
    present_naz, after_naz = nazario_status()
    present_health, after_health = healthcare_status()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summarize("SpamAssassin", before_spam, present_spam, after_spam))
    print(summarize("Nazario", before_naz, present_naz, after_naz))
    print(summarize("Healthcare", before_health, present_health, after_health))
    print()
    print(
        "No remote code execution in this pipeline: SpamAssassin reads parquet "
        "directly (no dataset loading script, no trust_remote_code), Nazario "
        "only invokes the wget binary, Healthcare only calls the Kaggle API "
        "client to fetch a data file."
    )


if __name__ == "__main__":
    main()
