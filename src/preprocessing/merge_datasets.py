"""Merge all raw/staged data sources into one unified dataset at
data/processed/merged_raw.csv.

Unified schema: text, label, source_dataset, subcategory, included_in_training
  - text: subject + body combined (or the source's single text field where
    there's no separate subject)
  - label: "phishing" or "legitimate" for rows used in training. SpamAssassin's
    "spam" rows are not part of this binary target and get "spam_excluded"
    instead, for reference only.
  - source_dataset: spamassassin | nazario | kaggle_chakraborty |
    synthetic_healthcare
  - subcategory: only populated for synthetic_healthcare rows, null otherwise
  - included_in_training: True for every row except SpamAssassin's spam
    group, which is general spam, not phishing, and is excluded from the
    phishing/legitimate binary target

No cleaning, tokenization, feature extraction, deduplication, or class
balancing happens here — that's a later step.
"""

import glob
import json
import mailbox
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SYNTHETIC_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "merged_raw.csv"

COLUMNS = ["text", "label", "source_dataset", "subcategory", "included_in_training"]


def _flatten_spamassassin_node(node) -> str:
    if isinstance(node, str):
        return "" if node == "…" else node
    if isinstance(node, list):
        return "\n".join(filter(None, (_flatten_spamassassin_node(n) for n in node)))
    return ""


def load_spamassassin() -> pd.DataFrame:
    """SpamAssassin's own labels are spam/ham, not phishing/legitimate.
    "ham" maps to "legitimate". "spam" is general spam, not phishing, so it
    is excluded from the binary training target: those rows are labeled
    "spam_excluded" for reference only and get included_in_training = False.
    """
    path = RAW_DIR / "spamassassin.csv"
    df = pd.read_csv(path)

    def flatten(raw_text: str) -> str:
        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return str(raw_text)
        return _flatten_spamassassin_node(parsed)

    text = df["text"].apply(flatten)
    label = df["label"].map({0: "spam_excluded", 1: "legitimate"})
    included_in_training = df["label"].map({0: False, 1: True})

    out = pd.DataFrame(
        {
            "text": text,
            "label": label,
            "source_dataset": "spamassassin",
            "subcategory": None,
            "included_in_training": included_in_training,
        }
    )
    return out[COLUMNS]


def _extract_nazario_message(msg) -> tuple[str, str]:
    subject = msg.get("Subject", "") or ""

    def decode_payload(part) -> str:
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            return ""
        if not payload:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")

    plain_parts, html_parts = [], []
    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            text = decode_payload(part)
            if not text:
                continue
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
    else:
        text = decode_payload(msg)
        if msg.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        body = "\n".join(plain_parts)
    elif html_parts:
        # Nazario is mostly HTML-only phishing pages; strip markup so "text"
        # holds readable content, not raw tags/attributes.
        body = "\n".join(
            BeautifulSoup(h, "html.parser").get_text(separator=" ", strip=True)
            for h in html_parts
        )
    else:
        body = ""

    return subject, body


def load_nazario() -> pd.DataFrame:
    nazario_dir = RAW_DIR / "nazario"
    rows = []
    for path in sorted(nazario_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            box = mailbox.mbox(str(path))
            keys = list(box.keys())
        except Exception:
            continue
        for key in keys:
            try:
                msg = box[key]
            except Exception:
                # mailbox's stdlib parser assumes ASCII on the "From " separator
                # line; a handful of messages have non-ASCII bytes there and
                # raise UnicodeDecodeError. Skip those rather than losing the
                # whole file — most messages in a file parse fine.
                continue
            subject, body = _extract_nazario_message(msg)
            text = f"{subject}\n\n{body}".strip() if subject else body.strip()
            if not text:
                continue
            rows.append(text)

    return pd.DataFrame(
        {
            "text": rows,
            "label": "phishing",
            "source_dataset": "nazario",
            "subcategory": None,
            "included_in_training": True,
        }
    )[COLUMNS]


def load_healthcare_kaggle() -> pd.DataFrame:
    path = RAW_DIR / "healthcare_phishing.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["Email Text"])
    label = df["Email Type"].map({"Phishing Email": "phishing", "Safe Email": "legitimate"})

    out = pd.DataFrame(
        {
            "text": df["Email Text"],
            "label": label,
            "source_dataset": "kaggle_chakraborty",
            "subcategory": None,
            "included_in_training": True,
        }
    )
    return out[COLUMNS]


def load_synthetic_healthcare() -> pd.DataFrame:
    batch_paths = sorted(glob.glob(str(SYNTHETIC_DIR / "healthcare_synthetic_*.csv")))
    frames = []
    for path in batch_paths:
        df = pd.read_csv(path)
        text = df["subject"].fillna("") + "\n\n" + df["body"].fillna("")
        label = df["email_type"].map({"Phishing Email": "phishing", "Safe Email": "legitimate"})
        frames.append(
            pd.DataFrame(
                {
                    "text": text,
                    "label": label,
                    "source_dataset": "synthetic_healthcare",
                    "subcategory": df["subcategory"],
                    "included_in_training": True,
                }
            )[COLUMNS]
        )
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parts = {
        "spamassassin": load_spamassassin(),
        "nazario": load_nazario(),
        "kaggle_chakraborty": load_healthcare_kaggle(),
        "synthetic_healthcare": load_synthetic_healthcare(),
    }

    for name, df in parts.items():
        print(f"[merge] {name}: {len(df)} rows")

    merged = pd.concat(parts.values(), ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(merged)}")
    print()
    print("By source_dataset:")
    print(merged["source_dataset"].value_counts())
    print()
    print("By label:")
    print(merged["label"].value_counts())
    print()
    print("By included_in_training:")
    print(merged["included_in_training"].value_counts())
    print()
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
