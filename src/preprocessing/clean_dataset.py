"""Clean the merged dataset (data/processed/merged_raw.csv) and write
data/processed/cleaned.csv, ready for sampling in a later step.

Per .claude/rules/training-data-integrity.md, this script filters to
included_in_training == True before doing anything else -- SpamAssassin's
excluded "spam_excluded" rows are never loaded past that point and never
appear in cleaned.csv.

Steps:
  1. Remove blank/placeholder rows
  2. Deduplicate (exact text match, within- and cross-source)
  3. Fix encoding artifacts (MIME encoded-words, mojibake) and strip HTML
  4. Detect (but do not remove) non-email content in SpamAssassin rows --
     requires --apply-non-email-filter to actually remove them

No sampling, train/val/test split, or feature engineering happens here.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from email.header import decode_header

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MERGED_PATH = PROCESSED_DIR / "merged_raw.csv"
OUTPUT_PATH = PROCESSED_DIR / "cleaned.csv"

PLACEHOLDER_VALUES = {
    "empty",
    "n/a",
    "na",
    "null",
    "none",
    "nil",
    "[no content]",
    "no content",
    "unknown",
    "undefined",
    "blank",
}

MIME_ENCODED_WORD_RE = re.compile(r"=\?[\w-]+\?[bBqQ]\?[^?]*\?=")

MOJIBAKE_HINT_RE = re.compile(
    r"(Ã¢â‚¬|Ã©|Ã¨|â€™|â€œ|â€\x9d|â€“|â€”|Â[\x80-\xbf])"
)

REPLACEMENT_CHAR = "�"

HTML_TAGS = (
    "html|head|body|div|span|p|br|a|img|table|tr|td|th|thead|tbody|ul|ol|li"
    "|h[1-6]|b|i|u|strong|em|font|style|script|meta|link|form|input|button"
    "|title|center|blockquote|pre|code|hr|iframe|small|sub|sup|nav|footer"
    "|header|section"
)
HTML_TAG_RE = re.compile(r"</?(?:" + HTML_TAGS + r")\b[^>]*>", re.I)

NON_EMAIL_PATTERNS = {
    "cvs_commit": re.compile(
        r"\b(cvs commit|Modified Files:|Log Message:|Index: |Revision \d|checked in|CVSROOT)",
        re.I,
    ),
    "git_diff": re.compile(r"(diff --git|^\+\+\+ |^--- |^@@ )", re.M),
    "code_snippet": re.compile(
        r"(#include\s|def \w+\(|public class |import java|<\?php|function\s+\w+\s*\()",
        re.I,
    ),
    "mailing_list_footer": re.compile(
        r"(to unsubscribe|mailman/listinfo|majordomo|list-unsubscribe"
        r"|_______________________________________________|exmh-workers"
        r"|ilug@linux\.ie|spamassassin-(devel|talk|sightings)@)",
        re.I,
    ),
    "linguistics": re.compile(
        r"\b(phoneme|morpholog|syntax tree|corpus linguistics|part-of-speech"
        r"|noun phrase|verb phrase)\w*",
        re.I,
    ),
}


def load_included() -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(MERGED_PATH)
    excluded_count = int((df["included_in_training"] == False).sum())  # noqa: E712
    df = df[df["included_in_training"] == True].reset_index(drop=True)  # noqa: E712
    return df, excluded_count


def is_blank_or_placeholder(text) -> bool:
    if pd.isna(text):
        return True
    s = str(text).strip()
    if s == "":
        return True
    return s.lower() in PLACEHOLDER_VALUES


def decode_mime_words(text: str) -> str:
    if not MIME_ENCODED_WORD_RE.search(text):
        return text

    def _decode_match(m: re.Match) -> str:
        try:
            parts = decode_header(m.group())
            return "".join(
                (p.decode(enc or "utf-8", errors="replace") if isinstance(p, bytes) else p)
                for p, enc in parts
            )
        except Exception:
            return m.group()

    return MIME_ENCODED_WORD_RE.sub(_decode_match, text)


def fix_mojibake(text: str) -> str:
    if not MOJIBAKE_HINT_RE.search(text):
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text
    # Reject the round-trip if it corrupted some other part of the string into
    # U+FFFD that wasn't already there -- a net-negative "fix" is worse than
    # leaving the original mojibake in place.
    if REPLACEMENT_CHAR in fixed and REPLACEMENT_CHAR not in text:
        return text
    # Only accept the fix if it actually reduced the mojibake hint count.
    if len(MOJIBAKE_HINT_RE.findall(fixed)) < len(MOJIBAKE_HINT_RE.findall(text)):
        return fixed
    return text


def strip_html(text: str) -> str:
    if not HTML_TAG_RE.search(text):
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def clean_text_encoding_and_html(text: str) -> str:
    text = decode_mime_words(text)
    text = fix_mojibake(text)
    text = strip_html(text)
    return text


def report_step1(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["text"].apply(is_blank_or_placeholder)
    removed = df[mask]
    print("=" * 70)
    print("STEP 1: blank / placeholder row removal")
    print("=" * 70)
    print(f"Removed: {len(removed)} / {len(df)}")
    print(removed.groupby("source_dataset").size().reindex(
        df["source_dataset"].unique(), fill_value=0
    ))
    print()
    return df[~mask].reset_index(drop=True)


def report_step2(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 2: deduplication (exact text match)")
    print("=" * 70)
    first_source = df.groupby("text")["source_dataset"].transform("first")
    is_dup = df.duplicated(subset="text", keep="first")
    removed = df[is_dup]
    cross_source = removed[removed["source_dataset"] != first_source[is_dup]]

    print(f"Removed: {len(removed)} / {len(df)}")
    print(removed.groupby("source_dataset").size().reindex(
        df["source_dataset"].unique(), fill_value=0
    ))
    print(f"Of those, cross-source duplicates (text's first occurrence was in "
          f"a different source_dataset): {len(cross_source)}")
    print()
    return df[~is_dup].reset_index(drop=True)


def report_step3(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 3: encoding normalization + HTML stripping")
    print("=" * 70)

    text = df["text"]
    before_html = text.apply(lambda s: bool(HTML_TAG_RE.search(s)))
    before_mime = text.apply(lambda s: bool(MIME_ENCODED_WORD_RE.search(s)))
    before_moji = text.apply(lambda s: bool(MOJIBAKE_HINT_RE.search(s)))
    before_repl = text.str.contains(REPLACEMENT_CHAR, regex=False)

    print("BEFORE, by source_dataset:")
    before_tbl = pd.DataFrame(
        {
            "html_tags": before_html,
            "mime_leftover": before_mime,
            "mojibake_hint": before_moji,
            "replacement_char": before_repl,
            "source_dataset": df["source_dataset"],
        }
    ).groupby("source_dataset").sum(numeric_only=True)
    print(before_tbl)
    print()

    cleaned_text = text.apply(clean_text_encoding_and_html)
    df = df.copy()
    df["text"] = cleaned_text

    after_html = df["text"].apply(lambda s: bool(HTML_TAG_RE.search(s)))
    after_mime = df["text"].apply(lambda s: bool(MIME_ENCODED_WORD_RE.search(s)))
    after_moji = df["text"].apply(lambda s: bool(MOJIBAKE_HINT_RE.search(s)))
    after_repl = df["text"].str.contains(REPLACEMENT_CHAR, regex=False)

    print("AFTER, by source_dataset:")
    after_tbl = pd.DataFrame(
        {
            "html_tags": after_html,
            "mime_leftover": after_mime,
            "mojibake_hint": after_moji,
            "replacement_char": after_repl,
            "source_dataset": df["source_dataset"],
        }
    ).groupby("source_dataset").sum(numeric_only=True)
    print(after_tbl)
    print()
    print(
        "Note: replacement_char (U+FFFD) rows are not fixable at this stage -- "
        "those bytes were already lost when the raw Nazario mbox files were "
        "decoded with errors='replace' during merge_datasets.py. Recovering "
        "them would require re-parsing the raw mbox files with corrected "
        "charset detection, which is outside this script's scope."
    )
    print()

    now_blank = df["text"].apply(is_blank_or_placeholder)
    if now_blank.any():
        print(f"Rows that became blank after encoding/HTML cleanup: {now_blank.sum()}")
        print(df[now_blank].groupby("source_dataset").size())
        df = df[~now_blank].reset_index(drop=True)
        print()

    return df


def detect_non_email_spamassassin(df: pd.DataFrame) -> pd.Series:
    sa = df[df["source_dataset"] == "spamassassin"]
    matched = pd.Series("", index=sa.index)
    for name, pattern in NON_EMAIL_PATTERNS.items():
        hit = sa["text"].apply(lambda s: bool(pattern.search(s)))
        matched.loc[hit] += name + ";"
    return matched[matched != ""]


def report_step4(df: pd.DataFrame) -> pd.Series:
    print("=" * 70)
    print("STEP 4: non-email content detection in SpamAssassin (DETECTION ONLY)")
    print("=" * 70)
    sa = df[df["source_dataset"] == "spamassassin"]
    flagged = detect_non_email_spamassassin(df)

    print(f"Flagged: {len(flagged)} / {len(sa)} SpamAssassin rows "
          f"({len(flagged) / len(sa):.1%})")
    print("By first matched pattern:")
    print(flagged.apply(lambda s: s.rstrip(";").split(";")[0]).value_counts())
    print()
    print("10 sample rows that WOULD be removed (not removed yet):")
    sample_idx = flagged.sample(min(10, len(flagged)), random_state=42).index
    for i in sample_idx:
        row = df.loc[i]
        preview = str(row["text"])[:220].replace("\n", " | ")
        print(f"--- idx {i} | matched: {flagged.loc[i]} ---")
        print(preview)
        print()
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply-non-email-filter",
        action="store_true",
        help="Apply step 4/5's SpamAssassin non-email content filter (requires review sign-off)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write data/processed/cleaned.csv. Without this flag, report-only.",
    )
    args = parser.parse_args()

    df, excluded_count = load_included()
    print(f"Loaded merged_raw.csv: {len(df) + excluded_count} total rows")
    print(f"Dropped included_in_training=False rows (SpamAssassin spam_excluded): "
          f"{excluded_count}")
    print(f"Rows entering cleaning pipeline: {len(df)}")
    print()

    df = report_step1(df)
    df = report_step2(df)
    df = report_step3(df)
    flagged = report_step4(df)

    if not args.write:
        print("--write not set: no output file written. Re-run with --write "
              "(and --apply-non-email-filter if approved) to save cleaned.csv.")
        return

    if args.apply_non_email_filter:
        print("=" * 70)
        print("STEP 5: applying approved non-email content filter")
        print("=" * 70)
        before = len(df)
        df = df.drop(index=flagged.index).reset_index(drop=True)
        print(f"Removed: {before - len(df)}")
        print()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("STEP 7: FINAL SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print()
    print("By source_dataset:")
    print(df["source_dataset"].value_counts())
    print()
    print("By label:")
    print(df["label"].value_counts())
    print()
    print("By source_dataset x label:")
    print(df.groupby(["source_dataset", "label"]).size())
    print()
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
