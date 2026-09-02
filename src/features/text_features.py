"""Text and lexical feature extraction for the phishing classifier.

Every function here takes a single email's text and returns a scalar, so
each one is unit-testable in isolation without a DataFrame or file on disk.
No header-based features live in this module; see project notes for why
header features were dropped from the architecture.
"""

import re

# Pressure / urgency phrases commonly used in phishing to push a fast,
# unconsidered response. Matched as whole phrases, case-insensitive, with
# word boundaries so "act now" doesn't match inside "react nowhere".
URGENCY_PHRASES = [
    r"act now",
    r"act immediately",
    r"acting immediately",
    r"immediate action required",
    r"immediate action is required",
    r"requires immediate attention",
    r"verify immediately",
    r"verify your account",
    r"verify your identity",
    r"verify your information",
    r"confirm your identity",
    r"confirm your account",
    r"update your information immediately",
    r"account will be suspended",
    r"account has been suspended",
    r"account will be locked",
    r"account will be closed",
    r"account will be terminated",
    r"account will be deactivated",
    r"avoid suspension",
    r"restore access",
    r"unauthorized access",
    r"unusual (sign-in|login) activity",
    r"suspicious activity",
    r"security alert",
    r"urgent action required",
    r"urgent(ly)?",
    r"final notice",
    r"final warning",
    r"last chance",
    r"limited time",
    r"time[- ]sensitive",
    r"expires? (today|soon|within \d+ hours?)",
    r"before it'?s too late",
    r"failure to (comply|respond|act)",
    r"respond within \d+ hours?",
    r"click here immediately",
    r"click below immediately",
    r"do not ignore this",
    r"this is not a drill",
]

URGENCY_PATTERN = re.compile(
    r"\b(" + "|".join(URGENCY_PHRASES) + r")\b", re.IGNORECASE
)

# Healthcare-context terms. Presence of these matters most for distinguishing
# the healthcare_synthetic phishing/legitimate pairs (which are lexically
# close to each other) from generic phishing, not for separating phishing
# from legitimate on their own.
HEALTHCARE_TERMS = [
    r"EHR",
    r"EMR",
    r"electronic health record",
    r"electronic medical record",
    r"patient chart",
    r"patient record",
    r"patient portal",
    r"provider portal",
    r"insurance claim",
    r"prior authorization",
    r"claim number",
    r"HIPAA",
    r"PHI",
    r"protected health information",
    r"lab results?",
    r"laboratory results?",
    r"prescription",
    r"medication",
    r"referral",
    r"transfer of care",
    r"clinical notes?",
    r"diagnosis",
    r"treatment plan",
    r"discharge summary",
    r"biopsy",
    r"telehealth",
    r"appointment confirmation",
    r"health insurance",
    r"medical record",
]

HEALTHCARE_PATTERN = re.compile(
    r"\b(" + "|".join(HEALTHCARE_TERMS) + r")\b", re.IGNORECASE
)

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(str(text))


def urgency_score(text: str) -> float:
    """Urgency-language density: matched pressure phrases per 100 words.

    Normalized by length rather than a raw count so a short, terse phishing
    email isn't scored lower than a long legitimate email that happens to
    contain one incidental match.
    """
    words = _words(text)
    if not words:
        return 0.0
    matches = len(URGENCY_PATTERN.findall(str(text)))
    return matches / len(words) * 100


def healthcare_term_count(text: str) -> int:
    """Raw count of healthcare-context term matches in the text."""
    return len(HEALTHCARE_PATTERN.findall(str(text)))


def word_count(text: str) -> int:
    return len(_words(text))


def avg_word_length(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def exclamation_count(text: str) -> int:
    return str(text).count("!")


def capitalized_word_count(text: str) -> int:
    """Count of shouted words: alphabetic tokens of length >= 2 that are
    entirely uppercase (e.g. "URGENT", "CLICK HERE"). Length >= 2 avoids
    counting single-letter tokens like the pronoun "I".
    """
    return sum(
        1
        for w in str(text).split()
        if len(w) >= 2 and w.isalpha() and w.isupper()
    )


STRUCTURAL_FEATURE_FUNCS = {
    "word_count": word_count,
    "avg_word_length": avg_word_length,
    "exclamation_count": exclamation_count,
    "capitalized_word_count": capitalized_word_count,
}

LEXICON_FEATURE_FUNCS = {
    "urgency_score": urgency_score,
    "healthcare_term_count": healthcare_term_count,
}


def extract_row_features(text: str) -> dict:
    """All lexicon + structural features for one email, as a flat dict."""
    row = {}
    for name, func in LEXICON_FEATURE_FUNCS.items():
        row[name] = func(text)
    for name, func in STRUCTURAL_FEATURE_FUNCS.items():
        row[name] = func(text)
    return row
