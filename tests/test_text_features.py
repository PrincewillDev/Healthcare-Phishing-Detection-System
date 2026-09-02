import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))

from text_features import (
    avg_word_length,
    capitalized_word_count,
    exclamation_count,
    healthcare_term_count,
    urgency_score,
    word_count,
)


def test_urgency_score_detects_pressure_phrases():
    text = "Act now! Verify your account immediately or it will be suspended."
    assert urgency_score(text) > 0


def test_urgency_score_zero_for_neutral_text():
    text = "Hi Torrey, attached is the Q3 report you requested. Thanks."
    assert urgency_score(text) == 0.0


def test_urgency_score_empty_text():
    assert urgency_score("") == 0.0


def test_healthcare_term_count_detects_terms():
    text = "Please review the patient chart and confirm the lab results."
    assert healthcare_term_count(text) == 2


def test_healthcare_term_count_zero_when_absent():
    text = "Let's catch up over coffee next week."
    assert healthcare_term_count(text) == 0


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0


def test_avg_word_length():
    assert avg_word_length("aa bbbb") == 3.0
    assert avg_word_length("") == 0.0


def test_exclamation_count():
    assert exclamation_count("Wow!! Really?!") == 3
    assert exclamation_count("no marks here") == 0


def test_capitalized_word_count():
    text = "URGENT please CLICK HERE now"
    assert capitalized_word_count(text) == 3


def test_capitalized_word_count_ignores_single_letters():
    text = "I am going now"
    assert capitalized_word_count(text) == 0
