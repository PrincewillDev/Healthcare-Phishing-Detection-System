import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))

from url_features import (
    avg_domain_entropy,
    extract_row_url_features,
    extract_urls,
    has_at_symbol,
    has_ip_literal,
    has_url_shortener,
    suspicious_tld,
    url_count,
)


def test_extract_urls_http_and_https():
    text = "Visit http://example.com and https://secure.example.org/path now."
    urls = extract_urls(text)
    assert urls == ["http://example.com", "https://secure.example.org/path"]


def test_extract_urls_hxxp_defanged():
    text = "See hxxp://referral-secure-portal.net for details."
    urls = extract_urls(text)
    assert urls == ["hxxp://referral-secure-portal.net"]


def test_extract_urls_bare_www():
    text = "Go to www.example.com/login today."
    urls = extract_urls(text)
    assert urls == ["www.example.com/login"]


def test_extract_urls_strips_trailing_punctuation():
    text = "Check this out: http://example.com/path."
    urls = extract_urls(text)
    assert urls == ["http://example.com/path"]


def test_extract_urls_none_found():
    assert extract_urls("Hi Torrey, attached is the report. Thanks.") == []


def test_url_count():
    assert url_count(["http://a.com", "http://b.com"]) == 2
    assert url_count([]) == 0


def test_has_ip_literal_true():
    assert has_ip_literal(["http://192.168.1.5/login"]) is True


def test_has_ip_literal_false_for_domain():
    assert has_ip_literal(["http://example.com/login"]) is False


def test_has_url_shortener_true():
    assert has_url_shortener(["http://bit.ly/abc123"]) is True


def test_has_url_shortener_false():
    assert has_url_shortener(["http://example.com"]) is False


def test_suspicious_tld_true():
    assert suspicious_tld(["http://free-prize.tk/claim"]) is True


def test_suspicious_tld_false_for_com():
    assert suspicious_tld(["http://example.com"]) is False


def test_has_at_symbol_obfuscation():
    assert has_at_symbol(["http://trusted.com@evil.com/login"]) is True


def test_has_at_symbol_false_when_absent():
    assert has_at_symbol(["http://example.com/login"]) is False


def test_avg_domain_entropy_zero_for_no_urls():
    assert avg_domain_entropy([]) == 0.0


def test_avg_domain_entropy_excludes_ip_literals():
    assert avg_domain_entropy(["http://192.168.1.5/login"]) == 0.0


def test_avg_domain_entropy_positive_for_domain():
    assert avg_domain_entropy(["http://example.com"]) > 0.0


def test_extract_row_url_features_defaults_for_no_urls():
    row = extract_row_url_features("No links in this email at all.")
    assert row == {
        "url_count": 0,
        "has_ip_literal": False,
        "has_url_shortener": False,
        "avg_domain_entropy": 0.0,
        "suspicious_tld": False,
        "has_at_symbol": False,
    }
