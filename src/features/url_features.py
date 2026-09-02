"""URL and domain feature extraction for the phishing classifier.

Every function here takes a single email's text (or a list of URLs already
extracted from it) and returns a scalar/list, so each one is unit-testable
in isolation without a DataFrame or file on disk.

domain_age is intentionally not implemented: computing it would require a
live or cached WHOIS lookup, which src/preprocessing and this module treat
as out of scope per the project's hard constraint against external API
calls in the data/feature pipeline (see .claude/rules/data-handling.md).

tldextract is configured with suffix_list_urls=() so it only ever uses the
public suffix list snapshot bundled with the package and never makes a
network call, keeping this pipeline fully offline.
"""

import math
import re
from urllib.parse import urlparse

import tldextract

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# Matches http(s):// URLs, the "hxxp(s)://" defanged convention (used
# throughout data/synthetic's healthcare emails instead of a live scheme),
# and bare "www."-prefixed strings. Bare domain mentions with neither a
# scheme nor a "www." prefix (e.g. plain "example.com" in prose) are not
# counted as URLs -- they're indistinguishable from ordinary text without
# much higher false-positive risk.
URL_RE = re.compile(
    r"\b(?:(?:https?|hxxps?)://[^\s<>\"'\)\]]+|www\.[^\s<>\"'\)\]]+)",
    re.IGNORECASE,
)

# Trailing punctuation that regularly gets swept up by the regex above
# because it ends a sentence rather than the URL itself.
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?\)\]\"']+$")

IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# Known URL-shortener domains. Compiled from the services most commonly
# cited in phishing/URL-analysis literature and abuse blocklists (e.g.
# CheckShortURL's and Wikipedia's lists of URL shortening services, plus
# the shorteners most frequently seen in Nazario/PhishTank-style phishing
# corpora). Not exhaustive -- shorteners are created faster than any
# static list can track -- but covers the services that show up
# repeatedly in practice.
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "cutt.ly", "rebrand.ly", "shorte.st", "tiny.cc",
    "s.id", "rb.gy", "v.gd", "soo.gd", "x.co", "po.st", "qr.ae", "tr.im",
    "snip.ly", "lnkd.in", "git.io", "shorturl.at", "clck.ru", "u.to",
}

# TLDs repeatedly flagged as most-abused for spam/phishing in industry
# threat research: Spamhaus's recurring "World's Most Abused TLDs" reports
# and the Blue Coat/Symantec "shady top-level domains" studies both
# consistently name the five Freenom free-registration TLDs (.tk, .ml,
# .ga, .cf, .gq) as the top offenders, since Freenom historically allowed
# registering them for free with minimal verification. The remainder are
# low-cost gTLDs (.xyz, .top, .club, .work, .click, .country, .men, .loan)
# that show up repeatedly in the same reports and in SURBL's abused-TLD
# statistics due to low registration cost and weak abuse enforcement.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",
    "xyz", "top", "club", "work", "click", "country", "men", "loan",
}


def extract_urls(text: str) -> list[str]:
    """All URL-like substrings found in text, trailing punctuation stripped."""
    matches = URL_RE.findall(str(text))
    return [_TRAILING_PUNCT_RE.sub("", m) for m in matches if m]


def _normalized_for_parsing(url: str) -> str:
    """urlparse needs a real "://" to populate netloc; www.-only strings
    and the hxxp(s) convention both need light normalization first."""
    if url.lower().startswith(("hxxp://", "hxxps://")):
        url = "http" + url[4:]
    if "://" not in url:
        url = "http://" + url
    return url


def _hostname(url: str) -> str:
    try:
        return urlparse(_normalized_for_parsing(url)).hostname or ""
    except ValueError:
        return ""


def url_count(urls: list[str]) -> int:
    return len(urls)


def has_ip_literal(urls: list[str]) -> bool:
    return any(IPV4_RE.match(_hostname(u)) for u in urls)


def has_url_shortener(urls: list[str]) -> bool:
    for u in urls:
        ext = _TLD_EXTRACTOR(_hostname(u))
        registered_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        if registered_domain.lower() in URL_SHORTENERS:
            return True
    return False


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def avg_domain_entropy(urls: list[str]) -> float:
    """Mean Shannon entropy of the registrable domain's second-level label
    (e.g. "evil23xk9" from "evil23xk9.tk"), averaged over URLs that resolve
    to an actual domain name. IP-literal URLs are excluded here since digit/
    dot entropy doesn't indicate randomly generated domain names -- see
    has_ip_literal for that signal instead.
    """
    entropies = []
    for u in urls:
        host = _hostname(u)
        if IPV4_RE.match(host):
            continue
        ext = _TLD_EXTRACTOR(host)
        if not ext.domain:
            continue
        entropies.append(_shannon_entropy(ext.domain))
    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def suspicious_tld(urls: list[str]) -> bool:
    for u in urls:
        ext = _TLD_EXTRACTOR(_hostname(u))
        if ext.suffix.lower() in SUSPICIOUS_TLDS:
            return True
    return False


def has_at_symbol(urls: list[str]) -> bool:
    """Detects the classic http://trusted.com@evil.com/ obfuscation, where
    everything before the @ is discarded as userinfo by real URL parsers."""
    return any("@" in u.split("://", 1)[-1].split("/", 1)[0] for u in urls)


URL_FEATURE_FUNCS = {
    "url_count": url_count,
    "has_ip_literal": has_ip_literal,
    "has_url_shortener": has_url_shortener,
    "avg_domain_entropy": avg_domain_entropy,
    "suspicious_tld": suspicious_tld,
    "has_at_symbol": has_at_symbol,
}


def extract_row_url_features(text: str) -> dict:
    """All URL/domain features for one email, as a flat dict. Emails with
    zero URLs get url_count=0 and every boolean/entropy feature at its
    default (False / 0.0) rather than null.
    """
    urls = extract_urls(text)
    row = {}
    for name, func in URL_FEATURE_FUNCS.items():
        row[name] = func(urls)
    return row
