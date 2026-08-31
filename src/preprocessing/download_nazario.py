"""Download the Nazario Phishing Corpus to data/raw/nazario/.

Idempotent: skips the download if data/raw/nazario/ already exists and has files.

Primary source: mirrors monkey.org/~jose/phishing/ via wget.
Fallback: a CSV mirror of the Nazario corpus on GitHub, saved to
data/raw/nazario_fallback.csv, used only if wget itself is not installed.
If wget is installed but the download fails for any other reason (network
error, host down, etc.), this stops and reports the exact error instead of
silently falling back.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
NAZARIO_DIR = RAW_DIR / "nazario"
FALLBACK_PATH = RAW_DIR / "nazario_fallback.csv"

PRIMARY_URL = "https://monkey.org/~jose/phishing/"
FALLBACK_URL = (
    "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nazario.csv"
)


def existing_file_count() -> int:
    return sum(1 for p in NAZARIO_DIR.rglob("*") if p.is_file())


def run_wget() -> None:
    """Run the primary wget mirror. Exits the process on any failure other than
    wget being missing (that case is handled by the caller before this runs)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-r",
        "--no-parent",
        "--level=1",
        "--reject",
        "index.html*",
        "--wait=1",
        PRIMARY_URL,
        "-P",
        str(RAW_DIR),
    ]
    print(f"[nazario] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[nazario] wget invocation failed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    if result.returncode != 0:
        print(f"[nazario] wget exited with code {result.returncode}")
        print(result.stderr.strip()[-2000:])
        sys.exit(1)

    # wget -r mirrors into a host-named directory (monkey.org/~jose/phishing/...);
    # normalize that into data/raw/nazario/. --reject only matches the URL path,
    # not query-string sort variants (index.html?C=N;O=D etc.), so those and
    # robots.txt are filtered out here rather than at the wget layer.
    mirrored_root = RAW_DIR / "monkey.org"
    if mirrored_root.exists():
        NAZARIO_DIR.mkdir(parents=True, exist_ok=True)
        for item in mirrored_root.rglob("*"):
            if not item.is_file():
                continue
            if item.name.startswith("index.html") or item.name == "robots.txt":
                continue
            dest = NAZARIO_DIR / item.name
            shutil.move(str(item), str(dest))
        shutil.rmtree(mirrored_root)

    if not (NAZARIO_DIR.exists() and existing_file_count() > 0):
        print("[nazario] wget exited 0 but no files ended up in data/raw/nazario/.")
        sys.exit(1)


def count_emails() -> int:
    """Best-effort mbox parse of downloaded files to report email volume."""
    import mailbox

    total = 0
    for path in sorted(NAZARIO_DIR.rglob("*")):
        if not path.is_file():
            continue
        try:
            box = mailbox.mbox(str(path))
            count = len(box)
        except Exception as exc:
            print(f"[nazario]   {path.name}: not parseable as mbox ({exc})")
            continue
        print(f"[nazario]   {path.name}: {count} messages")
        total += count
    return total


def download_fallback() -> None:
    print(f"[nazario] Falling back to {FALLBACK_URL}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(FALLBACK_URL, timeout=60)
    response.raise_for_status()
    FALLBACK_PATH.write_bytes(response.content)
    print(f"[nazario] Saved fallback CSV to {FALLBACK_PATH} ({len(response.content)} bytes)")
    print("[nazario] Path used: fallback (GitHub CSV mirror)")


def main() -> None:
    if NAZARIO_DIR.exists() and existing_file_count() > 0:
        print(f"[nazario] {NAZARIO_DIR} already exists, skipping download.")
        print(f"[nazario] Existing file count: {existing_file_count()}")
        return

    if FALLBACK_PATH.exists():
        print(f"[nazario] {FALLBACK_PATH} already exists, skipping download.")
        print(f"[nazario] Existing fallback file present at {FALLBACK_PATH}")
        return

    if shutil.which("wget") is None:
        print("[nazario] wget not found on PATH.")
        download_fallback()
        return

    run_wget()
    file_count = existing_file_count()
    print(f"[nazario] Path used: primary (wget mirror of {PRIMARY_URL})")
    print(f"[nazario] File count: {file_count}")

    print("[nazario] Parsing downloaded files as mbox to count emails...")
    email_count = count_emails()
    print(f"[nazario] Total email count (mbox-parseable files only): {email_count}")


if __name__ == "__main__":
    main()
