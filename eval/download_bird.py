"""Download the BIRD mini-dev benchmark dataset.

Usage:
    python eval/download_bird.py

Downloads to data/bird_mini_dev/ (gitignored).
BIRD source: https://bird-bench.github.io/
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Official BIRD mini-dev download link (check bird-bench.github.io for updates)
BIRD_MINI_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"

DATA_DIR = Path(__file__).parent.parent / "data"
BIRD_DIR = DATA_DIR / "bird_mini_dev"
ZIP_PATH = DATA_DIR / "minidev.zip"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download url to dest, showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip_path into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, dest)


def find_questions_file(base: Path) -> Path | None:
    """Locate the BIRD questions JSON after extraction (path may vary by version)."""
    for pattern in [
        "mini_dev_sqlite.json",
        "minidev/mini_dev_sqlite.json",
        "**/mini_dev_sqlite.json",
    ]:
        hits = list(base.glob(pattern))
        if hits:
            return hits[0]
    return None


def main() -> None:
    if BIRD_DIR.exists() and find_questions_file(BIRD_DIR):
        print(f"BIRD mini-dev already present at {BIRD_DIR}")
        print("Delete the directory to re-download.")
        return

    print(f"Downloading BIRD mini-dev from:\n  {BIRD_MINI_DEV_URL}\n")
    download_file(BIRD_MINI_DEV_URL, ZIP_PATH)

    print(f"\nExtracting to {BIRD_DIR} …")
    extract_zip(ZIP_PATH, BIRD_DIR)

    # Clean up zip
    ZIP_PATH.unlink(missing_ok=True)

    questions = find_questions_file(BIRD_DIR)
    if questions:
        print(f"\n✓ Questions file: {questions}")
    else:
        print("\n⚠ Could not find mini_dev_sqlite.json — check the extracted directory.")
        print(f"  Extracted to: {BIRD_DIR}")
        sys.exit(1)

    db_dirs = list((BIRD_DIR / "databases").glob("*")) if (BIRD_DIR / "databases").exists() else []
    print(f"✓ Databases: {len(db_dirs)} found")
    print("\nReady. Run the eval harness with:")
    questions_rel = questions.relative_to(Path.cwd()) if questions.is_relative_to(Path.cwd()) else questions
    print(f"  groundedsql eval --db-dir {BIRD_DIR}/databases --questions {questions_rel} --n 50")


if __name__ == "__main__":
    main()
