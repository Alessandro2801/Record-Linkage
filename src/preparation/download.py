#!/usr/bin/env python3
"""
download.py — Scarica i dataset grezzi da Kaggle.

Richiede kagglehub e credenziali Kaggle configurate
(variabili d'ambiente KAGGLE_USERNAME e KAGGLE_KEY oppure ~/.kaggle/kaggle.json).

Usage:
    python -m src.preparation.download
"""

from pathlib import Path

import kagglehub
from tqdm import tqdm

from src.config import RAW_DIR, KAGGLE_DATASETS, ensure_dirs


def download_dataset(name: str, kaggle_id: str, dest_dir: Path) -> None:
    """Scarica un dataset Kaggle e copia i file CSV in dest_dir."""
    print(f"\n--- {name} ---")
    print(f"Download: {kaggle_id}")
    cache_path = Path(kagglehub.dataset_download(kaggle_id))
    print(f"Cache: {cache_path}")

    csv_files = list(cache_path.rglob("*.csv"))
    if not csv_files:
        print(f"  Nessun CSV trovato in {cache_path}")
        return

    for csv in csv_files:
        dest = dest_dir / csv.name
        # Symlink instead of copy: avoids EOVERFLOW on large files
        # and saves disk space.
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(csv.resolve())
        size_mb = csv.stat().st_size / 1e6
        print(f"  -> {dest.name} ({size_mb:.1f} MB) [symlink]")


def main():
    ensure_dirs()
    print(f"Directory di destinazione: {RAW_DIR}")

    for name, kaggle_id in tqdm(KAGGLE_DATASETS.items(),
                                 total=len(KAGGLE_DATASETS),
                                 desc="Download dataset",
                                 unit="ds"):
        download_dataset(name, kaggle_id, RAW_DIR)

    print("\nDownload completato.")


if __name__ == "__main__":
    main()
