#!/usr/bin/env python3
"""
download.py
Script per il download dei dataset da Kaggle e copia in data/raw/.

Usage:
    python -m src.download_data.download [--dataset all|craigslist|used_cars]
"""

import argparse
import shutil
from pathlib import Path
import kagglehub

# Path del progetto
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_craigslist_data() -> str:
    """
    Download del dataset Craigslist Cars/Trucks.
    
    Returns:
        Path alla directory contenente i file del dataset.
    """
    print("Downloading Craigslist Cars/Trucks dataset...")
    path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")
    print(f"Dataset scaricato in: {path}")
    
    # Copia in data/raw/
    _copy_to_raw(path, "craigslist")
    
    return path


def download_used_cars_data() -> str:
    """
    Download del dataset US Used Cars.
    
    Returns:
        Path alla directory contenente i file del dataset.
    """
    print("Downloading US Used Cars dataset...")
    path = kagglehub.dataset_download("ananaymital/us-used-cars-dataset")
    print(f"Dataset scaricato in: {path}")
    
    # Copia in data/raw/
    _copy_to_raw(path, "used_cars")
    
    return path


def _copy_to_raw(source_path: str, dataset_name: str):
    """Copia i file CSV nella cartella data/raw/."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    source = Path(source_path)
    csv_files = list(source.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  Nessun CSV trovato in {source_path}")
        return
    
    print(f"📁 Copia file in {RAW_DATA_DIR}...")
    for csv_file in csv_files:
        dest_name = f"{dataset_name}_{csv_file.name}"
        dest_path = RAW_DATA_DIR / dest_name
        shutil.copy2(csv_file, dest_path)
        print(f"   ✅ {csv_file.name} -> {dest_name}")


def download_all_datasets() -> dict:
    """
    Download di tutti i dataset necessari.
    
    Returns:
        Dizionario con i path dei dataset scaricati.
    """
    paths = {}
    
    print("=" * 60)
    print("DOWNLOAD DATASETS")
    print("=" * 60)
    
    paths["craigslist"] = download_craigslist_data()
    paths["used_cars"] = download_used_cars_data()
    
    print("\n✅ Tutti i dataset sono stati scaricati!")
    for name, path in paths.items():
        print(f"   {name}: {path}")
    
    return paths


def main():
    parser = argparse.ArgumentParser(description='Download datasets da Kaggle')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'craigslist', 'used_cars'],
                        help='Dataset da scaricare (default: all)')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        download_all_datasets()
    elif args.dataset == 'craigslist':
        download_craigslist_data()
    elif args.dataset == 'used_cars':
        download_used_cars_data()


if __name__ == '__main__':
    main()
