#!/usr/bin/env python3
"""
download.py
Script per il download dei dataset da Kaggle.

Usage:
    python -m src.data.download [--dataset all|craigslist|used_cars]
"""

import argparse
import kagglehub


def download_craigslist_data() -> str:
    """
    Download del dataset Craigslist Cars/Trucks.
    
    Returns:
        Path alla directory contenente i file del dataset.
    """
    print("Downloading Craigslist Cars/Trucks dataset...")
    path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")
    print(f"Dataset scaricato in: {path}")
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
    return path


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
