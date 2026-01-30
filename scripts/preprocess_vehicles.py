#!/usr/bin/env python3
"""
Script per il preprocessing del dataset vehicles (Craigslist).
Converto da: eda_preprocessing/eda_dataset_vehicles.ipynb

Questo script:
1. Carica il dataset raw vehicles.csv
2. Rimuove colonne con >70% di valori mancanti
3. Rimuove righe con valori mancanti nelle colonne critiche
4. Salva il dataset processato in datasets/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root() -> Path:
    """Trova la directory root del progetto."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def profiling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola statistiche per ogni colonna del dataset."""
    missing_pct = (df.isnull().sum() / len(df)) * 100
    unique_pct = (df.nunique() / len(df)) * 100
    dtypes = df.dtypes.astype(str)
    
    summary_df = pd.DataFrame(
        [missing_pct, unique_pct, dtypes], 
        index=['missing_values_%', 'unique_values_%', 'type']
    )
    return summary_df


def preprocess_vehicles(input_path: Path, output_path: Path) -> None:
    """Preprocessa il dataset vehicles."""
    print(f"📂 Caricamento dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Shape originale: {df.shape}")
    
    # Calcolo statistiche
    df_stats = profiling_dataset(df)
    
    # 1. Rimozione colonne con >70% di valori mancanti
    soglia = 70.0
    cols_to_drop = df_stats.columns[df_stats.loc['missing_values_%'] > soglia].tolist()
    print(f"   Colonne eliminate (>{soglia}% missing): {cols_to_drop}")
    df_processed = df.drop(columns=cols_to_drop)
    
    # 2. Rimozione righe con valori mancanti nelle colonne critiche
    colonne_critiche = ['manufacturer', 'model', 'year', 'odometer']
    df_processed = df_processed.dropna(subset=colonne_critiche, how='any')
    print(f"   Shape dopo rimozione righe critiche: {df_processed.shape}")
    
    # 3. Salvataggio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"   ✅ Dataset salvato in: {output_path}")
    
    return df_processed


def main():
    """Funzione principale."""
    print("=" * 60)
    print("🚗 Preprocessing Dataset Vehicles (Craigslist)")
    print("=" * 60)
    
    project_root = get_project_root()
    input_path = project_root / "datasets" / "raw" / "vehicles.csv"
    output_path = project_root / "datasets" / "processed" / "vehicles_processed.csv"
    
    if not input_path.exists():
        print(f"❌ File non trovato: {input_path}")
        print("   Esegui prima setup_datasets.py per scaricare i dati.")
        return 1
    
    preprocess_vehicles(input_path, output_path)
    print("\n✅ Preprocessing completato!")
    return 0


if __name__ == "__main__":
    exit(main())
