#!/usr/bin/env python3
"""
Script per il preprocessing del dataset used_cars (US Used Cars).
Convertito da: eda_preprocessing/eda_dataset_cars.ipynb

Questo script:
1. Carica il dataset raw used_cars_data.csv
2. Genera nuovi ID univoci (rimuovendo sp_id)
3. Rimuove colonne con >70% di valori mancanti
4. Rimuove righe con valori mancanti nelle colonne critiche
5. Salva il dataset processato in datasets/processed/
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


def preprocess_used_cars(input_path: Path, output_path: Path) -> None:
    """Preprocessa il dataset used_cars."""
    print(f"📂 Caricamento dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Shape originale: {df.shape}")
    
    # 1. Rimuoviamo sp_id (non affidabile) e creiamo nuovi ID univoci
    if 'sp_id' in df.columns:
        df = df.drop(columns=['sp_id'])
    df = df.reset_index(drop=True)
    df['id'] = "S2_" + df.index.astype(str)
    print(f"   ID generati. Esempio: {df['id'].iloc[0]}")
    print(f"   ID univoci? {df['id'].is_unique}")
    
    # Calcolo statistiche
    df_stats = profiling_dataset(df)
    
    # 2. Rimozione colonne con >70% di valori mancanti
    soglia = 70.0
    cols_to_drop = df_stats.columns[df_stats.loc['missing_values_%'] > soglia].tolist()
    print(f"   Colonne eliminate (>{soglia}% missing): {cols_to_drop}")
    df_processed = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Rimozione righe con valori mancanti nelle colonne critiche
    colonne_critiche = ['make_name', 'model_name', 'year', 'mileage']
    df_processed = df_processed.dropna(subset=colonne_critiche, how='any')
    print(f"   Shape dopo rimozione righe critiche: {df_processed.shape}")
    
    # 4. Salvataggio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"   ✅ Dataset salvato in: {output_path}")
    
    return df_processed


def main():
    """Funzione principale."""
    print("=" * 60)
    print("🚗 Preprocessing Dataset Used Cars (US Used Cars)")
    print("=" * 60)
    
    project_root = get_project_root()
    input_path = project_root / "datasets" / "raw" / "used_cars_data.csv"
    output_path = project_root / "datasets" / "processed" / "used_cars_data_processed.csv"
    
    if not input_path.exists():
        print(f"❌ File non trovato: {input_path}")
        print("   Esegui prima setup_datasets.py per scaricare i dati.")
        return 1
    
    preprocess_used_cars(input_path, output_path)
    print("\n✅ Preprocessing completato!")
    return 0


if __name__ == "__main__":
    exit(main())
