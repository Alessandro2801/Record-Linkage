#!/usr/bin/env python3
"""
mediated_schema.py
Script per la creazione dello schema mediato tra i due dataset sorgente.

Usage:
    python mediated_schema.py [--output PATH]
"""

import argparse
import os
import pandas as pd
import numpy as np


def load_datasets(base_path: str, sample_size: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carica i dataset da data/raw/.
    
    Args:
        base_path: Path base del progetto
        sample_size: Se specificato, campiona N righe per dataset (utile per file grandi)
    """
    # Prima prova raw, poi processed (per retrocompatibilità)
    df1_path_raw = os.path.join(base_path, 'data/raw/craigslist_vehicles.csv')
    df1_path_processed = os.path.join(base_path, 'data/processed/vehicles_processed.csv')
    
    df2_path_raw = os.path.join(base_path, 'data/raw/used_cars_used_cars_data.csv')
    df2_path_processed = os.path.join(base_path, 'data/processed/used_cars_data_processed.csv')
    
    # Colonne rilevanti per il matching (riduce memoria)
    usecols_df1 = ['id', 'VIN', 'price', 'year', 'manufacturer', 'model', 'fuel', 
                   'odometer', 'transmission', 'drive', 'type', 'paint_color',
                   'lat', 'long', 'posting_date', 'region', 'description']
    
    usecols_df2 = ['vin', 'price', 'year', 'make_name', 'model_name', 'fuel_type',
                   'mileage', 'transmission', 'wheel_system', 'body_type', 'listing_color',
                   'latitude', 'longitude', 'listed_date', 'city', 'description']
    
    # Dataset 1: Craigslist
    df1_path = df1_path_raw if os.path.exists(df1_path_raw) else df1_path_processed
    print(f"Caricamento {df1_path}...")
    
    # Leggi solo colonne esistenti
    df1_temp = pd.read_csv(df1_path, nrows=0)
    usecols_df1_exist = [c for c in usecols_df1 if c in df1_temp.columns]
    
    if sample_size:
        # Sampling per file grandi
        df1 = pd.read_csv(df1_path, usecols=usecols_df1_exist, nrows=sample_size)
    else:
        df1 = pd.read_csv(df1_path, usecols=usecols_df1_exist)
    
    # Dataset 2: Used Cars (file grande, usa chunking)
    df2_path = df2_path_raw if os.path.exists(df2_path_raw) else df2_path_processed
    print(f"Caricamento {df2_path}...")
    
    # Leggi solo colonne esistenti
    df2_temp = pd.read_csv(df2_path, nrows=0)
    usecols_df2_exist = [c for c in usecols_df2 if c in df2_temp.columns]
    
    if sample_size:
        df2 = pd.read_csv(df2_path, usecols=usecols_df2_exist, nrows=sample_size, low_memory=False)
    else:
        # Carica a chunks per file molto grandi
        print("   Caricamento in chunks (file grande)...")
        chunks = []
        for chunk in pd.read_csv(df2_path, usecols=usecols_df2_exist, chunksize=100000, low_memory=False):
            chunks.append(chunk)
            print(f"   Caricati {len(chunks) * 100000:,} record...", end='\r')
        df2 = pd.concat(chunks, ignore_index=True)
        print()
    
    print(f"Dataset 1 (vehicles): {df1.shape}")
    print(f"Dataset 2 (used_cars): {df2.shape}")
    
    return df1, df2


def remove_non_descriptive_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rimuove colonne non descrittive delle entità."""
    df1_features_to_remove = {'url', 'region_url', 'image_url'}
    df2_features_to_remove = {'engine_type', 'listing_id', 'main_picture_url'}
    
    df1_cols_to_drop = [c for c in df1_features_to_remove if c in df1.columns]
    df2_cols_to_drop = [c for c in df2_features_to_remove if c in df2.columns]
    
    df1 = df1.drop(columns=df1_cols_to_drop)
    df2 = df2.drop(columns=df2_cols_to_drop)
    
    print(f"Rimosse {len(df1_cols_to_drop)} colonne da df1, {len(df2_cols_to_drop)} colonne da df2")
    
    return df1, df2


def apply_column_mapping(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applica il mapping delle colonne per allineare gli schemi."""
    mapping_df1 = {
        'VIN': 'vin',
        'id': 'id_source_vehicles',
        'type': 'body_type',
        'paint_color': 'main_color',
        'long': 'longitude',
        'lat': 'latitude',
        'posting_date': 'pubblication_date',
        'drive': 'traction',
        'title_status': 'status',
        'odometer': 'mileage',
        'fuel': 'fuel_type',
        'region': 'location'
    }
    
    mapping_df2 = {
        'make_name': 'manufacturer',
        'model_name': 'model',
        'listing_color': 'main_color',
        'id': 'id_source_used_cars',
        'engine_cylinders': 'cylinders',
        'listed_date': 'pubblication_date',
        'wheel_system': 'traction',
        'theft_title': 'theft_status',
        'city': 'location',
    }
    
    df1_mapped = df1.rename(columns=mapping_df1)
    df2_mapped = df2.rename(columns=mapping_df2)
    
    return df1_mapped, df2_mapped


def fix_id_format(df: pd.DataFrame) -> pd.DataFrame:
    """Corregge il formato degli ID per la sorgente vehicles."""
    if 'id_source_vehicles' in df.columns:
        df['id_source_vehicles'] = (
            pd.to_numeric(df['id_source_vehicles'], errors='coerce')
            .astype('Int64')
            .astype(str)
            .replace('<NA>', np.nan)
        )
    return df


def create_mediated_schema(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Crea lo schema mediato unendo i due dataset."""
    df_mediated = pd.concat([df1, df2], ignore_index=True)
    print(f"Schema mediato: {df_mediated.shape}")
    return df_mediated


def main():
    parser = argparse.ArgumentParser(description='Crea lo schema mediato dai dataset sorgente')
    parser.add_argument('--base-path', type=str, default='.', 
                        help='Path base del progetto Record-Linkage')
    parser.add_argument('--output', type=str, default=None,
                        help='Path output per lo schema mediato')
    parser.add_argument('--sample', type=int, default=None,
                        help='Numero di record da campionare per dataset (per test veloci)')
    args = parser.parse_args()
    
    base_path = args.base_path
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(base_path, 'data/mediated_schema/mediated_schema.csv')
    
    # Path anche per versione normalizzata
    normalized_path = output_path.replace('.csv', '_normalized.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 60)
    print("CREAZIONE SCHEMA MEDIATO")
    print("=" * 60)
    
    df1, df2 = load_datasets(base_path, sample_size=args.sample)
    df1, df2 = remove_non_descriptive_columns(df1, df2)
    df1_mapped, df2_mapped = apply_column_mapping(df1, df2)
    df_mediated = create_mediated_schema(df1_mapped, df2_mapped)
    df_mediated = fix_id_format(df_mediated)
    
    print(f"\nSalvataggio in {output_path}...")
    df_mediated.to_csv(output_path, index=False)
    
    # Salva anche versione normalizzata (copia per ora)
    print(f"Salvataggio versione normalizzata in {normalized_path}...")
    df_mediated.to_csv(normalized_path, index=False)
    
    print(f"\n✅ Schema mediato salvato con {len(df_mediated):,} record")
    print(f"   Colonne: {list(df_mediated.columns)}")


if __name__ == '__main__':
    main()
