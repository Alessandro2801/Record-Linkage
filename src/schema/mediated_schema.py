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


def load_datasets(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica i dataset preprocessati."""
    df1_path = os.path.join(base_path, 'data/processed/vehicles_processed.csv')
    df2_path = os.path.join(base_path, 'data/processed/used_cars_data_processed.csv')
    
    print(f"Caricamento {df1_path}...")
    df1 = pd.read_csv(df1_path)
    
    print(f"Caricamento {df2_path}...")
    df2 = pd.read_csv(df2_path, low_memory=False)
    
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
    args = parser.parse_args()
    
    base_path = args.base_path
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(base_path, 'data/mediated_schema/mediated_schema.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 60)
    print("CREAZIONE SCHEMA MEDIATO")
    print("=" * 60)
    
    df1, df2 = load_datasets(base_path)
    df1, df2 = remove_non_descriptive_columns(df1, df2)
    df1_mapped, df2_mapped = apply_column_mapping(df1, df2)
    df_mediated = create_mediated_schema(df1_mapped, df2_mapped)
    df_mediated = fix_id_format(df_mediated)
    
    print(f"\nSalvataggio in {output_path}...")
    df_mediated.to_csv(output_path, index=False)
    
    print(f"\n✅ Schema mediato salvato con {len(df_mediated)} record")
    print(f"   Colonne: {list(df_mediated.columns)}")


if __name__ == '__main__':
    main()
