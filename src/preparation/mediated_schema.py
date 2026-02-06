#!/usr/bin/env python3
"""
mediated_schema.py — Generazione dello schema mediato normalizzato.

Carica i due dataset processati (vehicles, used_cars), applica il mapping
delle colonne risultante da COMA schema matching, concatena e normalizza.

Usage:
    python -m src.preparation.mediated_schema
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from src.config import (
    VEHICLES_PROCESSED_PATH,
    USED_CARS_PROCESSED_PATH,
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    print_hw_info,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAPPING COLONNE (risultato di COMA schema matching)
# ═══════════════════════════════════════════════════════════════════════════════

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
    'region': 'location',
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

# Colonne non descrittive delle entità (da rimuovere)
DF1_FEATURES_REMOVED = {'url', 'region_url', 'image_url'}
DF2_FEATURES_REMOVED = {'engine_type', 'listing_id', 'main_picture_url'}

# Colonne finali per lo schema mediato
COLONNE_IN_MATCH = [
    'id_source_vehicles', 'id_source_used_cars', 'location', 'price', 'year',
    'manufacturer', 'model', 'cylinders', 'fuel_type', 'mileage', 'transmission',
    'vin', 'traction', 'body_type', 'main_color', 'description', 'latitude',
    'longitude', 'pubblication_date',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  NORMALIZZAZIONE
# ═══════════════════════════════════════════════════════════════════════════════

def normalizzazione_universale(df):
    """
    Normalizza il dataset per Record Linkage, Dedupe e DITTO evitando errori su valori nulli.
    """
    df_norm = df.copy()

    colonne_testo = [
        'location', 'manufacturer', 'model', 'fuel_type',
        'transmission', 'traction', 'body_type', 'main_color', 'cylinders'
    ]

    # Pulizia vettorizzata con pandas str methods (molto più veloce di apply)
    for col in tqdm(colonne_testo, desc="  Normalizzazione colonne testo", unit="col"):
        if col in df_norm.columns:
            s = df_norm[col].astype(str).str.lower().str.strip()
            # Marca i valori nulli/assenti
            null_mask = s.isin(['nan', 'none', '', '<na>']) | df_norm[col].isna()
            # Rimuovi caratteri non alfanumerici (tranne spazi)
            s = s.str.replace(r'[^a-z0-9\s]', '', regex=True)
            # Collassa spazi multipli
            s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
            # Stringhe vuote dopo pulizia -> NaN
            s = s.where(s != '', other=np.nan)
            s = s.where(~null_mask, other=np.nan)
            df_norm[col] = s

    # Pulizia description (vettorizzata)
    if 'description' in df_norm.columns:
        print("  Normalizzazione colonna description...")
        desc = df_norm['description'].astype(str).str.lower()
        null_mask = desc.isin(['nan', 'none', '', '<na>']) | df_norm['description'].isna()
        desc = desc.str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        desc = desc.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
        desc = desc.str.replace(r'\s+', ' ', regex=True).str.strip()
        desc = desc.where(desc != '', other=np.nan)
        desc = desc.where(~null_mask, other=np.nan)
        df_norm['description'] = desc

    # Normalizzazione numerica
    colonne_numeriche = ['year', 'price', 'mileage']
    for col in tqdm(colonne_numeriche, desc="  Normalizzazione colonne numeriche", unit="col"):
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')

    return df_norm


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()
    print_hw_info()

    # 1. Carica dataset processati (in parallelo con ThreadPool)
    from concurrent.futures import ThreadPoolExecutor
    print(f"Caricamento dataset processati...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(pd.read_csv, VEHICLES_PROCESSED_PATH, low_memory=False)
        fut2 = pool.submit(pd.read_csv, USED_CARS_PROCESSED_PATH, low_memory=False)
        df1 = fut1.result()
        df2 = fut2.result()

    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")

    # 2. Rimuovi colonne non descrittive
    df1 = df1.drop(columns=[c for c in DF1_FEATURES_REMOVED if c in df1.columns])
    df2 = df2.drop(columns=[c for c in DF2_FEATURES_REMOVED if c in df2.columns])

    # 3. Rinomina colonne
    df1_mapped = df1.rename(columns=mapping_df1)
    df2_mapped = df2.rename(columns=mapping_df2)

    # 4. Concatena
    df_mediated = pd.concat([df1_mapped, df2_mapped], ignore_index=True)
    print(f"Dimensioni schema mediato: {df_mediated.shape}")

    # 5. Ripristina ID della Sorgente 1
    df_mediated['id_source_vehicles'] = (
        pd.to_numeric(df_mediated['id_source_vehicles'], errors='coerce')
        .astype('Int64')
        .astype(str)
        .replace('<NA>', np.nan)
    )

    # 6. Seleziona solo colonne coinvolte nel matching
    df_mediated_cleaned = df_mediated[[c for c in COLONNE_IN_MATCH if c in df_mediated.columns]]

    # 7. Normalizzazione universale
    df_mediated_norm = normalizzazione_universale(df_mediated_cleaned)

    # 8. Normalizzazione valori specifici
    transmission_map = {'a': 'automatic', 'm': 'manual'}
    fuel_type_map = {'gasoline': 'gas'}

    df_mediated_norm['transmission'] = df_mediated_norm['transmission'].str.lower().replace(transmission_map)
    df_mediated_norm['fuel_type'] = df_mediated_norm['fuel_type'].str.lower().replace(fuel_type_map)

    print("Normalizzazione completata.")

    # 9. Salvataggio
    MEDIATED_SCHEMA_NORMALIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_mediated_norm.to_csv(MEDIATED_SCHEMA_NORMALIZED_PATH, index=False)
    print(f"Schema mediato normalizzato salvato in: {MEDIATED_SCHEMA_NORMALIZED_PATH}")
    print(f"Shape finale: {df_mediated_norm.shape}")


if __name__ == '__main__':
    main()
