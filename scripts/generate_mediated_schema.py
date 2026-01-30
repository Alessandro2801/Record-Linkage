#!/usr/bin/env python3
"""
Script per la generazione dello schema mediato.
Convertito da: schema_alignment/mediated_schema.ipynb

Questo script:
1. Carica i dataset preprocessati (vehicles_processed.csv, used_cars_data_processed.csv)
2. Applica il mapping delle colonne per allineare gli schemi
3. Unisce i dataset in uno schema mediato
4. Applica normalizzazione universale per Record Linkage
5. Salva il dataset normalizzato in datasets/mediated_schema/
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


def get_project_root() -> Path:
    """Trova la directory root del progetto."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def normalizzazione_universale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza il dataset per Record Linkage, Dedupe e DITTO evitando errori su valori nulli.
    """
    df_norm = df.copy()
    
    colonne_testo = [
        'location', 'manufacturer', 'model', 'fuel_type', 
        'transmission', 'traction', 'body_type', 'main_color', 'cylinders'
    ]
    
    def clean_text_safe(x):
        """Pulisce il testo in modo sicuro."""
        if pd.isna(x) or str(x).lower() in ['nan', 'none', '']:
            return np.nan
        
        s = str(x).lower().strip()
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s if s != '' else np.nan

    # Applicazione della pulizia alle colonne testuali
    for col in colonne_testo:
        if col in df_norm.columns:            
            df_norm[col] = df_norm[col].apply(clean_text_safe)

    # Pulizia specifica per la colonna 'description'
    if 'description' in df_norm.columns:
        def clean_description_safe(x):
            if pd.isna(x) or str(x).lower() in ['nan', 'none', '']:
                return np.nan
            
            s = str(x).lower()
            s = re.sub(r'http\S+|www\S+|https\S+', '', s, flags=re.MULTILINE)
            s = re.sub(r'[^a-z0-9\s]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s if s != '' else np.nan

        df_norm['description'] = df_norm['description'].apply(clean_description_safe)

    # Normalizzazione numerica
    colonne_numeriche = ['year', 'price', 'mileage']
    for col in colonne_numeriche:
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')

    return df_norm


def generate_mediated_schema(
    vehicles_path: Path, 
    used_cars_path: Path, 
    output_path: Path
) -> pd.DataFrame:
    """Genera lo schema mediato dai due dataset preprocessati."""
    
    print(f"📂 Caricamento dataset...")
    df1 = pd.read_csv(vehicles_path)
    df2 = pd.read_csv(used_cars_path)
    print(f"   Dataset 1 (vehicles): {df1.shape}")
    print(f"   Dataset 2 (used_cars): {df2.shape}")
    
    # Rimozione colonne non descrittive delle entità
    df1_features_removed = {'url', 'region_url', 'image_url'}
    df2_features_removed = {'engine_type', 'listing_id', 'main_picture_url'}
    
    df1 = df1.drop(columns=[c for c in df1_features_removed if c in df1.columns], errors='ignore')
    df2 = df2.drop(columns=[c for c in df2_features_removed if c in df2.columns], errors='ignore')
    
    # Mapping delle colonne per lo schema mediato
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
    
    # Rinomina colonne e unione
    print("🔗 Allineamento schemi...")
    df1_mapped = df1.rename(columns=mapping_df1)
    df2_mapped = df2.rename(columns=mapping_df2)
    
    df_mediated = pd.concat([df1_mapped, df2_mapped], ignore_index=True)
    print(f"   Dimensioni schema mediato: {df_mediated.shape}")
    
    # Ripristino formato ID Sorgente 1
    df_mediated['id_source_vehicles'] = (
        pd.to_numeric(df_mediated['id_source_vehicles'], errors='coerce')
        .astype('Int64')
        .astype(str)
        .replace('<NA>', np.nan)
    )
    
    # Selezione colonne coinvolte nel matching
    colonne_in_match = [
        'id_source_vehicles', 'id_source_used_cars', 'location', 'price', 'year', 
        'manufacturer', 'model', 'cylinders', 'fuel_type', 'mileage', 'transmission', 
        'vin', 'traction', 'body_type', 'main_color', 'description', 'latitude',
        'longitude', 'pubblication_date'
    ]
    
    # Filtra solo le colonne esistenti
    colonne_presenti = [c for c in colonne_in_match if c in df_mediated.columns]
    df_mediated_cleaned = df_mediated[colonne_presenti]
    print(f"   Colonne selezionate: {len(colonne_presenti)}")
    
    # Normalizzazione universale
    print("🔄 Applicazione normalizzazione...")
    df_mediated_norm = normalizzazione_universale(df_mediated_cleaned)
    
    # Salvataggio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_mediated_norm.to_csv(output_path, index=False)
    print(f"   ✅ Schema mediato salvato in: {output_path}")
    
    return df_mediated_norm


def main():
    """Funzione principale."""
    print("=" * 60)
    print("🔗 Generazione Schema Mediato")
    print("=" * 60)
    
    project_root = get_project_root()
    vehicles_path = project_root / "datasets" / "processed" / "vehicles_processed.csv"
    used_cars_path = project_root / "datasets" / "processed" / "used_cars_data_processed.csv"
    output_path = project_root / "datasets" / "mediated_schema" / "mediated_schema_normalized.csv"
    
    # Verifica esistenza file di input
    if not vehicles_path.exists():
        print(f"❌ File non trovato: {vehicles_path}")
        print("   Esegui prima preprocess_vehicles.py")
        return 1
    
    if not used_cars_path.exists():
        print(f"❌ File non trovato: {used_cars_path}")
        print("   Esegui prima preprocess_used_cars.py")
        return 1
    
    generate_mediated_schema(vehicles_path, used_cars_path, output_path)
    print("\n✅ Generazione schema mediato completata!")
    return 0


if __name__ == "__main__":
    exit(main())
