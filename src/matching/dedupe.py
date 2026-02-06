#!/usr/bin/env python3
"""
dedupe.py — Pipeline di deduplicazione con la libreria Dedupe 3.0.

Workflow:
    1. Carica dataset unificato e blocking splits
    2. Prepara i dati nel formato dizionario richiesto da Dedupe
    3. Definisce campi (String, Price, Categorical con has_missing)
    4. Addestra il modello con le coppie etichettate dalla GT
    5. Salva il modello (.pickle)

Usage:
    python -m src.matching.dedupe --train --strategy B1
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

from src.config import (
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    RANDOM_SEED,
    N_JOBS,
    MP_POOL_SIZE,
    print_hw_info,
    blocking_split_path,
    model_path as get_model_path,
    ensure_dirs,
)

try:
    import dedupe
    DEDUPE_AVAILABLE = True
except ImportError:
    DEDUPE_AVAILABLE = False
    print("Warning: dedupe library not available. Install with: pip install dedupe")


# Definizione colonne per il modello Dedupe
COLS_STRING = ['location', 'manufacturer', 'model', 'cylinders',
               'year', 'latitude', 'longitude', 'body_type', 'main_color']
COLS_NUMERIC = ['price', 'mileage']
COLS_CATEGORICAL = ['fuel_type', 'traction', 'transmission']
ALL_COLS = COLS_STRING + COLS_NUMERIC + COLS_CATEGORICAL


def load_data(strategy: str) -> tuple:
    """Carica i dati necessari dai blocking splits."""
    print(f"Caricamento dati per strategia {strategy}...")
    df_unificato = pd.read_csv(
        MEDIATED_SCHEMA_NORMALIZED_PATH,
        dtype={'id_source_vehicles': 'object'}, low_memory=False,
    )
    gt_train = pd.read_csv(blocking_split_path(strategy, 'train'))
    gt_val = pd.read_csv(blocking_split_path(strategy, 'val'))
    gt_test = pd.read_csv(blocking_split_path(strategy, 'test'))

    # Rimuovi colonne non necessarie
    for col in ['vin', 'description']:
        if col in df_unificato.columns:
            df_unificato.drop(columns=[col], inplace=True)

    for gt in [gt_train, gt_val, gt_test]:
        for col in ['description_A', 'description_B']:
            if col in gt.columns:
                gt.drop(columns=[col], inplace=True)

    print(f"Dataset unificato: {df_unificato.shape}")
    print(f"Train: {len(gt_train)}, Val: {len(gt_val)}, Test: {len(gt_test)}")

    return df_unificato, gt_train, gt_val, gt_test


def to_clean_string(val):
    """Converte un valore in stringa pulita."""
    if pd.isnull(val):
        return ""
    if isinstance(val, (float, int)):
        if val == int(val):
            return str(int(val))
    return str(val).strip()


def prepare_data_for_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara i dati nel formato richiesto da Dedupe 3.0."""
    df = df.copy()

    df['id_unificato'] = (
        df['id_source_vehicles']
        .fillna(df['id_source_used_cars'])
    )
    df = df.set_index('id_unificato')

    cols_to_fix = ['year', 'latitude', 'longitude']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(to_clean_string)

    for col in COLS_STRING:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', 'NaN', '<NA>'], '')

    for col in COLS_NUMERIC:
        if col in df.columns:
            df[col] = df[col].replace({np.nan: None})

    for col in COLS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].replace({np.nan: None})

    return df


def create_dedupe_dict(df: pd.DataFrame, ids: set) -> dict:
    """Crea un dizionario di record per Dedupe con tutte le colonne richieste."""
    df_subset = df.loc[df.index.isin(ids)]

    records = {}
    for idx, row in df_subset.iterrows():
        record = {}
        for col in ALL_COLS:
            if col in df_subset.columns:
                val = row[col]
                if col in COLS_NUMERIC:
                    record[col] = None if pd.isna(val) else val
                elif col in COLS_CATEGORICAL:
                    record[col] = None if pd.isna(val) or val == 'nan' else val
                else:
                    if pd.isna(val) or val == 'nan' or str(val).strip() == '':
                        record[col] = None
                    else:
                        record[col] = str(val)
            else:
                record[col] = None
        records[str(idx)] = record

    return records


def get_dedupe_fields():
    """Definisce i campi per Dedupe 3.0."""
    if not DEDUPE_AVAILABLE:
        return []

    fields = [
        dedupe.variables.String("manufacturer", has_missing=True),
        dedupe.variables.String("model", has_missing=True),
        dedupe.variables.String("year", has_missing=True),
        dedupe.variables.String("location", has_missing=True),
        dedupe.variables.String("cylinders", has_missing=True),
        dedupe.variables.String("body_type", has_missing=True),
        dedupe.variables.String("main_color", has_missing=True),
        dedupe.variables.String("latitude", has_missing=True),
        dedupe.variables.String("longitude", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),
        dedupe.variables.Categorical(
            "transmission",
            categories=['other', 'automatic', 'manual', 'cvt', 'dual clutch'],
            has_missing=True,
        ),
        dedupe.variables.Categorical(
            "fuel_type",
            categories=['gas', 'other', 'diesel', 'hybrid', 'electric',
                        'biodiesel', 'flex fuel vehicle', 'compressed natural gas', 'propane'],
            has_missing=True,
        ),
        dedupe.variables.Categorical(
            "traction",
            categories=['rwd', '4wd', 'fwd', 'awd', '4x2'],
            has_missing=True,
        ),
    ]

    return fields


def prepare_training_pairs(gt_df: pd.DataFrame, data_dict: dict) -> tuple:
    """Prepara le coppie di training dalla ground truth."""
    matches = []
    distinct = []
    skipped = 0

    for _, row in gt_df.iterrows():
        id_a = str(row['id_A'])
        id_b = str(row['id_B'])

        if id_a in data_dict and id_b in data_dict:
            pair = (data_dict[id_a], data_dict[id_b])
            if row['label'] == 1:
                matches.append(pair)
            else:
                distinct.append(pair)
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Saltate {skipped} coppie (ID non trovati nel dizionario)")

    return matches, distinct


def train_dedupe(strategy: str, sample_size: int = 5000):
    """Addestra il modello Dedupe."""
    if not DEDUPE_AVAILABLE:
        print("Error: dedupe library is required. Install with: pip install dedupe")
        sys.exit(1)

    ensure_dirs()

    print("=" * 60)
    print(f"DEDUPE TRAINING PIPELINE — Strategia {strategy}")
    print("=" * 60)

    # 1. Carica dati
    df_unificato, gt_train, gt_val, gt_test = load_data(strategy)

    # 2. Prepara DataFrame
    print("\nPreparazione dati...")
    df_prepared = prepare_data_for_dedupe(df_unificato)
    print(f"Dataset preparato: {df_prepared.shape}")

    # 3. Identifica gli ID necessari
    ids_gt_train_val = (set(gt_train['id_A'].astype(str)) | set(gt_train['id_B'].astype(str)) |
                        set(gt_val['id_A'].astype(str)) | set(gt_val['id_B'].astype(str)))

    # Escludi ID del test set per purezza metodologica
    ids_test = set(gt_test['id_A'].astype(str)) | set(gt_test['id_B'].astype(str))
    df_no_test = df_prepared.loc[~df_prepared.index.isin(ids_test)]

    ids_sample = set(
        df_no_test.sample(n=min(sample_size, len(df_no_test)), random_state=RANDOM_SEED)
        .index.astype(str)
    )

    ids_finali = ids_gt_train_val | ids_sample
    print(f"ID totali per training: {len(ids_finali)}")

    # 4. Crea dizionario
    print("\nCreazione dizionario per Dedupe...")
    data_dict = create_dedupe_dict(df_prepared, ids_finali)
    print(f"Dizionario creato con {len(data_dict)} record")

    if data_dict:
        sample_record = list(data_dict.values())[0]
        print(f"Campi nel record: {list(sample_record.keys())}")
        for col in ALL_COLS:
            if col not in sample_record:
                print(f"  Campo mancante: {col}")

    # 5. Prepara training pairs
    print("\nPreparazione coppie di training...")
    matches, distinct = prepare_training_pairs(gt_train, data_dict)
    print(f"Coppie positive (match): {len(matches)}")
    print(f"Coppie negative (distinct): {len(distinct)}")

    if len(matches) == 0:
        print("Errore: Nessuna coppia positiva trovata!")
        return

    # 6. Inizializza Dedupe (usa tutti i core disponibili)
    n_cores = MP_POOL_SIZE
    print(f"\nInizializzazione Dedupe (cores: {n_cores}, in_memory=True)...")
    fields = get_dedupe_fields()
    deduper = dedupe.Dedupe(fields, num_cores=n_cores, in_memory=True)

    # 7. Passa gli esempi
    print("Marcatura coppie di training...")
    deduper.mark_pairs({'match': matches, 'distinct': distinct})

    # 8. Prepare training
    print("\nPreparazione addestramento (campionamento blocking)...")
    deduper.prepare_training(
        data=data_dict,
        sample_size=min(5000, len(data_dict)),
        blocked_proportion=0.2,
    )

    # 9. Train
    print("\nAddestramento modello...")
    deduper.train()

    print("\nTraining completato!")

    # 10. Salva modello
    _model_path = get_model_path('dedupe', strategy)
    _model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(_model_path, 'wb') as f:
        deduper.write_settings(f)

    print(f"Modello salvato in: {_model_path}")

    return deduper


def main():
    parser = argparse.ArgumentParser(description='Pipeline Dedupe per deduplicazione')
    parser.add_argument('--train', action='store_true', help='Esegui training del modello')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Numero di record per campionamento blocking')
    parser.add_argument('--strategy', type=str, required=True, choices=['B1', 'B2'],
                        help='Strategia di blocking (B1 o B2)')
    args = parser.parse_args()

    if args.train:
        print_hw_info()
        train_dedupe(args.strategy, args.sample_size)
    else:
        print("Usa --train per addestrare il modello Dedupe")
        print("Esempio: python -m src.matching.dedupe --train --strategy B1")


if __name__ == '__main__':
    main()
