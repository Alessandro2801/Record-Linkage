#!/usr/bin/env python3
"""
dedupe.py
Pipeline per deduplicazione usando la libreria Dedupe 3.0.

Usage:
    python -m src.pipelines.dedupe [--base-path PATH] [--train]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Prova ad importare dedupe, ma gestisci il caso in cui non sia installato
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


def load_data(base_path: str) -> tuple:
    """
    Carica i dati necessari per la deduplicazione.
    
    Args:
        base_path: Path base del progetto
        
    Returns:
        Tuple di (df_unificato, gt_train, gt_val, gt_test)
    """
    # Paths
    mediated_path = os.path.join(base_path, 'data/mediated_schema/mediated_schema_normalized.csv')
    train_path = os.path.join(base_path, 'data/ground_truth/GT_train/train.csv')
    val_path = os.path.join(base_path, 'data/ground_truth/GT_train/val.csv')
    test_path = os.path.join(base_path, 'data/ground_truth/GT_train/test.csv')
    
    print("Caricamento dati...")
    df_unificato = pd.read_csv(mediated_path, dtype={'id_source_vehicles': 'object'}, low_memory=False)
    gt_train = pd.read_csv(train_path)
    gt_val = pd.read_csv(val_path)
    gt_test = pd.read_csv(test_path)
    
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
    """
    Prepara i dati nel formato richiesto da Dedupe 3.0.
    
    Args:
        df: DataFrame con i dati
        
    Returns:
        DataFrame preparato con index impostato
    """
    df = df.copy()
    
    # Crea ID unificato
    df['id_unificato'] = (
        df['id_source_vehicles']
        .fillna(df['id_source_used_cars'])
    )
    df = df.set_index('id_unificato')
    
    # Colonne da convertire in stringa pulita
    cols_to_fix = ['year', 'latitude', 'longitude']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].apply(to_clean_string)
    
    # Pulizia colonne testuali - convertiamo in stringhe, NaN diventa stringa vuota
    for col in COLS_STRING:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'None', 'NaN', '<NA>'], '')
    
    # Pulizia colonne numeriche - NaN diventa None
    for col in COLS_NUMERIC:
        if col in df.columns:
            df[col] = df[col].replace({np.nan: None})
    
    # Pulizia colonne categoriche - NaN diventa None
    for col in COLS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].replace({np.nan: None})
    
    return df


def create_dedupe_dict(df: pd.DataFrame, ids: set) -> dict:
    """
    Crea un dizionario di record per Dedupe con TUTTE le colonne richieste dal modello.
    
    Args:
        df: DataFrame con indici impostati
        ids: Set di ID da includere
        
    Returns:
        Dizionario {id: {field: value, ...}}
    """
    # Filtra per gli ID richiesti
    df_subset = df.loc[df.index.isin(ids)]
    
    # Seleziona TUTTE le colonne necessarie per il modello
    # Questo è il FIX: il notebook originale non includeva tutte le colonne
    records = {}
    for idx, row in df_subset.iterrows():
        record = {}
        for col in ALL_COLS:
            if col in df_subset.columns:
                val = row[col]
                # Gestisci None/NaN per colonne numeriche
                if col in COLS_NUMERIC:
                    record[col] = None if pd.isna(val) else val
                # Gestisci None/NaN per colonne categoriche
                elif col in COLS_CATEGORICAL:
                    record[col] = None if pd.isna(val) or val == 'nan' else val
                # Colonne stringa - IMPORTANTE: usa None invece di "" per has_missing=True
                else:
                    if pd.isna(val) or val == 'nan' or str(val).strip() == '':
                        record[col] = None  # Dedupe richiede None, non ""
                    else:
                        record[col] = str(val)
            else:
                # Se la colonna non esiste, metti valore di default
                record[col] = None
        records[str(idx)] = record
    
    return records


def get_dedupe_fields():
    """
    Definisce i campi per Dedupe 3.0.
    
    Returns:
        Lista di variabili Dedupe
    """
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
            has_missing=True
        ),
        dedupe.variables.Categorical(
            "fuel_type", 
            categories=['gas', 'other', 'diesel', 'hybrid', 'electric',
                       'biodiesel', 'flex fuel vehicle', 'compressed natural gas', 'propane'],
            has_missing=True
        ),
        dedupe.variables.Categorical(
            "traction", 
            categories=['rwd', '4wd', 'fwd', 'awd', '4x2'], 
            has_missing=True
        ),
    ]
    
    return fields


def prepare_training_pairs(gt_df: pd.DataFrame, data_dict: dict) -> tuple:
    """
    Prepara le coppie di training dalla ground truth.
    
    Args:
        gt_df: DataFrame ground truth con colonne id_A, id_B, label
        data_dict: Dizionario di record
        
    Returns:
        Tuple di (matches, distinct)
    """
    matches = []
    distinct = []
    skipped = 0
    
    for _, row in gt_df.iterrows():
        id_a = str(row['id_A'])
        id_b = str(row['id_B'])
        
        # Verifica che entrambi gli ID siano nel dizionario
        if id_a in data_dict and id_b in data_dict:
            pair = (data_dict[id_a], data_dict[id_b])
            if row['label'] == 1:
                matches.append(pair)
            else:
                distinct.append(pair)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"⚠️  Saltate {skipped} coppie (ID non trovati nel dizionario)")
    
    return matches, distinct


def train_dedupe(base_path: str, sample_size: int = 5000):
    """
    Addestra il modello Dedupe.
    
    Args:
        base_path: Path base del progetto
        sample_size: Numero di record casuali per le statistiche di blocking
    """
    if not DEDUPE_AVAILABLE:
        print("Error: dedupe library is required. Install with: pip install dedupe")
        sys.exit(1)
    
    print("=" * 60)
    print("DEDUPE TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Carica dati
    df_unificato, gt_train, gt_val, gt_test = load_data(base_path)
    
    # 2. Prepara DataFrame
    print("\nPreparazione dati...")
    df_prepared = prepare_data_for_dedupe(df_unificato)
    print(f"Dataset preparato: {df_prepared.shape}")
    
    # 3. Identifica gli ID necessari
    # - Tutti gli ID dalla ground truth di training e validazione
    # - Un campione casuale per le statistiche di blocking
    
    ids_gt_train_val = set(gt_train['id_A'].astype(str)) | set(gt_train['id_B'].astype(str)) | \
                       set(gt_val['id_A'].astype(str)) | set(gt_val['id_B'].astype(str))
    
    # Escludi ID del test set per purezza metodologica
    ids_test = set(gt_test['id_A'].astype(str)) | set(gt_test['id_B'].astype(str))
    df_no_test = df_prepared.loc[~df_prepared.index.isin(ids_test)]
    
    ids_sample = set(df_no_test.sample(n=min(sample_size, len(df_no_test)), random_state=42).index.astype(str))
    
    ids_finali = ids_gt_train_val | ids_sample
    print(f"ID totali per training: {len(ids_finali)}")
    
    # 4. Crea dizionario con TUTTE le colonne richieste
    print("\nCreazione dizionario per Dedupe...")
    data_dict = create_dedupe_dict(df_prepared, ids_finali)
    print(f"Dizionario creato con {len(data_dict)} record")
    
    # 5. Verifica campi
    if data_dict:
        sample_record = list(data_dict.values())[0]
        print(f"Campi nel record: {list(sample_record.keys())}")
        
        # Verifica che tutti i campi richiesti siano presenti
        for col in ALL_COLS:
            if col not in sample_record:
                print(f"⚠️  Campo mancante: {col}")
    
    # 6. Prepara training pairs
    print("\nPreparazione coppie di training...")
    matches, distinct = prepare_training_pairs(gt_train, data_dict)
    print(f"Coppie positive (match): {len(matches)}")
    print(f"Coppie negative (distinct): {len(distinct)}")
    
    if len(matches) == 0:
        print("❌ Errore: Nessuna coppia positiva trovata!")
        return
    
    # 7. Inizializza Dedupe con massime prestazioni
    # - num_cores=None: usa tutti i core disponibili
    # - in_memory=True: genera coppie in RAM per velocità massima
    print("\nInizializzazione Dedupe...")
    print(f"   Cores disponibili: {os.cpu_count()}")
    print(f"   Modalità: in_memory=True (RAM)")
    fields = get_dedupe_fields()
    deduper = dedupe.Dedupe(fields, num_cores=None, in_memory=True)
    
    # 8. Passa gli esempi
    print("Marcatura coppie di training...")
    deduper.mark_pairs({'match': matches, 'distinct': distinct})
    
    # 9. Prepare training
    print("\nPreparazione addestramento (campionamento blocking)...")
    deduper.prepare_training(
        data=data_dict,
        sample_size=min(5000, len(data_dict)),
        blocked_proportion=0.2
    )
    
    # 10. Train
    print("\nAddestramento modello...")
    deduper.train()
    
    print("\n✅ Training completato!")
    
    # 11. Salva modello (opzionale)
    model_path = os.path.join(base_path, 'models/dedupe_model.pickle')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        deduper.write_settings(f)
    
    print(f"✅ Modello salvato in: {model_path}")
    
    return deduper


def main():
    parser = argparse.ArgumentParser(description='Pipeline Dedupe per deduplicazione')
    parser.add_argument('--base-path', type=str, default='.', 
                        help='Path base del progetto Record-Linkage')
    parser.add_argument('--train', action='store_true',
                        help='Esegui training del modello')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Numero di record per campionamento blocking')
    args = parser.parse_args()
    
    if args.train:
        train_dedupe(args.base_path, args.sample_size)
    else:
        print("Usa --train per addestrare il modello Dedupe")
        print("Esempio: python -m src.pipelines.dedupe --base-path . --train")


if __name__ == '__main__':
    main()
