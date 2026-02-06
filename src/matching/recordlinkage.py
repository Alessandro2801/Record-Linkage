#!/usr/bin/env python3
"""
recordlinkage.py — Pipeline Record Linkage con la libreria recordlinkage + Logistic Regression.

Workflow:
    1. Carica dataset unificato e blocking splits
    2. Calcola feature di similarità (Jaro-Winkler, exact, gaussian)
    3. Tuning iperparametri su validation set
    4. Valutazione finale su test set
    5. Salva modello e metriche

Usage:
    python -m src.matching.recordlinkage --train --strategy B1
    python -m src.matching.recordlinkage --evaluate --strategy B1
"""

import argparse
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import recordlinkage as rl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, precision_recall_fscore_support,
)
import joblib

from src.config import (
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    MODEL_RESULTS_DIR,
    RANDOM_SEED,
    N_JOBS,
    print_hw_info,
    blocking_split_path,
    model_path as get_model_path,
    ensure_dirs,
)


def load_data(strategy: str):
    """Carica dataset e blocking splits per la strategia specificata."""
    print(f"Caricamento dati per strategia {strategy}...")
    df_unificato = pd.read_csv(MEDIATED_SCHEMA_NORMALIZED_PATH, dtype={'id_source_vehicles': 'object'}, low_memory=False)
    gt_train = pd.read_csv(blocking_split_path(strategy, 'train'))
    gt_val = pd.read_csv(blocking_split_path(strategy, 'val'))
    gt_test = pd.read_csv(blocking_split_path(strategy, 'test'))

    # Rimuovi colonne non necessarie
    drop_cols = ['description', 'vin']
    for col in drop_cols:
        if col in df_unificato.columns:
            df_unificato.drop(columns=[col], inplace=True)

    for gt in [gt_train, gt_val, gt_test]:
        for col in ['description_A', 'description_B']:
            if col in gt.columns:
                gt.drop(columns=[col], inplace=True)

    # Crea ID unificato e imposta come indice
    df_unificato['id_unificato'] = (
        df_unificato['id_source_vehicles']
        .fillna(df_unificato['id_source_used_cars'])
    )
    df_unificato = df_unificato.set_index('id_unificato')

    print(f"Dataset unificato: {df_unificato.shape}")
    print(f"Train: {len(gt_train)}, Val: {len(gt_val)}, Test: {len(gt_test)}")

    return df_unificato, gt_train, gt_val, gt_test


def setup_comparator():
    """Configura il comparatore per il calcolo delle feature di similarità."""
    compare = rl.Compare()

    # Stringhe (Fuzzy)
    compare.string('manufacturer', 'manufacturer', method='jarowinkler', threshold=0.85, label='manufacturer')
    compare.string('model', 'model', method='jarowinkler', threshold=0.85, label='model')
    compare.string('location', 'location', method='jarowinkler', threshold=0.85, label='location')
    compare.string('cylinders', 'cylinders', method='jarowinkler', threshold=0.70, label='cylinders')

    # Esatti (Categorie)
    compare.exact('year', 'year', label='year')
    compare.exact('fuel_type', 'fuel_type', label='fuel_type')
    compare.exact('traction', 'traction', label='traction')
    compare.exact('body_type', 'body_type', label='body_type')
    compare.exact('main_color', 'main_color', label='main_color')
    compare.exact('transmission', 'transmission', label='transmission')

    # Numerici (Gaussiani)
    compare.numeric('price', 'price', method='gauss', offset=500, scale=2000, label='price')
    compare.numeric('mileage', 'mileage', method='gauss', offset=1000, scale=10000, label='mileage')
    compare.numeric('latitude', 'latitude', method='gauss', offset=0.01, scale=0.1, label='lat')
    compare.numeric('longitude', 'longitude', method='gauss', offset=0.01, scale=0.1, label='lon')

    return compare


def compute_features(compare, pairs, df, gt):
    """Calcola la matrice delle feature per le coppie specificate."""
    X = compare.compute(pairs, df, df)
    y = gt['label']
    return X, y


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Esegue tuning degli iperparametri per LogisticRegression."""
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    weights = [None, 'balanced']

    best_f1 = 0
    best_params = {}
    best_model = None

    print("\n--- INIZIO TUNING IPERPARAMETRI ---")
    print(f"Distribuzione Training: {y_train.value_counts().to_dict()}")
    print(f"Parallelismo: n_jobs={N_JOBS}")

    start_time = time.time()

    configs = [(c, w) for c in c_values for w in weights]
    for c, w in tqdm(configs, desc="Tuning iperparametri", unit="config"):
        model = LogisticRegression(
            C=c, class_weight=w, max_iter=1000,
            random_state=RANDOM_SEED, n_jobs=N_JOBS,
            solver='lbfgs'
        )
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        current_f1 = f1_score(y_val, y_val_pred)
        current_prec = precision_score(y_val, y_val_pred, zero_division=0)
        current_rec = recall_score(y_val, y_val_pred, zero_division=0)

        print(f"C: {c:7} | Weight: {str(w):10} | F1: {current_f1:.4f} (P: {current_prec:.2f}, R: {current_rec:.2f})")

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model = model
            best_params = {'C': c, 'class_weight': w}

    tuning_duration = time.time() - start_time

    print(f"\n--- RISULTATI TUNING ---")
    print(f"Miglior F1-Score su Validation: {best_f1:.4f}")
    print(f"Migliori Parametri: {best_params}")
    print(f"Tempo impiegato: {tuning_duration:.2f} secondi")

    return best_model, best_params


def evaluate_model(model, X_test, y_test, strategy):
    """Valuta il modello sul test set e salva le metriche in JSON."""
    print("\n--- VALUTAZIONE FINALE SUL TEST SET ---")

    y_test_pred = model.predict(X_test)

    print(classification_report(y_test, y_test_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary', pos_label=1
    )

    metrics = {
        "strategy": strategy,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1-measure": round(float(f1), 4),
    }

    output_path = MODEL_RESULTS_DIR / "recordlinkage.json"
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metriche salvate in: {output_path}")

    return y_test_pred


def save_model(model, output_path):
    """Salva il modello addestrato."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModello salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Pipeline Record Linkage')
    parser.add_argument('--train', action='store_true', help='Esegui training del modello')
    parser.add_argument('--evaluate', action='store_true', help='Valuta modello esistente')
    parser.add_argument('--strategy', type=str, required=True, choices=['B1', 'B2'],
                        help='Strategia di blocking (B1 o B2)')
    args = parser.parse_args()

    ensure_dirs()
    print_hw_info()
    strategy = args.strategy
    _model_path = get_model_path('recordlinkage', strategy)

    print("=" * 60)
    print("RECORD LINKAGE PIPELINE")
    print("=" * 60)

    # Caricamento dati
    df_unificato, gt_train, gt_val, gt_test = load_data(strategy)

    # Setup comparatore
    compare = setup_comparator()

    # Generazione coppie
    print("\nCalcolo Feature Matrix...")
    training_pairs = gt_train.set_index(['id_A', 'id_B']).index
    val_pairs = gt_val.set_index(['id_A', 'id_B']).index
    test_pairs = gt_test.set_index(['id_A', 'id_B']).index

    X_train, y_train = compute_features(compare, training_pairs, df_unificato, gt_train)
    X_val, y_val = compute_features(compare, val_pairs, df_unificato, gt_val)

    if args.train or not _model_path.exists():
        # Training
        best_model, best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

        # Valutazione sul test set
        X_test, y_test = compute_features(compare, test_pairs, df_unificato, gt_test)
        evaluate_model(best_model, X_test, y_test, strategy)

        # Salvataggio modello
        save_model(best_model, _model_path)

    elif args.evaluate:
        # Carica modello esistente
        print(f"\nCaricamento modello da {_model_path}...")
        model = joblib.load(_model_path)

        X_test, y_test = compute_features(compare, test_pairs, df_unificato, gt_test)
        evaluate_model(model, X_test, y_test, strategy)

    else:
        print("\nSpecifica --train per addestrare o --evaluate per valutare")


if __name__ == '__main__':
    main()
