#!/usr/bin/env python3
"""
dedupe_eval.py — Valutazione del modello Dedupe su validation e test set.

Workflow:
    1. Carica modello salvato (StaticDedupe)
    2. Ottimizza soglia su validation set
    3. Valuta su test set
    4. Salva metriche in JSON

Usage:
    python -m src.matching.dedupe_eval --strategy B1
"""

import argparse
import numpy as np
import dedupe
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    MODEL_RESULTS_DIR,
    blocking_split_path,
    model_path as get_model_path,
    ensure_dirs,
)
from src.matching.dedupe import load_data, prepare_data_for_dedupe, create_dedupe_dict


def get_pair_scores(linker, gt_df, data_dict):
    """Calcola i punteggi di probabilità per le coppie nella Ground Truth."""
    pairs = []
    labels = []

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="  Scoring coppie", unit="coppia"):
        id_a, id_b = str(row['id_A']), str(row['id_B'])

        if id_a in data_dict and id_b in data_dict:
            pairs.append(((id_a, data_dict[id_a]), (id_b, data_dict[id_b])))
            labels.append(row['label'])

    if not pairs:
        return np.array([]), np.array([])

    scored = linker.score(pairs)
    scores = scored['score']

    return np.array(scores), np.array(labels)


def main(strategy: str = None):
    # Supporto chiamata diretta e da CLI
    if strategy is None:
        parser = argparse.ArgumentParser(description='Valutazione modello Dedupe')
        parser.add_argument('--strategy', type=str, required=True, choices=['B1', 'B2'],
                            help='Strategia di blocking (B1 o B2)')
        args = parser.parse_args()
        strategy = args.strategy

    ensure_dirs()

    # Caricamento dati
    print(f"Caricamento dati per strategia {strategy}...")
    df_unificato, gt_train, gt_val, gt_test = load_data(strategy)
    df_prepared = prepare_data_for_dedupe(df_unificato)
    df_prepared.index = df_prepared.index.astype(str)

    ids_val_test = (set(gt_val['id_A'].astype(str)) | set(gt_val['id_B'].astype(str)) |
                    set(gt_test['id_A'].astype(str)) | set(gt_test['id_B'].astype(str)))

    data_dict = create_dedupe_dict(df_prepared, ids_val_test)
    print(f"Record nel dizionario dati: {len(data_dict)}")

    # Caricamento modello
    _model_path = get_model_path('dedupe', strategy)

    if _model_path.exists():
        with open(_model_path, 'rb') as f:
            linker = dedupe.StaticDedupe(f)
        print(f"Modello caricato da {_model_path}")
    else:
        raise FileNotFoundError(f"Modello non trovato: {_model_path}")

    # Ottimizzazione soglia su validation
    print("\nOttimizzazione soglia su validation set...")
    val_scores, val_labels = get_pair_scores(linker, gt_val, data_dict)

    best_threshold = 0.1
    best_f1 = 0

    if len(val_scores) > 0:
        for threshold in tqdm(np.arange(0.1, 1.0, 0.05), desc="  Ottimizzazione soglia", unit="th"):
            predictions = (val_scores > threshold).astype(int)
            f1 = f1_score(val_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Miglior soglia: {best_threshold:.2f} con F1: {best_f1:.4f}")
    else:
        print("Errore: Nessuna coppia di validation trovata nel dizionario dati.")

    # Valutazione finale su test set
    print("\nValutazione finale su test set...")
    test_scores, test_labels = get_pair_scores(linker, gt_test, data_dict)

    if len(test_scores) > 0:
        test_predictions = (test_scores > best_threshold).astype(int)

        precision = precision_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions)
        f1 = f1_score(test_labels, test_predictions)

        metrics = {
            "strategy": strategy,
            "best_threshold": round(float(best_threshold), 2),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1-measure": round(float(f1), 4),
        }

        output_path = MODEL_RESULTS_DIR / "dedupe.json"
        MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

        print(f"--- Risultati Test (Soglia: {best_threshold:.2f}) ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nMetriche salvate in: {output_path}")
    else:
        print("Errore: Nessuna coppia di test trovata nel dizionario dati.")


if __name__ == "__main__":
    main()
