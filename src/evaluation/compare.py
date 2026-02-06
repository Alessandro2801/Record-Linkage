#!/usr/bin/env python3
"""
compare.py — Valutazione comparativa di tutte le pipeline di matching.

Confronta le prestazioni di 6 combinazioni (B1/B2 × RL/Dedupe/Ditto)
in termini di precision, recall (modello × blocking), F1 e tempi di inferenza.

Usage:
    python -m src.evaluation.compare
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import dedupe
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from src.config import (
    BLOCKING_STATS_DIR,
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    DITTO_CHECKPOINTS_DIR,
    DITTO_VENDOR_DIR,
    PIPELINE_REPORT_PATH,
    DITTO_BATCH_SIZE,
    N_JOBS,
    print_hw_info,
    blocking_split_path,
    model_path as get_model_path,
    ditto_data_path,
    ensure_dirs,
)

# Aggiungi vendor/FAIR-DA4ER/ditto/ a sys.path per importare ditto_light
sys.path.insert(0, str(DITTO_VENDOR_DIR))

try:
    from src.matching.recordlinkage import setup_comparator, compute_features
    from src.matching.dedupe import prepare_data_for_dedupe, create_dedupe_dict

    import ditto_light.ditto as ditto_module
    import ditto_light.dataset as dataset_module

    DittoModel = ditto_module.DittoModel
    DittoDataset = dataset_module.DittoDataset

    print("Import di Ditto e moduli locali completato.")
except ImportError as e:
    print(f"Errore di importazione: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI DI SUPPORTO
# ═══════════════════════════════════════════════════════════════════════════════

def get_blocking_recall(strategy):
    """Recupera la recall del blocking dalle statistiche salvate."""
    stats_path = BLOCKING_STATS_DIR / f'stats_{strategy}.json'
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            return stats.get('recall_percentage', 100.0) / 100.0
    print(f"Statistiche non trovate in {stats_path}. Uso recall=1.0.")
    return 1.0


def load_data_and_models(strategy):
    """Carica dati (test split) e modelli per una strategia."""

    # 1. Coppie candidate — test split
    pairs_path = blocking_split_path(strategy, 'test')
    if not pairs_path.exists():
        raise FileNotFoundError(f"File non trovato: {pairs_path}")
    df_pairs = pd.read_csv(pairs_path)

    # 2. Dataset unificato
    df_unificato = pd.read_csv(
        MEDIATED_SCHEMA_NORMALIZED_PATH,
        dtype={'id_source_vehicles': 'object', 'id_source_used_cars': 'object'},
        low_memory=False,
    )
    df_unificato['id_unificato'] = (
        df_unificato['id_source_vehicles']
        .fillna(df_unificato['id_source_used_cars'])
        .astype(str).str.strip()
    )
    df_unificato = df_unificato.set_index('id_unificato')

    # 3. Modelli classici
    model_rl = joblib.load(get_model_path('recordlinkage', strategy))
    path_dedupe = get_model_path('dedupe', strategy)
    with open(path_dedupe, 'rb') as f:
        model_dedupe = dedupe.StaticDedupe(f)

    # 4. Modello Ditto (PyTorch)
    ditto_path = DITTO_CHECKPOINTS_DIR / 'automotive_task' / 'model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(ditto_path):
        checkpoint = torch.load(ditto_path, map_location=device)
        model_ditto = DittoModel(device=device, lm='roberta')
        model_ditto.load_state_dict(checkpoint['model'])
        model_ditto.to(device)
        model_ditto.eval()
        print(f"Modello Ditto caricato da {ditto_path} su {device}.")
    else:
        raise FileNotFoundError(f"Modello Ditto non trovato in {ditto_path}")

    return df_pairs, model_rl, model_dedupe, model_ditto, df_unificato, device


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI DI INFERENZA
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference_rl(model, df_pairs, model_name, df_unificato, blocking_recall):
    """Inferenza con Record Linkage (Logistic Regression)."""
    compare = setup_comparator()
    test_pairs = df_pairs.set_index(['id_A', 'id_B']).index
    X_test, y_test = compute_features(compare, test_pairs, df_unificato, df_pairs)

    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    precision = precision_score(y_test, y_pred, zero_division=0)
    model_recall = recall_score(y_test, y_pred, zero_division=0)
    total_recall = model_recall * blocking_recall
    f1_total = (2 * precision * total_recall) / (precision + total_recall) if (precision + total_recall) > 0 else 0

    return {
        "model": model_name, "precision": round(precision, 4),
        "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4), "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4), "inference_time_sec": round(inference_time, 6),
    }


def run_inference_dedupe(linker, df_pairs, model_name, df_unificato, blocking_recall):
    """Inferenza con Dedupe."""
    df_prepared = prepare_data_for_dedupe(df_unificato)
    df_prepared.index = df_prepared.index.astype(str)
    relevant_ids = set(df_pairs['id_A'].astype(str)) | set(df_pairs['id_B'].astype(str))
    data_dict = create_dedupe_dict(df_prepared, relevant_ids)

    pairs, y_true = [], []
    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Dedupe pairs"):
        id_a, id_b = str(row['id_A']), str(row['id_B'])
        if id_a in data_dict and id_b in data_dict:
            pairs.append(((id_a, data_dict[id_a]), (id_b, data_dict[id_b])))
            y_true.append(row['label'])

    start_time = time.time()
    scored_pairs = linker.score(pairs)
    y_pred = (scored_pairs['score'] > 0.5).astype(int)
    inference_time = time.time() - start_time

    precision = precision_score(y_true, y_pred, zero_division=0)
    model_recall = recall_score(y_true, y_pred, zero_division=0)
    total_recall = model_recall * blocking_recall
    f1_total = (2 * precision * total_recall) / (precision + total_recall) if (precision + total_recall) > 0 else 0

    return {
        "model": model_name, "precision": round(precision, 4),
        "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4), "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4), "inference_time_sec": round(inference_time, 6),
    }


def run_inference_ditto(model, txt_path, model_name, blocking_recall, device):
    """Inferenza con Ditto (Transformer-based)."""
    if not os.path.exists(txt_path):
        return {"model": model_name, "error": f"File {txt_path} non trovato"}

    processed_lines = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            s1, s2, label = parts[0], parts[1], parts[-1]
            if label not in ('0', '1'):
                continue
            processed_lines.append(f"{s1}\t{s2}\t{label}")

    if not processed_lines:
        return {"model": model_name, "error": "Nessuna riga valida nel file .txt"}

    print(f"  Ditto: {len(processed_lines)} coppie caricate da {txt_path}")

    dataset = DittoDataset(processed_lines, lm='roberta', max_len=256, da=None)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=DITTO_BATCH_SIZE, shuffle=False,
        collate_fn=DittoDataset.pad,
        num_workers=min(4, N_JOBS), pin_memory=(device == 'cuda'),
    )

    all_probs = []
    y_true = []

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ditto inference"):
            if len(batch) == 2:
                x, y = batch
                x = x.to(device)
                logits = model(x)
            else:
                x1, x2, y = batch
                x1, x2 = x1.to(device), x2.to(device)
                logits = model(x1, x2)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())

    inference_time = time.time() - start_time

    # Threshold ottimale
    best_th, best_f1 = 0.5, 0.0
    for th in tqdm(np.arange(0.0, 1.0, 0.05), desc="Ditto threshold opt"):
        pred = [1 if p > th else 0 for p in all_probs]
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    y_pred = np.array([1 if p > best_th else 0 for p in all_probs])
    y_true = np.array(y_true)

    precision = precision_score(y_true, y_pred, zero_division=0)
    model_recall = recall_score(y_true, y_pred, zero_division=0)
    total_recall = model_recall * blocking_recall
    f1_total = (2 * precision * total_recall) / (precision + total_recall) if (precision + total_recall) > 0 else 0

    print(f"  Ditto threshold ottimale: {best_th:.2f}")

    return {
        "model": model_name, "precision": round(precision, 4),
        "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4), "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4), "f1_model": round(best_f1, 4),
        "threshold": round(best_th, 2), "inference_time_sec": round(inference_time, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    strategies = ['B1', 'B2']
    all_results = []
    print_hw_info()

    for strategy in strategies:
        print(f"\n--- Valutazione Strategia: {strategy} ---")
        try:
            blocking_recall = get_blocking_recall(strategy)
            df_pairs, model_rl, model_dedupe, model_ditto, df_unificato, device = \
                load_data_and_models(strategy)

            # 1. Record Linkage
            all_results.append(
                run_inference_rl(model_rl, df_pairs, f"{strategy}_RL", df_unificato, blocking_recall)
            )

            # 2. Dedupe
            all_results.append(
                run_inference_dedupe(model_dedupe, df_pairs, f"{strategy}_Dedupe", df_unificato, blocking_recall)
            )

            # 3. Ditto
            txt_test_path = str(ditto_data_path(strategy, 'test'))
            all_results.append(
                run_inference_ditto(model_ditto, txt_test_path, f"{strategy}_Ditto", blocking_recall, device)
            )

        except Exception as e:
            print(f"Errore nella strategia {strategy}: {e}")

    # Report finale
    results_df = pd.DataFrame(all_results)

    PIPELINE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(str(PIPELINE_REPORT_PATH), orient='records', indent=4)

    print(f"\nReport salvato in: {PIPELINE_REPORT_PATH}")
    print(results_df[["model", "f1_total", "inference_time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
