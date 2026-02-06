import pandas as pd
import os
import sys
import joblib
import time
import json
import dedupe
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Aggiunta path per import dai tuoi file di training
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainings.record_linkage import setup_comparator, compute_features
from trainings.train_dedupe import prepare_data_for_dedupe, create_dedupe_dict

def get_blocking_recall(base_path, strategy):
    """
    Recupera la recall del blocking salvata precedentemente nel file JSON delle statistiche.
    """
    stats_path = os.path.join(base_path, 'output/blocking/', f'stats_blocking_{strategy}.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            # La recall nel JSON è in percentuale (es 98.5), la trasformiamo in decimale (0.985)
            return stats.get('recall_percentage', 100.0) / 100.0
    print(f"⚠️ Attenzione: Statistiche blocking non trovate in {stats_path}. Uso 1.0 (100%).")
    return 1.0

def load_data_and_models(base_path, strategy):
    """Carica l'oracolo filtrato dal blocking e i modelli addestrati."""
    pairs_path = os.path.join(base_path, 'data/blocking', f'candidate_pairs_{strategy}.csv')
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"File non trovato: {pairs_path}. Esegui prima il blocking.")
    
    df_pairs = pd.read_csv(pairs_path)

    mediated_path = os.path.join(base_path, 'data/mediated_schema/mediated_schema_normalized.csv')
    df_unificato = pd.read_csv(mediated_path, dtype={'id_source_vehicles': 'object', 'id_source_used_cars': 'object'}, low_memory=False)

    df_unificato['id_unificato'] = df_unificato['id_source_vehicles'].fillna(df_unificato['id_source_used_cars']).astype(str).str.strip()
    df_unificato = df_unificato.set_index('id_unificato')
    
    model_rl = joblib.load(os.path.join(base_path, 'models/recordlinkage_model.joblib'))
    path_dedupe = os.path.join(base_path, 'models/dedupe_model.pickle')

    if os.path.exists(path_dedupe):
        with open(path_dedupe, 'rb') as f:
            model_dedupe = dedupe.StaticDedupe(f)
        print(f"Modello Dedupe caricato con successo.")
    else:
        raise FileNotFoundError(f"Impossibile trovare il modello in {path_dedupe}")
    
    return df_pairs, model_rl, model_dedupe, df_unificato

def run_inference_rl(model, df_pairs, model_name, df_unificato, blocking_recall):
    """Esegue la predizione RL e calcola la Recall totale del sistema."""
    compare = setup_comparator()
    test_pairs = df_pairs.set_index(['id_A', 'id_B']).index

    X_test, y_test = compute_features(compare, test_pairs, df_unificato, df_pairs)
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    precision = precision_score(y_test, y_pred, zero_division=0)
    model_recall = recall_score(y_test, y_pred, zero_division=0)
    
    # CALCOLO RECALL TOTALE: Recall_Blocking * Recall_Modello
    total_recall = model_recall * blocking_recall
    
    # F1-Score totale ricalcolato
    f1_total = (2 * precision * total_recall) / (precision + total_recall) if (precision + total_recall) > 0 else 0

    return {
        "model": model_name,
        "precision": round(precision, 4),
        "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4),
        "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4),
        "inference_time_sec": round(inference_time, 6)
    }

def run_inference_dedupe(linker, df_pairs, model_name, df_unificato, blocking_recall):
    """Esegue la predizione Dedupe e calcola la Recall totale del sistema."""
    df_prepared = prepare_data_for_dedupe(df_unificato)
    df_prepared.index = df_prepared.index.astype(str)

    relevant_ids = set(df_pairs['id_A'].astype(str)) | set(df_pairs['id_B'].astype(str))
    data_dict = create_dedupe_dict(df_prepared, relevant_ids)
    
    pairs = []
    y_true = []
    for _, row in df_pairs.iterrows():
        id_a, id_b = str(row['id_A']), str(row['id_B'])
        if id_a in data_dict and id_b in data_dict:
            pairs.append(((id_a, data_dict[id_a]), (id_b, data_dict[id_b])))
            y_true.append(row['label'])

    if not pairs:
        return {"model": model_name, "error": "No valid pairs"}

    start_time = time.time()
    scored_pairs = linker.score(pairs)
    scores = scored_pairs['score']
    
    y_pred = (scores > 0.5).astype(int)
    inference_time = time.time() - start_time

    precision = precision_score(y_true, y_pred, zero_division=0)
    model_recall = recall_score(y_true, y_pred, zero_division=0)
    
    # CALCOLO RECALL TOTALE: Recall_Blocking * Recall_Modello
    total_recall = model_recall * blocking_recall
    
    # F1-Score totale ricalcolato
    f1_total = (2 * precision * total_recall) / (precision + total_recall) if (precision + total_recall) > 0 else 0

    return {
        "model": model_name,
        "precision": round(precision, 4),
        "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4),
        "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4),
        "inference_time_sec": round(inference_time, 6)
    }

def main():
    base_path = "."
    strategies = ['B1', 'B2']
    all_results = []

    for strategy in strategies:
        print(f"\n--- Test Pipeline con Strategia Blocking: {strategy} ---")
        try:
            # Recuperiamo la recall specifica del blocking per questa strategia
            blocking_recall = get_blocking_recall(base_path, strategy)
            
            df_pairs, model_rl, model_dedupe, df_unificato = load_data_and_models(base_path, strategy)
            
            # Pipeline 1: Record Linkage
            res_rl = run_inference_rl(model_rl, df_pairs, f"{strategy}_RecordLinkage", df_unificato, blocking_recall)
            all_results.append(res_rl)
            
            # Pipeline 2: Dedupe
            res_dedupe = run_inference_dedupe(model_dedupe, df_pairs, f"{strategy}_Dedupe", df_unificato, blocking_recall)
            all_results.append(res_dedupe)
            
        except Exception as e:
            print(f"Errore durante il test della strategia {strategy}: {e}")

    # Visualizzazione Risultati
    results_df = pd.DataFrame(all_results)
    
    # Riordiniamo le colonne per una lettura più chiara
    cols_to_show = ["model", "precision", "recall_blocking", "recall_model", "recall_total", "f1_total", "inference_time_sec"]
    print("\n--- RISULTATI FINALI PIPELINES (Recall di Sistema) ---")
    print(results_df[cols_to_show].to_string(index=False))

    # Salvataggio report
    output_report = os.path.join(base_path, 'output', 'pipeline_performance_report.json')
    with open(output_report, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✅ Report salvato in: {output_report}")

if __name__ == "__main__":
    main()