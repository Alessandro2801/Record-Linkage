import pandas as pd
import os
import sys
import joblib
import time
import json
import dedupe
import torch
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# CONFIGURAZIONE PERCORSI BASE
# ==========================================
BASE_PATH = "."

# Risolviamo il percorso assoluto della root del progetto
project_root = Path(BASE_PATH).resolve()

# Path alla cartella ditto della repo clonata
ditto_parent = project_root / "src" / "pipelines" / "ditto" / "FAIR-DA4ER" / "ditto"

# Configurazione sys.path per importazioni
sys.path.insert(0, str(project_root / "src")) # Per trainings
sys.path.insert(0, str(ditto_parent))         # Per ditto_light

# --- IMPORT MODULI ---
try:
    from trainings.record_linkage import setup_comparator, compute_features
    from trainings.train_dedupe import prepare_data_for_dedupe, create_dedupe_dict
    
    import ditto_light.ditto as ditto_module
    import ditto_light.dataset as dataset_module
    
    DittoModel = ditto_module.DittoModel
    DittoDataset = dataset_module.DittoDataset
    
    print("✅ Import di Ditto e moduli locali completato con successo.")
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    sys.exit(1)

# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================

def get_blocking_recall(strategy):
    """Recupera la recall del blocking partendo da BASE_PATH."""
    stats_path = os.path.join(BASE_PATH, 'output', 'blocking', f'stats_blocking_{strategy}.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            return stats.get('recall_percentage', 100.0) / 100.0
    print(f"⚠️ Attenzione: Statistiche non trovate in {stats_path}. Uso 1.0.")
    return 1.0

def load_data_and_models(strategy):
    """Carica dati e modelli partendo da BASE_PATH."""
    
    # 1. Caricamento Coppie Candidate (CSV)
    pairs_path = os.path.join(BASE_PATH, 'data', 'blocking', f'candidate_pairs_{strategy}.csv')
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"File non trovato: {pairs_path}")
    df_pairs = pd.read_csv(pairs_path)

    # 2. Dataset Unificato
    mediated_path = os.path.join(BASE_PATH, 'data', 'mediated_schema', 'mediated_schema_normalized.csv')
    df_unificato = pd.read_csv(mediated_path, dtype={'id_source_vehicles': 'object', 'id_source_used_cars': 'object'}, low_memory=False)
    df_unificato['id_unificato'] = df_unificato['id_source_vehicles'].fillna(df_unificato['id_source_used_cars']).astype(str).str.strip()
    df_unificato = df_unificato.set_index('id_unificato')
    
    # 3. Caricamento Modelli Classici
    model_rl = joblib.load(os.path.join(BASE_PATH, 'models', 'recordlinkage_model.joblib'))
    path_dedupe = os.path.join(BASE_PATH, 'models', 'dedupe_model.pickle')
    with open(path_dedupe, 'rb') as f:
        model_dedupe = dedupe.StaticDedupe(f)

    # 4. Caricamento Modello DITTO (PyTorch)
    ditto_path = os.path.join(BASE_PATH, 'src', 'pipelines', 'ditto', 'FAIR-DA4ER', 'ditto', 'checkpoints', 'automotive_task', 'model.pt')
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

# ==========================================
# FUNZIONI DI INFERENZA
# ==========================================

def run_inference_rl(model, df_pairs, model_name, df_unificato, blocking_recall):
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
        "model": model_name, "precision": round(precision, 4), "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4), "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4), "inference_time_sec": round(inference_time, 6)
    }

def run_inference_dedupe(linker, df_pairs, model_name, df_unificato, blocking_recall):
    df_prepared = prepare_data_for_dedupe(df_unificato)
    df_prepared.index = df_prepared.index.astype(str)
    relevant_ids = set(df_pairs['id_A'].astype(str)) | set(df_pairs['id_B'].astype(str))
    data_dict = create_dedupe_dict(df_prepared, relevant_ids)
    
    pairs, y_true = [], []
    for _, row in df_pairs.iterrows():
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
        "model": model_name, "precision": round(precision, 4), "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4), "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4), "inference_time_sec": round(inference_time, 6)
    }

def run_inference_ditto(model, txt_path, model_name, blocking_recall, device):
    """
    Inferenza Ditto seguendo la logica originale della repo FAIR-DA4ER.
    
    Il DittoDataset senza data augmentation (da=None) restituisce (x, label).
    Il metodo pad() restituisce quindi (x_batch, y_batch) con 2 elementi.
    Il modello forward accetta model(x1, x2=None) dove x2 è opzionale.
    """
    if not os.path.exists(txt_path):
        return {"model": model_name, "error": f"File {txt_path} non trovato"}

    # Pre-processing: pulisce le righe e garantisce formato s1\ts2\tlabel
    processed_lines = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue  # Salta righe malformate
            
            s1 = parts[0]
            s2 = parts[1]
            label = parts[-1]  # Label è sempre l'ultimo elemento
            
            # Verifica che label sia 0 o 1
            if label not in ('0', '1'):
                continue
                
            processed_lines.append(f"{s1}\t{s2}\t{label}")

    if not processed_lines:
        return {"model": model_name, "error": "Nessuna riga valida trovata nel file .txt"}

    print(f"  📊 Ditto: {len(processed_lines)} coppie caricate da {txt_path}")

    # Creazione dataset SENZA data augmentation (da=None è il default)
    dataset = DittoDataset(processed_lines, lm='roberta', max_len=256, da=None)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=DittoDataset.pad
    )
    
    all_probs = []
    y_true = []
    
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # IMPORTANTE: senza data augmentation, batch ha solo 2 elementi (x, y)
            # Con data augmentation avrebbe 3 elementi (x1, x2, y)
            if len(batch) == 2:
                x, y = batch
                x = x.to(device)
                logits = model(x)  # forward con x2=None
            else:
                # Caso con data augmentation (non usato qui, ma per completezza)
                x1, x2, y = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                logits = model(x1, x2)
            
            # Calcola le probabilità per la classe positiva (match)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    
    inference_time = time.time() - start_time
    
    # Threshold ottimale (come nella funzione evaluate originale)
    best_th = 0.5
    best_f1 = 0.0
    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in all_probs]
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    
    # Predizioni finali con threshold ottimale
    y_pred = np.array([1 if p > best_th else 0 for p in all_probs])
    y_true = np.array(y_true)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    model_recall = recall_score(y_true, y_pred, zero_division=0)
    total_recall = model_recall * blocking_recall
    f1_total = (2 * precision * total_recall) / (precision + total_recall) if (precision + total_recall) > 0 else 0

    print(f"  🎯 Ditto threshold ottimale: {best_th:.2f}")

    return {
        "model": model_name, 
        "precision": round(precision, 4), 
        "recall_blocking": round(blocking_recall, 4),
        "recall_model": round(model_recall, 4), 
        "recall_total": round(total_recall, 4),
        "f1_total": round(f1_total, 4), 
        "f1_model": round(best_f1, 4),  # F1 del solo modello (senza blocking)
        "threshold": round(best_th, 2),
        "inference_time_sec": round(inference_time, 6)
    }

# ==========================================
# MAIN
# ==========================================

def main():
    strategies = ['B1', 'B2']
    all_results = []

    for strategy in strategies:
        print(f"\n--- Valutazione Strategia Blocking: {strategy} ---")
        try:
            blocking_recall = get_blocking_recall(strategy)
            df_pairs, model_rl, model_dedupe, model_ditto, df_unificato, device = load_data_and_models(strategy)
            
            # 1. Record Linkage
            all_results.append(run_inference_rl(model_rl, df_pairs, f"{strategy}_RL", df_unificato, blocking_recall))
            
            # 2. Dedupe
            all_results.append(run_inference_dedupe(model_dedupe, df_pairs, f"{strategy}_Dedupe", df_unificato, blocking_recall))
            
            # 3. Ditto
            txt_test_path = os.path.join(BASE_PATH, 'data', 'blocking', f'candidate_pairs_{strategy}.txt')
            all_results.append(run_inference_ditto(model_ditto, txt_test_path, f"{strategy}_Ditto", blocking_recall, device))
            
        except Exception as e:
            print(f"Errore nella strategia {strategy}: {e}")

    # Report Finale
    results_df = pd.DataFrame(all_results)
    output_report = os.path.join(BASE_PATH, 'output', 'pipeline_performance_report.json')
    
    # Assicuriamoci che la cartella output esista
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    
    results_df.to_json(output_report, orient='records', indent=4)
    print(f"\n✅ Report salvato in: {output_report}")
    print(results_df[["model", "f1_total", "inference_time_sec"]].to_string(index=False))

if __name__ == "__main__":
    main()