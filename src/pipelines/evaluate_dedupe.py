import numpy as np
import dedupe
import os
import sys
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# --- CONFIGURAZIONE PATH E CARICAMENTO DATI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/pipelines -> src -> Record-Linkage
project_root = os.path.dirname(os.path.dirname(current_dir)) 
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipelines.train_dedupe import load_data, prepare_data_for_dedupe, create_dedupe_dict

print(f"Project root: {project_root}")
df_unificato, gt_train, gt_val, gt_test = load_data(project_root)
df_prepared = prepare_data_for_dedupe(df_unificato)

# Converti l'indice del DataFrame in stringhe per matchare con gli ID della GT
df_prepared.index = df_prepared.index.astype(str)

ids_val_test = set(gt_val['id_A'].astype(str)) | set(gt_val['id_B'].astype(str)) | \
               set(gt_test['id_A'].astype(str)) | set(gt_test['id_B'].astype(str))

data_dict = create_dedupe_dict(df_prepared, ids_val_test)
print(f"Record nel dizionario dati: {len(data_dict)}")

# 1. CARICAMENTO DEL MODELLO SALVATO
model_path = os.path.join(project_root, 'models/dedupe_model.pickle')

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        # Usiamo StaticDedupe per caricare un modello già addestrato
        linker = dedupe.StaticDedupe(f)
    print(f"Modello caricato con successo da {model_path}")
else:
    raise FileNotFoundError(f"Impossibile trovare il modello in {model_path}")

# 2. FUNZIONE PER OTTENERE I PUNTEGGI
def get_pair_scores(linker, gt_df, data_dict):
    """Calcola i punteggi di probabilità per le coppie nella Ground Truth."""
    pairs = []
    labels = []
    
    for _, row in gt_df.iterrows():
        id_a, id_b = str(row['id_A']), str(row['id_B'])
        
        if id_a in data_dict and id_b in data_dict:
            # Formato corretto per dedupe.score(): tuple di (id, record)
            pairs.append(((id_a, data_dict[id_a]), (id_b, data_dict[id_b])))
            labels.append(row['label'])
    
    if not pairs:
        return np.array([]), np.array([])

    # score() restituisce un structured array con colonne 'pairs' e 'score'
    scored = linker.score(pairs)
    scores = scored['score']
        
    return np.array(scores), np.array(labels)

# --- FASE DI OTTIMIZZAZIONE SOGLIA ---
print("\nOttimizzazione della soglia su gt_val...")
val_scores, val_labels = get_pair_scores(linker, gt_val, data_dict)

if len(val_scores) > 0:
    best_threshold = 0.1
    best_f1 = 0

    for threshold in np.arange(0.1, 1.0, 0.05):
        predictions = (val_scores > threshold).astype(int)
        f1 = f1_score(val_labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Miglior Soglia trovata: {best_threshold:.2f} con F1-Score: {best_f1:.4f}")
else:
    print("Errore: Nessuna coppia di gt_val trovata nel dizionario dati.")

# --- FASE DI VALUTAZIONE FINALE ---
print("\nValutazione finale su gt_test...")
test_scores, test_labels = get_pair_scores(linker, gt_test, data_dict)

if len(test_scores) > 0:
    test_predictions = (test_scores > best_threshold).astype(int)

    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)

    print(f"--- Risultati Test (Soglia: {best_threshold:.2f}) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
else:
    print("Errore: Nessuna coppia di gt_test trovata nel dizionario dati.")