import pandas as pd
import os
import jellyfish
import json

def load_and_concat_oracle(path_eval, path_test):
    """Carica e unisce i due file di Ground Truth."""
    print(f"Caricamento oracoli da {path_eval} e {path_test}...")
    gt_eval = pd.read_csv(path_eval)
    gt_test = pd.read_csv(path_test)
    return pd.concat([gt_eval, gt_test], ignore_index=True)

def apply_fuzzy_blocking_to_oracle(gt, strategy='B1', mfr_threshold=0.92, model_threshold=0.88):
    """
    Applica il blocking fuzzy all'oracolo usando Jaro-Winkler.
    """
    # 1. Normalizzazione preventiva
    print("Normalizzazione dati in corso...")
    for col in ['year_A', 'year_B']:
        gt[col] = gt[col].fillna(0).astype(int).astype(str)
    
    string_fields = ['manufacturer', 'model', 'fuel_type']
    for field in string_fields:
        gt[f'{field}_A'] = gt[f'{field}_A'].astype(str).str.lower().str.strip()
        gt[f'{field}_B'] = gt[f'{field}_B'].astype(str).str.lower().str.strip()

    # 2. Definizione della funzione di confronto Fuzzy
    def check_fuzzy_row(row):
        if row['year_A'] != row['year_B']:
            return False
        
        mfr_sim = jellyfish.jaro_winkler_similarity(row['manufacturer_A'], row['manufacturer_B'])
        
        if strategy == 'B1':
            return mfr_sim >= mfr_threshold
        
        elif strategy == 'B2':
            model_sim = jellyfish.jaro_winkler_similarity(row['model_A'], row['model_B'])
            fuel_match = (row['fuel_type_A'] == row['fuel_type_B'])
            return (mfr_sim >= mfr_threshold) and (model_sim >= model_threshold) and fuel_match
        
        return False

    print(f"--- Esecuzione Fuzzy Blocking {strategy} (JW) ---")
    
    # Applichiamo il filtro
    mask = gt.apply(check_fuzzy_row, axis=1)
    filtered_gt = gt[mask].copy()

    # 3. Calcolo Statistiche per il JSON
    positives_in_gt = int(len(gt[gt['label'] == 1]))
    total_pairs_selected = int(len(filtered_gt))
    positives_captured = int(len(filtered_gt[filtered_gt['label'] == 1]))
    
    recall = (positives_captured / positives_in_gt) * 100 if positives_in_gt > 0 else 0
    
    stats = {
        "strategy": strategy,
        "mfr_threshold": mfr_threshold,
        "model_threshold": model_threshold if strategy == 'B2' else None,
        "positives_in_gt": positives_in_gt,
        "total_pairs_selected": total_pairs_selected,
        "positives_captured": positives_captured,
        "recall_percentage": round(recall, 2)
    }

    print(f"Recall: {recall:.2f}% ({positives_captured}/{positives_in_gt})")
    print(f"Coppie totali selezionate: {total_pairs_selected}")

    return filtered_gt, stats

def main():
    base_path = "." 
    path_eval = os.path.join(base_path, 'data/ground_truth/GT_eval.csv')
    path_test = os.path.join(base_path, 'data/ground_truth/GT_train/test.csv')
    output_dir = os.path.join(base_path, 'data/blocking/')
    json_dir = os.path.join(base_path, 'output/blocking/')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Caricamento
    gt_unified = load_and_concat_oracle(path_eval, path_test)
    
    # 2. Esecuzione Blocking (Parametri suggeriti per il test)
    current_strategy = 'B2' 
    filtered_oracle, stats = apply_fuzzy_blocking_to_oracle(
        gt_unified, 
        strategy=current_strategy,
        mfr_threshold=0.95, 
        model_threshold=0.85
    )
    
    # 3. Salvataggio CSV
    csv_name = f'candidate_pairs_{current_strategy}.csv'
    filtered_oracle.to_csv(os.path.join(output_dir, csv_name), index=False)
    
    # 4. Salvataggio JSON Statistiche
    json_name = f'stats_blocking_{current_strategy}.json'
    with open(os.path.join(json_dir, json_name), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n✅ File JSON salvato: {json_name}")
    print(f"✅ File CSV salvato: {csv_name}")

if __name__ == "__main__":
    main()