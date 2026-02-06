import pandas as pd
import os

def serialize_row(row, columns_base):
    """
    Trasforma dinamicamente tutte le feature di un record nel formato Ditto.
    columns_base: lista dei nomi delle colonne senza il suffisso _A o _B.
    """
    serialized = []
    for col in columns_base:
        # Recuperiamo il valore (gestendo i nulli come stringa vuota)
        val = str(row[col]).replace('\n', ' ').strip()
        if val.lower() == 'nan' or val == '':
            val = "NULL" # Ditto capisce meglio se c'è un token per il vuoto
        
        # Formato: COL nome_colonna VAL valore_colonna
        serialized.append(f"COL {col} VAL {val}")
    
    return " ".join(serialized)

def convert_to_ditto(input_path, output_path):
    print(f"Conversione in corso: {input_path}...")
    df = pd.read_csv(input_path)
    
    # Identifichiamo dinamicamente tutte le colonne 'base' partendo da quelle _A
    # Escludiamo id_A e id_B perché sono identificativi, non feature semantiche
    all_cols = df.columns.tolist()
    base_features = [c[:-2] for c in all_cols if c.endswith('_A') and not c.startswith('id_')]

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Prepariamo due subset per la riga corrente per facilitare la serializzazione
            row_a = {col: row[f"{col}_A"] for col in base_features}
            row_b = {col: row[f"{col}_B"] for col in base_features}
            
            part_a = serialize_row(row_a, base_features)
            part_b = serialize_row(row_b, base_features)
            
            label = str(int(row['label']))
            
            # Formato finale: [Record A] [SEP] [Record B] [TAB] [Label]
            f.write(f"{part_a} [SEP] {part_b}\t{label}\n")
            
    print(f"✅ File salvato in: {output_path} (Feature incluse: {len(base_features)})")

# --- ESECUZIONE ---
input_files = {
    "train": "data/ground_truth/GT_train/train.csv",
    "valid": "data/ground_truth/GT_train/val.csv",
    "test": "data/ground_truth/GT_eval.csv"
}

output_dir = "data/ditto_data"
os.makedirs(output_dir, exist_ok=True)

for name, path in input_files.items():
    if os.path.exists(path):
        convert_to_ditto(path, os.path.join(output_dir, f"{name}.txt"))
    else:
        print(f"⚠️ File non trovato: {path}")