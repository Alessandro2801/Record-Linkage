import pandas as pd
from pathlib import Path
import os

# Lista delle colonne presenti nel tuo dataset mediato (senza suffisso)
MEDIATED_FIELDS = [
    "manufacturer", "year", "model", "mileage", "price", "main_color",
    "transmission", "traction", "body_type", "fuel_type", "cylinders",
    "latitude", "longitude", "description", "location", "pubblication_date",
]

def serialize(row, suffix):
    """
    Trasforma le feature di un record nel formato Ditto usando il suffisso _A o _B.
    """
    parts = []
    for f in MEDIATED_FIELDS:
        col_name = f"{f}_{suffix}"
        val = str(row.get(col_name, "")).strip()
        
        if val and val.lower() != "nan":
            # Pulizia per evitare rotture nel parsing di Ditto
            val_clean = val.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            parts.append(f"COL {f} VAL {val_clean}")
            
    return " ".join(parts)

def convert_folder(in_dir: Path, out_dir: Path):
    """Converte i file CSV in formato TXT con doppia tabulazione per Ditto."""
    files_map = {
        "train.csv": "train.txt",
        "val.csv": "val.txt", 
        "test.csv": "test.txt"
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_name, txt_name in files_map.items():
        in_file = in_dir / csv_name
        if not in_file.exists(): 
            continue
            
        print(f"  -> Conversione: {in_file} in {out_dir / txt_name}")
        df = pd.read_csv(in_file, dtype=str)
        
        with open(out_dir / txt_name, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record_a = serialize(row, "A")
                record_b = serialize(row, "B")
                
                # Formato cruciale: RecordA [TAB] RecordB [TAB] Label
                f.write(f"{record_a}\t{record_b}\t{row['label']}\n")

if __name__ == "__main__":
    # Definizione base_path
    base_path = "."
    
    # Percorso di Input: data/ground_truth/GT_train/
    input_dir = Path(base_path) / "data" / "ground_truth" / "GT_train"
    
    # Percorso di Output: src/pipelines/ditto/FAIR-DA4ER/ditto/data/ditto_data
    output_dir = Path(base_path) / "src" / "pipelines" / "ditto" / "FAIR-DA4ER" / "ditto" / "data" / "ditto_data"

    print(f"Base Path: {os.path.abspath(base_path)}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}\n")

    if input_dir.exists():
        convert_folder(input_dir, output_dir)
        
        # Gestione specifica per test.csv se si trova in data/ground_truth/ (un livello sopra GT_train)
        test_csv_extra = input_dir.parent / "test.csv"
        if test_csv_extra.exists():
            print(f"  -> Trovato test.csv extra in {input_dir.parent}")
            convert_folder(input_dir.parent, output_dir)
    else:
        print(f"Errore: La cartella di input {input_dir} non esiste.")

    print(f"\n✅ Elaborazione completata. I file .txt sono in: {output_dir}")