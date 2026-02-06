import pandas as pd
from pathlib import Path
import os

# Lista delle colonne aggiornata secondo il tuo schema
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
            # Pulizia per evitare rotture nel parsing di Ditto (cruciale per A100)
            val_clean = val.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            parts.append(f"COL {f} VAL {val_clean}")
            
    return " ".join(parts)

def convert_blocking_files(input_dir: Path):
    """Converte i file di blocking specifici in formato TXT per l'inferenza."""
    
    # Lista esatta dei file da convertire
    target_files = ["candidate_pairs_B1.csv", "candidate_pairs_B2.csv"]

    for csv_name in target_files:
        in_file = input_dir / csv_name
        # Il file di output avrà lo stesso nome ma estensione .txt
        out_file = input_dir / csv_name.replace(".csv", ".txt")
        
        if not in_file.exists(): 
            print(f"⚠️ Salto: {csv_name} non trovato in {input_dir}")
            continue
            
        print(f"🚀 Conversione in corso: {csv_name} -> {out_file.name}")
        
        # Caricamento dati
        df = pd.read_csv(in_file, dtype=str)
        
        with open(out_file, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record_a = serialize(row, "A")
                record_b = serialize(row, "B")
                
                # Label (0 o 1). Se per qualche motivo manca nel blocking, mettiamo 0 di default
                label = str(row.get('label', '0'))
                
                # Formato Ditto standard: RecordA [TAB] RecordB [TAB] Label
                f.write(f"{record_a}\t{record_b}\t{label}\n")

if __name__ == "__main__":
    base_path = "."
    
    # Directory specifica richiesta
    blocking_dir = Path(base_path) / "data" / "blocking"

    print(f"Inizio conversione file di inferenza in: {blocking_dir}\n")

    if blocking_dir.exists():
        convert_blocking_files(blocking_dir)
        print(f"\n✅ Elaborazione completata. I file .txt sono pronti per l'inferenza.")
    else:
        print(f"❌ Errore: La cartella {blocking_dir} non esiste.")