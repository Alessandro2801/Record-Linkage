#!/usr/bin/env python3
"""
ditto_format.py — Conversione dati in formato Ditto (serializzazione colonnare).

Converte i CSV delle coppie candidate (blocking splits e file completo)
nel formato testuale richiesto da Ditto:
    COL field1 VAL value1 COL field2 VAL value2 ...  [TAB]  ... [TAB] label

Modalità:
    splits     — Converte train/val/test per una strategia
    inference  — Converte il file completo dei candidati
    all        — Entrambe (default)

Usage:
    python -m src.blocking.ditto_format --strategy B1
    python -m src.blocking.ditto_format --strategy B1 --mode splits
"""

import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.config import (
    blocking_candidates_path,
    blocking_split_path,
    blocking_strategy_dir,
    ditto_data_path,
    print_hw_info,
    ensure_dirs,
)

# Colonne dello schema mediato da serializzare
MEDIATED_FIELDS = [
    "manufacturer", "year", "model", "mileage", "price", "main_color",
    "transmission", "traction", "body_type", "fuel_type", "cylinders",
    "latitude", "longitude", "description", "location", "pubblication_date",
]


def serialize(row, suffix: str) -> str:
    """Serializza un record nel formato colonnare Ditto."""
    parts = []
    for f in MEDIATED_FIELDS:
        col_name = f"{f}_{suffix}"
        val = str(row.get(col_name, "")).strip()
        if val and val.lower() != "nan":
            val_clean = val.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            parts.append(f"COL {f} VAL {val_clean}")
    return " ".join(parts)


def _serialize_vectorized(df: pd.DataFrame, suffix: str) -> pd.Series:
    """Serializza tutti i record in modo vettorizzato (molto più veloce di iterrows)."""
    parts_list = []
    for f in MEDIATED_FIELDS:
        col_name = f"{f}_{suffix}"
        if col_name in df.columns:
            vals = df[col_name].astype(str).str.strip()
            # Pulisci tab/newline
            vals = vals.str.replace('\t', ' ', regex=False)
            vals = vals.str.replace('\n', ' ', regex=False)
            vals = vals.str.replace('\r', ' ', regex=False)
            # Maschera i NaN
            mask = vals.str.lower().ne('nan') & vals.ne('')
            col_str = ("COL " + f + " VAL " + vals).where(mask, '')
        else:
            col_str = pd.Series([''] * len(df), index=df.index)
        parts_list.append(col_str)

    # Concatena tutti i campi per ogni riga
    result = parts_list[0]
    for p in parts_list[1:]:
        # Aggiungi spazio solo se entrambe le parti sono non vuote
        result = result.where(result == '', result + ' ') + p
        # Pulisci spazi iniziali per righe che iniziavano con campo vuoto
        result = result.str.strip()

    return result


def convert_csv_to_ditto(csv_path: Path, txt_path: Path) -> None:
    """Converte un singolo CSV in formato Ditto .txt (vettorizzato)."""
    if not csv_path.exists():
        print(f"  File non trovato: {csv_path}")
        return

    print(f"  {csv_path.name} -> {txt_path.name}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Serializzazione vettorizzata
    records_a = _serialize_vectorized(df, "A")
    records_b = _serialize_vectorized(df, "B")
    labels = df.get('label', pd.Series(['0'] * len(df))).astype(str)

    # Componi le righe finali
    lines = records_a + '\t' + records_b + '\t' + labels + '\n'

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines.tolist())


def convert_splits(strategy: str) -> None:
    """Converte i blocking splits (train/val/test) in formato Ditto."""
    print(f"\n  Conversione splits per {strategy}...")

    split_map = {"train": "train", "val": "valid", "test": "test"}
    for split_name, ditto_name in split_map.items():
        csv_path = blocking_split_path(strategy, split_name)
        txt_path = ditto_data_path(strategy, ditto_name)
        convert_csv_to_ditto(csv_path, txt_path)


def convert_candidates(strategy: str) -> None:
    """Converte il file completo dei candidati in formato Ditto (per inferenza)."""
    print(f"\n  Conversione candidati completi per {strategy}...")
    csv_path = blocking_candidates_path(strategy)
    txt_path = blocking_strategy_dir(strategy) / "candidates_ditto.txt"
    convert_csv_to_ditto(csv_path, txt_path)


def convert_all(strategy: str) -> None:
    """Converte sia splits che candidati completi in parallelo."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(convert_splits, strategy)
        pool.submit(convert_candidates, strategy)


def main():
    parser = argparse.ArgumentParser(description='Conversione dati in formato Ditto')
    parser.add_argument('--strategy', type=str, required=True, choices=['B1', 'B2'],
                        help='Strategia di blocking')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['splits', 'inference', 'all'],
                        help='Cosa convertire (default: all)')
    args = parser.parse_args()

    ensure_dirs()
    print_hw_info()
    strategy = args.strategy

    print(f"Conversione formato Ditto — Strategia {strategy}")

    if args.mode in ('splits', 'all'):
        convert_splits(strategy)
    if args.mode in ('inference', 'all'):
        convert_candidates(strategy)

    print(f"\nConversione completata per {strategy}.")


if __name__ == "__main__":
    main()
