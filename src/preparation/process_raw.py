#!/usr/bin/env python3
"""
process_raw.py — Pulizia e preprocessing dei dataset grezzi.

Automatizza le operazioni di data-cleaning che erano nei notebook EDA:
  • vehicles.csv       →  vehicles_processed.csv
  • used_cars_data.csv →  used_cars_data_processed.csv

Logica applicata (per ciascun dataset):
  1. Rimozione colonne con più del 70 % di valori mancanti.
  2. Rimozione righe con NULL sulle colonne critiche per il matching.
  3. (Solo used_cars) Rimozione colonna `sp_id` e generazione ID univoco.
  4. Salvataggio in storage/processed/.

I notebook EDA restano disponibili per l'analisi esplorativa e i grafici,
ma non sono più necessari per la pipeline automatizzata.

Usage:
    python -m src.preparation.process_raw
"""

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    VEHICLES_PROCESSED_PATH,
    USED_CARS_PROCESSED_PATH,
    N_JOBS,
    print_hw_info,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETRI
# ═══════════════════════════════════════════════════════════════════════════════

MISSING_THRESHOLD = 70.0   # Colonne con più del 70 % di NULL vengono rimosse

# Colonne critiche: righe con NULL su queste vengono eliminate
VEHICLES_CRITICAL_COLS   = ["manufacturer", "model", "year", "odometer", "fuel"]
USED_CARS_CRITICAL_COLS  = ["make_name", "model_name", "year", "mileage", "fuel_type"]


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI DI PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _drop_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rimuove colonne con percentuale di NULL superiore alla soglia."""
    print("  Analisi valori mancanti...")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if cols_to_drop:
        print(f"  Colonne rimosse (>{threshold}% missing): {cols_to_drop}")
    else:
        print(f"  Nessuna colonna supera la soglia di {threshold}% di missing.")
    return df.drop(columns=cols_to_drop)


def _drop_null_critical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rimuove righe con valori NULL sulle colonne critiche."""
    before = len(df)
    df = df.dropna(subset=cols, how="any")
    dropped = before - len(df)
    print(f"  Righe rimosse (NULL su colonne critiche): {dropped}")
    return df


def process_vehicles() -> None:
    """Processa vehicles.csv → vehicles_processed.csv."""
    raw_path = RAW_DIR / "vehicles.csv"
    print(f"\n{'─'*60}")
    print(f"  Processing: {raw_path.name}")
    print(f"{'─'*60}")

    print("  Lettura CSV...")
    df = pd.read_csv(raw_path, low_memory=False, engine='c')
    print(f"  Shape originale: {df.shape}")

    df = _drop_high_missing(df, MISSING_THRESHOLD)
    df = _drop_null_critical(df, VEHICLES_CRITICAL_COLS)

    print(f"  Shape finale:    {df.shape}")

    print("  Scrittura CSV...")
    df.to_csv(VEHICLES_PROCESSED_PATH, index=False)
    print(f"  Salvato → {VEHICLES_PROCESSED_PATH}")


def process_used_cars() -> None:
    """Processa used_cars_data.csv → used_cars_data_processed.csv."""
    raw_path = RAW_DIR / "used_cars_data.csv"
    print(f"\n{'─'*60}")
    print(f"  Processing: {raw_path.name}")
    print(f"{'─'*60}")

    print("  Lettura CSV (file grande, attendere)...")
    df = pd.read_csv(raw_path, low_memory=False, engine='c')
    print(f"  Shape originale: {df.shape}")

    # 1. Rimozione colonna sp_id (non affidabile)
    if "sp_id" in df.columns:
        df = df.drop(columns=["sp_id"])
        print("  Rimossa colonna 'sp_id'.")

    # 2. Generazione ID univoco per Sorgente 2
    df = df.reset_index(drop=True)
    df["id"] = "S2_" + df.index.astype(str)
    print(f"  Generati {len(df)} ID univoci (prefisso S2_).")

    # 3. Pulizia colonne e righe
    df = _drop_high_missing(df, MISSING_THRESHOLD)
    df = _drop_null_critical(df, USED_CARS_CRITICAL_COLS)

    print(f"  Shape finale:    {df.shape}")

    print("  Scrittura CSV...")
    df.to_csv(USED_CARS_PROCESSED_PATH, index=False)
    print(f"  Salvato → {USED_CARS_PROCESSED_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    print("=" * 60)
    print("  PREPROCESSING DATASET GREZZI")
    print("=" * 60)
    print_hw_info()

    # 125+ GB di RAM e 32 core: entrambi i dataset ci stanno in memoria (~35 GB).
    # ProcessPoolExecutor → vero parallelismo (bypassa il GIL),
    # ogni processo sfrutta i core per le operazioni pandas vettorizzate.
    with ProcessPoolExecutor(max_workers=2) as pool:
        fut_v = pool.submit(process_vehicles)
        fut_u = pool.submit(process_used_cars)
        fut_v.result()
        fut_u.result()

    print(f"\n{'='*60}")
    print("  Preprocessing completato.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
