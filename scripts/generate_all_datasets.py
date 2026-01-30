#!/usr/bin/env python3
"""
Script principale per generare tutti i dataset necessari all'addestramento dei modelli.

Questo script orchestra l'intera pipeline:
1. Scarica i dataset raw da Kaggle (setup_datasets.py)
2. Preprocessa vehicles.csv (preprocess_vehicles.py)
3. Preprocessa used_cars_data.csv (preprocess_used_cars.py)
4. Genera lo schema mediato normalizzato (generate_mediated_schema.py)
5. Genera la Ground Truth con split train/val/test (generate_ground_truth.py)

Uso:
    python scripts/generate_all_datasets.py

Opzioni:
    --skip-download     Salta il download da Kaggle (usa dati già scaricati)
    --skip-preprocessing Salta il preprocessing (usa dati già processati)
"""

import subprocess
import sys
from pathlib import Path
import argparse


def get_project_root() -> Path:
    """Trova la directory root del progetto."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def run_script(script_name: str, description: str) -> bool:
    """Esegue uno script Python e restituisce True se ha successo."""
    project_root = get_project_root()
    script_path = project_root / "scripts" / script_name
    
    print(f"\n{'='*60}")
    print(f"📌 Step: {description}")
    print(f"   Script: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore durante l'esecuzione di {script_name}")
        print(f"   Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script non trovato: {script_path}")
        return False


def check_file_exists(relative_path: str) -> bool:
    """Verifica se un file esiste nel progetto."""
    project_root = get_project_root()
    full_path = project_root / relative_path
    exists = full_path.exists()
    if exists:
        size_mb = full_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.2f} MB"
        print(f"   ✓ {relative_path} ({size_str})")
    else:
        print(f"   ✗ {relative_path} (non trovato)")
    return exists


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Genera tutti i dataset necessari all'addestramento dei modelli."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Salta il download da Kaggle (usa dati già scaricati)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Salta il preprocessing (usa dati già processati)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 PIPELINE GENERAZIONE DATASET - Homework 6")
    print("=" * 60)
    print("""
Pipeline completa:
  1. Download dataset raw da Kaggle
  2. Preprocessing vehicles.csv
  3. Preprocessing used_cars_data.csv
  4. Generazione schema mediato
  5. Generazione Ground Truth e split
""")
    
    # Step 1: Download dataset da Kaggle
    if not args.skip_download:
        if not run_script("setup_datasets.py", "Download dataset da Kaggle"):
            print("\n⚠️  Download fallito. Continuare solo se i dati raw esistono già.")
    else:
        print("\n⏭️  Step 1 saltato (--skip-download)")
    
    # Verifica esistenza dati raw
    print("\n📂 Verifica dati raw...")
    raw_vehicles = check_file_exists("datasets/raw/vehicles.csv")
    raw_used_cars = check_file_exists("datasets/raw/used_cars_data.csv")
    
    if not (raw_vehicles and raw_used_cars):
        print("\n❌ Dati raw mancanti. Esegui senza --skip-download.")
        return 1
    
    # Step 2 & 3: Preprocessing
    if not args.skip_preprocessing:
        if not run_script("preprocess_vehicles.py", "Preprocessing dataset Vehicles"):
            return 1
        
        if not run_script("preprocess_used_cars.py", "Preprocessing dataset Used Cars"):
            return 1
    else:
        print("\n⏭️  Step 2 e 3 saltati (--skip-preprocessing)")
    
    # Verifica esistenza dati processati
    print("\n📂 Verifica dati processati...")
    proc_vehicles = check_file_exists("datasets/processed/vehicles_processed.csv")
    proc_used_cars = check_file_exists("datasets/processed/used_cars_data_processed.csv")
    
    if not (proc_vehicles and proc_used_cars):
        print("\n❌ Dati processati mancanti. Esegui senza --skip-preprocessing.")
        return 1
    
    # Step 4: Generazione schema mediato
    if not run_script("generate_mediated_schema.py", "Generazione Schema Mediato"):
        return 1
    
    # Step 5: Generazione Ground Truth
    if not run_script("generate_ground_truth.py", "Generazione Ground Truth"):
        return 1
    
    # Riepilogo finale
    print("\n" + "=" * 60)
    print("📊 RIEPILOGO FINALE")
    print("=" * 60)
    
    print("\n📂 Dataset generati:")
    check_file_exists("datasets/raw/vehicles.csv")
    check_file_exists("datasets/raw/used_cars_data.csv")
    check_file_exists("datasets/processed/vehicles_processed.csv")
    check_file_exists("datasets/processed/used_cars_data_processed.csv")
    check_file_exists("datasets/mediated_schema/mediated_schema_normalized.csv")
    check_file_exists("datasets/ground_truth/ground_truth.csv")
    check_file_exists("datasets/ground_truth/GT_eval.csv")
    check_file_exists("datasets/ground_truth/GT_train/train.csv")
    check_file_exists("datasets/ground_truth/GT_train/val.csv")
    check_file_exists("datasets/ground_truth/GT_train/test.csv")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETATA CON SUCCESSO!")
    print("=" * 60)
    print("""
I seguenti dataset sono pronti per l'addestramento:
  - datasets/ground_truth/GT_train/train.csv
  - datasets/ground_truth/GT_train/val.csv
  - datasets/ground_truth/GT_train/test.csv
  - datasets/ground_truth/GT_eval.csv
""")
    
    return 0


if __name__ == "__main__":
    exit(main())
