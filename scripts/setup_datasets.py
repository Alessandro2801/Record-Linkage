#!/usr/bin/env python3
"""
Script per scaricare e configurare i dataset per il progetto homework6.

Questo script scarica automaticamente i dataset da Kaggle e li copia nella
cartella datasets/raw/ del progetto.

Requisiti:
    - kagglehub: pip install kagglehub
    - Credenziali Kaggle configurate (vedi istruzioni sotto)

Configurazione Kaggle:
    1. Vai su https://www.kaggle.com/settings
    2. Nella sezione "API", clicca "Create New Token"
    3. Verrà scaricato un file kaggle.json
    4. Posiziona il file in ~/.kaggle/kaggle.json
    5. Imposta i permessi: chmod 600 ~/.kaggle/kaggle.json

Uso:
    python scripts/setup_datasets.py
"""

import os
import shutil
import sys
from pathlib import Path

# Verifica che kagglehub sia installato
try:
    import kagglehub
except ImportError:
    print("❌ kagglehub non è installato.")
    print("   Esegui: pip install kagglehub")
    sys.exit(1)


# Configurazione dei dataset
DATASETS = [
    {
        "name": "austinreese/craigslist-carstrucks-data",
        "description": "Craigslist Cars and Trucks Data",
        "expected_files": ["vehicles.csv"],
    },
    {
        "name": "ananaymital/us-used-cars-dataset",
        "description": "US Used Cars Dataset",
        "expected_files": ["used_cars_data.csv"],
    },
]


def get_project_root() -> Path:
    """Trova la directory root del progetto."""
    script_dir = Path(__file__).resolve().parent
    # Lo script è in scripts/, quindi il progetto è un livello sopra
    return script_dir.parent


def download_dataset(dataset_name: str, description: str) -> Path:
    """Scarica un dataset da Kaggle."""
    print(f"\n📥 Scaricando: {description}")
    print(f"   Dataset: {dataset_name}")
    
    try:
        path = kagglehub.dataset_download(dataset_name)
        print(f"   ✅ Scaricato in: {path}")
        return Path(path)
    except Exception as e:
        print(f"   ❌ Errore durante il download: {e}")
        raise


def copy_files_to_raw(source_dir: Path, target_dir: Path, expected_files: list[str]):
    """Copia i file del dataset nella cartella raw."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in expected_files:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if source_file.exists():
            print(f"   📁 Copiando {filename}...")
            # Usa shutil.copy2 per preservare i metadata
            shutil.copy2(source_file, target_file)
            
            # Mostra la dimensione del file
            size_mb = target_file.stat().st_size / (1024 * 1024)
            if size_mb > 1024:
                size_str = f"{size_mb / 1024:.2f} GB"
            else:
                size_str = f"{size_mb:.2f} MB"
            print(f"   ✅ {filename} copiato ({size_str})")
        else:
            # Cerca il file ricorsivamente
            found = list(source_dir.rglob(filename))
            if found:
                shutil.copy2(found[0], target_file)
                size_mb = target_file.stat().st_size / (1024 * 1024)
                size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.2f} MB"
                print(f"   ✅ {filename} copiato ({size_str})")
            else:
                print(f"   ⚠️  File non trovato: {filename}")


def main():
    """Funzione principale."""
    print("=" * 60)
    print("🚗 Setup Dataset - Homework 6")
    print("=" * 60)
    
    # Determina la directory del progetto
    project_root = get_project_root()
    raw_dir = project_root / "datasets" / "raw"
    
    print(f"\n📂 Directory progetto: {project_root}")
    print(f"📂 Directory dataset: {raw_dir}")
    
    # Verifica credenziali Kaggle
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n⚠️  Attenzione: ~/.kaggle/kaggle.json non trovato!")
        print("   Potrebbe essere necessario configurare le credenziali Kaggle.")
        print("   Vedi le istruzioni all'inizio di questo script.")
        print("")
        response = input("   Vuoi continuare comunque? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Scarica e copia ogni dataset
    success_count = 0
    for dataset in DATASETS:
        try:
            download_path = download_dataset(
                dataset["name"], 
                dataset["description"]
            )
            copy_files_to_raw(
                download_path, 
                raw_dir, 
                dataset["expected_files"]
            )
            success_count += 1
        except Exception as e:
            print(f"   ❌ Fallito: {e}")
            continue
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("📊 Riepilogo")
    print("=" * 60)
    print(f"   Dataset scaricati: {success_count}/{len(DATASETS)}")
    
    if raw_dir.exists():
        print(f"\n📂 File in {raw_dir}:")
        for f in raw_dir.iterdir():
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.2f} MB"
                print(f"   - {f.name}: {size_str}")
    
    if success_count == len(DATASETS):
        print("\n✅ Setup completato con successo!")
    else:
        print("\n⚠️  Alcuni dataset non sono stati scaricati correttamente.")
        sys.exit(1)


if __name__ == "__main__":
    main()
