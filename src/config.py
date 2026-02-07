"""
config.py — Configurazione centralizzata per il progetto Record-Linkage.

Definisce tutti i percorsi, i parametri sperimentali e le funzioni helper
per la gestione dei file. Ogni modulo del progetto importa i path da qui.
"""

import os
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
#  PERCORSI
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ── Dati ──────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "storage"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MEDIATED_SCHEMA_DIR = DATA_DIR / "mediated_schema"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
GT_SPLITS_DIR = GROUND_TRUTH_DIR / "splits"
BLOCKING_DIR = DATA_DIR / "blocking"
DITTO_DATA_DIR = DATA_DIR / "ditto"

# ── Modelli addestrati ────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"

# ── Risultati e metriche ──────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"
BLOCKING_STATS_DIR = RESULTS_DIR / "blocking"
MODEL_RESULTS_DIR = RESULTS_DIR / "models"
PIPELINE_REPORT_PATH = RESULTS_DIR / "pipeline_report.json"

# ── Codice esterno (Ditto / FAIR-DA4ER) ──────────────────────────────────────
VENDOR_DIR = PROJECT_ROOT / "vendor"
DITTO_VENDOR_DIR = VENDOR_DIR / "FAIR-DA4ER" / "ditto"
DITTO_CHECKPOINTS_DIR = DITTO_VENDOR_DIR / "checkpoints"
DITTO_VENDOR_DATA_DIR = DITTO_VENDOR_DIR / "data" / "ditto_data"

# ═══════════════════════════════════════════════════════════════════════════════
#  FILE SPECIFICI
# ═══════════════════════════════════════════════════════════════════════════════

VEHICLES_PROCESSED_PATH = PROCESSED_DIR / "vehicles_processed.csv"
USED_CARS_PROCESSED_PATH = PROCESSED_DIR / "used_cars_data_processed.csv"
MEDIATED_SCHEMA_NORMALIZED_PATH = MEDIATED_SCHEMA_DIR / "mediated_schema_normalized.csv"
GROUND_TRUTH_PATH = GROUND_TRUTH_DIR / "ground_truth.csv"
CLEANED_DATASET_PATH = GROUND_TRUTH_DIR / "cleaned_dataset.csv"
GT_TRAIN_PATH = GT_SPLITS_DIR / "train.csv"
GT_VAL_PATH = GT_SPLITS_DIR / "val.csv"
GT_TEST_PATH = GT_SPLITS_DIR / "test.csv"

# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETRI SPERIMENTALI
# ═══════════════════════════════════════════════════════════════════════════════

GT_NEGATIVE_RATIO = 2.0
RANDOM_SEED = 42
VIN_MAX_DIFF = 3
BLOCKING_SAMPLE_SIZE = 50_000
BLOCKING_MFR_THRESHOLD = 0.95
BLOCKING_MODEL_THRESHOLD = 0.85

# ═══════════════════════════════════════════════════════════════════════════════
#  KAGGLE
# ═══════════════════════════════════════════════════════════════════════════════

KAGGLE_DATASETS = {
    "craigslist": "austinreese/craigslist-carstrucks-data",
    "used_cars": "ananaymital/us-used-cars-dataset",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  RILEVAMENTO AUTOMATICO RISORSE HW
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_hardware():
    """Rileva CPU cores, RAM disponibile e GPU."""
    n_cpu = os.cpu_count() or 1

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        ram_gb = int(line.split()[1]) / (1024 ** 2)
                        break
                else:
                    ram_gb = 8.0
        except Exception:
            ram_gb = 8.0

    gpu_available = False
    gpu_name = "N/A"
    gpu_mem_gb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        pass

    return {
        "n_cpu": n_cpu,
        "ram_gb": round(ram_gb, 1),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_mem_gb": round(gpu_mem_gb, 1),
    }


HW = _detect_hardware()

N_JOBS = max(1, HW["n_cpu"] - 1)
CSV_CHUNK_SIZE = max(50_000, int(HW["ram_gb"] * 15_000))

if HW["gpu_available"]:
    DITTO_BATCH_SIZE = min(256, max(32, int(HW["gpu_mem_gb"] * 16)))
else:
    DITTO_BATCH_SIZE = 16

MP_POOL_SIZE = max(1, HW["n_cpu"] - 1)


def print_hw_info():
    """Stampa un riepilogo delle risorse HW rilevate."""
    print(f"  CPU cores:   {HW['n_cpu']}")
    print(f"  RAM:         {HW['ram_gb']:.1f} GB")
    print(f"  N_JOBS:      {N_JOBS}")
    print(f"  Chunk CSV:   {CSV_CHUNK_SIZE:,}")
    if HW["gpu_available"]:
        print(f"  GPU:         {HW['gpu_name']} ({HW['gpu_mem_gb']:.1f} GB)")
        print(f"  Batch Ditto: {DITTO_BATCH_SIZE}")
    else:
        print("  GPU:         non disponibile")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def gt_split_path(split: str) -> Path:
    """Path di uno split della GT: storage/ground_truth/splits/{train|val|test}.csv."""
    valid = {"train", "val", "test"}
    if split not in valid:
        raise ValueError(f"split non valido: {split}. Ammessi: {sorted(valid)}")
    return GT_SPLITS_DIR / f"{split}.csv"


def blocking_strategy_dir(strategy: str) -> Path:
    """Directory di una strategia di blocking: storage/blocking/{B1|B2}/."""
    return BLOCKING_DIR / strategy


def blocking_candidates_path(strategy: str) -> Path:
    """Path legacy del CSV completo delle coppie candidate."""
    return blocking_strategy_dir(strategy) / "candidates.csv"


def blocking_split_path(strategy: str, split: str) -> Path:
    """Path legacy di uno split (train/val/test) per una strategia di blocking."""
    return blocking_strategy_dir(strategy) / f"{split}.csv"


def blocking_test_candidates_path(strategy: str) -> Path:
    """Path canonico candidate pairs sul test GT per una strategia."""
    return blocking_strategy_dir(strategy) / "test_candidates.csv"


def blocking_test_stats_path(strategy: str) -> Path:
    """Path delle statistiche blocking su test GT."""
    return BLOCKING_STATS_DIR / f"test_stats_{strategy}.json"


def ditto_data_path(strategy: str, split: str) -> Path:
    """Path di un file Ditto (.txt) per strategia e split."""
    return DITTO_DATA_DIR / strategy / f"{split}.txt"


def model_path(model_type: str, strategy: Optional[str] = None) -> Path:
    """
    Path di un modello salvato.

    Nuovo formato (globale):
      - models/recordlinkage.joblib
      - models/dedupe.pickle

    Legacy (compatibilità):
      - models/{type}_{strategy}.{ext}
    """
    ext_map = {
        "recordlinkage": "joblib",
        "dedupe": "pickle",
    }
    if model_type not in ext_map:
        raise ValueError(f"model_type non supportato: {model_type}")

    ext = ext_map[model_type]
    if strategy:
        return MODELS_DIR / f"{model_type}_{strategy}.{ext}"
    return MODELS_DIR / f"{model_type}.{ext}"


def model_meta_path(model_type: str, strategy: Optional[str] = None) -> Path:
    """Path dei metadati modello (JSON), con supporto legacy per strategia."""
    if strategy:
        return MODELS_DIR / f"{model_type}_{strategy}_meta.json"
    return MODELS_DIR / f"{model_type}_meta.json"


def ensure_dirs():
    """Crea tutte le directory necessarie se non esistono."""
    dirs = [
        RAW_DIR,
        PROCESSED_DIR,
        MEDIATED_SCHEMA_DIR,
        GT_SPLITS_DIR,
        MODELS_DIR,
        DITTO_DATA_DIR,
        BLOCKING_STATS_DIR,
        MODEL_RESULTS_DIR,
        DITTO_VENDOR_DATA_DIR,
    ]
    for strategy in ("B1", "B2"):
        dirs.append(blocking_strategy_dir(strategy))
        dirs.append(DITTO_DATA_DIR / strategy)
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
