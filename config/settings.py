#!/usr/bin/env python3
"""
settings.py
Configurazione centralizzata per il progetto Record-Linkage.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Root del progetto (directory contenente questo file config/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Dati
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MEDIATED_SCHEMA_DIR = DATA_DIR / "mediated_schema"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

# Modelli
MODELS_DIR = PROJECT_ROOT / "models"

# Output
OUTPUTS_DIR = DATA_DIR / "outputs"

# =============================================================================
# FILE PATHS
# =============================================================================

# Dataset sorgente
VEHICLES_PROCESSED_PATH = PROCESSED_DATA_DIR / "vehicles_processed.csv"
USED_CARS_PROCESSED_PATH = PROCESSED_DATA_DIR / "used_cars_data_processed.csv"

# Schema mediato
MEDIATED_SCHEMA_PATH = MEDIATED_SCHEMA_DIR / "mediated_schema.csv"
MEDIATED_SCHEMA_NORMALIZED_PATH = MEDIATED_SCHEMA_DIR / "mediated_schema_normalized.csv"

# Ground Truth
GT_TRAIN_PATH = GROUND_TRUTH_DIR / "GT_train" / "train.csv"
GT_VAL_PATH = GROUND_TRUTH_DIR / "GT_train" / "val.csv"
GT_TEST_PATH = GROUND_TRUTH_DIR / "GT_train" / "test.csv"

# Modello
RECORD_LINKAGE_MODEL_PATH = MODELS_DIR / "recordlinkage_model.joblib"

# =============================================================================
# PARAMETRI PIPELINE
# =============================================================================

# Ground Truth generation
GT_NEGATIVE_RATIO = 2.0  # Rapporto negativi/positivi

# Random seed per riproducibilità
RANDOM_SEED = 42

# VIN deduplication
VIN_MAX_DIFF = 3  # Max differenze per considerare record simili

# =============================================================================
# KAGGLE DATASETS
# =============================================================================

KAGGLE_DATASETS = {
    "craigslist": "austinreese/craigslist-carstrucks-data",
    "used_cars": "ananaymital/us-used-cars-dataset"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Crea le directory necessarie se non esistono."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MEDIATED_SCHEMA_DIR,
        GROUND_TRUTH_DIR / "GT_train",
        MODELS_DIR,
        OUTPUTS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_path(name: str) -> Path:
    """Ottiene un path per nome (per retrocompatibilità)."""
    paths = {
        "vehicles_processed": VEHICLES_PROCESSED_PATH,
        "used_cars_processed": USED_CARS_PROCESSED_PATH,
        "mediated_schema": MEDIATED_SCHEMA_PATH,
        "mediated_schema_normalized": MEDIATED_SCHEMA_NORMALIZED_PATH,
        "gt_train": GT_TRAIN_PATH,
        "gt_val": GT_VAL_PATH,
        "gt_test": GT_TEST_PATH,
        "model": RECORD_LINKAGE_MODEL_PATH
    }
    return paths.get(name)
