#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Dataset & model configuration  (edit these three lines if needed)
# ---------------------------------------------------------------------------
DIR_NAME="fair/iTunes-Amazon"                  # sub-folder under data/

# ------------------------------------------------------------------------------
# Model name (Hugging Face model identifier)
# ------------------------------------------------------------------------------
MODEL_NAME="google/flan-t5-large"            # e.g. facebook/bart-large, google/t5-large
SEED=1                                      # for reproducibility

# ---------------------------------------------------------------------------
# Derived paths (usually no need to touch these)
# ---------------------------------------------------------------------------
INPUT_PATH="data/${DIR_NAME}"
OUTPUT_PATH="data/${DIR_NAME}"
INPUT_FILE="${INPUT_PATH}/train.txt"
OUTPUT_FILE="${OUTPUT_PATH}/train_DA.txt"

# Make sure the output directory exists
mkdir -p "${OUTPUT_PATH}"

# ---------------------------------------------------------------------------
# Launch the mix-up generator
# ---------------------------------------------------------------------------
python mixup/ditto_dataset_generator.py \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --model "${MODEL_NAME}" \
  --seed "${SEED}" \
  --generations 3 \
  --alpha 0.5 \
  --alpha-delta 0.08 \
  --preserve-label-ratio \
  --augmentation-budget-ratio 0.5 \
  --num-beams 4 \
  --batch-size 32 \
