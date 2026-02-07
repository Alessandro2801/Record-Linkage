#!/usr/bin/env bash
# train_ditto.sh — Script di training Ditto.
# Accetta parametri dalla command line, con default ragionevoli.
set -euo pipefail

TASK="${1:-automotive_task}"
BATCH_SIZE="${2:-32}"
MAX_LEN="${3:-256}"
LR="${4:-3e-5}"
N_EPOCHS="${5:-7}"
LM="${6:-roberta}"
FP16="${7:-1}"
DEVICE="${8:-cuda}"
LOGDIR="${9:-checkpoints/}"
RUN_ID="${10:-42}"

# Crea la cartella se non esiste
mkdir -p "$LOGDIR"

FP16_FLAG=""
if [ "$FP16" = "1" ]; then
  FP16_FLAG="--fp16"
fi

python train_ditto.py \
  --task "$TASK" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --lr "$LR" \
  --n_epochs "$N_EPOCHS" \
  --lm "$LM" \
  $FP16_FLAG \
  --save_model \
  --logdir "$LOGDIR" \
  --device "$DEVICE" \
  --run_id "$RUN_ID"