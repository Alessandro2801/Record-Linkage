INPUT_PATH="data/er_magellan/Structured/Amazon-Google/"
OUTPUT_PATH="data/fair/Amazon-Google/"

INPUT="${INPUT_PATH}train.txt"
OUTPUT="${OUTPUT_PATH}train.txt"

DISCRIMINATOR_INPUT="${OUTPUT_PATH}train.txt"
OUTPUT_VALID="${OUTPUT_PATH}train_clean.txt"
OUTPUT_INVALID="${OUTPUT_PATH}train_reject.txt"

python mixupv2/main.py \
  --input ${INPUT} \
  --output ${OUTPUT} \
  --generations 1 \
  --alpha 0.5 \
  --alpha-delta 0.08 \
  --augmented-labels 1 \
  --seed 42 \
  --size 10000 \
  --preserve-label-ratio \
  --augmentation-budget-ratio 0.6 \
  --synthetic-ratio 0.8

python mixup/discriminator.py \
  --input ${DISCRIMINATOR_INPUT} \
  --output-valid ${OUTPUT_VALID} \
  --output-invalid ${OUTPUT_INVALID} \
  --remove-stopwords \
  --dedup-ngrams 2
