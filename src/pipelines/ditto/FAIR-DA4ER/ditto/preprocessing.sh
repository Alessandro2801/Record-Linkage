NAME="Abt-Buy"  # Change this to your dataset name
DIR="fair/${NAME}"

DATA_DIR="data/${DIR}"
INPUT_PATH="${DATA_DIR}/train_DA.txt"
OUTPUT_PATH="${DATA_DIR}/"
ORIGINAL_INPUT_PATH="${DATA_DIR}/train.txt"

# Ensure output directory exists
mkdir -p "${OUTPUT_PATH}"


python mixup/discriminator.py \
    --input ${INPUT_PATH} \
    --original-input ${ORIGINAL_INPUT_PATH} \
    --output-valid  ${OUTPUT_PATH}train_DA_valid.txt \
    --output-invalid ${OUTPUT_PATH}train_DA_invalid.txt \
    --delta 2.5 \
    --max-length auto \
    --auto-scope label1 \
    --max-dup-ngrams auto \
    --min-tfidf auto \
    --min-dict-hit-rate auto \
    --min-zipf auto