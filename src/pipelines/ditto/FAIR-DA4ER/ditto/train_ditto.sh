# Crea la cartella se non esiste
mkdir -p checkpoints

python train_ditto.py \
  --task automotive_task \
  --batch_size 32 \
  --max_len 256 \
  --lr 3e-5 \
  --n_epochs 15 \
  --lm roberta \
  --fp16 \
  --save_model \
  --logdir checkpoints/ \
  --device cuda