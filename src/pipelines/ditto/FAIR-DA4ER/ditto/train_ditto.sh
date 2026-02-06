OUTPUT_PATH="data/fair/Abt-Buy/"

python train_ditto.py \
  --task DA_iTunes-Amazon \
  --batch_size 64 \
  --run_id 11 \
  --logdir ${OUTPUT_PATH}log \
  --lm roberta \
  --finetuning \
  --lr 1.5e-5 \
  --n_epochs 20 \
  --device cuda \
