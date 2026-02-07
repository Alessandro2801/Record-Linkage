#!/usr/bin/env python3
"""
ditto.py — Integrazione Ditto per training/tuning/inferenza su split GT.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.blocking.ditto_format import dataframe_to_ditto_lines
from src.config import (
    DITTO_BATCH_SIZE,
    DITTO_CHECKPOINTS_DIR,
    DITTO_FP16,
    DITTO_LM,
    DITTO_LR,
    DITTO_MAX_LEN,
    DITTO_N_EPOCHS,
    DITTO_SCRIPT,
    DITTO_TASK,
    DITTO_TRAIN_BATCH_SIZE,
    DITTO_VENDOR_DATA_DIR,
    DITTO_VENDOR_DIR,
    RANDOM_SEED,
    ensure_dirs,
    gt_split_path,
    model_meta_path,
    print_hw_info,
)

THRESHOLDS = np.arange(0.10, 1.00, 0.05)
DEFAULT_CHECKPOINT = DITTO_CHECKPOINTS_DIR / DITTO_TASK / "model.pt"


def _import_ditto_runtime():
    import importlib

    if str(DITTO_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(DITTO_VENDOR_DIR))

    ditto_module = importlib.import_module("ditto_light.ditto")
    dataset_module = importlib.import_module("ditto_light.dataset")

    return ditto_module.DittoModel, dataset_module.DittoDataset


def _select_device(preferred: str = "auto") -> str:
    import torch

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Iterable[float] = THRESHOLDS):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def load_gt_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_train = pd.read_csv(gt_split_path("train"))
    gt_val = pd.read_csv(gt_split_path("val"))
    gt_test = pd.read_csv(gt_split_path("test"))
    return gt_train, gt_val, gt_test


def prepare_ditto_training_files() -> None:
    from src.blocking.ditto_format import convert_gt_splits

    convert_gt_splits()


def run_ditto_training(
    lm: str = DITTO_LM,
    max_len: int = DITTO_MAX_LEN,
    lr: float = DITTO_LR,
    n_epochs: int = DITTO_N_EPOCHS,
    batch_size: int = DITTO_TRAIN_BATCH_SIZE,
    device: str = "auto",
    task: str = DITTO_TASK,
    fp16: bool = DITTO_FP16,
) -> None:
    """Esegue il training Ditto invocando train_ditto.sh via subprocess
    con cwd impostata a DITTO_VENDOR_DIR, così che i path relativi
    (configs.json, data/, checkpoints/) funzionino correttamente."""
    selected_device = _select_device(device)

    # Verifica che i file di training esistano
    for name in ("train.txt", "val.txt", "test.txt"):
        p = DITTO_VENDOR_DATA_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"File Ditto non trovato: {p}")

    cmd = [
        "bash", str(DITTO_SCRIPT),
        task,                       # $1 TASK
        str(batch_size),            # $2 BATCH_SIZE
        str(max_len),               # $3 MAX_LEN
        str(lr),                    # $4 LR
        str(n_epochs),              # $5 N_EPOCHS
        lm,                         # $6 LM
        "1" if fp16 else "0",       # $7 FP16
        selected_device,            # $8 DEVICE
        str(DITTO_CHECKPOINTS_DIR) + "/",  # $9 LOGDIR (path assoluto)
        str(RANDOM_SEED),           # $10 RUN_ID
    ]

    print(f"Esecuzione training Ditto: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(DITTO_VENDOR_DIR), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training Ditto fallito (exit code {result.returncode})")


def load_model_and_meta(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    lm: str = "roberta",
    device: str = "auto",
):
    model, selected_device = load_model(checkpoint_path=checkpoint_path, lm=lm, device=device)
    meta_file = model_meta_path("ditto")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadati Ditto non trovati: {meta_file}")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta, selected_device


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    lm: str = "roberta",
    device: str = "auto",
):
    import torch

    DittoModel, _ = _import_ditto_runtime()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint Ditto non trovato: {checkpoint_path}")

    selected_device = _select_device(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=selected_device)

    model = DittoModel(device=selected_device, lm=lm)
    model.load_state_dict(checkpoint["model"])
    model.to(selected_device)
    model.eval()
    return model, selected_device


def infer_pairs(
    model,
    pairs_df: pd.DataFrame,
    threshold: float,
    device: str,
    lm: str = "roberta",
    max_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    _, DittoDataset = _import_ditto_runtime()

    if pairs_df.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    work_df = pairs_df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = 0

    lines = dataframe_to_ditto_lines(work_df)
    dataset = DittoDataset(lines, lm=lm, max_len=max_len, da=None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=DITTO_BATCH_SIZE,
        shuffle=False,
        collate_fn=DittoDataset.pad,
        num_workers=0,
    )

    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, _ = batch
                x = x.to(device)
                logits = model(x)
            else:
                x1, x2, _ = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                logits = model(x1, x2)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.detach().cpu().numpy().tolist())

    prob_array = np.asarray(probs, dtype=float)
    pred_array = (prob_array >= threshold).astype(int)
    return pred_array, prob_array


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def save_meta(
    best_threshold: float,
    best_val_f1: float,
    hyperparams: Dict[str, object],
    checkpoint_path: Path,
) -> None:
    payload = {
        "model": "ditto",
        "best_threshold": round(float(best_threshold), 4),
        "best_val_f1": round(float(best_val_f1), 6),
        "checkpoint_path": str(checkpoint_path),
        "hyperparams": hyperparams,
    }
    output = model_meta_path("ditto")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"Metadati salvati in: {output}")


def train_model(
    lm: str = DITTO_LM,
    max_len: int = DITTO_MAX_LEN,
    lr: float = DITTO_LR,
    n_epochs: int = DITTO_N_EPOCHS,
    batch_size: int = DITTO_TRAIN_BATCH_SIZE,
    device: str = "auto",
    task: str = DITTO_TASK,
) -> None:
    ensure_dirs()
    print_hw_info()

    prepare_ditto_training_files()

    run_ditto_training(
        lm=lm,
        max_len=max_len,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        task=task,
    )

    gt_train, gt_val, gt_test = load_gt_splits()
    del gt_train  # usato solo nel training diretto

    checkpoint_path = DITTO_CHECKPOINTS_DIR / task / "model.pt"
    selected_device = _select_device(device)

    # tuning threshold su val
    model, selected_device = load_model(
        checkpoint_path=checkpoint_path,
        lm=lm,
        device=selected_device,
    )

    _, val_prob = infer_pairs(model, gt_val, threshold=0.5, device=selected_device, lm=lm, max_len=max_len)
    val_true = gt_val["label"].astype(int).to_numpy()
    best_threshold, best_val_f1 = _best_threshold(val_true, val_prob)

    hyperparams = {
        "task": task,
        "lm": lm,
        "max_len": max_len,
        "lr": lr,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "device": selected_device,
    }
    save_meta(best_threshold, best_val_f1, hyperparams, checkpoint_path)

    # valutazione rapida su test GT
    test_pred, _ = infer_pairs(model, gt_test, threshold=best_threshold, device=selected_device, lm=lm, max_len=max_len)
    test_true = gt_test["label"].astype(int).to_numpy()
    metrics = evaluate_predictions(test_true, test_pred)

    print(f"Best threshold val: {best_threshold:.2f} (F1={best_val_f1:.4f})")
    print(f"Test metrics: {metrics}")


def evaluate_saved_model(device: str = "auto") -> Dict[str, float]:
    ensure_dirs()

    gt_train, gt_val, gt_test = load_gt_splits()
    del gt_train, gt_val

    model, meta, selected_device = load_model_and_meta(device=device)
    threshold = float(meta.get("best_threshold", 0.5))

    hyperparams = meta.get("hyperparams", {})
    lm = str(hyperparams.get("lm", "roberta"))
    max_len = int(hyperparams.get("max_len", 256))

    pred, _ = infer_pairs(model, gt_test, threshold=threshold, device=selected_device, lm=lm, max_len=max_len)
    y_true = gt_test["label"].astype(int).to_numpy()

    metrics = evaluate_predictions(y_true, pred)
    metrics["threshold"] = round(float(threshold), 2)
    print(f"Test metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Training/eval Ditto")
    parser.add_argument("--train", action="store_true", help="Esegue training + tuning")
    parser.add_argument("--evaluate", action="store_true", help="Valuta checkpoint salvato")
    parser.add_argument("--task", type=str, default=DITTO_TASK)
    parser.add_argument("--lm", type=str, default=DITTO_LM)
    parser.add_argument("--max-len", type=int, default=DITTO_MAX_LEN)
    parser.add_argument("--lr", type=float, default=DITTO_LR)
    parser.add_argument("--n-epochs", type=int, default=DITTO_N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DITTO_TRAIN_BATCH_SIZE)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.train:
        train_model(
            lm=args.lm,
            max_len=args.max_len,
            lr=args.lr,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=args.device,
            task=args.task,
        )
    elif args.evaluate:
        evaluate_saved_model(device=args.device)
    else:
        raise SystemExit("Specifica --train o --evaluate")


if __name__ == "__main__":
    main()
