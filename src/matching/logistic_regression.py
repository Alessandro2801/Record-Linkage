#!/usr/bin/env python3
"""
logistic_regression.py — Training/inferenza Record Linkage con Logistic Regression.

Workflow:
    1. Train su GT train (no blocking)
    2. Tuning iperparametri + threshold su GT val (no blocking)
    3. Salvataggio modello globale e metadati
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import recordlinkage as rl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    RANDOM_SEED,
    ensure_dirs,
    gt_split_path,
    model_meta_path,
    model_path,
    print_hw_info,
)

THRESHOLDS = np.arange(0.10, 1.00, 0.05)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carica dataset mediato e split GT (train/val/test)."""
    df_unified = pd.read_csv(
        MEDIATED_SCHEMA_NORMALIZED_PATH,
        dtype={"id_source_vehicles": "object", "id_source_used_cars": "object"},
        low_memory=False,
    )

    gt_train = pd.read_csv(gt_split_path("train"))
    gt_val = pd.read_csv(gt_split_path("val"))
    gt_test = pd.read_csv(gt_split_path("test"))

    for col in ("description", "vin"):
        if col in df_unified.columns:
            df_unified.drop(columns=[col], inplace=True)

    for frame in (gt_train, gt_val, gt_test):
        for col in ("description_A", "description_B"):
            if col in frame.columns:
                frame.drop(columns=[col], inplace=True)
        frame["id_A"] = frame["id_A"].astype(str)
        frame["id_B"] = frame["id_B"].astype(str)

    df_unified["id_unificato"] = (
        df_unified["id_source_vehicles"].fillna(df_unified["id_source_used_cars"]).astype(str).str.strip()
    )
    df_unified = df_unified.set_index("id_unificato")

    return df_unified, gt_train, gt_val, gt_test


def setup_comparator() -> rl.Compare:
    compare = rl.Compare()

    compare.string("manufacturer", "manufacturer", method="jarowinkler", threshold=0.85, label="manufacturer")
    compare.string("model", "model", method="jarowinkler", threshold=0.85, label="model")
    compare.string("location", "location", method="jarowinkler", threshold=0.85, label="location")
    compare.string("cylinders", "cylinders", method="jarowinkler", threshold=0.70, label="cylinders")

    compare.exact("year", "year", label="year")
    compare.exact("fuel_type", "fuel_type", label="fuel_type")
    compare.exact("traction", "traction", label="traction")
    compare.exact("body_type", "body_type", label="body_type")
    compare.exact("main_color", "main_color", label="main_color")
    compare.exact("transmission", "transmission", label="transmission")

    compare.numeric("price", "price", method="gauss", offset=500, scale=2000, label="price")
    compare.numeric("mileage", "mileage", method="gauss", offset=1000, scale=10000, label="mileage")
    compare.numeric("latitude", "latitude", method="gauss", offset=0.01, scale=0.1, label="lat")
    compare.numeric("longitude", "longitude", method="gauss", offset=0.01, scale=0.1, label="lon")

    return compare


def _build_pairs_index(gt_df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([gt_df["id_A"].astype(str), gt_df["id_B"].astype(str)])


def compute_features(compare: rl.Compare, pairs: pd.MultiIndex, df_unified: pd.DataFrame) -> pd.DataFrame:
    return compare.compute(pairs, df_unified, df_unified)


def compute_features_for_gt(
    compare: rl.Compare,
    df_unified: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    pairs = _build_pairs_index(gt_df)
    X = compute_features(compare, pairs, df_unified)
    y = gt_df["label"].astype(int).to_numpy()
    return X, y


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Iterable[float] = THRESHOLDS):
    best_threshold = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_prec = precision_score(y_true, pred, zero_division=0)
            best_rec = recall_score(y_true, pred, zero_division=0)

    return best_threshold, best_f1, best_prec, best_rec


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
):
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    class_weights = [None, "balanced"]

    best_model = None
    best_params: Dict[str, object] = {}
    best_threshold = 0.5
    best_val_f1 = -1.0

    print("\nTuning Logistic Regression su GT val...")
    for c in c_values:
        for class_weight in class_weights:
            model = LogisticRegression(
                C=c,
                class_weight=class_weight,
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
            model.fit(X_train, y_train)

            val_prob = model.predict_proba(X_val)[:, 1]
            threshold, f1, prec, rec = find_best_threshold(y_val, val_prob)

            print(
                f"  C={c:<7} class_weight={str(class_weight):<8} "
                f"th={threshold:.2f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}"
            )

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model = model
                best_threshold = threshold
                best_params = {
                    "C": c,
                    "class_weight": class_weight,
                }

    if best_model is None:
        raise RuntimeError("Nessun modello valido trovato durante il tuning")

    return best_model, best_params, best_threshold, best_val_f1


def infer_pairs(
    model: LogisticRegression,
    compare: rl.Compare,
    df_unified: pd.DataFrame,
    pairs_df: pd.DataFrame,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inferenza su DataFrame di coppie. Restituisce predizioni e probabilita."""
    if pairs_df.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    ids_set = set(df_unified.index.astype(str))
    mask_valid = pairs_df["id_A"].astype(str).isin(ids_set) & pairs_df["id_B"].astype(str).isin(ids_set)

    pred = np.zeros(len(pairs_df), dtype=int)
    prob = np.zeros(len(pairs_df), dtype=float)

    if not mask_valid.any():
        return pred, prob

    valid_pairs = pairs_df.loc[mask_valid, ["id_A", "id_B"]].astype(str)
    pair_index = pd.MultiIndex.from_arrays([valid_pairs["id_A"], valid_pairs["id_B"]])
    X = compute_features(compare, pair_index, df_unified)
    valid_prob = model.predict_proba(X)[:, 1]
    valid_pred = (valid_prob >= threshold).astype(int)

    prob[mask_valid.to_numpy()] = valid_prob
    pred[mask_valid.to_numpy()] = valid_pred
    return pred, prob


def evaluate_on_gt_test(
    model: LogisticRegression,
    threshold: float,
    compare: rl.Compare,
    df_unified: pd.DataFrame,
    gt_test: pd.DataFrame,
) -> Dict[str, float]:
    pred, _ = infer_pairs(model, compare, df_unified, gt_test, threshold)
    y_true = gt_test["label"].astype(int).to_numpy()

    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": round(float(threshold), 2),
    }
    print(f"Test metrics: {metrics}")
    return metrics


def save_model_and_meta(
    model: LogisticRegression,
    best_params: Dict[str, object],
    best_threshold: float,
    best_val_f1: float,
) -> None:
    model_output = model_path("recordlinkage")
    meta_output = model_meta_path("recordlinkage")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)

    payload = {
        "model": "recordlinkage",
        "best_params": best_params,
        "best_threshold": round(float(best_threshold), 4),
        "best_val_f1": round(float(best_val_f1), 6),
    }
    with open(meta_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Modello salvato in: {model_output}")
    print(f"Metadati salvati in: {meta_output}")


def load_model_and_meta():
    model_file = model_path("recordlinkage")
    meta_file = model_meta_path("recordlinkage")

    if not model_file.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadati non trovati: {meta_file}")

    model = joblib.load(model_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def train_model() -> None:
    ensure_dirs()
    print_hw_info()

    df_unified, gt_train, gt_val, gt_test = load_data()
    compare = setup_comparator()

    X_train, y_train = compute_features_for_gt(compare, df_unified, gt_train)
    X_val, y_val = compute_features_for_gt(compare, df_unified, gt_val)

    best_model, best_params, best_threshold, best_val_f1 = tune_hyperparameters(X_train, y_train, X_val, y_val)
    save_model_and_meta(best_model, best_params, best_threshold, best_val_f1)

    evaluate_on_gt_test(best_model, best_threshold, compare, df_unified, gt_test)


def evaluate_saved_model() -> None:
    ensure_dirs()
    print_hw_info()

    model, meta = load_model_and_meta()
    threshold = float(meta.get("best_threshold", 0.5))

    df_unified, _, _, gt_test = load_data()
    compare = setup_comparator()
    evaluate_on_gt_test(model, threshold, compare, df_unified, gt_test)


def main():
    parser = argparse.ArgumentParser(description="Training/eval Logistic Regression")
    parser.add_argument("--train", action="store_true", help="Esegue training + tuning")
    parser.add_argument("--evaluate", action="store_true", help="Valuta modello salvato su GT test")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_saved_model()
    else:
        raise SystemExit("Specifica --train o --evaluate")


if __name__ == "__main__":
    main()
