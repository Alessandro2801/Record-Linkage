#!/usr/bin/env python3
"""
dedupe.py — Training/inferenza Dedupe 3.0 su split GT non bloccati.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.config import (
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    MP_POOL_SIZE,
    RANDOM_SEED,
    ensure_dirs,
    gt_split_path,
    model_meta_path,
    model_path,
    print_hw_info,
)

try:
    import dedupe

    DEDUPE_AVAILABLE = True
except ImportError:
    dedupe = None
    DEDUPE_AVAILABLE = False


THRESHOLDS = np.arange(0.10, 1.00, 0.05)

COLS_STRING = [
    "location",
    "manufacturer",
    "model",
    "cylinders",
    "year",
    "latitude",
    "longitude",
    "body_type",
    "main_color",
]
COLS_NUMERIC = ["price", "mileage"]
COLS_CATEGORICAL = ["fuel_type", "traction", "transmission"]
ALL_COLS = COLS_STRING + COLS_NUMERIC + COLS_CATEGORICAL


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_unified = pd.read_csv(
        MEDIATED_SCHEMA_NORMALIZED_PATH,
        dtype={"id_source_vehicles": "object", "id_source_used_cars": "object"},
        low_memory=False,
    )
    gt_train = pd.read_csv(gt_split_path("train"))
    gt_val = pd.read_csv(gt_split_path("val"))
    gt_test = pd.read_csv(gt_split_path("test"))

    for col in ("vin", "description"):
        if col in df_unified.columns:
            df_unified.drop(columns=[col], inplace=True)

    for frame in (gt_train, gt_val, gt_test):
        for col in ("description_A", "description_B"):
            if col in frame.columns:
                frame.drop(columns=[col], inplace=True)
        frame["id_A"] = frame["id_A"].astype(str)
        frame["id_B"] = frame["id_B"].astype(str)

    return df_unified, gt_train, gt_val, gt_test


def to_clean_string(val):
    if pd.isnull(val):
        return ""
    if isinstance(val, (float, int)):
        try:
            if val == int(val):
                return str(int(val))
        except Exception:
            return str(val).strip()
    return str(val).strip()


def prepare_data_for_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["id_unificato"] = df["id_source_vehicles"].fillna(df["id_source_used_cars"]).astype(str).str.strip()
    df = df.set_index("id_unificato")

    for col in ("year", "latitude", "longitude"):
        if col in df.columns:
            df[col] = df[col].apply(to_clean_string)

    for col in COLS_STRING:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(["nan", "None", "NaN", "<NA>"], "")

    for col in COLS_NUMERIC + COLS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].replace({np.nan: None})

    return df


def create_dedupe_dict(df: pd.DataFrame, ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
    ids_set = set(str(i) for i in ids)
    subset = df.loc[df.index.astype(str).isin(ids_set)]

    records: Dict[str, Dict[str, object]] = {}
    for idx, row in subset.iterrows():
        item: Dict[str, object] = {}
        for col in ALL_COLS:
            if col not in subset.columns:
                item[col] = None
                continue

            value = row[col]
            if col in COLS_NUMERIC:
                if pd.isna(value):
                    item[col] = None
                else:
                    try:
                        item[col] = float(value)
                    except (TypeError, ValueError):
                        item[col] = None
            elif col in COLS_CATEGORICAL:
                item[col] = None if pd.isna(value) or str(value).lower() == "nan" else str(value)
            else:
                if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none"}:
                    item[col] = None
                else:
                    item[col] = str(value)
        records[str(idx)] = item

    return records


def get_dedupe_fields():
    if not DEDUPE_AVAILABLE:
        return []

    return [
        dedupe.variables.String("manufacturer", has_missing=True),
        dedupe.variables.String("model", has_missing=True),
        dedupe.variables.String("year", has_missing=True),
        dedupe.variables.String("location", has_missing=True),
        dedupe.variables.String("cylinders", has_missing=True),
        dedupe.variables.String("body_type", has_missing=True),
        dedupe.variables.String("main_color", has_missing=True),
        dedupe.variables.String("latitude", has_missing=True),
        dedupe.variables.String("longitude", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),
        dedupe.variables.Categorical(
            "transmission",
            categories=["other", "automatic", "manual", "cvt", "dual clutch"],
            has_missing=True,
        ),
        dedupe.variables.Categorical(
            "fuel_type",
            categories=[
                "gas",
                "other",
                "diesel",
                "hybrid",
                "electric",
                "biodiesel",
                "flex fuel vehicle",
                "compressed natural gas",
                "propane",
            ],
            has_missing=True,
        ),
        dedupe.variables.Categorical(
            "traction",
            categories=["rwd", "4wd", "fwd", "awd", "4x2"],
            has_missing=True,
        ),
    ]


def prepare_training_pairs(gt_df: pd.DataFrame, data_dict: Dict[str, Dict[str, object]]):
    matches: List[Tuple[dict, dict]] = []
    distinct: List[Tuple[dict, dict]] = []

    for _, row in gt_df.iterrows():
        id_a = str(row["id_A"])
        id_b = str(row["id_B"])
        if id_a not in data_dict or id_b not in data_dict:
            continue

        pair = (data_dict[id_a], data_dict[id_b])
        if int(row["label"]) == 1:
            matches.append(pair)
        else:
            distinct.append(pair)

    return matches, distinct


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


def score_pairs(linker, pairs_df: pd.DataFrame, data_dict: Dict[str, Dict[str, object]]) -> np.ndarray:
    """Ritorna score allineati all'ordine di pairs_df (default 0 per ID mancanti)."""
    if pairs_df.empty:
        return np.array([], dtype=float)

    scored_input = []
    valid_positions: List[int] = []

    for pos, (_, row) in enumerate(pairs_df.iterrows()):
        id_a = str(row["id_A"])
        id_b = str(row["id_B"])
        if id_a in data_dict and id_b in data_dict:
            scored_input.append(((id_a, data_dict[id_a]), (id_b, data_dict[id_b])))
            valid_positions.append(pos)

    scores = np.zeros(len(pairs_df), dtype=float)
    if scored_input:
        scored = linker.score(scored_input)
        scores[np.array(valid_positions, dtype=int)] = np.asarray(scored["score"], dtype=float)
    return scores


def infer_pairs(
    linker,
    pairs_df: pd.DataFrame,
    df_prepared: pd.DataFrame,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ids = set(pairs_df["id_A"].astype(str)) | set(pairs_df["id_B"].astype(str))
    data_dict = create_dedupe_dict(df_prepared, ids)

    probs = score_pairs(linker, pairs_df, data_dict)
    preds = (probs >= threshold).astype(int)
    return preds, probs


def save_meta(best_threshold: float, best_val_f1: float) -> None:
    payload = {
        "model": "dedupe",
        "best_threshold": round(float(best_threshold), 4),
        "best_val_f1": round(float(best_val_f1), 6),
    }
    output = model_meta_path("dedupe")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"Metadati salvati in: {output}")


def load_linker_and_meta():
    model_file = model_path("dedupe")
    meta_file = model_meta_path("dedupe")

    if not model_file.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadati non trovati: {meta_file}")

    with open(model_file, "rb") as f:
        linker = dedupe.StaticDedupe(f)
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return linker, meta


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def train_model(sample_size: int = 5000):
    if not DEDUPE_AVAILABLE:
        raise SystemExit("dedupe non disponibile. Installa la libreria dedupe prima del training.")

    ensure_dirs()
    print_hw_info()

    df_unified, gt_train, gt_val, gt_test = load_data()
    df_prepared = prepare_data_for_dedupe(df_unified)
    df_prepared.index = df_prepared.index.astype(str)

    ids_train_val = (
        set(gt_train["id_A"].astype(str))
        | set(gt_train["id_B"].astype(str))
        | set(gt_val["id_A"].astype(str))
        | set(gt_val["id_B"].astype(str))
    )

    remaining_ids = [idx for idx in df_prepared.index if idx not in ids_train_val]
    if remaining_ids:
        rng = np.random.RandomState(RANDOM_SEED)
        extra_size = min(sample_size, len(remaining_ids))
        extra_ids = set(rng.choice(remaining_ids, size=extra_size, replace=False).tolist())
    else:
        extra_ids = set()

    ids_for_training = ids_train_val | extra_ids
    data_dict = create_dedupe_dict(df_prepared, ids_for_training)

    matches, distinct = prepare_training_pairs(gt_train, data_dict)
    if not matches:
        raise RuntimeError("Nessuna coppia positiva disponibile per training Dedupe")

    fields = get_dedupe_fields()
    deduper = dedupe.Dedupe(fields, num_cores=MP_POOL_SIZE, in_memory=True)
    deduper.mark_pairs({"match": matches, "distinct": distinct})
    deduper.prepare_training(
        data=data_dict,
        sample_size=min(sample_size, len(data_dict)),
        blocked_proportion=0.2,
    )
    # Dedupe usa una LogisticRegression interna (max_iter=100) che puo emettere
    # numerosi ConvergenceWarning pur completando correttamente il training.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        deduper.train()

    model_output = model_path("dedupe")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as f:
        deduper.write_settings(f)
    print(f"Modello salvato in: {model_output}")

    with open(model_output, "rb") as f:
        linker = dedupe.StaticDedupe(f)

    val_pred, val_prob = infer_pairs(linker, gt_val, df_prepared, threshold=0.5)
    val_true = gt_val["label"].astype(int).to_numpy()
    best_threshold, best_val_f1 = _best_threshold(val_true, val_prob)

    save_meta(best_threshold, best_val_f1)

    test_pred, _ = infer_pairs(linker, gt_test, df_prepared, threshold=best_threshold)
    test_true = gt_test["label"].astype(int).to_numpy()
    metrics = evaluate_predictions(test_true, test_pred)

    print(f"Best threshold val: {best_threshold:.2f} (F1={best_val_f1:.4f})")
    print(f"Test metrics: {metrics}")


def evaluate_saved_model_on_gt_test() -> Dict[str, float]:
    if not DEDUPE_AVAILABLE:
        raise SystemExit("dedupe non disponibile. Installa la libreria dedupe prima della valutazione.")

    ensure_dirs()

    linker, meta = load_linker_and_meta()
    threshold = float(meta.get("best_threshold", 0.5))

    df_unified, _, _, gt_test = load_data()
    df_prepared = prepare_data_for_dedupe(df_unified)
    df_prepared.index = df_prepared.index.astype(str)

    pred, _ = infer_pairs(linker, gt_test, df_prepared, threshold=threshold)
    y_true = gt_test["label"].astype(int).to_numpy()

    metrics = evaluate_predictions(y_true, pred)
    metrics["threshold"] = round(float(threshold), 2)
    print(f"Test metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Training/eval Dedupe")
    parser.add_argument("--train", action="store_true", help="Esegue training + tuning")
    parser.add_argument("--evaluate", action="store_true", help="Valuta modello salvato su GT test")
    parser.add_argument("--sample-size", type=int, default=5000, help="Campione extra record per training")
    args = parser.parse_args()

    if args.train:
        train_model(sample_size=args.sample_size)
    elif args.evaluate:
        evaluate_saved_model_on_gt_test()
    else:
        raise SystemExit("Specifica --train o --evaluate")


if __name__ == "__main__":
    main()
