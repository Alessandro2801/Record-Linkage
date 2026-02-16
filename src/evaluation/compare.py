#!/usr/bin/env python3
"""
compare.py — Valutazione comparativa delle 6 pipeline (B1/B2 x RL/Dedupe/Ditto).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    PIPELINE_REPORT_PATH,
    blocking_test_candidates_path,
    blocking_test_stats_path,
    ensure_dirs,
    gt_split_path,
    print_hw_info,
)


def canonical_pair(id_a, id_b) -> Tuple[str, str]:
    a = str(id_a)
    b = str(id_b)
    return (a, b) if a <= b else (b, a)


def build_global_predictions(
    gt_test: pd.DataFrame,
    candidates_df: pd.DataFrame,
    candidate_pred: np.ndarray,
) -> np.ndarray:
    """
    Ricostruisce predizioni globali su tutto il test:
      - default 0 su tutte le coppie
      - overwrite predizioni sulle coppie candidate
    """
    y_pred_global = np.zeros(len(gt_test), dtype=int)

    gt_lookup: Dict[Tuple[str, str], int] = {}
    for idx, row in gt_test[["id_A", "id_B"]].iterrows():
        gt_lookup[canonical_pair(row["id_A"], row["id_B"])] = idx

    for pos, row in enumerate(candidates_df[["id_A", "id_B"]].itertuples(index=False)):
        key = canonical_pair(row.id_A, row.id_B)
        gt_idx = gt_lookup.get(key)
        if gt_idx is not None:
            y_pred_global[gt_idx] = int(candidate_pred[pos])

    return y_pred_global


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _candidate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _load_blocking_stats(strategy: str) -> Dict[str, float]:
    path = blocking_test_stats_path(strategy)
    if not path.exists():
        raise FileNotFoundError(f"Statistiche blocking non trovate: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_rl_inference(candidates_df: pd.DataFrame):
    from src.matching import logistic_regression as rl_module

    model, meta = rl_module.load_model_and_meta()
    threshold = float(meta.get("best_threshold", 0.5))

    df_unified, _, _, _ = rl_module.load_data()
    compare = rl_module.setup_comparator()

    t0 = time.time()
    pred, prob = rl_module.infer_pairs(model, compare, df_unified, candidates_df, threshold)
    elapsed = time.time() - t0

    return pred, prob, elapsed, threshold


def _run_dedupe_inference(candidates_df: pd.DataFrame):
    from src.matching import dedupe as dedupe_module

    linker, meta = dedupe_module.load_linker_and_meta()
    threshold = float(meta.get("best_threshold", 0.5))

    df_unified, _, _, _ = dedupe_module.load_data()
    relevant_ids = set(candidates_df["id_A"].astype(str)) | set(candidates_df["id_B"].astype(str))
    id_series = df_unified["id_source_vehicles"].fillna(df_unified["id_source_used_cars"]).astype(str).str.strip()
    df_subset = df_unified.loc[id_series.isin(relevant_ids)].copy()
    df_prepared = dedupe_module.prepare_data_for_dedupe(df_subset)
    df_prepared.index = df_prepared.index.astype(str)

    t0 = time.time()
    pred, prob = dedupe_module.infer_pairs(linker, candidates_df, df_prepared, threshold)
    elapsed = time.time() - t0

    return pred, prob, elapsed, threshold


def _run_ditto_inference(candidates_df: pd.DataFrame):
    from src.matching import ditto as ditto_module

    model, meta, device = ditto_module.load_model_and_meta()
    threshold = float(meta.get("best_threshold", 0.5))
    hyper = meta.get("hyperparams", {})
    lm = str(hyper.get("lm", "roberta"))
    max_len = int(hyper.get("max_len", 256))

    t0 = time.time()
    pred, prob = ditto_module.infer_pairs(
        model,
        candidates_df,
        threshold=threshold,
        device=device,
        lm=lm,
        max_len=max_len,
    )
    elapsed = time.time() - t0

    return pred, prob, elapsed, threshold


def _evaluate_model_for_strategy(
    model_name: str,
    strategy: str,
    gt_test: pd.DataFrame,
    candidates_df: pd.DataFrame,
    blocking_stats: Dict[str, float],
) -> Dict[str, object]:
    if model_name == "Logistic Regression":
        candidate_pred, _, inference_time, threshold = _run_rl_inference(candidates_df)
    elif model_name == "Dedupe":
        candidate_pred, _, inference_time, threshold = _run_dedupe_inference(candidates_df)
    elif model_name == "Ditto":
        candidate_pred, _, inference_time, threshold = _run_ditto_inference(candidates_df)
    else:
        raise ValueError(f"Modello non supportato: {model_name}")

    y_true_candidates = candidates_df["label"].astype(int).to_numpy()
    cand_metrics = _candidate_metrics(y_true_candidates, candidate_pred)

    y_true_global = gt_test["label"].astype(int).to_numpy()
    y_pred_global = build_global_predictions(gt_test, candidates_df, candidate_pred)
    global_metrics = compute_confusion_metrics(y_true_global, y_pred_global)

    result = {
        "strategy": strategy,
        "model": model_name,
        "blocking_recall": round(float(blocking_stats["blocking_recall"]), 6),
        "n_test_pairs": int(blocking_stats["n_test_pairs"]),
        "n_candidates": int(blocking_stats["n_candidates"]),
        "reduction_ratio_vs_test": round(float(blocking_stats["reduction_ratio_vs_test"]), 6),
        "tp": global_metrics["tp"],
        "fp": global_metrics["fp"],
        "fn": global_metrics["fn"],
        "tn": global_metrics["tn"],
        "precision_global": round(global_metrics["precision"], 6),
        "recall_global": round(global_metrics["recall"], 6),
        "f1_global": round(global_metrics["f1"], 6),
        "precision_candidates": round(cand_metrics["precision"], 6),
        "recall_candidates": round(cand_metrics["recall"], 6),
        "f1_candidates": round(cand_metrics["f1"], 6),
        "threshold_used": round(float(threshold), 4),
        "inference_time_sec": round(float(inference_time), 6),
    }
    return result


def run_comparison(strategies: Iterable[str]) -> List[Dict[str, object]]:
    gt_test = pd.read_csv(gt_split_path("test"))
    results: List[Dict[str, object]] = []

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Valutazione strategia {strategy}")
        print(f"{'=' * 60}")

        candidates_path = blocking_test_candidates_path(strategy)
        if not candidates_path.exists():
            raise FileNotFoundError(f"Candidate test non trovate: {candidates_path}")

        candidates_df = pd.read_csv(candidates_path)
        blocking_stats = _load_blocking_stats(strategy)

        for model_name in ("Logistic Regression", "Dedupe", "Ditto"):
            print(f"  -> {model_name}")
            row = _evaluate_model_for_strategy(model_name, strategy, gt_test, candidates_df, blocking_stats)
            results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="Confronto finale pipeline")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["B1", "B2"],
        default=["B1", "B2"],
        help="Strategie da valutare (default: B1 B2)",
    )
    args = parser.parse_args()

    ensure_dirs()
    print_hw_info()

    rows = run_comparison(args.strategies)

    PIPELINE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)

    print(f"\nReport salvato in: {PIPELINE_REPORT_PATH}")
    if rows:
        df = pd.DataFrame(rows)
        print(df[["strategy", "model", "f1_global", "inference_time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
