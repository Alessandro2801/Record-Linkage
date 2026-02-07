#!/usr/bin/env python3
"""
generate.py — Blocking pair-level su split test della Ground Truth.

Strategie:
    B1: year uguale + manufacturer[:3] uguale + Jaro-Winkler manufacturer >= soglia
    B2: B1 + model[:3] uguale + Jaro-Winkler model >= soglia + fuel_type uguale

Usage:
    python -m src.blocking.generate
    python -m src.blocking.generate --strategies B1
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List

import jellyfish
import pandas as pd

from src.config import (
    BLOCKING_MFR_THRESHOLD,
    BLOCKING_MODEL_THRESHOLD,
    GT_TEST_PATH,
    RANDOM_SEED,
    blocking_test_candidates_path,
    blocking_test_stats_path,
    ensure_dirs,
    print_hw_info,
)


def _norm_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _norm_year(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _prefix3(value: str) -> str:
    return value[:3] if value else ""


def pair_passes_b1(
    row: pd.Series,
    mfr_threshold: float = BLOCKING_MFR_THRESHOLD,
) -> bool:
    """Predicato B1 applicato a una coppia della GT test."""
    year_a = _norm_year(row.get("year_A"))
    year_b = _norm_year(row.get("year_B"))
    if not year_a or year_a != year_b:
        return False

    mfr_a = _norm_str(row.get("manufacturer_A"))
    mfr_b = _norm_str(row.get("manufacturer_B"))
    if _prefix3(mfr_a) != _prefix3(mfr_b):
        return False

    mfr_sim = jellyfish.jaro_winkler_similarity(mfr_a, mfr_b)
    return mfr_sim >= mfr_threshold


def pair_passes_b2(
    row: pd.Series,
    mfr_threshold: float = BLOCKING_MFR_THRESHOLD,
    model_threshold: float = BLOCKING_MODEL_THRESHOLD,
) -> bool:
    """Predicato B2 applicato a una coppia della GT test."""
    if not pair_passes_b1(row, mfr_threshold=mfr_threshold):
        return False

    model_a = _norm_str(row.get("model_A"))
    model_b = _norm_str(row.get("model_B"))
    if _prefix3(model_a) != _prefix3(model_b):
        return False

    model_sim = jellyfish.jaro_winkler_similarity(model_a, model_b)
    if model_sim < model_threshold:
        return False

    fuel_a = _norm_str(row.get("fuel_type_A"))
    fuel_b = _norm_str(row.get("fuel_type_B"))
    return fuel_a == fuel_b


def _apply_blocking(df_test: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "B1":
        mask = df_test.apply(pair_passes_b1, axis=1)
    elif strategy == "B2":
        mask = df_test.apply(pair_passes_b2, axis=1)
    else:
        raise ValueError(f"Strategia non supportata: {strategy}")
    return df_test.loc[mask].copy()


def evaluate_blocking_on_test(candidates: pd.DataFrame, gt_test: pd.DataFrame) -> Dict[str, float]:
    """Metriche di blocking rispetto al test split completo."""
    positives_in_test = int(gt_test["label"].sum())
    positives_captured = int(candidates["label"].sum())

    n_test_pairs = int(len(gt_test))
    n_candidates = int(len(candidates))

    blocking_recall = (positives_captured / positives_in_test) if positives_in_test > 0 else 0.0
    reduction_ratio_vs_test = 1.0 - (n_candidates / n_test_pairs) if n_test_pairs > 0 else 0.0

    return {
        "positives_in_test": positives_in_test,
        "positives_captured": positives_captured,
        "blocking_recall": round(float(blocking_recall), 6),
        "n_test_pairs": n_test_pairs,
        "n_candidates": n_candidates,
        "reduction_ratio_vs_test": round(float(reduction_ratio_vs_test), 6),
    }


def run_for_strategy(df_test: pd.DataFrame, strategy: str) -> Dict[str, float]:
    print(f"\n{'=' * 60}")
    print(f"BLOCKING TEST — Strategia {strategy}")
    print(f"{'=' * 60}")

    candidates = _apply_blocking(df_test, strategy)
    stats = evaluate_blocking_on_test(candidates, df_test)

    stats["strategy"] = strategy
    stats["mfr_threshold"] = BLOCKING_MFR_THRESHOLD
    stats["model_threshold"] = BLOCKING_MODEL_THRESHOLD if strategy == "B2" else None
    stats["random_seed"] = RANDOM_SEED

    csv_path = blocking_test_candidates_path(strategy)
    json_path = blocking_test_stats_path(strategy)

    candidates.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"  Test pairs:       {stats['n_test_pairs']:,}")
    print(f"  Candidates:       {stats['n_candidates']:,}")
    print(f"  Positives test:   {stats['positives_in_test']:,}")
    print(f"  Positives cap.:   {stats['positives_captured']:,}")
    print(f"  Blocking recall:  {stats['blocking_recall']:.4f}")
    print(f"  Reduction ratio:  {stats['reduction_ratio_vs_test']:.4f}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

    return stats


def main(strategies: Iterable[str] | None = None):
    ensure_dirs()
    print_hw_info()

    if not GT_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Split test non trovato: {GT_TEST_PATH}\n"
            "Eseguire prima: python -m src.preparation.ground_truth"
        )

    df_test = pd.read_csv(GT_TEST_PATH)
    print(f"Caricato GT test: {GT_TEST_PATH} ({len(df_test):,} coppie)")

    selected = list(strategies) if strategies else ["B1", "B2"]
    for strategy in selected:
        run_for_strategy(df_test, strategy)


def cli():
    parser = argparse.ArgumentParser(description="Blocking pair-level sul GT test")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["B1", "B2"],
        default=["B1", "B2"],
        help="Strategie da eseguire (default: B1 B2)",
    )
    args = parser.parse_args()
    main(strategies=args.strategies)


if __name__ == "__main__":
    cli()
