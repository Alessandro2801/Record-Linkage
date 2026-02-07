#!/usr/bin/env python3
"""
ditto_format.py — Conversione dati in formato Ditto.

Modalità:
    gt-splits   -> converte GT train/val/test in vendor/FAIR-DA4ER/ditto/data/ditto_data/
    blocked-test -> converte test bloccato per una strategia (B1/B2)
    all         -> esegue entrambe

Usage:
    python -m src.blocking.ditto_format --mode gt-splits
    python -m src.blocking.ditto_format --mode blocked-test --strategy B1
    python -m src.blocking.ditto_format --mode all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.config import (
    DITTO_VENDOR_DATA_DIR,
    GT_TEST_PATH,
    GT_TRAIN_PATH,
    GT_VAL_PATH,
    blocking_test_candidates_path,
    ditto_data_path,
    ensure_dirs,
    print_hw_info,
)

MEDIATED_FIELDS = [
    "manufacturer",
    "year",
    "model",
    "mileage",
    "price",
    "main_color",
    "transmission",
    "traction",
    "body_type",
    "fuel_type",
    "cylinders",
    "latitude",
    "longitude",
    "description",
    "location",
    "pubblication_date",
]


def serialize_record(row: pd.Series, suffix: str) -> str:
    """Serializza un record nel formato colonnare Ditto."""
    parts: List[str] = []
    for field in MEDIATED_FIELDS:
        col_name = f"{field}_{suffix}"
        value = str(row.get(col_name, "")).strip()
        if not value or value.lower() == "nan":
            continue
        value = value.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        parts.append(f"COL {field} VAL {value}")
    return " ".join(parts)


def row_to_ditto_line(row: pd.Series, label_col: str = "label") -> str:
    left = serialize_record(row, "A")
    right = serialize_record(row, "B")
    label = str(row.get(label_col, 0)).strip()
    if label not in {"0", "1"}:
        label = "0"
    return f"{left}\t{right}\t{label}\n"


def dataframe_to_ditto_lines(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    return [row_to_ditto_line(row, label_col=label_col) for _, row in df.iterrows()]


def convert_csv_to_ditto(csv_path: Path, txt_path: Path, label_col: str = "label") -> None:
    """Converte un CSV pair-level nel formato .txt usato da Ditto."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    print(f"  {csv_path.name} -> {txt_path} ({len(df):,} righe)")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = dataframe_to_ditto_lines(df, label_col=label_col)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def convert_gt_splits() -> None:
    """Converte GT train/val/test in file Ditto usati per training."""
    mapping = {
        GT_TRAIN_PATH: DITTO_VENDOR_DATA_DIR / "train.txt",
        GT_VAL_PATH: DITTO_VENDOR_DATA_DIR / "val.txt",
        GT_TEST_PATH: DITTO_VENDOR_DATA_DIR / "test.txt",
    }
    print("\nConversione GT splits -> Ditto vendor data")
    for csv_path, txt_path in mapping.items():
        convert_csv_to_ditto(csv_path, txt_path)


def convert_blocked_test(strategy: str) -> None:
    """Converte le candidate del test bloccato per una strategia."""
    csv_path = blocking_test_candidates_path(strategy)
    txt_path = ditto_data_path(strategy, "test_candidates")
    print(f"\nConversione blocked test per {strategy}")
    convert_csv_to_ditto(csv_path, txt_path)


def convert_all(strategies: Iterable[str]) -> None:
    convert_gt_splits()
    for strategy in strategies:
        convert_blocked_test(strategy)


def main():
    parser = argparse.ArgumentParser(description="Conversione dati in formato Ditto")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["gt-splits", "blocked-test", "all"],
        help="Modalita di conversione",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["B1", "B2"],
        default=None,
        help="Strategia per mode blocked-test",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["B1", "B2"],
        default=["B1", "B2"],
        help="Strategie usate in mode all",
    )
    args = parser.parse_args()

    ensure_dirs()
    print_hw_info()

    if args.mode == "gt-splits":
        convert_gt_splits()
    elif args.mode == "blocked-test":
        if not args.strategy:
            raise ValueError("--strategy e obbligatorio in mode blocked-test")
        convert_blocked_test(args.strategy)
    else:
        convert_all(args.strategies)

    print("\nConversione completata.")


if __name__ == "__main__":
    main()
