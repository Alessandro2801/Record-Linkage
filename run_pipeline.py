#!/usr/bin/env python3
"""
run_pipeline.py — Orchestratore del workflow sperimentale di Record Linkage.

Pipeline metodologica:
    Step 1  ─  Download dataset da Kaggle
    Step 2  ─  Preprocessing dataset grezzi
    Step 3  ─  Costruzione schema mediato
    Step 4  ─  Generazione ground truth e split GT (train/val/test)
    Step 5  ─  Blocking pair-level su GT test (B1/B2)
    Step 6  ─  Conversione GT split in formato Ditto
    Step 7  ─  Training Logistic Regression (globale)
    Step 8  ─  Training Dedupe (globale)
    Step 9  ─  Training Ditto (globale)
    Step 10 ─  Valutazione comparativa (B1/B2 x 3 modelli)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import List

from src.config import print_hw_info


STEPS = [
    (1, "Download dataset da Kaggle", "src.preparation.download", []),
    (2, "Preprocessing dataset grezzi", "src.preparation.process_raw", []),
    (3, "Costruzione schema mediato", "src.preparation.mediated_schema", []),
    (4, "Generazione ground truth", "src.preparation.ground_truth", []),
    (5, "Blocking pair-level su test GT", "src.blocking.generate", []),
    (6, "Conversione GT split in formato Ditto", "src.blocking.ditto_format", ["--mode", "gt-splits"]),
    (7, "Training Logistic Regression", "src.matching.logistic_regression", ["--train"]),
    (8, "Training Dedupe", "src.matching.dedupe", ["--train"]),
    (9, "Training Ditto", "src.matching.ditto", ["--train"]),
    (10, "Valutazione comparativa pipeline", "src.evaluation.compare", []),
]


def run_module(module: str, extra_args: List[str] | None = None) -> bool:
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def step_args(module: str, base_args: List[str], strategies: List[str]) -> List[str]:
    args = list(base_args)
    if module in {"src.blocking.generate", "src.evaluation.compare"}:
        args.extend(["--strategies", *strategies])
    return args


def run_step(num: int, name: str, module: str, base_args: List[str], strategies: List[str]) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  STEP {num}: {name}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    args = step_args(module, base_args, strategies)
    if not run_module(module, args):
        print(f"\n  ERRORE nello step {num}")
        return False

    elapsed = time.time() - t0
    print(f"\n  Step {num} completato in {elapsed:.1f}s")
    return True


def run_pipeline(from_step: int = 1, only_step: int | None = None, strategies: List[str] | None = None):
    if strategies is None:
        strategies = ["B1", "B2"]

    print("=" * 70)
    print("  RECORD LINKAGE — PIPELINE METODOLOGICA")
    print("=" * 70)
    print_hw_info()
    print(f"  Strategie blocking/eval: {', '.join(strategies)}")
    if only_step:
        print(f"  Esecuzione: solo step {only_step}")
    else:
        print(f"  Esecuzione: step {from_step} -> {STEPS[-1][0]}")
    print()

    t_start = time.time()
    completed = []

    for num, name, module, base_args in STEPS:
        if only_step and num != only_step:
            continue
        if not only_step and num < from_step:
            continue

        ok = run_step(num, name, module, base_args, strategies)
        if ok:
            completed.append(num)
        else:
            print(f"\nPipeline interrotta allo step {num}.")
            break

    elapsed_total = time.time() - t_start

    print(f"\n{'=' * 70}")
    print("  RIEPILOGO")
    print(f"{'=' * 70}")
    print(f"  Step completati: {completed}")
    print(f"  Tempo totale: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Record Linkage — Pipeline metodologica riproducibile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step disponibili:
  1  Download dataset da Kaggle
  2  Preprocessing dataset grezzi
  3  Costruzione schema mediato
  4  Generazione ground truth
  5  Blocking pair-level su test GT
  6  Conversione GT split in formato Ditto
  7  Training Logistic Regression
  8  Training Dedupe
  9  Training Ditto
  10 Valutazione comparativa pipeline

Esempi:
  python run_pipeline.py
  python run_pipeline.py --from-step 5
  python run_pipeline.py --only-step 7
  python run_pipeline.py --strategies B1
""",
    )
    parser.add_argument("--from-step", type=int, default=1, help="Parti da questo step (default: 1)")
    parser.add_argument("--only-step", type=int, default=None, help="Esegui solo questo step")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["B1", "B2"],
        choices=["B1", "B2"],
        help="Strategie usate da blocking (step 5) e compare (step 10)",
    )

    args = parser.parse_args()
    run_pipeline(from_step=args.from_step, only_step=args.only_step, strategies=args.strategies)


if __name__ == "__main__":
    main()
