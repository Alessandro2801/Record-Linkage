#!/usr/bin/env python3
"""
run_pipeline.py — Orchestratore del workflow sperimentale di Record Linkage.

Riproduce l'intero esperimento o singoli step, dalla acquisizione dati
alla valutazione comparativa delle pipeline di matching.

Pipeline:
    Step 1  ─  Download dataset da Kaggle            (src.preparation.download)
    Step 2  ─  Preprocessing dataset grezzi           (src.preparation.process_raw)
    Step 3  ─  Costruzione schema mediato             (src.preparation.mediated_schema)
    Step 4  ─  Generazione ground truth da VIN        (src.preparation.ground_truth)
    Step 5  ─  Blocking: coppie candidate B1, B2      (src.blocking.generate)
    Step 6  ─  Conversione dati in formato Ditto      (src.blocking.ditto_format)
    Step 7  ─  Training RecordLinkage (per strategia) (src.matching.recordlinkage)
    Step 8  ─  Training + eval Dedupe (per strategia) (src.matching.dedupe + dedupe_eval)
    Step 9  ─  Valutazione comparativa pipeline       (src.evaluation.compare)

Usage:
    python run_pipeline.py                    # Esegui tutto
    python run_pipeline.py --from-step 5      # Riprendi dallo step 5
    python run_pipeline.py --only-step 7      # Esegui solo lo step 7
    python run_pipeline.py --strategies B1    # Solo strategia B1

Note:
    - Lo Step 1 richiede credenziali Kaggle configurate.
    - I notebook EDA (notebooks/) restano disponibili per analisi esplorativa
      e visualizzazioni, ma non sono più necessari per la pipeline.
"""

import argparse
import subprocess
import sys
import time

from src.config import print_hw_info


# ═══════════════════════════════════════════════════════════════════════════════
#  DEFINIZIONE STEP
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = [
    # (num, nome, modulo, richiede strategia, args extra)
    (1, "Download dataset da Kaggle",            "src.preparation.download",          False, []),
    (2, "Preprocessing dataset grezzi",          "src.preparation.process_raw",       False, []),
    (3, "Costruzione schema mediato",            "src.preparation.mediated_schema",   False, []),
    (4, "Generazione ground truth",              "src.preparation.ground_truth",      False, []),
    (5, "Blocking: coppie candidate B1 + B2",    "src.blocking.generate",             False, []),
    (6, "Conversione dati formato Ditto",        "src.blocking.ditto_format",         True,  []),
    (7, "Training RecordLinkage",                "src.matching.recordlinkage",        True,  ["--train"]),
    (8, "Training + eval Dedupe",                "src.matching.dedupe",               True,  ["--train"]),
    (9, "Valutazione comparativa pipeline",      "src.evaluation.compare",            False, []),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  ESECUZIONE
# ═══════════════════════════════════════════════════════════════════════════════

def run_module(module: str, extra_args: list = None) -> bool:
    """Esegue un modulo Python come subprocess."""
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_step(num: int, name: str, module: str, per_strategy: bool,
             extra_args: list, strategies: list) -> bool:
    """Esegue uno step della pipeline."""
    print(f"\n{'='*70}")
    print(f"  STEP {num}: {name}")
    print(f"{'='*70}\n")

    t0 = time.time()

    if per_strategy:
        for strategy in strategies:
            print(f"\n  --- Strategia {strategy} ---\n")
            args = extra_args + ["--strategy", strategy]
            if not run_module(module, args):
                print(f"\n  ERRORE nello step {num} (strategia {strategy})")
                return False

        # Per Dedupe: dopo il training, esegui anche la valutazione
        if module == "src.matching.dedupe":
            for strategy in strategies:
                print(f"\n  --- Valutazione Dedupe — Strategia {strategy} ---\n")
                if not run_module("src.matching.dedupe_eval", ["--strategy", strategy]):
                    print(f"\n  ERRORE nella valutazione Dedupe (strategia {strategy})")
                    return False
    else:
        if not run_module(module, extra_args):
            print(f"\n  ERRORE nello step {num}")
            return False

    elapsed = time.time() - t0
    print(f"\n  Step {num} completato in {elapsed:.1f}s")
    return True


def run_pipeline(from_step: int = 1, only_step: int = None,
                 strategies: list = None):
    """Esegue la pipeline completa o parziale."""
    if strategies is None:
        strategies = ["B1", "B2"]

    print("=" * 70)
    print("  RECORD LINKAGE — PIPELINE SPERIMENTALE")
    print("=" * 70)
    print_hw_info()
    print(f"  Strategie: {', '.join(strategies)}")
    if only_step:
        print(f"  Esecuzione: solo step {only_step}")
    else:
        print(f"  Esecuzione: step {from_step} -> {STEPS[-1][0]}")
    print()

    t_start = time.time()
    completed = []

    for num, name, module, per_strategy, extra_args in STEPS:
        if only_step and num != only_step:
            continue
        if not only_step and num < from_step:
            continue

        success = run_step(num, name, module, per_strategy, extra_args, strategies)
        if success:
            completed.append(num)
        else:
            print(f"\nPipeline interrotta allo step {num}.")
            break

    elapsed_total = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  RIEPILOGO")
    print(f"{'='*70}")
    print(f"  Step completati: {completed}")
    print(f"  Tempo totale: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Record Linkage — Pipeline sperimentale riproducibile',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step disponibili:
  1  Download dataset da Kaggle
  2  Preprocessing dataset grezzi
  3  Costruzione schema mediato
  4  Generazione ground truth
  5  Blocking (B1 + B2)
  6  Conversione formato Ditto
  7  Training RecordLinkage
  8  Training + eval Dedupe
  9  Valutazione comparativa pipeline

Esempi:
  python run_pipeline.py                       # Tutto
  python run_pipeline.py --from-step 5         # Da step 5
  python run_pipeline.py --only-step 7         # Solo step 7
  python run_pipeline.py --strategies B1       # Solo B1
""")
    parser.add_argument('--from-step', type=int, default=1,
                        help='Parti da questo step (default: 1)')
    parser.add_argument('--only-step', type=int, default=None,
                        help='Esegui solo questo step')
    parser.add_argument('--strategies', nargs='+', default=['B1', 'B2'],
                        choices=['B1', 'B2'],
                        help='Strategie di blocking da processare (default: B1 B2)')

    args = parser.parse_args()

    run_pipeline(
        from_step=args.from_step,
        only_step=args.only_step,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
