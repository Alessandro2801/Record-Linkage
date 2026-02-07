#!/usr/bin/env python3
"""
dedupe_eval.py — Wrapper legacy per valutazione modello Dedupe.

Mantiene compatibilita CLI, delegando la logica al modulo src.matching.dedupe.
"""

from src.matching.dedupe import evaluate_saved_model_on_gt_test


def main():
    evaluate_saved_model_on_gt_test()


if __name__ == "__main__":
    main()
