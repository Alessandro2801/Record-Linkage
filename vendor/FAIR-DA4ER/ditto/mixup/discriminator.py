#!/usr/bin/env python
"""discriminator.py
========================================================================
Filters Ditto‑style lines, writing good tuples to *output‑valid* and noisy
ones (with rejection reasons) to *output‑invalid*.

Upgrade (June 2025)
───────────────────
* **Auto‑tuned thresholds** – every numeric heuristic parameter can be a
  fixed **number** or the string **``auto``**. When ``auto`` is used the
  script derives the cutoff from the input corpus using the formula:
  ``µ ± Δ σ`` where **Δ** is configurable via the global flag
  ``--delta`` (default **2.0**).

  | Flag                     | Meaning of ``auto``                                  |
  |--------------------------|-------------------------------------------------------|
  | ``--max-length``         | upper‑tail: **µ + Δ σ** of *VAL* lengths              |
  | ``--max-dup-ngrams``     | upper‑tail: **µ + Δ σ** of repeated‑ngram counts      |
  | ``--min-tfidf``          | lower‑tail: **µ − Δ σ** of mean TF‑IDF                |
  | ``--min-dict-hit-rate``  | lower‑tail: **µ − Δ σ** of dictionary hit‑rates        |
  | ``--min-zipf``           | lower‑tail: **µ − Δ σ** of mean Zipf frequencies       |

* **Selective stats scope** – pass ``--auto-scope label1`` if you want the
  automatic‑threshold statistics (µ, σ) to be computed *only* on examples
  whose label is **1**. The default ``all`` keeps the old behaviour (all
  labels).

* **Label‑aware validation pass** – rows whose label is **not** 1 are now
  *preserved verbatim* in the *output‑valid* file. They are no longer
  discarded. Only label‑1 rows undergo the expensive heuristics.

* **Robust ``TOKEN_RE``** – allows multi‑word column names.
* Threshold summary is printed so you can copy the actual numbers.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import pandas as pd  # pip install pandas
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer  # lightweight LM proxy
from wordfreq import zipf_frequency  # pip install wordfreq

###############################################################################
# Regexes & constants
###############################################################################

# Accept multi‑word column names ➜ non‑greedy capture up to the next " VAL "
TOKEN_RE = re.compile(r"COL\s+(.+?)\s+VAL\s+([^\t]+)", re.UNICODE)
CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")
BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]

###############################################################################
# CLI helpers
###############################################################################

def _num_or_auto(value: str, caster: Callable[[str], Union[int, float]]):
    """Return a number or the string 'auto'."""
    if value.lower() == "auto":
        return "auto"
    try:
        return caster(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Ditto dataset lines")
    p.add_argument("--input", required=True)
    p.add_argument("--original-input", required=True)
    p.add_argument("--output-valid", required=True)
    p.add_argument("--output-invalid", required=True)

    # heuristics: each can be <number> or 'auto'
    p.add_argument("--max-length", type=lambda s: _num_or_auto(s, int), default="auto",
                   help="Reject values longer than N chars, or 'auto'")
    p.add_argument("--max-dup-ngrams", type=lambda s: _num_or_auto(s, int), default="auto",
                   help="Detect repeated n‑grams of this size (0=off), or 'auto'")
    p.add_argument("--min-tfidf", type=lambda s: _num_or_auto(s, float), default="auto",
                   help="Minimum average TF‑IDF, or 'auto'")
    p.add_argument("--min-dict-hit-rate", type=lambda s: _num_or_auto(s, float), default="auto",
                   help="Reject if dictionary hit‑rate below this float, or 'auto'")
    p.add_argument("--min-zipf", type=lambda s: _num_or_auto(s, float), default="auto",
                   help="Reject if mean Zipf frequency below this, or 'auto'")
    p.add_argument("--delta", type=float, default=2.0,
                   help="σ‑multiplier used when any threshold is set to 'auto' (default 2.0)")

    p.add_argument("--auto-scope", choices=["all", "label1"], default="all",
                   help="When resolving 'auto' thresholds, compute statistics on all lines (default) "
                        "or only on lines with label 1 ('label1').")
    return p.parse_args()

###############################################################################
# Utility functions
###############################################################################

def ngram_repetition(text: str, n: int) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return max(Counter(ngrams).values(), default=0)


def has_unbalanced_brackets(text: str) -> bool:
    for o, c in BRACKET_PAIRS:
        if text.count(o) != text.count(c):
            return True
    return False


def dict_hit_rate(tokens: Sequence[str]) -> float:
    if not tokens:
        return 1.0
    hits = sum(zipf_frequency(t, "en") > 0 for t in tokens)
    return hits / len(tokens)


def mean_zipf(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return sum(zipf_frequency(t, "en") for t in tokens) / len(tokens)


def _mu_pm_k_sigma(data: Sequence[float], k: float = 2.0, direction: str = "upper") -> float:
    """Return µ ± k σ (upper or lower tail). If σ=0 or len<2, returns µ."""
    if not data:
        return 0.0
    mu = statistics.mean(data)
    if len(data) < 2:
        return mu
    sigma = statistics.stdev(data)
    return mu + k * sigma if direction == "upper" else mu - k * sigma

###############################################################################
# Core validation
###############################################################################

def validate_line(
    line: str,
    *,
    max_length: int,
    max_dup_ngrams: int,
    min_tfidf: float,
    min_dict_hit_rate: float,
    min_zipf: float,
    vectorizer: TfidfVectorizer,
) -> Tuple[bool, List[str]]:
    """Return (is_valid, reasons)"""
    try:
        rec1, rec2, _label = line.rstrip("\n").rsplit("\t", 2)
    except ValueError:
        return False, ["bad_format"]

    fulltext = f"{rec1} {rec2}"
    reasons: List[str] = []

    # length check per VAL
    vals = [m.group(2).strip() for m in TOKEN_RE.finditer(line)]
    if any(len(v) > max_length for v in vals):
        reasons.append("too_long")

    # control characters
    if CONTROL_RE.search(fulltext):
        reasons.append("control_char")

    # bracket balance
    if has_unbalanced_brackets(fulltext):
        reasons.append("unbalanced_brackets")

    # n‑gram repetition
    if max_dup_ngrams and ngram_repetition(fulltext, max_dup_ngrams) > 1:
        reasons.append("dup_ngrams")

    # TF‑IDF
    if min_tfidf > 0:
        if float(vectorizer.transform([fulltext]).mean()) < min_tfidf:
            reasons.append("low_tfidf")

    # tokens for dict/zipf tests
    alpha_tokens = re.findall(r"[A-Za-z]+", fulltext.lower())

    if dict_hit_rate(alpha_tokens) < min_dict_hit_rate:
        reasons.append("low_dict_hit")

    if mean_zipf(alpha_tokens) < min_zipf:
        reasons.append("low_zipf_entropy")

    return not reasons, reasons

###############################################################################
# Main
###############################################################################

def canonicalise(line: str) -> str:
    """
    Strip trailing/leading whitespace from every TAB-separated field so that
    lines that are identical modulo spacing are treated as the same row.
    """
    return "\t".join(part.strip() for part in line.rstrip("\n").split("\t"))


def main() -> None:
    args = get_args()

    # ------------------------------------------------------------------ paths
    input_path          = Path(args.input)
    original_input_path = Path(args.original_input)
    valid_path          = Path(args.output_valid)
    invalid_path        = Path(args.output_invalid)

    # read files -------------------------------------------------------------
    lines_raw          = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    original_lines_raw = original_input_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # canonicalised versions (trimmed fields ⇒ robust matching)
    lines          = [canonicalise(l) for l in lines_raw]
    original_set   = {canonicalise(l) for l in original_lines_raw}

    # ── 1st pass ── gather corpus-wide statistics ---------------------------
    corpus, val_lens, dict_hits, zipf_means, dup_ngram_counts = [], [], [], [], []

    for l in lines:
        try:
            r1, r2, lbl = l.rsplit("\t", 2)
            lbl = lbl.strip()
        except ValueError:
            continue  # malformed ► skip in stats

        if args.auto_scope == "label1" and lbl != "1":
            continue  # scope restriction

        # NOTE: originals now *included* in statistics
        fulltext = f"{r1} {r2}"
        corpus.append(fulltext)

        val_lens.extend(len(m.group(2).strip()) for m in TOKEN_RE.finditer(l))
        dup_ngram_counts.append(ngram_repetition(fulltext, 3))

        toks = re.findall(r"[A-Za-z]+", fulltext.lower())
        dict_hits.append(dict_hit_rate(toks))
        zipf_means.append(mean_zipf(toks))

    # ── TF-IDF stats --------------------------------------------------------
    vectorizer = TfidfVectorizer(min_df=2, max_features=10_000)
    X = vectorizer.fit_transform(corpus) if corpus else None
    tfidf_means = X.mean(axis=1).A.ravel().tolist() if X is not None else []

    # ── Resolve any 'auto' thresholds ---------------------------------------
    k = args.delta
    if args.max_length == "auto":
        args.max_length = int(_mu_pm_k_sigma(val_lens, k, "upper")) or 1
    if args.max_dup_ngrams == "auto":
        args.max_dup_ngrams = max(0, int(_mu_pm_k_sigma(dup_ngram_counts, k, "upper")))
    if args.min_tfidf == "auto":
        args.min_tfidf = _mu_pm_k_sigma(tfidf_means, k, "lower")
    if args.min_dict_hit_rate == "auto":
        args.min_dict_hit_rate = max(0.0, _mu_pm_k_sigma(dict_hits, k, "lower"))
    if args.min_zipf == "auto":
        args.min_zipf = max(0.0, _mu_pm_k_sigma(zipf_means, k, "lower"))

    print(
        f"Thresholds (Δ = {k:.2f} σ, scope = {args.auto_scope}) → "
        f"max_length={args.max_length}, "
        f"max_dup_ngrams={args.max_dup_ngrams}, "
        f"min_tfidf={args.min_tfidf:.4f}, "
        f"min_dict_hit_rate={args.min_dict_hit_rate:.3f}, "
        f"min_zipf={args.min_zipf:.3f}"
    )


    # ── 2nd pass ── validate line-by-line -----------------------------------
    total = kept = rejected = 0
    kept_lines: List[str] = []

    with valid_path.open("w", encoding="utf-8") as good, \
         invalid_path.open("w", encoding="utf-8") as bad:

        for line in lines:
            total += 1

            # originals (label 0 or 1) ► always keep verbatim
            if line in original_set:
                kept += 1
                good.write(f"{line}\n")
                kept_lines.append(line)
                continue

            # parse label
            try:
                _r1, _r2, label = line.rsplit("\t", 2)
            except ValueError:
                rejected += 1
                bad.write(f"{line}\t# REJECTED: bad_format\n")
                continue

            label = label.strip()

            if label == "1":
                ok, reasons = validate_line(
                    line,
                    max_length=args.max_length,
                    max_dup_ngrams=args.max_dup_ngrams,
                    min_tfidf=args.min_tfidf,
                    min_dict_hit_rate=args.min_dict_hit_rate,
                    min_zipf=args.min_zipf,
                    vectorizer=vectorizer,
                )
                if ok:
                    kept += 1
                    good.write(f"{line}\n")
                    kept_lines.append(line)
                else:
                    rejected += 1
                    bad.write(f"{line}\t# REJECTED: {','.join(reasons)}\n")
            else:
                kept += 1
                good.write(f"{line}\n")
                kept_lines.append(line)

    print(f"Processed {total:,} · kept {kept:,} · rejected {rejected:,}.")

    # ---- label distribution for kept lines ---------------------------------
    if kept_lines:
        df = pd.DataFrame(
            [l.rsplit("\t", 2) for l in kept_lines],
            columns=["rec1", "rec2", "Label"]
        )
        print("Label counts (kept):")
        for lbl, cnt in df["Label"].value_counts().sort_index().items():
            print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    main()
