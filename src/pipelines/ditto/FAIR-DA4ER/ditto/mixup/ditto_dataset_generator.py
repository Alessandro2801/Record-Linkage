#!/usr/bin/env python

"""
Ditto Dataset Generator — TAB‑separated TXT ↔︎ TXT (parallel, seed‑aware, generic seq2seq)
========================================================================================

This version extends the original script with

* **Generic model loading** – pass *any* HuggingFace seq‑to‑seq checkpoint
  (BART, T5, etc.).
* **Batched / GPU‑friendly generation** – hidden‑state mix‑up and decoding are
  performed in chunks (`--batch-size`) for significant speed‑ups.
* **Seed‑exposure end‑to‑end** – one `--seed` flag controls *all* randomness:
  Python, NumPy, Torch, and the hidden‑state mixing weights.
* **Improved decoding quality** – configurable `--num-beams`,
  `--top-k`, `--top-p`, `--temperature` give more fluent English output.

Usage example ────────────────────────────────────────────────────────────────

```bash
python ditto_dataset_generator_parallel.py \
  --input beers_pairs.txt \
  --output beers_augmented.txt \
  --generations 3 \
  --alpha 0.5 \
  --alpha-delta 0.05 \
  --model google/t5-large-lm-adapt \
  --num-beams 4 \
  --batch-size 16 \
  --seed 1337
```

"""

from __future__ import annotations

import argparse
import csv
import os
import json
import random
from typing import Dict, List, Tuple
import datetime as _dt
import time         

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


###############################################################################
# Reproducibility helpers
###############################################################################

def set_seed(seed: int) -> None:
    """Fix *all* PRNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


###############################################################################
# Ditto parsing helpers 
###############################################################################

def _parse_side(tokens: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        if tokens[i] != "COL":
            raise ValueError(f"Expected 'COL' at pos {i}, got '{tokens[i]}'")
        if i + 2 >= len(tokens) or tokens[i + 2] != "VAL":
            raise ValueError("Malformed segment; expected 'COL <attr> VAL'.")
        attr = tokens[i + 1]
        j = i + 3
        val_tokens: List[str] = []
        while j < len(tokens) and tokens[j] != "COL":
            val_tokens.append(tokens[j])
            j += 1
        out[attr] = " ".join(val_tokens)
        i = j
    return out

def parse_ditto_line(line: str) -> Tuple[Dict[str, str], List[str]]:
    try:
        rec1_txt, rec2_txt, label_part = line.rstrip("\n").rsplit("\t", 2)
    except ValueError as exc:
        raise ValueError("Line must contain two TABs: rec1<TAB>rec2<TAB>label") from exc

    label = int(label_part.strip())
    rec1_attrs = _parse_side(rec1_txt.split())
    rec2_attrs = _parse_side(rec2_txt.split())

    attributes: List[str] = []
    for a in list(rec1_attrs) + list(rec2_attrs):
        if a not in attributes:
            attributes.append(a)

    record: Dict[str, str] = {"Label": label}
    for a in attributes:
        record[f"{a}_1"] = rec1_attrs.get(a, "")
        record[f"{a}_2"] = rec2_attrs.get(a, "")
    return record, attributes


###############################################################################
# Ditto re‑serialization helper 
###############################################################################

def _emit(tokens: List[str], attr: str, value: str | None) -> None:
    if value is None:
        return
    tokens += ["COL", attr, "VAL"]
    val = str(value).strip()
    if val:
        tokens.append(val)

def row_to_ditto(row: pd.Series, columns: List[str]) -> str:
    toks1: List[str] = []
    toks2: List[str] = []
    for col in columns:
        v1 = row.get(f"{col}_1", "")
        v2 = row.get(f"{col}_2", "")
        if v1 or v2:
            _emit(toks1, col, v1)
            _emit(toks2, col, v2)
    rec1 = " ".join(toks1)
    rec2 = " ".join(toks2)
    return f"{rec1}\t{rec2}\t{int(row['Label'])}"


###############################################################################
# Mix‑up internals ─ batched version
###############################################################################

def _batched_interpolate(
    tokenizer,
    model,
    s1_list: List[str],
    s2_list: List[str],
    device: torch.device,
    alpha: float,
    alpha_delta: float,
    num_beams: int,
    top_k: int,
    top_p: float,
    temperature: float,
    max_len: int = 512,
) -> List[str]:
    """Create synthetic sentences for many *pairs* at once, forwarding
    sampling flags **only** when sampling is enabled."""
    batch_size = len(s1_list)

    # 1) Encode the two sides
    toks = tokenizer(
        s1_list + s2_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        enc_out = model.get_encoder()(
            toks.input_ids, attention_mask=toks.attention_mask
        )
    hidden = enc_out.last_hidden_state  # (2B, L, D)
    h1, h2 = hidden[:batch_size], hidden[batch_size:]

    # 2) Mix the hidden states
    w = torch.tensor(
        np.random.uniform(alpha, alpha + alpha_delta, size=(batch_size, 1, 1)),
       dtype=h1.dtype,
        device=device,
    )
    # Draw w ~ 𝓝(μ=alpha, σ=alpha_delta) and clip to [0,1]
    w = np.random.normal(loc=alpha, scale=alpha_delta, size=(batch_size, 1, 1))
    w = np.clip(w, 0.0, 1.0)              # keep valid mixing weights
    w = torch.as_tensor(w, dtype=h1.dtype, device=device)

    h_mix = (1 - w) * h1 + w * h2
    enc_mix = BaseModelOutput(last_hidden_state=h_mix)

    # 3) Build generation kwargs *dynamically*
    start_ids = torch.full(
        (batch_size, 1), model.config.decoder_start_token_id, device=device
    )

    gen_kwargs = dict(
        decoder_input_ids=start_ids,
        encoder_outputs=enc_mix,
        max_new_tokens=max_len,
        num_beams=num_beams,
        early_stopping=True,
    )

    # Enable sampling only if we are **not** doing beam search
    sampling = (num_beams == 1) and (
        (top_k > 0) or (top_p < 1.0) or (temperature != 1.0)
    )
    if sampling:
        gen_kwargs.update(
            do_sample=True,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            temperature=temperature,
        )

    # strip out any None values so HF doesn’t complain
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # 4) Decode
    ids = model.generate(**gen_kwargs)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)

    """Create synthetic sentences for many *pairs* at once."""
    batch_size = len(s1_list)
    toks = tokenizer(
        s1_list + s2_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        enc_out = model.get_encoder()(toks.input_ids, attention_mask=toks.attention_mask)
    hidden = enc_out.last_hidden_state
    h1, h2 = hidden[:batch_size], hidden[batch_size:]

    w = torch.tensor(
        np.random.uniform(alpha, alpha + alpha_delta, size=(batch_size, 1, 1)),
        dtype=h1.dtype,
        device=device,
    )
    h_mix = (1 - w) * h1 + w * h2
    enc_mix = BaseModelOutput(last_hidden_state=h_mix)

    start_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)
    ids = model.generate(
        decoder_input_ids=start_ids,
        encoder_outputs=enc_mix,
        max_new_tokens=max_len,
        num_beams=num_beams,
        do_sample=top_k > 0 or top_p < 1.0,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        early_stopping=True,
    )
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


###############################################################################
# Augmentation (batched)
###############################################################################

def augment(
    df: pd.DataFrame,
    attrs: List[str],
    tokenizer,
    model,
    device: torch.device,
    alpha: float,
    alpha_delta: float,
    gen: int,
    batch_size: int,
    num_beams: int,
    top_k: int,
    top_p: float,
    temperature: float,
) -> pd.DataFrame:
    """One generation of synthetic pairs – *batched* for speed."""

    new_rows: List[Dict[str, str]] = []
    jobs: List[Tuple[str, str, int, str]] = []
    for idx, row in df.iterrows():
        for a in attrs:
            v1 = row.get(f"{a}_1", "").strip()
            v2 = row.get(f"{a}_2", "").strip()
            if v1 and v2:
                jobs.append((v1, v2, idx, a))
            else:
                df.at[idx, f"{a}_3"] = v1 or v2

    for start in tqdm(range(0, len(jobs), batch_size), desc="Augmenting", unit="batch"):
        chunk = jobs[start : start + batch_size]
        s1_chunk, s2_chunk, idx_chunk, attr_chunk = zip(*chunk)
        mixed = _batched_interpolate(
            tokenizer,
            model,
            list(s1_chunk),
            list(s2_chunk),
            device,
            alpha,
            alpha_delta,
            num_beams,
            top_k,
            top_p,
            temperature,
        )
        for val, row_idx, attr in zip(mixed, idx_chunk, attr_chunk):
            df.at[row_idx, f"{attr}_3"] = val

    for idx, row in df.iterrows():
        merged = row.to_dict()
        for left, right in (("_1", "_3"), ("_2", "_3")):
            new_row = {f"{a}_1": merged[f"{a}{left}"] for a in attrs}
            new_row.update({f"{a}_2": merged[f"{a}{right}"] for a in attrs})
            new_row["Label"] = 1
            new_row["generation"] = gen + 1
            new_row["parent"] = idx
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)


###############################################################################
# CLI
###############################################################################

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Augment Ditto TAB‑separated TXT (vector mix‑up, parallel)")
    ap.add_argument("--input", required=True, help="Path to input .txt (Ditto format)")
    ap.add_argument("--output", required=True, help="Path to output .txt")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alpha-delta", type=float, default=0.05)
    ap.add_argument("--model", default="facebook/bart-large", help="Any HF seq2seq checkpoint (BART/T5/…)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for GPU generation (default: 8)")
    ap.add_argument("--num-beams", type=int, default=1, help="Beam size for decoding")
    ap.add_argument("--top-k", type=int, default=0, help="top‑k sampling (0 = disabled)")
    ap.add_argument("--top-p", type=float, default=1.0, help="top‑p nucleus sampling (1.0 = disabled)")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmented-labels", default="1")
    ap.add_argument("--size", type=int, help="Total size of the dataset after augmentation (default: +inf)", default=float("inf"))
    ap.add_argument("--preserve-label-ratio", action="store_true")
    ap.add_argument("--augmentation-budget-ratio", type=float, default=0.25)
    return ap.parse_args()


# ──  Label-wise size diagnostics  ───────────────────────────────────
def _label_counts(frame: pd.DataFrame) -> Dict[str, int]:
    """Return counts as plain {label: n} dict with str keys for JSON."""
    return {str(k): int(v) for k, v in frame["Label"].value_counts().sort_index().items()}


###############################################################################
# Main
###############################################################################

def main() -> None:
    args = get_args()
    set_seed(args.seed)

    # ── 1. Beam-vs-sampling guard ───────────────────────────────────────────
    sampling_flags_used = (
        (args.top_k > 0) or (args.top_p < 1.0) or (args.temperature != 1.0)
    )
    if args.num_beams > 1 and sampling_flags_used:
        print(
            "ℹ️  Ignoring --top-k/--top-p/--temperature because beam search is "
            "active (num_beams > 1)"
        )
        args.top_k = 0
        args.top_p = 1.0
        args.temperature = 1.0

    # ── 2. Read & parse input ───────────────────────────────────────────────
    print("🔄 Reading input …")
    attr_union: set[str] = set()
    attr_order: List[str] = []
    row_dicts: List[Dict[str, str]] = []
    other_dicts: List[Dict[str, str]] = []

    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec, attrs = parse_ditto_line(line)
            label_str = str(rec.get("Label", ""))
            if label_str in args.augmented_labels.split(","):
                row_dicts.append(rec)
                for a in attrs:
                    if a not in attr_union:
                        attr_order.append(a)
                    attr_union.add(a)
            else:
                other_dicts.append(rec)

    df = pd.DataFrame(row_dicts)            # eligible for augmentation
    other_df = pd.DataFrame(other_dicts)    # kept as-is
    orig_considered = len(df)

    # if generations is 0, we skip augmentation but still use the input labeling budget
    if args.generations <= 0:
        print("ℹ️  Generations is set to 0, skipping augmentation.")
        

   # ── 3a. Sub-sample to --size and budget ────────────────────────────────────
    combined = (
        pd.concat([df, other_df], ignore_index=True)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    if len(combined) > args.size:
        combined = combined.iloc[: args.size]

    augmented_label_set = set(map(int, args.augmented_labels.split(",")))

    if 0 in augmented_label_set:
        raise ValueError(
            "Augmenting label 0 is not permitted. "
            "Remove 0 from --augmented-labels."
        )

    df = combined[combined["Label"].isin(augmented_label_set)]
    other_df = combined[~combined["Label"].isin(augmented_label_set)]
    initial_counts = _label_counts(pd.concat([df, other_df], ignore_index=True))
    print(f"📊 Initial label distribution: {initial_counts}")


    df = df.sample(frac=args.augmentation_budget_ratio, random_state=args.seed)

    # ── 3b. Ensure every attr_1/_2/_3 column exists ────────────────────────
    for a in attr_union:
        for suffix in ("_1", "_2", "_3"):
            col = f"{a}{suffix}"
            if col not in df.columns:
                df[col] = ""
    

    post_budget_counts = _label_counts(df)
    print(f"📊 After budget ratio        : {post_budget_counts}")

    # ── EARLY EXIT WHEN NO GENERATIONS ARE REQUESTED ────────────────────────
    if args.generations <= 0:
        print("ℹ️  Generations is 0 → skipping augmentation steps, "
              "but keeping label-budget split.")

        # ✨ NEW — drop rows that belong to the augmented-label set
        aug_labels = set(map(int, args.augmented_labels.split(",")))
        other_df = other_df[~other_df["Label"].isin(aug_labels)]

        # merge real rows
        full_df = pd.concat([df, other_df], ignore_index=True)
        lines = full_df.apply(lambda r: row_to_ditto(r, attr_order), axis=1)
        lines = lines.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        # 7. Write output
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        print(f"🔄 Writing output to {args.output} …")
        lines.to_csv(
            args.output,
            index=False,
            header=False,
            quoting=csv.QUOTE_NONE,
            escapechar=",",
        )
        final_size = len(lines)
        print(f"💾 Wrote {final_size:,} lines → {args.output}")

        # 8. Log
        final_counts = _label_counts(full_df)
        print(f"📊 Final (no augmentation): {final_counts}")
        log_entry = {
            "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds"),
            "input": args.input,
            "output": args.output,
            "sizes": {
                "initial": initial_counts,
                "post_budget": post_budget_counts,
                "final": final_counts,
            },
            "params": {
                "generations": args.generations,
                "alpha": args.alpha,
                "alpha_delta": args.alpha_delta,
                "model": args.model,
                "batch_size": args.batch_size,
                "num_beams": args.num_beams,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "seed": args.seed,
            },
        }
        LOG_FILE = os.path.join(os.path.dirname(args.output), "run_log.json")
        with open(LOG_FILE, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry) + "\n")
        print(f"📜 Appended run log to {LOG_FILE}")
        return 


    # ── 4. Load model ───────────────────────────────────────────────────────
    print("🔄 Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── 5. Iterative generations ────────────────────────────────────────────
    current = df.assign(generation=0, parent=-1)
    synth_frames: List[pd.DataFrame] = []


    print(f"🔄 Starting {args.generations} generations with {len(current):,} rows")

    for g in range(args.generations):
        print(f"=== Generation {g + 1}/{args.generations} ===")

        t0 = time.perf_counter()                       # ▶︎ START TIMER
        nxt = augment(
            current.copy(),
            attr_order,
            tokenizer,
            model,
            device,
            args.alpha,
            args.alpha_delta,
            g,
            args.batch_size,
            args.num_beams,
            args.top_k,
            args.top_p,
            args.temperature,
        )
        dt = time.perf_counter() - t0                  # ◀︎ STOP TIMER

        n_rows = len(nxt)
        if n_rows:                                     # avoid div-by-zero
            rows_per_sec   = n_rows / dt
            ms_per_row     = (dt / n_rows) * 1000
            print(f"⏱️  Throughput: {rows_per_sec:,.1f} rows/s "
                f"({ms_per_row:,.1f} ms per row) over {dt:,.2f}s")

        print(f"✨ Generated {n_rows:,} synthetic rows")

        synth_frames.append(nxt)
        current = nxt


    # ── 6. Merge & shuffle ─────────────────────────────────────────────────
    full_df = pd.concat([df, *synth_frames, other_df], ignore_index=True)
    lines = full_df.apply(lambda r: row_to_ditto(r, attr_order), axis=1)
    lines = lines.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── 7. Write output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"🔄 Writing output to {args.output} …")
    lines.to_csv(
        args.output,
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        escapechar=",",
    )
    final_size = len(lines)
    print(f"💾 Wrote {final_size:,} lines → {args.output}")

    # ── 8. Append run log ──────────────────────────────────────────────────
    final_counts = _label_counts(full_df)
    print(f"📊 After augmentation (final): {final_counts}")

    log_entry = {
        "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds"),
        "input": args.input,
        "output": args.output,
        "sizes": {
            "initial": initial_counts,
            "post_budget": post_budget_counts,
            "final": final_counts,
        },
        "params": {
            "generations": args.generations,
            "alpha": args.alpha,
            "alpha_delta": args.alpha_delta,
            "model": args.model,
            "batch_size": args.batch_size,
            "num_beams": args.num_beams,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    }
    LOG_FILE = os.path.join(os.path.dirname(args.output), "run_log.json")
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(log_entry) + "\n")
    print(f"📜 Appended run log to {LOG_FILE}")


if __name__ == "__main__":
    main()
