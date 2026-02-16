
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import csv
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# BART vector mix‑up
# --------------------------------------------------------------------------- #

def interpolate_sentences(
    tokenizer: BartTokenizer,
    model: BartForConditionalGeneration,
    s1: str,
    s2: str,
    device: torch.device,
    alpha: float,
    alpha_delta: float,
    max_len: int = 512,
) -> str:
    toks = tokenizer([s1, s2], return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        enc_out = model.get_encoder()(toks.input_ids,
                                      attention_mask=toks.attention_mask)
    h1, h2 = enc_out.last_hidden_state
    
    w = alpha + np.random.uniform(0, alpha_delta)
    h_mix = (1 - w) * h1 + w * h2
    enc_mix = BaseModelOutput(last_hidden_state=h_mix.unsqueeze(0))

    start = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    ids = model.generate(
        decoder_input_ids=start,
        encoder_outputs=enc_mix,
        max_new_tokens=max_len,
        num_return_sequences=1,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)



# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

def augment(
    df: pd.DataFrame,
    attrs: List[str],
    tokenizer: BartTokenizer,
    model: BartForConditionalGeneration,
    device: torch.device,
    alpha: float,
    alpha_delta: float,
    gen: int,
) -> pd.DataFrame:
    """One generation of synthetic pairs."""
    new_rows: List[Dict[str, str]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        side3 = {}
        for a in attrs:
            v1 = row.get(f"{a}_1", "").strip()
            v2 = row.get(f"{a}_2", "").strip()
            side3[f"{a}_3"] = (
                interpolate_sentences(tokenizer, model, v1, v2, device,
                                      alpha, alpha_delta)
                if v1 and v2 else v1 or v2
            )

        # create (1,3) and (2,3) pairs
        merged = {**row.to_dict(), **side3}
        for left, right in (("_1", "_3"), ("_2", "_3")):
            new_row = {f"{a}_1": merged[f"{a}{left}"] for a in attrs}
            new_row.update({f"{a}_2": merged[f"{a}{right}"] for a in attrs})
            new_row["Label"] = 1
            new_row["generation"] = gen + 1
            new_row["parent"] = idx
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)

