#!/usr/bin/env python3
"""
generate.py — Generazione coppie candidate con due strategie di blocking.

Strategie:
    B1: Stesso anno + prefisso manufacturer (3 char) + Jaro-Winkler manufacturer >= 0.95
    B2: B1 + Jaro-Winkler model >= 0.85 + stesso fuel_type

Usage:
    python -m src.blocking.generate
"""

import pandas as pd
import jellyfish
import json
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import (
    CLEANED_DATASET_PATH,
    GROUND_TRUTH_PATH,
    BLOCKING_DIR,
    BLOCKING_STATS_DIR,
    BLOCKING_MFR_THRESHOLD,
    BLOCKING_MODEL_THRESHOLD,
    RANDOM_SEED,
    MP_POOL_SIZE,
    print_hw_info,
    blocking_candidates_path,
    blocking_split_path,
    ensure_dirs,
)


def build_inverted_index(df, year_col='year', mfr_col='manufacturer'):
    """
    Costruisce dizionario {(year_str, mfr_prefix_3): [lista_indici_dataframe]}.
    """
    index = defaultdict(list)
    years = df[year_col].fillna(0).astype(int).astype(str).values
    mfrs = df[mfr_col].fillna('unknown').astype(str).str.lower().str.strip().values

    for i in range(len(df)):
        key = (years[i], mfrs[i][:3])
        index[key].append(i)

    return index


def _process_bucket(args):
    """Worker: processa un singolo bucket e restituisce le coppie trovate."""
    bucket, mfrs, models, fuels, ids, strategy, mfr_threshold, model_threshold = args
    pairs = []
    n_bucket = len(bucket)

    for i_pos in range(n_bucket):
        for j_pos in range(i_pos + 1, n_bucket):
            idx_i = bucket[i_pos]
            idx_j = bucket[j_pos]

            mfr_sim = jellyfish.jaro_winkler_similarity(mfrs[idx_i], mfrs[idx_j])
            if mfr_sim < mfr_threshold:
                continue

            if strategy == 'B2':
                model_sim = jellyfish.jaro_winkler_similarity(models[idx_i], models[idx_j])
                if model_sim < model_threshold:
                    continue
                if fuels[idx_i] != fuels[idx_j]:
                    continue

            id_a, id_b = ids[idx_i], ids[idx_j]
            if id_a > id_b:
                id_a, id_b = id_b, id_a
                idx_i, idx_j = idx_j, idx_i

            pairs.append((id_a, id_b, idx_i, idx_j))

    return pairs


def generate_candidate_pairs(df, strategy, mfr_threshold=0.95, model_threshold=0.85):
    """
    Genera coppie candidate (cross-source E within-source) usando inverted index + filtri fuzzy.
    Parallelizzato su tutti i core disponibili.
    """
    attributi = [
        'id_univoco', 'location', 'price', 'year', 'manufacturer', 'model',
        'cylinders', 'fuel_type', 'mileage', 'transmission', 'traction',
        'body_type', 'main_color', 'description', 'latitude', 'longitude',
        'pubblication_date'
    ]

    cols = [c for c in attributi if c in df.columns]
    df_work = df[cols].copy().reset_index(drop=True)

    # Normalizza per confronto
    df_work['_mfr'] = df_work['manufacturer'].fillna('unknown').astype(str).str.lower().str.strip()
    df_work['_year'] = df_work['year'].fillna(0).astype(int).astype(str)
    df_work['_model'] = df_work['model'].fillna('').astype(str).str.lower().str.strip()
    df_work['_fuel'] = df_work['fuel_type'].fillna('unknown').astype(str).str.lower().str.strip()

    print(f"  Costruzione inverted index su {len(df_work)} record...")
    inv_index = build_inverted_index(df_work, year_col='_year', mfr_col='_mfr')

    mfrs = df_work['_mfr'].values
    models = df_work['_model'].values
    fuels = df_work['_fuel'].values
    ids = df_work['id_univoco'].values

    # Sub-partizionamento di TUTTI i bucket per model_prefix_3
    final_buckets = []
    for bucket in inv_index.values():
        sub = defaultdict(list)
        for idx in bucket:
            model_prefix = models[idx][:3] if models[idx] else ''
            sub[model_prefix].append(idx)
        final_buckets.extend(sub.values())

    n_buckets = len(final_buckets)
    print(f"  Blocking key: (year, mfr[:3], model[:3]) -> {n_buckets:,} bucket")
    print(f"  Generazione coppie con strategia {strategy} ({n_buckets:,} bucket, {MP_POOL_SIZE} workers)...")

    # Prepara argomenti per multiprocessing
    bucket_args = [
        (bucket, mfrs, models, fuels, ids, strategy, mfr_threshold, model_threshold)
        for bucket in final_buckets
    ]

    # Processa bucket in parallelo
    seen = set()
    candidate_pairs = []

    if MP_POOL_SIZE > 1 and n_buckets > 50:
        # Usa chunk_size per ridurre overhead IPC
        chunk = max(1, n_buckets // (MP_POOL_SIZE * 4))
        with Pool(MP_POOL_SIZE) as pool:
            for batch_pairs in tqdm(
                pool.imap_unordered(_process_bucket, bucket_args, chunksize=chunk),
                total=n_buckets, desc="  Blocking", unit="bucket"
            ):
                for id_a, id_b, idx_i, idx_j in batch_pairs:
                    pair_key = (id_a, id_b)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        candidate_pairs.append((idx_i, idx_j))
    else:
        for args in tqdm(bucket_args, desc="  Blocking", unit="bucket"):
            for id_a, id_b, idx_i, idx_j in _process_bucket(args):
                pair_key = (id_a, id_b)
                if pair_key not in seen:
                    seen.add(pair_key)
                    candidate_pairs.append((idx_i, idx_j))

    print(f"  Totale coppie candidate: {len(candidate_pairs)}")

    if len(candidate_pairs) == 0:
        return pd.DataFrame()

    a_indices = [p[0] for p in candidate_pairs]
    b_indices = [p[1] for p in candidate_pairs]

    attr_cols = [c for c in attributi if c != 'id_univoco']

    result_A = df_work.loc[a_indices, ['id_univoco'] + attr_cols].reset_index(drop=True)
    result_B = df_work.loc[b_indices, ['id_univoco'] + attr_cols].reset_index(drop=True)

    result_A.columns = ['id_A'] + [f'{c}_A' for c in attr_cols]
    result_B.columns = ['id_B'] + [f'{c}_B' for c in attr_cols]

    candidates = pd.concat([result_A, result_B], axis=1)

    drop_cols = [c for c in candidates.columns if c.startswith('_')]
    if drop_cols:
        candidates = candidates.drop(columns=drop_cols)

    return candidates


def evaluate_blocking(candidates, gt, n_total_records):
    """Valuta la qualità del blocking: pairs completeness e reduction ratio."""
    gt_pos = gt[gt['label'] == 1]
    gt_pos_pairs = set(zip(gt_pos['id_A'].astype(str), gt_pos['id_B'].astype(str)))

    cand_pairs = set(zip(candidates['id_A'].astype(str), candidates['id_B'].astype(str)))
    cand_pairs_inv = set(zip(candidates['id_B'].astype(str), candidates['id_A'].astype(str)))
    cand_pairs_all = cand_pairs | cand_pairs_inv

    captured = gt_pos_pairs & cand_pairs_all
    positives_captured = len(captured)
    positives_in_gt = len(gt_pos_pairs)

    recall = (positives_captured / positives_in_gt * 100) if positives_in_gt > 0 else 0

    total_possible = n_total_records * (n_total_records - 1) // 2
    reduction_ratio = 1 - (len(candidates) / total_possible) if total_possible > 0 else 0

    stats = {
        "positives_in_gt": positives_in_gt,
        "total_pairs_selected": len(candidates),
        "positives_captured": positives_captured,
        "recall_percentage": round(recall, 2),
        "reduction_ratio": round(reduction_ratio, 8)
    }

    print(f"  Recall: {recall:.2f}% ({positives_captured}/{positives_in_gt})")
    print(f"  Coppie selezionate: {len(candidates)}")
    print(f"  Reduction Ratio: {reduction_ratio:.6f}")

    return stats


def label_candidates(candidates, gt):
    """Incrocia le coppie candidate con la GT per assegnare label (vettorizzato)."""
    gt_pos = gt[gt['label'] == 1]
    gt_pos_pairs = set(zip(gt_pos['id_A'].astype(str), gt_pos['id_B'].astype(str)))
    gt_pos_pairs_inv = set(zip(gt_pos['id_B'].astype(str), gt_pos['id_A'].astype(str)))
    all_pos = gt_pos_pairs | gt_pos_pairs_inv

    # Vettorizzato: crea tuple e controlla appartenenza
    pair_tuples = list(zip(candidates['id_A'].astype(str), candidates['id_B'].astype(str)))
    candidates['label'] = [1 if p in all_pos else 0 for p in pair_tuples]

    print(f"  Labeling: {candidates['label'].sum()} positivi, {(candidates['label'] == 0).sum()} negativi")
    return candidates


def main():
    ensure_dirs()
    print_hw_info()

    # 1. Carica dataset campionato (generato da ground_truth.py)
    print(f"Caricamento dataset campionato da {CLEANED_DATASET_PATH}...")
    if not CLEANED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset campionato non trovato: {CLEANED_DATASET_PATH}\n"
            "Eseguire prima: python -m src.preparation.ground_truth"
        )
    df_combined = pd.read_csv(CLEANED_DATASET_PATH, dtype={'id_source_vehicles': 'object'}, low_memory=False)
    df_combined['id_univoco'] = df_combined['id_source_vehicles'].fillna(df_combined['id_source_used_cars']).astype(str)
    print(f"Dataset campionato: {len(df_combined)} record")

    # 4. Carica GT per labeling
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    print(f"\nGround Truth caricata: {len(gt)} coppie ({gt['label'].sum()} positivi)")

    for strategy in ['B1', 'B2']:
        print(f"\n{'='*60}")
        print(f"STRATEGIA: {strategy}")
        print(f"{'='*60}")

        # 5. Genera coppie candidate
        candidates = generate_candidate_pairs(
            df_combined, strategy=strategy,
            mfr_threshold=BLOCKING_MFR_THRESHOLD,
            model_threshold=BLOCKING_MODEL_THRESHOLD,
        )

        if len(candidates) == 0:
            print(f"  ATTENZIONE: nessuna coppia generata per {strategy}!")
            continue

        # 6. Assegna label dalla GT
        candidates = label_candidates(candidates, gt)

        # 7. Split stratificato train/val/test (70/15/15)
        print(f"\n  Split stratificato train/val/test...")
        train_df, temp_df = train_test_split(
            candidates, test_size=0.30, stratify=candidates['label'], random_state=RANDOM_SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df['label'], random_state=RANDOM_SEED
        )
        print(f"  Train: {len(train_df)} (pos: {train_df['label'].sum()})")
        print(f"  Val:   {len(val_df)} (pos: {val_df['label'].sum()})")
        print(f"  Test:  {len(test_df)} (pos: {test_df['label'].sum()})")

        # 8. Valuta blocking (sul file completo)
        stats = evaluate_blocking(candidates, gt, len(df_combined))
        stats['strategy'] = strategy
        stats['sample_size'] = len(df_combined)
        stats['mfr_threshold'] = BLOCKING_MFR_THRESHOLD
        stats['model_threshold'] = BLOCKING_MODEL_THRESHOLD if strategy == 'B2' else None

        # 9. Salva file completo + split
        csv_path = blocking_candidates_path(strategy)
        train_path = blocking_split_path(strategy, 'train')
        val_path = blocking_split_path(strategy, 'val')
        test_path = blocking_split_path(strategy, 'test')
        json_path = BLOCKING_STATS_DIR / f'stats_{strategy}.json'

        print(f"\n  Salvataggio CSV completo ({len(candidates):,} righe)...", flush=True)
        candidates.to_csv(csv_path, index=False)
        print(f"  -> {csv_path}")

        print(f"  Salvataggio train ({len(train_df):,} righe)...", flush=True)
        train_df.to_csv(train_path, index=False)
        print(f"  -> {train_path}")

        print(f"  Salvataggio val ({len(val_df):,} righe)...", flush=True)
        val_df.to_csv(val_path, index=False)
        print(f"  -> {val_path}")

        print(f"  Salvataggio test ({len(test_df):,} righe)...", flush=True)
        test_df.to_csv(test_path, index=False)
        print(f"  -> {test_path}")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
