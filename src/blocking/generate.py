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
import numpy as np
import jellyfish
import json
from collections import defaultdict
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

from src.config import (
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    GROUND_TRUTH_PATH,
    BLOCKING_DIR,
    BLOCKING_STATS_DIR,
    BLOCKING_SAMPLE_SIZE,
    BLOCKING_MFR_THRESHOLD,
    BLOCKING_MODEL_THRESHOLD,
    RANDOM_SEED,
    MP_POOL_SIZE,
    print_hw_info,
    blocking_candidates_path,
    blocking_split_path,
    ensure_dirs,
)


def load_mediated_schema(path):
    """Carica il CSV mediato, crea id_univoco, separa le due sorgenti."""
    print(f"Caricamento mediated schema da {path}...")
    df = pd.read_csv(path, dtype={'id_source_vehicles': 'object'}, low_memory=False)
    df['id_univoco'] = df['id_source_vehicles'].fillna(df['id_source_used_cars']).astype(str)
    return df


def stratified_sample(df, n, stratify_cols=['year', 'manufacturer'], random_state=42):
    """
    Campiona N record stratificati per anno e manufacturer.
    Usa groupby + campionamento proporzionale per gruppo.
    """
    if len(df) <= n:
        return df.copy()

    rng = np.random.RandomState(random_state)

    df = df.copy()
    df['_strat_key'] = (
        df[stratify_cols[0]].fillna(0).astype(int).astype(str) + '_' +
        df[stratify_cols[1]].fillna('unknown').astype(str).str.lower().str.strip()
    )

    # Calcola quota proporzionale per gruppo
    group_sizes = df.groupby('_strat_key').size()
    proportions = group_sizes / len(df)
    quotas = (proportions * n).apply(np.floor).astype(int)

    # Distribuisci il residuo ai gruppi più grandi
    deficit = n - quotas.sum()
    if deficit > 0:
        remainder = (proportions * n) - quotas
        top_groups = remainder.nlargest(deficit).index
        quotas[top_groups] += 1

    sampled = []
    for key, group in df.groupby('_strat_key'):
        quota = quotas.get(key, 0)
        if quota > 0:
            sample_size = min(quota, len(group))
            sampled.append(group.sample(n=sample_size, random_state=rng))

    result = pd.concat(sampled, ignore_index=True).drop(columns=['_strat_key'])

    # Se ancora sotto quota (per arrotondamenti), aggiungi random
    if len(result) < n:
        remaining_idx = df.index.difference(result.index)
        extra = df.loc[remaining_idx].sample(n=n - len(result), random_state=rng)
        result = pd.concat([result, extra.drop(columns=['_strat_key'])], ignore_index=True)

    print(f"  Campionamento stratificato: {len(df)} -> {len(result)} record")
    return result


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

    n_buckets = len(inv_index)
    print(f"  Generazione coppie con strategia {strategy} ({n_buckets:,} bucket, {MP_POOL_SIZE} workers)...")

    # Prepara argomenti per multiprocessing
    bucket_args = [
        (bucket, mfrs, models, fuels, ids, strategy, mfr_threshold, model_threshold)
        for bucket in inv_index.values()
    ]

    # Processa bucket in parallelo
    seen = set()
    candidate_pairs = []

    if MP_POOL_SIZE > 1 and n_buckets > 50:
        # Usa chunk_size per ridurre overhead IPC
        chunk = max(1, n_buckets // (MP_POOL_SIZE * 4))
        with Pool(MP_POOL_SIZE) as pool:
            for batch_pairs in pool.imap_unordered(_process_bucket, bucket_args, chunksize=chunk):
                for id_a, id_b, idx_i, idx_j in batch_pairs:
                    pair_key = (id_a, id_b)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        candidate_pairs.append((idx_i, idx_j))
    else:
        for b_idx, args in enumerate(bucket_args):
            for id_a, id_b, idx_i, idx_j in _process_bucket(args):
                pair_key = (id_a, id_b)
                if pair_key not in seen:
                    seen.add(pair_key)
                    candidate_pairs.append((idx_i, idx_j))
            if b_idx % 500 == 0 and b_idx > 0:
                print(f"    Progresso: {b_idx}/{n_buckets} bucket ({len(candidate_pairs)} coppie)")

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

    # 1. Carica dataset mediato
    df = load_mediated_schema(str(MEDIATED_SCHEMA_NORMALIZED_PATH))

    # 2. Separa sorgenti per campionamento bilanciato
    df_vehicles = df[df['id_source_vehicles'].notna()].copy()
    df_usedcars = df[df['id_source_used_cars'].notna()].copy()
    print(f"Vehicles (Craigslist): {len(df_vehicles)} record")
    print(f"UsedCars: {len(df_usedcars)} record")

    # 3. Campiona stratificato per sorgente, poi combina
    print(f"\nCampionamento stratificato di {BLOCKING_SAMPLE_SIZE} record per sorgente...")
    df_veh_sample = stratified_sample(df_vehicles, BLOCKING_SAMPLE_SIZE, random_state=RANDOM_SEED)
    df_uc_sample = stratified_sample(df_usedcars, BLOCKING_SAMPLE_SIZE, random_state=RANDOM_SEED)

    df_combined = pd.concat([df_veh_sample, df_uc_sample], ignore_index=True)
    print(f"Dataset combinato: {len(df_combined)} record")

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
        stats['sample_size_per_source'] = BLOCKING_SAMPLE_SIZE
        stats['mfr_threshold'] = BLOCKING_MFR_THRESHOLD
        stats['model_threshold'] = BLOCKING_MODEL_THRESHOLD if strategy == 'B2' else None

        # 9. Salva file completo + split
        csv_path = blocking_candidates_path(strategy)
        train_path = blocking_split_path(strategy, 'train')
        val_path = blocking_split_path(strategy, 'val')
        test_path = blocking_split_path(strategy, 'test')
        json_path = BLOCKING_STATS_DIR / f'stats_{strategy}.json'

        candidates.to_csv(csv_path, index=False)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)

        print(f"\n  Salvato CSV completo: {csv_path}")
        print(f"  Salvato train: {train_path}")
        print(f"  Salvato val:   {val_path}")
        print(f"  Salvato test:  {test_path}")
        print(f"  Salvato stats: {json_path}")


if __name__ == "__main__":
    main()
