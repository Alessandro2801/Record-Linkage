#!/usr/bin/env python3
"""
ground_truth.py — Generazione della Ground Truth per Record Linkage.

Carica il mediated schema normalizzato, esegue pulizia VIN avanzata,
genera coppie positive e negative stratificate, e produce gli split
train/val/test.

Usage:
    python -m src.preparation.ground_truth
"""

import pandas as pd
import numpy as np
import re
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import train_test_split

from src.config import (
    MEDIATED_SCHEMA_NORMALIZED_PATH,
    GROUND_TRUTH_DIR,
    GT_SPLITS_DIR,
    GT_EVAL_PATH,
    GT_NEGATIVE_RATIO,
    RANDOM_SEED,
    VIN_MAX_DIFF,
    MP_POOL_SIZE,
    print_hw_info,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_vin_checksum(vin):
    """
    Algoritmo ufficiale per la validazione della 9a cifra (Check Digit) del VIN.
    """
    if not isinstance(vin, str) or len(vin) != 17:
        return False

    val_map = {
        'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8,
        'J':1, 'K':2, 'L':3, 'M':4, 'N':5, 'P':7, 'R':9, 'S':2,
        'T':3, 'U':4, 'V':5, 'W':6, 'X':7, 'Y':8, 'Z':9,
        '0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9
    }
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    try:
        total = sum(val_map[vin[i]] * weights[i] for i in range(17) if i != 8)
        remainder = total % 11
        check_digit = 'X' if remainder == 10 else str(remainder)
        return vin[8] == check_digit
    except KeyError:
        return False


def _validate_vin_batch(vins):
    """Valida un batch di VIN in un singolo worker (per multiprocessing)."""
    return [is_valid_vin_checksum(v) for v in vins]


def validate_vins_parallel(vin_series, n_workers=None):
    """Valida i VIN in parallelo con multiprocessing."""
    if n_workers is None:
        n_workers = MP_POOL_SIZE

    vins = vin_series.tolist()
    n = len(vins)

    if n < 10_000 or n_workers <= 1:
        # Non conviene parallelizzare per pochi record
        return [is_valid_vin_checksum(v) for v in vins]

    chunk_size = max(1, n // n_workers)
    chunks = [vins[i:i + chunk_size] for i in range(0, n, chunk_size)]

    print(f"  Validazione VIN: {n:,} record su {n_workers} workers...")
    with Pool(n_workers) as pool:
        results = pool.map(_validate_vin_batch, chunks)

    return [v for batch in results for v in batch]


def _dedup_group(args):
    """Worker function per deduplicare un singolo gruppo VIN."""
    group_values, colonne_da_confrontare, max_diff = args
    gruppo = pd.DataFrame(group_values)

    if len(gruppo) == 1:
        return [gruppo.iloc[0].to_dict()]

    usati = set()
    result = []

    for i in range(len(gruppo)):
        if i in usati:
            continue
        row_i = gruppo.iloc[i]
        cluster = [i]
        for j in range(i + 1, len(gruppo)):
            if j in usati:
                continue
            row_j = gruppo.iloc[j]
            diff = 0
            for col in colonne_da_confrontare:
                v1, v2 = row_i[col], row_j[col]
                if pd.isna(v1) and pd.isna(v2):
                    continue
                if v1 != v2:
                    diff += 1
                    if diff > max_diff:
                        break
            if diff <= max_diff:
                cluster.append(j)

        for idx in cluster:
            usati.add(idx)

        result.append(gruppo.iloc[cluster[0]].to_dict())

    return result


def deduplica_vin_per_similarita(df, vin_col='vin', date_col='pubblication_date', max_diff=3):
    colonne_da_confrontare = [
        c for c in df.columns
        if c not in {vin_col, date_col, 'id_source_vehicles', 'id_source_used_cars', 'description'}
    ]

    df = df.copy()
    df['temp_date_sort'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(by='temp_date_sort', ascending=False)

    # Prepara argomenti per multiprocessing
    groups = []
    single_records = []
    for vin, gruppo in df.groupby(vin_col):
        if len(gruppo) == 1:
            single_records.append(gruppo.iloc[0].to_dict())
        else:
            groups.append((gruppo.to_dict('records'), colonne_da_confrontare, max_diff))

    print(f"  Dedup VIN: {len(single_records):,} singoli + {len(groups):,} gruppi da deduplicare")

    # Parallelizza solo i gruppi con più di un VIN
    record_finali = list(single_records)

    if groups:
        n_workers = min(MP_POOL_SIZE, len(groups))
        if n_workers > 1 and len(groups) > 100:
            with Pool(n_workers) as pool:
                results = pool.map(_dedup_group, groups)
            for batch in results:
                record_finali.extend(batch)
        else:
            for g in groups:
                record_finali.extend(_dedup_group(g))

    df_res = pd.DataFrame(record_finali).reset_index(drop=True)
    if 'temp_date_sort' in df_res.columns:
        df_res = df_res.drop(columns=['temp_date_sort'])

    return df_res


def pulizia_vin_avanzata(df, colonna_vin='vin', colonna_data='pubblication_date'):
    """
    Esegue pulizia alfanumerica, rimozione null, validazione checksum e deduplicazione.
    """
    # 1. RIMOZIONE RECORD CON VIN NULLI
    df_clean = df.dropna(subset=[colonna_vin]).copy()

    # 2. NORMALIZZAZIONE E PULIZIA STRINGA
    df_clean[colonna_vin] = df_clean[colonna_vin].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

    # 3. FILTRI FORMATO E CARATTERI VIETATI (I, O, Q)
    regex_legale = r'^[A-HJ-NPR-Z0-9]{17}$'
    df_clean = df_clean[df_clean[colonna_vin].str.contains(regex_legale, regex=True)]

    # 4. RIMOZIONE PLACEHOLDER
    regex_placeholder = r'^(.)\1{16}$|12345678|ABCDEFGH'
    df_clean = df_clean[~df_clean[colonna_vin].str.contains(regex_placeholder, regex=True)]

    # 5. VALIDAZIONE CHECKSUM (parallelizzata)
    df_clean['vin_valido'] = validate_vins_parallel(df_clean[colonna_vin])
    df_clean = df_clean[df_clean['vin_valido'] == True].drop(columns=['vin_valido'])

    # 6. DEDUPLICAZIONE AVANZATA VIN
    df_clean = deduplica_vin_per_similarita(
        df_clean, vin_col=colonna_vin, date_col=colonna_data, max_diff=VIN_MAX_DIFF
    )

    return df_clean


def genera_negativi_stratificati(df, n_target, positivi_ids, random_state=42):
    """
    Genera negativi stratificati in 2 fasce di difficoltà coerenti con il blocking:
    - HARD (60%): stesso manufacturer + year + body_type
    - MEDIUM (40%): stesso manufacturer + year
    Fallback a random puro se non si raggiunge la quota.
    """
    rng = np.random.RandomState(random_state)

    df_neg = df.copy()
    df_neg['_mfr'] = df_neg['manufacturer'].fillna('unknown').astype(str).str.lower().str.strip()
    df_neg['_year'] = df_neg['year'].fillna(0).astype(int)
    df_neg['_body'] = df_neg['body_type'].fillna('unknown').astype(str).str.lower().str.strip()

    seen_pairs = set(positivi_ids)

    def sample_pairs_from_groups(grouped, quota, label_name):
        pairs = []
        for _, group in grouped:
            if len(group) < 2:
                continue
            ids = group['id_univoco'].values
            vins = group['vin'].values
            n_group = len(ids)

            if n_group <= 50:
                for i in range(n_group):
                    for j in range(i + 1, n_group):
                        if vins[i] != vins[j]:
                            a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                            if (a, b) not in seen_pairs:
                                pairs.append((a, b))
            else:
                n_samples = min(n_group * 3, 500)
                idx1 = rng.randint(0, n_group, size=n_samples)
                idx2 = rng.randint(0, n_group, size=n_samples)
                for i, j in zip(idx1, idx2):
                    if i != j and vins[i] != vins[j]:
                        a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                        if (a, b) not in seen_pairs:
                            pairs.append((a, b))

        pairs = list(set(pairs))

        if len(pairs) > quota:
            selected_idx = rng.choice(len(pairs), size=quota, replace=False)
            pairs = [pairs[i] for i in selected_idx]

        print(f"  {label_name}: richiesti {quota}, disponibili {len(pairs)}")
        return pairs

    # HARD (60%)
    quota_hard = int(n_target * 0.6)
    grouped_hard = df_neg.groupby(['_mfr', '_year', '_body'])
    hard_pairs = sample_pairs_from_groups(grouped_hard, quota_hard, "HARD")
    seen_pairs.update(hard_pairs)

    # MEDIUM (40% + deficit)
    quota_medium = n_target - len(hard_pairs)
    grouped_medium = df_neg.groupby(['_mfr', '_year'])
    medium_pairs = sample_pairs_from_groups(grouped_medium, quota_medium, "MEDIUM")
    seen_pairs.update(medium_pairs)

    # FALLBACK
    all_pairs = hard_pairs + medium_pairs
    if len(all_pairs) < n_target:
        remaining = n_target - len(all_pairs)
        print(f"  FALLBACK random (stesso anno): generando {remaining} coppie aggiuntive...")
        grouped_year = df_neg.groupby(['_year'])
        fallback = sample_pairs_from_groups(grouped_year, remaining, "FALLBACK")
        seen_pairs.update(fallback)
        all_pairs.extend(fallback)

    total = len(all_pairs)
    n_h, n_m = len(hard_pairs), len(medium_pairs)
    n_f = total - n_h - n_m
    print(f"\n  Distribuzione negativi: HARD={n_h} ({n_h/total*100:.1f}%), "
          f"MEDIUM={n_m} ({n_m/total*100:.1f}%), FALLBACK={n_f} ({n_f/total*100:.1f}%)")

    return all_pairs[:n_target]


def genera_ground_truth(df_mediato, ratio_negativi=2.0, random_state=42):
    """
    Genera GT includendo match A-B, A-A e B-B.
    Formato: (id_A, attr_A, id_B, attr_B, label)
    Negativi stratificati per difficoltà (hard/medium), coerenti con il blocking.
    """
    np.random.seed(random_state)

    attributi = [
        'location', 'price', 'year', 'manufacturer', 'model', 'cylinders',
        'fuel_type', 'mileage', 'transmission', 'traction',
        'body_type', 'main_color', 'description',
        'latitude', 'longitude', 'pubblication_date'
    ]

    df = df_mediato.copy()
    df['id_univoco'] = df['id_source_vehicles'].fillna(df['id_source_used_cars']).astype(str)

    df_per_join = df[['id_univoco', 'vin'] + attributi]

    # GENERAZIONE POSITIVI (Label 1)
    positivi = pd.merge(
        df_per_join, df_per_join, on='vin', suffixes=('_A', '_B')
    )
    positivi = positivi[positivi['id_univoco_A'] < positivi['id_univoco_B']].copy()
    positivi['label'] = 1
    positivi = positivi.rename(columns={'id_univoco_A': 'id_A', 'id_univoco_B': 'id_B'})
    positivi = positivi.drop(columns=['vin'])

    n_positivi = len(positivi)
    print(f"Coppie positive totali (A-B, A-A, B-B): {n_positivi}")

    # GENERAZIONE NEGATIVI STRATIFICATI (Label 0)
    n_negativi_target = int(n_positivi * ratio_negativi)
    positivi_ids = set(zip(positivi['id_A'], positivi['id_B']))

    print(f"\nGenerazione {n_negativi_target} negativi stratificati...")
    neg_pairs = genera_negativi_stratificati(
        df_per_join, n_negativi_target, positivi_ids, random_state
    )

    df_lookup = df_per_join.drop_duplicates(subset=['id_univoco']).set_index('id_univoco')

    neg_A_ids = [p[0] for p in neg_pairs]
    neg_B_ids = [p[1] for p in neg_pairs]

    neg_A = df_lookup.loc[neg_A_ids].reset_index()
    neg_B = df_lookup.loc[neg_B_ids].reset_index()

    negativi = pd.concat([
        neg_A.rename(columns={c: f'{c}_A' if c != 'id_univoco' else 'id_A' for c in neg_A.columns}),
        neg_B.rename(columns={c: f'{c}_B' if c != 'id_univoco' else 'id_B' for c in neg_B.columns})
    ], axis=1)

    negativi = negativi.drop(columns=['vin_A', 'vin_B'])
    negativi['label'] = 0

    print(f"Coppie negative generate: {len(negativi)}")

    # ASSEMBLAGGIO FINALE
    ground_truth = pd.concat([positivi, negativi], ignore_index=True)
    ground_truth = ground_truth.sample(frac=1, random_state=random_state).reset_index(drop=True)

    cols_A = ['id_A'] + [f"{a}_A" for a in attributi]
    cols_B = ['id_B'] + [f"{a}_B" for a in attributi]
    ground_truth = ground_truth[cols_A + cols_B + ['label']]

    return ground_truth, positivi, negativi


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()
    print_hw_info()

    # 1. Carica mediated schema normalizzato
    print(f"Caricamento mediated schema da {MEDIATED_SCHEMA_NORMALIZED_PATH}...")
    df_mediated_norm = pd.read_csv(
        MEDIATED_SCHEMA_NORMALIZED_PATH,
        dtype={'id_source_vehicles': 'object'}
    )
    print(f"Record pre-pulizia: {len(df_mediated_norm)}")

    # 2. Pulizia VIN avanzata
    df_sanificato_norm = pulizia_vin_avanzata(df_mediated_norm)
    print(f"Record post-pulizia: {len(df_sanificato_norm)}")

    # 3. Genera Ground Truth
    ground_truth_df, positivi, negativi = genera_ground_truth(
        df_sanificato_norm, ratio_negativi=GT_NEGATIVE_RATIO, random_state=RANDOM_SEED
    )
    print(f"\nShape finale Ground Truth: {ground_truth_df.shape}")
    print(f"Distribuzione Label:\n{ground_truth_df['label'].value_counts()}")

    # 4. Salva GT completa
    gt_path = GROUND_TRUTH_DIR / "ground_truth.csv"
    ground_truth_df.to_csv(gt_path, index=False)
    print(f"\nGround Truth salvata in: {gt_path}")

    # 5. Split GT -> train vs eval (STRATIFICATO)
    GT_train, GT_eval = train_test_split(
        ground_truth_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=ground_truth_df['label']
    )
    GT_eval.to_csv(GT_EVAL_PATH, index=False)

    print(f"Split completato: Train={len(GT_train)}, Eval={len(GT_eval)}")

    # 6. Split train -> train/val/test (70/15/15)
    train_set, temp_set = train_test_split(
        GT_train,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=GT_train['label']
    )
    val_set, test_set = train_test_split(
        temp_set,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_set['label']
    )

    GT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_set.to_csv(GT_SPLITS_DIR / "train.csv", index=False)
    val_set.to_csv(GT_SPLITS_DIR / "val.csv", index=False)
    test_set.to_csv(GT_SPLITS_DIR / "test.csv", index=False)

    print(f"Split train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print("Generazione Ground Truth completata.")


if __name__ == '__main__':
    main()
