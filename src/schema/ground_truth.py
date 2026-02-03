#!/usr/bin/env python3
"""
ground_truth.py
Script per la generazione della Ground Truth per Record Linkage.

Usage:
    python ground_truth.py [--base-path PATH] [--ratio FLOAT]
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def is_valid_vin_checksum(vin: str) -> bool:
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


def count_differences(row1, row2, colonne_da_confrontare):
    """Conta le differenze tra due righe."""
    diff = 0
    for col in colonne_da_confrontare:
        v1, v2 = row1[col], row2[col]
        if pd.isna(v1) and pd.isna(v2):
            continue
        if v1 != v2:
            diff += 1
    return diff


def deduplica_vin_per_similarita(df, vin_col='vin', date_col='pubblication_date', max_diff=3):
    """Deduplica record con stesso VIN basandosi sulla similarità."""
    colonne_da_confrontare = [
        c for c in df.columns
        if c not in {vin_col, date_col, 'id_source_vehicles', 'id_source_used_cars', 'description'}
    ]

    df['temp_date_sort'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    record_finali = []

    for vin, gruppo in df.groupby(vin_col):
        if len(gruppo) == 1:
            record_finali.append(gruppo.iloc[0])
            continue

        gruppo = gruppo.sort_values(by='temp_date_sort', ascending=False)
        usati = set()

        for i, row_i in gruppo.iterrows():
            if i in usati:
                continue
            cluster = [i]
            for j, row_j in gruppo.iterrows():
                if j == i or j in usati:
                    continue
                diff = count_differences(row_i, row_j, colonne_da_confrontare)
                if diff <= max_diff:
                    cluster.append(j)
            
            for idx in cluster:
                usati.add(idx)
            
            record_finali.append(gruppo.loc[cluster].iloc[0])

    df_res = pd.DataFrame(record_finali).reset_index(drop=True)
    if 'temp_date_sort' in df_res.columns:
        df_res = df_res.drop(columns=['temp_date_sort'])
    
    return df_res


def pulizia_vin_avanzata(df, colonna_vin='vin', colonna_data='pubblication_date'):
    """
    Esegue pulizia alfanumerica, rimozione null, validazione checksum e deduplicazione.
    """
    print("Pulizia VIN in corso...")
    
    # 1. Rimozione record con VIN nulli
    df_clean = df.dropna(subset=[colonna_vin]).copy()
    
    # 2. Normalizzazione e pulizia stringa
    df_clean[colonna_vin] = df_clean[colonna_vin].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    
    # 3. Filtri formato e caratteri vietati (I, O, Q)
    regex_legale = r'^[A-HJ-NPR-Z0-9]{17}$'
    df_clean = df_clean[df_clean[colonna_vin].str.contains(regex_legale, regex=True)]
    
    # 4. Rimozione placeholder
    regex_placeholder = r'^(.)\1{16}$|12345678|ABCDEFGH'
    df_clean = df_clean[~df_clean[colonna_vin].str.contains(regex_placeholder, regex=True)]
    
    # 5. Validazione checksum
    print("Validazione checksum VIN...")
    df_clean['vin_valido'] = df_clean[colonna_vin].apply(is_valid_vin_checksum)
    df_clean = df_clean[df_clean['vin_valido'] == True].drop(columns=['vin_valido'])
    
    # 6. Deduplicazione avanzata VIN
    print("Deduplicazione VIN...")
    df_clean = deduplica_vin_per_similarita(df_clean, vin_col=colonna_vin, date_col=colonna_data, max_diff=3)
    
    return df_clean


def genera_ground_truth(df_mediato, ratio_negativi=2.0, random_state=42):
    """
    Genera GT includendo match A-B, A-A e B-B.
    Formato: (id_A, attr_A, id_B, attr_B, label)
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

    # Generazione POSITIVI (Label 1)
    print("Generazione coppie positive...")
    positivi = pd.merge(
        df_per_join,
        df_per_join,
        on='vin',
        suffixes=('_A', '_B')
    )

    positivi = positivi[positivi['id_univoco_A'] < positivi['id_univoco_B']].copy()
    positivi['label'] = 1
    positivi = positivi.rename(columns={'id_univoco_A': 'id_A', 'id_univoco_B': 'id_B'})
    positivi = positivi.drop(columns=['vin'])
    
    n_positivi = len(positivi)
    print(f"Coppie positive totali (A-B, A-A, B-B): {n_positivi}")

    # Generazione NEGATIVI (Label 0)
    print("Generazione coppie negative...")
    n_negativi_target = int(n_positivi * ratio_negativi)
    negativi_list = []
    
    while len(negativi_list) == 0 or len(pd.concat(negativi_list).drop_duplicates(subset=['id_univoco_A', 'id_univoco_B'])) < n_negativi_target:
        idx1 = np.random.choice(df.index, size=n_negativi_target)
        idx2 = np.random.choice(df.index, size=n_negativi_target)
        
        tmp_A = df.loc[idx1, ['id_univoco', 'vin'] + attributi].reset_index(drop=True)
        tmp_B = df.loc[idx2, ['id_univoco', 'vin'] + attributi].reset_index(drop=True)
        
        mask = (tmp_A['id_univoco'] != tmp_B['id_univoco']) & (tmp_A['vin'] != tmp_B['vin'])
        
        swap_mask = tmp_A['id_univoco'] > tmp_B['id_univoco']
        tmp_A.loc[swap_mask], tmp_B.loc[swap_mask] = tmp_B.loc[swap_mask].values, tmp_A.loc[swap_mask].values
        
        batch_neg = pd.concat([
            tmp_A.add_suffix('_A'), 
            tmp_B.add_suffix('_B')
        ], axis=1)[mask]
        
        negativi_list.append(batch_neg)
        combined_neg = pd.concat(negativi_list).drop_duplicates(subset=['id_univoco_A', 'id_univoco_B'])
        if len(combined_neg) >= n_negativi_target:
            negativi = combined_neg.head(n_negativi_target).copy()
            break

    negativi = negativi.rename(columns={'id_univoco_A': 'id_A', 'id_univoco_B': 'id_B'})
    negativi = negativi.drop(columns=['vin_A', 'vin_B'])
    negativi['label'] = 0
    
    print(f"Coppie negative generate: {len(negativi)}")

    # Assemblaggio finale
    ground_truth = pd.concat([positivi, negativi], ignore_index=True)
    ground_truth = ground_truth.sample(frac=1, random_state=random_state).reset_index(drop=True)

    cols_A = ['id_A'] + [f"{a}_A" for a in attributi]
    cols_B = ['id_B'] + [f"{a}_B" for a in attributi]
    ground_truth = ground_truth[cols_A + cols_B + ['label']]

    return ground_truth


def main():
    parser = argparse.ArgumentParser(description='Genera Ground Truth per Record Linkage')
    parser.add_argument('--base-path', type=str, default='.', 
                        help='Path base del progetto Record-Linkage')
    parser.add_argument('--ratio', type=float, default=2.0,
                        help='Rapporto negativi/positivi (default: 2.0)')
    args = parser.parse_args()
    
    base_path = args.base_path
    
    # Paths
    input_path = os.path.join(base_path, 'data/mediated_schema/mediated_schema_normalized.csv')
    output_dir = os.path.join(base_path, 'data/ground_truth/GT_train')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERAZIONE GROUND TRUTH")
    print("=" * 60)
    
    # Caricamento
    print(f"\nCaricamento {input_path}...")
    df_mediated_norm = pd.read_csv(input_path, dtype={'id_source_vehicles': 'object'})
    print(f"Record caricati: {len(df_mediated_norm)}")
    
    # Pulizia VIN
    df_sanificato = pulizia_vin_avanzata(df_mediated_norm)
    print(f"Record post-pulizia: {len(df_sanificato)}")
    
    # Generazione Ground Truth
    ground_truth_df = genera_ground_truth(df_sanificato, ratio_negativi=args.ratio)
    
    print(f"\nShape finale Ground Truth: {ground_truth_df.shape}")
    print(f"Distribuzione Label:\n{ground_truth_df['label'].value_counts()}")
    
    # Split train/val/test (70/15/15)
    print("\nSplit train/val/test...")
    train_df, temp_df = train_test_split(ground_truth_df, test_size=0.3, random_state=42, stratify=ground_truth_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Salvataggio
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Ground Truth salvata:")
    print(f"   Train: {train_path} ({len(train_df)} record)")
    print(f"   Val:   {val_path} ({len(val_df)} record)")
    print(f"   Test:  {test_path} ({len(test_df)} record)")


if __name__ == '__main__':
    main()
