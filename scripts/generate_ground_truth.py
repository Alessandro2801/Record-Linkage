#!/usr/bin/env python3
"""
Script per la generazione della Ground Truth per il Record Linkage.
Convertito da: schema_alignment/ground_truth.ipynb

Questo script:
1. Carica il dataset dello schema mediato normalizzato
2. Pulisce e valida i VIN (rimuove VIN non validi, placeholder, checksum errati)
3. Genera coppie positive (stesso VIN) e negative (VIN diversi)
4. Crea gli split train/val/test per l'addestramento dei modelli
5. Salva tutti i dataset in datasets/ground_truth/
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    """Trova la directory root del progetto."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def is_valid_vin_checksum(vin: str) -> bool:
    """
    Calcola e verifica il check digit (9° carattere) di un VIN.
    Restituisce True se il VIN supera la validazione del checksum.
    """
    if not isinstance(vin, str) or len(vin) != 17:
        return False
    
    # Mappa di traslitterazione carattere -> valore numerico
    transliteration = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
        'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    }
    
    # Pesi per ogni posizione
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    
    try:
        vin_upper = vin.upper()
        total = sum(transliteration.get(char, 0) * weight for char, weight in zip(vin_upper, weights))
        check_digit_calculated = total % 11
        
        if check_digit_calculated == 10:
            check_digit_char = 'X'
        else:
            check_digit_char = str(check_digit_calculated)
        
        return vin_upper[8] == check_digit_char
    except Exception:
        return False


def deduplica_vin_per_similarita(
    df: pd.DataFrame, 
    vin_col: str = 'vin', 
    date_col: str = 'pubblication_date', 
    max_diff: int = 3
) -> pd.DataFrame:
    """
    Rimuove duplicati soft dove i VIN differiscono per pochi caratteri.
    Mantiene il record più recente.
    """
    df_sorted = df.sort_values(by=date_col, ascending=False, na_position='last')
    df_vin_series = df_sorted[vin_col].str.upper().fillna('')
    
    da_eliminare = set()
    processed = list(df_vin_series.items())
    
    for i, (idx1, vin1) in enumerate(processed):
        if idx1 in da_eliminare or len(vin1) != 17:
            continue
        for j, (idx2, vin2) in enumerate(processed[i+1:], start=i+1):
            if idx2 in da_eliminare or len(vin2) != 17:
                continue
            diff = sum(c1 != c2 for c1, c2 in zip(vin1, vin2))
            if diff <= max_diff:
                da_eliminare.add(idx2)
    
    return df_sorted.drop(index=list(da_eliminare))


def pulizia_vin_avanzata(
    df: pd.DataFrame, 
    colonna_vin: str = 'vin', 
    colonna_data: str = 'pubblication_date'
) -> pd.DataFrame:
    """
    Pipeline completa di pulizia dei VIN.
    """
    df_clean = df.copy()
    
    # 1. RIMOZIONE VALORI NULLI
    df_clean = df_clean.dropna(subset=[colonna_vin])
    
    # 2. STANDARDIZZAZIONE
    df_clean[colonna_vin] = (
        df_clean[colonna_vin]
        .astype(str)
        .str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
    )
    
    # 3. FILTRI FORMATO E CARATTERI VIETATI (I, O, Q)
    regex_legale = r'^[A-HJ-NPR-Z0-9]{17}$'
    df_clean = df_clean[df_clean[colonna_vin].str.contains(regex_legale, regex=True)]
    
    # 4. RIMOZIONE PLACEHOLDER
    regex_placeholder = r'^(.)\1{16}$|12345678|ABCDEFGH'
    df_clean = df_clean[~df_clean[colonna_vin].str.contains(regex_placeholder, regex=True)]
    
    # 5. VALIDAZIONE CHECKSUM
    df_clean['vin_valido'] = df_clean[colonna_vin].apply(is_valid_vin_checksum)
    df_clean = df_clean[df_clean['vin_valido']].drop(columns=['vin_valido'])
    
    # 6. DEDUPLICAZIONE VIN
    df_clean = deduplica_vin_per_similarita(
        df_clean, 
        vin_col=colonna_vin, 
        date_col=colonna_data, 
        max_diff=3
    )
    
    return df_clean


def genera_ground_truth(
    df_mediato: pd.DataFrame, 
    ratio_negativi: float = 2.0, 
    random_state: int = 42
) -> pd.DataFrame:
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

    # GENERAZIONE POSITIVI (Label 1)
    print("   Generazione coppie positive...")
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
    print(f"   Coppie positive totali: {n_positivi}")

    # GENERAZIONE NEGATIVI (Label 0)
    print("   Generazione coppie negative...")
    n_negativi_target = int(n_positivi * ratio_negativi)
    negativi_list = []
    
    while len(negativi_list) < n_negativi_target:
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
    
    print(f"   Coppie negative generate: {len(negativi)}")

    # ASSEMBLAGGIO FINALE
    ground_truth = pd.concat([positivi, negativi], ignore_index=True)
    ground_truth = ground_truth.sample(frac=1, random_state=random_state).reset_index(drop=True)

    cols_A = ['id_A'] + [f"{a}_A" for a in attributi]
    cols_B = ['id_B'] + [f"{a}_B" for a in attributi]
    ground_truth = ground_truth[cols_A + cols_B + ['label']]

    return ground_truth


def main():
    """Funzione principale."""
    print("=" * 60)
    print("🎯 Generazione Ground Truth per Record Linkage")
    print("=" * 60)
    
    project_root = get_project_root()
    input_path = project_root / "datasets" / "mediated_schema" / "mediated_schema_normalized.csv"
    output_dir = project_root / "datasets" / "ground_truth"
    gt_train_dir = output_dir / "GT_train"
    
    # Verifica esistenza file di input
    if not input_path.exists():
        print(f"❌ File non trovato: {input_path}")
        print("   Esegui prima generate_mediated_schema.py")
        return 1
    
    # Caricamento dataset
    print(f"\n📂 Caricamento schema mediato: {input_path}")
    df_mediated_norm = pd.read_csv(input_path, dtype={'id_source_vehicles': 'object'})
    print(f"   Record caricati: {len(df_mediated_norm)}")
    
    # Pulizia VIN
    print("\n🔧 Pulizia e validazione VIN...")
    print(f"   Record pre-pulizia: {len(df_mediated_norm)}")
    df_sanificato = pulizia_vin_avanzata(df_mediated_norm)
    print(f"   Record post-pulizia: {len(df_sanificato)}")
    
    # Generazione Ground Truth
    print("\n📊 Generazione Ground Truth...")
    ground_truth_df = genera_ground_truth(df_sanificato, ratio_negativi=2.0)
    print(f"   Shape finale: {ground_truth_df.shape}")
    print(f"   Distribuzione Label:\n{ground_truth_df['label'].value_counts()}")
    
    # Validazione
    assert not ground_truth_df.isna().all(axis=1).any(), "Righe completamente vuote!"
    
    # Salvataggio Ground Truth completo
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_df.to_csv(output_dir / "ground_truth.csv", index=False)
    print(f"\n   ✅ Ground Truth salvato: {output_dir / 'ground_truth.csv'}")
    
    # Split train vs eval (stratificato)
    print("\n📂 Creazione split Train/Eval...")
    GT_train, GT_eval = train_test_split(
        ground_truth_df,
        test_size=0.30,
        random_state=42,
        stratify=ground_truth_df['label']
    )
    print(f"   GT_train: {len(GT_train)} righe")
    print(f"   GT_eval: {len(GT_eval)} righe")
    
    GT_eval.to_csv(output_dir / "GT_eval.csv", index=False)
    print(f"   ✅ GT_eval salvato: {output_dir / 'GT_eval.csv'}")
    
    # Split train/val/test (stratificato)
    print("\n📂 Creazione split Train/Val/Test...")
    train_set, temp_set = train_test_split(
        GT_train, 
        test_size=0.30, 
        random_state=42, 
        stratify=GT_train['label']
    )
    val_set, test_set = train_test_split(
        temp_set, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_set['label']
    )
    
    print(f"   Train: {len(train_set)} righe")
    print(f"   Val: {len(val_set)} righe")
    print(f"   Test: {len(test_set)} righe")
    
    gt_train_dir.mkdir(parents=True, exist_ok=True)
    train_set.to_csv(gt_train_dir / "train.csv", index=False)
    val_set.to_csv(gt_train_dir / "val.csv", index=False)
    test_set.to_csv(gt_train_dir / "test.csv", index=False)
    
    print(f"\n   ✅ Split salvati in: {gt_train_dir}")
    print("\n✅ Generazione Ground Truth completata!")
    return 0


if __name__ == "__main__":
    exit(main())
