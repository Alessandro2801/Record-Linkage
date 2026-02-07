# Record-Linkage

Pipeline di **Entity Resolution** tra due dataset di auto usate negli USA:
- [Craigslist Cars/Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- [US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

## Metodologia

La pipeline implementa questo flusso:
1. Training dei modelli su `storage/ground_truth/splits/train.csv` (senza blocking).
2. Tuning soglia su `storage/ground_truth/splits/val.csv` (senza blocking).
3. Blocking B1/B2 applicato solo a `storage/ground_truth/splits/test.csv`.
4. Valutazione globale sul test completo (coppie non candidate → predizione negativa).

## Struttura progetto

```text
.
├── README.md
├── docs/
│   └── HOMEWORK.md
├── notebooks/
│   ├── eda_cars.ipynb
│   └── eda_vehicles.ipynb
├── pyproject.toml
├── pyrightconfig.json
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configurazione centralizzata (path, parametri, HW)
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── ditto_format.py             # Conversione dati → formato Ditto
│   │   └── generate.py                 # Generazione candidate pairs B1/B2
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── compare.py                  # Valutazione comparativa B x modello
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── dedupe.py                   # Training/inferenza Dedupe
│   │   ├── dedupe_eval.py              # Evaluation Dedupe (legacy)
│   │   ├── ditto.py                    # Training/inferenza Ditto
│   │   └── logistic_regression.py      # Logistic Regression via recordlinkage
│   └── preparation/
│       ├── __init__.py
│       ├── download.py                 # Download automatico da Kaggle
│       ├── ground_truth.py             # Ground truth + split train/val/test
│       ├── mediated_schema.py          # Schema mediato normalizzato
│       └── process_raw.py              # Preprocessing dataset grezzi
├── tests/
│   ├── test_blocking_generate.py
│   └── test_compare_metrics.py
└── vendor/
    └── FAIR-DA4ER/
        └── ditto/                      # Fork Ditto (training via train_ditto.sh)
            ├── train_ditto.sh
            ├── train_ditto.py
            ├── configs.json
            └── ditto_light/            # Runtime Ditto (modello, dataset, augment)
```

## Requisiti

- Python ≥ 3.10
- Credenziali Kaggle configurate (per lo step download)

Installazione:

```bash
pip install -r requirements.txt
```

## Esecuzione pipeline completa

```bash
python run_pipeline.py
```

Esempi:

```bash
python run_pipeline.py --from-step 5      # riparti dallo step 5
python run_pipeline.py --only-step 7       # esegui solo step 7
python run_pipeline.py --strategies B1     # solo strategia B1
```

## Step della pipeline

| Step | Descrizione | Modulo |
|------|-------------|--------|
| 1 | Download dataset da Kaggle | `src.preparation.download` |
| 2 | Preprocessing dataset grezzi | `src.preparation.process_raw` |
| 3 | Costruzione schema mediato | `src.preparation.mediated_schema` |
| 4 | Generazione ground truth e split GT | `src.preparation.ground_truth` |
| 5 | Blocking pair-level su GT test | `src.blocking.generate` |
| 6 | Conversione GT split in formato Ditto | `src.blocking.ditto_format` |
| 7 | Training Logistic Regression | `src.matching.logistic_regression` |
| 8 | Training Dedupe | `src.matching.dedupe` |
| 9 | Training Ditto | `src.matching.ditto` |
| 10 | Valutazione comparativa finale | `src.evaluation.compare` |

## Comandi singoli

```bash
# 1) Download
python -m src.preparation.download

# 2) Preprocessing
python -m src.preparation.process_raw

# 3) Schema mediato
python -m src.preparation.mediated_schema

# 4) Ground truth + split GT
python -m src.preparation.ground_truth

# 5) Blocking pair-level sul test GT
python -m src.blocking.generate --strategies B1 B2

# 6) Conversione Ditto (GT split)
python -m src.blocking.ditto_format --mode gt-splits

# (opzionale) Conversione Ditto candidate bloccate
python -m src.blocking.ditto_format --mode blocked-test --strategy B1
python -m src.blocking.ditto_format --mode blocked-test --strategy B2

# 7) Logistic Regression
python -m src.matching.logistic_regression --train
python -m src.matching.logistic_regression --evaluate

# 8) Dedupe
python -m src.matching.dedupe --train
python -m src.matching.dedupe --evaluate

# 9) Ditto
python -m src.matching.ditto --train
python -m src.matching.ditto --evaluate

# 10) Compare finale
python -m src.evaluation.compare --strategies B1 B2
```

## Artefatti prodotti

### Blocking su test
- `storage/blocking/B1/test_candidates.csv`
- `storage/blocking/B2/test_candidates.csv`
- `results/blocking/test_stats_B1.json`
- `results/blocking/test_stats_B2.json`

### Modelli e metadati
| Modello | File modello | Metadati |
|---------|-------------|----------|
| Logistic Regression | `models/recordlinkage.joblib` | `models/recordlinkage_meta.json` |
| Dedupe | `models/dedupe.pickle` | `models/dedupe_meta.json` |
| Ditto | `vendor/.../checkpoints/automotive_task/model.pt` | `models/ditto_meta.json` |

### Report finale
- `results/pipeline_report.json`

Contiene 6 righe (B1/B2 × Logistic Regression / Dedupe / Ditto) con:
- metriche di blocking (recall, reduction ratio)
- metriche candidate-only (precision, recall, F1)
- metriche globali su test completo
- tempi di inferenza e soglia usata

## Configurazione

Tutti i path e i parametri sperimentali sono centralizzati in `src/config.py`:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `RANDOM_SEED` | 42 | Seed per riproducibilità |
| `GT_NEGATIVE_RATIO` | 2.0 | Rapporto negativi/positivi nella GT |
| `DITTO_TASK` | `automotive_task` | Nome task Ditto |
| `DITTO_LM` | `roberta` | Language model per Ditto |
| `DITTO_N_EPOCHS` | 7 | Epoche di training Ditto |
| `DITTO_TRAIN_BATCH_SIZE` | 32 | Batch size training Ditto |
| `DITTO_LR` | 3e-5 | Learning rate Ditto |
| `DITTO_FP16` | `True` | Mixed precision training |
| `DITTO_BATCH_SIZE` | auto | Batch size inferenza (calcolato in base alla GPU) |

## Test

```bash
pytest -q
```

Verifica sintattica:

```bash
python -m compileall -q src run_pipeline.py
```

## Mapping Homework → Codice

| Task HW | Descrizione | Modulo |
|---------|-------------|--------|
| 4.A | Ground truth via VIN | `src/preparation/ground_truth.py` |
| 4.B | Rimozione VIN dai dataset | `src/preparation/ground_truth.py` |
| 4.C | Split train/val/test | `src/preparation/ground_truth.py` |
| 4.D | Blocking B1/B2 | `src/blocking/generate.py` |
| 4.E | Logistic Regression (recordlinkage) | `src/matching/logistic_regression.py` |
| 4.F | Dedupe | `src/matching/dedupe.py` |
| 4.G | Ditto | `src/matching/ditto.py` |
| 4.H | Valutazione comparativa | `src/evaluation/compare.py` |
| 4.F | Dedupe | `src/matching/dedupe.py` |
| 4.G | Ditto | `src/matching/ditto.py` |
| 4.H | Valutazione comparativa | `src/evaluation/compare.py` |
