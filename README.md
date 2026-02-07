# Record-Linkage

Pipeline di **Entity Resolution** tra due dataset di auto usate negli USA:
- [Craigslist Cars/Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- [US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

## Metodologia attuale

La pipeline implementa questo flusso:
1. training dei modelli su `storage/ground_truth/splits/train.csv` (senza blocking),
2. tuning su `storage/ground_truth/splits/val.csv` (senza blocking),
3. blocking B1/B2 applicato solo a `storage/ground_truth/splits/test.csv`,
4. valutazione globale sul test completo (coppie non candidate => predizione negativa).

## Struttura progetto

```text
.
├── README.md
├── docs
│   └── HOMEWORK.md
├── notebooks
│   ├── eda_cars.ipynb
│   └── eda_vehicles.ipynb
├── pyproject.toml
├── pyrightconfig.json
├── requirements.txt
├── results
│   ├── blocking
│   │   ├── stats_B1.json
│   │   ├── stats_B2.json
│   │   ├── test_stats_B1.json
│   │   └── test_stats_B2.json
│   └── pipeline_report.json
├── run_pipeline.py
├── src
│   ├── __init__.py
│   ├── blocking
│   │   ├── __init__.py
│   │   ├── ditto_format.py
│   │   └── generate.py
│   ├── config.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   └── compare.py
│   ├── matching
│   │   ├── __init__.py
│   │   ├── dedupe.py
│   │   ├── dedupe_eval.py
│   │   ├── ditto.py
│   │   └── logistic_regression.py
│   └── preparation
│       ├── __init__.py
│       ├── download.py
│       ├── ground_truth.py
│       ├── mediated_schema.py
│       └── process_raw.py
├── tests
│   ├── test_blocking_generate.py
│   └── test_compare_metrics.py
└── vendor
    └── FAIR-DA4ER
        ├── README.md
        ├── ditto
        └── requirements.txt
```

## Requisiti

- Python >= 3.10
- Credenziali Kaggle configurate (per lo step download)

Installazione:

```bash
pip install -r requirements.txt
```

## Esecuzione pipeline completa

```bash
python run_pipeline.py
```

Esempi utili:

```bash
python run_pipeline.py --from-step 5
python run_pipeline.py --only-step 7
python run_pipeline.py --strategies B1
```

## Step della pipeline

| Step | Descrizione | Modulo |
|---|---|---|
| 1 | Download dataset da Kaggle | `src.preparation.download` |
| 2 | Preprocessing dataset grezzi | `src.preparation.process_raw` |
| 3 | Costruzione schema mediato | `src.preparation.mediated_schema` |
| 4 | Generazione ground truth e split GT | `src.preparation.ground_truth` |
| 5 | Blocking pair-level su GT test | `src.blocking.generate` |
| 6 | Conversione GT split in formato Ditto | `src.blocking.ditto_format --mode gt-splits` |
| 7 | Training Logistic Regression | `src.matching.logistic_regression --train` |
| 8 | Training Dedupe | `src.matching.dedupe --train` |
| 9 | Training Ditto | `src.matching.ditto --train` |
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
- `models/recordlinkage.joblib`
- `models/recordlinkage_meta.json`
- `models/dedupe.pickle`
- `models/dedupe_meta.json`
- `models/ditto_meta.json`

Nota: i nomi artefatto `recordlinkage*` sono mantenuti per compatibilità, anche se il modulo è `logistic_regression.py`.

### Report finale
- `results/pipeline_report.json`

Contiene 6 righe (`B1/B2 x Logistic Regression/Dedupe/Ditto`) con:
- metriche di blocking,
- metriche candidate-only,
- metriche globali su test completo,
- tempi di inferenza,
- soglia usata.

## Test e validazione

Se disponibile `pytest`:

```bash
pytest -q
```

Verifica sintattica minima:

```bash
python -m compileall -q src run_pipeline.py
```

## Configurazione

Path e parametri sperimentali centralizzati in `src/config.py`.

## Mapping Homework -> Codice

| Task HW | Descrizione | Modulo |
|---|---|---|
| 4.C | Split ground truth train/val/test | `src/preparation/ground_truth.py` |
| 4.D | Blocking B1/B2 su test | `src/blocking/generate.py` |
| 4.E | Logistic Regression | `src/matching/logistic_regression.py` |
| 4.F | Dedupe | `src/matching/dedupe.py` |
| 4.G | Ditto | `src/matching/ditto.py` |
| 4.H | Valutazione comparativa | `src/evaluation/compare.py` |
