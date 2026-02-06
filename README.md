# Record-Linkage

Pipeline di **Entity Resolution** tra due dataset di auto usate negli Stati Uniti ([Craigslist Cars/Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) e [US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)).

Confronta tre approcci di matching — **Record Linkage** (Logistic Regression), **Dedupe** e **Ditto** (Transformer) — su due strategie di blocking (B1, B2), valutando precision, recall, F1 e tempi di inferenza.

## Struttura del progetto

```
Record-Linkage/
├── run_pipeline.py               # Orchestratore: riproduce l'intero workflow
├── pyproject.toml                # Metadati e dipendenze del progetto
├── requirements.txt              # Dipendenze (pip)
│
├── src/                          # Codice sorgente
│   ├── config.py                 # Configurazione centralizzata (path, parametri)
│   ├── preparation/              # Acquisizione e preparazione dati
│   │   ├── download.py           #   Download dataset da Kaggle
│   │   ├── process_raw.py        #   Pulizia e preprocessing dataset grezzi
│   │   ├── mediated_schema.py    #   Schema mediato + normalizzazione
│   │   └── ground_truth.py       #   Generazione GT da VIN
│   ├── blocking/                 # Strategie di blocking
│   │   ├── generate.py           #   Coppie candidate B1, B2
│   │   └── ditto_format.py       #   Conversione in formato Ditto
│   ├── matching/                 # Modelli di matching
│   │   ├── recordlinkage.py      #   Record Linkage + Logistic Regression
│   │   ├── dedupe.py             #   Dedupe 3.0
│   │   └── dedupe_eval.py        #   Valutazione Dedupe
│   └── evaluation/               # Valutazione comparativa
│       └── compare.py            #   Confronto 6 pipeline (B×M)
│
├── storage/                      # Dati (non versionati)
│   ├── raw/                      #   Dataset grezzi
│   ├── processed/                #   Dataset puliti (output process_raw.py)
│   ├── mediated_schema/          #   Schema unificato
│   ├── ground_truth/             #   GT completa + splits/
│   ├── blocking/B1/, B2/         #   Coppie candidate per strategia
│   └── ditto/B1/, B2/            #   Dati formato Ditto
│
├── models/                       # Modelli addestrati (.joblib, .pickle)
├── results/                      # Metriche e report
│   ├── blocking/                 #   Statistiche blocking
│   ├── models/                   #   Metriche singoli modelli
│   └── pipeline_report.json      #   Report comparativo finale
│
├── notebooks/                    # Analisi esplorativa (EDA)
├── vendor/FAIR-DA4ER/            # Codice esterno Ditto
└── docs/                         # Traccia homework
```

## Requisiti

- Python >= 3.10
- Credenziali Kaggle configurate (per il download dei dataset)

```bash
pip install -r requirements.txt
```

## Esecuzione

### Pipeline completa

```bash
python run_pipeline.py
```

### Singoli step

```bash
python run_pipeline.py --from-step 5          # Riprendi dallo step 5
python run_pipeline.py --only-step 7          # Esegui solo lo step 7
python run_pipeline.py --strategies B1        # Solo strategia B1
```

### Step individuali

```bash
# Step 1 — Download dataset
python -m src.preparation.download

# Step 2 — Preprocessing dataset grezzi
python -m src.preparation.process_raw

# Step 3 — Schema mediato
python -m src.preparation.mediated_schema

# Step 4 — Ground truth
python -m src.preparation.ground_truth

# Step 5 — Blocking
python -m src.blocking.generate

# Step 6 — Formato Ditto
python -m src.blocking.ditto_format --strategy B1

# Step 7 — Training Record Linkage
python -m src.matching.recordlinkage --train --strategy B1

# Step 8 — Training Dedupe
python -m src.matching.dedupe --train --strategy B1

# Step 8b — Valutazione Dedupe
python -m src.matching.dedupe_eval --strategy B1

# Step 9 — Valutazione comparativa
python -m src.evaluation.compare
```

## Configurazione

Tutti i percorsi e i parametri sperimentali sono centralizzati in [`src/config.py`](src/config.py).
Nessun path hardcoded nei singoli moduli.

## Mapping Homework → Codice

| Task HW | Descrizione | Modulo |
|---------|-------------|--------|
| 1 | Analisi sorgenti + preprocessing | `notebooks/` + `src/preparation/process_raw.py` |
| 2-3 | Schema mediato + allineamento | `src/preparation/mediated_schema.py` |
| 4.A-C | Ground truth da VIN + split | `src/preparation/ground_truth.py` |
| 4.D | Strategie di blocking B1, B2 | `src/blocking/generate.py` |
| 4.E | Record Linkage | `src/matching/recordlinkage.py` |
| 4.F | Dedupe | `src/matching/dedupe.py` |
| 4.G | Ditto | `vendor/FAIR-DA4ER/` + `src/blocking/ditto_format.py` |
| 4.H | Valutazione pipeline | `src/evaluation/compare.py` |
