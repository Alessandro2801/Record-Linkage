# Relazione — Progetto Record Linkage (Homework 6 — Ingegneria dei Dati 2025/2026)

## 1. Introduzione (Record Linkage su dataset “big”)
- Definizione del problema: riconciliare entità (auto) provenienti da sorgenti diverse, identificando record che si riferiscono allo stesso veicolo.
- Motivazione “big data”: confronto naive all-pairs è impraticabile (crescita O(n²)); su dati reali serve ridurre drasticamente lo spazio di ricerca.
- Contesto del progetto e sorgenti (da traccia):
  - Craigslist Cars/Trucks vs US Used Cars.
  - Dimensioni in analisi: ~3.0M righe (us_cars_dataset) e ~400k righe (vehicles).
- Principali difficoltà su scala:
  - Dati sporchi/incompleti (missing), attributi non uniformi, valori testuali non standardizzati.
  - Schema eterogeneo (colonne diverse tra fonti) ⇒ necessità di schema mediato.
  - Labeling: ground truth affidabile e “pulita” per training/valutazione.
  - Necessità di blocking per rendere il RL computazionalmente fattibile.
- Obiettivo operativo della pipeline (visione end-to-end):
  - (1) EDA e pulizia → (2) schema mediato e normalizzazione → (3) ground truth (VIN) → (4) blocking (B1, B2) → (5) modelli (RecordLinkage, Dedupe, Ditto) → (6) valutazione per-pipeline e di sistema.

## 2. Requisiti (tecnici e di progetto)
- Requisiti funzionali (dalla traccia HOMEWORK):
  - Analisi sorgenti: percentuali di nulli e unici per attributo.
  - Definizione schema mediato e allineamento sorgenti.
  - Generazione ground-truth con VIN, rimozione VIN dagli input e split train/val/test.
  - Definizione di due strategie di blocking (B1 e B2).
  - Addestramento/uso di: RecordLinkage library, Dedupe, Ditto.
  - Valutazione pipeline in precision/recall/F1 e tempi (training e inferenza).
- Requisiti non funzionali:
  - Scalabilità: gestione dataset multi-milione di righe (RAM/IO).
  - Riproducibilità: salvataggio artefatti (CSV, modelli, report JSON) e pipeline deterministiche (random_state).
- Ambiente di calcolo (vincolo chiave):
  - Lightning AI Studio: macchine con **64 GB RAM** e **16 GB VRAM** utilizzate per training intensivo dei modelli di RL, Dedupe e Ditto.
- Stack software/librerie (in base al codice):
  - Python + pandas, numpy; scikit-learn (LogisticRegression, metriche).
  - recordlinkage (comparators e feature computation).
  - dedupe (training e scoring con modello salvato).
  - jellyfish (Jaro-Winkler per blocking fuzzy).
  - Ditto (repo FAIR-DA4ER in `src/pipelines/ditto/...`, dipendenze torch/GPU).
- Artefatti prodotti (output di progetto):
  - `output/blocking/stats_blocking_*.json` (statistiche blocking).
  - `output/result_models/evaluate_*.json` (metriche modello).
  - `output/pipeline_performance_report.json` (metriche end-to-end per pipeline).

## 3. Dataset & EDA (caratterizzazione e pulizia)
- Descrizione delle sorgenti e differenze di schema:
  - Vehicles (Craigslist): molte colonne “listing-like” (region, posting_date, ecc.), VIN presente ma con missing significativo.
  - US Used Cars: molte colonne tecniche (motore, dimensioni, ecc.), VIN presente e altamente distintivo.
- Analisi qualità dati (nulli/unici) svolta in:
  - `eda/eda_cars.ipynb`: profiling per dataset ~3,000,040 righe, 66 colonne; evidenza impatto RAM (~1.4GB in info()).
  - `eda/eda_vehicles.ipynb`: profiling e scelta colonne da eliminare.
- Scelte di pulizia e riduzione dimensionalità (EDA):
  - Eliminazione colonne con >70% di missing:
    - Vehicles: rimozione `['size', 'county']`, poi dataset ~426,880 righe → dopo drop di nulli critici ~397,874 righe.
    - Used cars: rimozione `['bed','bed_height','bed_length','cabin','combine_fuel_economy','is_certified','is_cpo','is_oemcpo','vehicle_damage_category']`, poi 3,000,040 righe → dopo drop nulli critici ~2,779,118 righe.
  - Definizione “colonne critiche” per record linkage (drop righe con null in campi core):
    - Vehicles: `manufacturer, model, year, odometer, fuel`.
    - Used cars: `make_name, model_name, year, mileage, fuel_type`.
  - Generazione ID per record:
    - In `eda_cars.ipynb` viene generato un ID sintetico per la sorgente (`"S2_" + index`) per garantire chiave univoca.
- Deliverable EDA da includere in relazione (figure/tabelle):
  - Tabella (o grafico) missing% e unique% per attributo per ciascuna sorgente.
  - Razionale della scelta di rimozione colonne e definizione dei campi core.

## 4. Schema mediato & allineamento sorgenti
- Obiettivo: rendere confrontabili i record tramite un set comune di attributi.
- Supporto allo schema matching (semi-automatico):
  - Uso di Valentine/COMA in `src/schema/mediated_schema.ipynb` per suggerire corrispondenze (con `java_xmx="16G"`).
  - Discussione: perché le similarity proposte non bastano (ambiguità semantiche, campi “simili” ma non equivalenti).
- Mapping manuale finale (esempi da notebook):
  - Veicoli (table_1): `VIN→vin`, `id→id_source_vehicles`, `odometer→mileage`, `fuel→fuel_type`, ecc.
  - Used cars (table_2): `make_name→manufacturer`, `model_name→model`, `engine_cylinders→cylinders`, `listing_color→main_color`, `id→id_source_used_cars`, ecc.
- Creazione dello schema mediato (stacking/concat):
  - Concat delle due sorgenti allineate ⇒ dataset mediato da **3,176,992** record.
  - Normalizzazione/pulizia “universale” dei campi testuali per evitare NaN/valori sporchi nei modelli (lowercase, trim, rimozione caratteri speciali, compattazione spazi).
- Output generato:
  - Salvataggio in `data/mediated_schema/mediated_schema_normalized.csv`.
  - Schema finale normalizzato (19 colonne) e footprint in RAM (~460MB indicato dal notebook).
- Elementi da presentare:
  - Tabella dello schema mediato (colonne + descrizione semantica + sorgente).

## 5. Ground Truth (VIN) e split train/val/test
- Razionale: usare VIN come “chiave quasi-univoca” per costruire coppie match/non-match.
- Pulizia avanzata VIN (da `src/schema/ground_truth.ipynb`):
  - Rimozione VIN nulli.
  - Normalizzazione alfanumerica (uppercase, rimozione caratteri non [A-Z0-9]).
  - Filtri formato: lunghezza 17, esclusione caratteri vietati (I,O,Q).
  - Rimozione placeholder/pattern sospetti.
  - Validazione checksum (algoritmo ufficiale della 9a cifra).
  - Deduplicazione “avanzata” per VIN: clustering per similarità su attributi con priorità su record più recente (basata su `pubblication_date`).
  - Risultato: dataset “sanificato” passa da 3,176,992 a ~2,889,790 record (post-pulizia VIN).
- Generazione coppie:
  - Positive (label=1): self-join su VIN, con vincolo `id_A < id_B` per evitare duplicati e self-pairs.
  - Negative (label=0): campionamento casuale di coppie con VIN diversi e ID diversi (target ratio_negativi=2.0).
  - Numeri prodotti (notebook):
    - Positive: 5,921; Negative: 11,842; Totale GT: 17,763 righe (35 colonne).
- Eliminazione VIN dai dataset e dalla GT (requisito 4.B):
  - Spiegare perché è necessario (evitare leakage: il modello non deve “vedere” l’identificatore).
- Split:
  - Split stratificato GT in train/eval; poi train in train/val/test:
    - Train: 8,703; Val: 1,865; Test: 1,866.
  - File salvati in `data/ground_truth/GT_train/{train,val,test}.csv` e `data/ground_truth/GT_eval.csv`.
- Materiale da includere:
  - Distribuzione label e motivazione del rapporto 1:2 (positivi:negativi).
  - Nota sulla qualità/limiti di VIN come ground truth (VIN mancante/errato ⇒ bias).

## 6. Blocking (B1 e B2) per ridurre lo spazio di confronto
- Motivazione teorica: rendere il problema computazionalmente gestibile su milioni di record, sacrificando il meno possibile la recall.
- Implementazione (da `src/pipelines/pairs_blocking.py`):
  - Blocking “fuzzy” basato su Jaro-Winkler.
  - Normalizzazione preventiva campi e confronto riga-per-riga sulle coppie dell’oracolo (per stimare recall del blocking).
- Strategia B1 (più “larga”, alta recall):
  - Vincoli: `year_A == year_B` e similarity(manufacturer) ≥ soglia.
  - Output e statistiche:
    - `output/blocking/stats_blocking_B1.json`: recall 98.08% (2352/2398), candidate pairs 2435.
- Strategia B2 (più “stretta”, meno candidate pairs):
  - Vincoli: `year` uguale + similarity(manufacturer) ≥ 0.95 + similarity(model) ≥ 0.85 + match esatto `fuel_type`.
  - Output e statistiche:
    - `output/blocking/stats_blocking_B2.json`: recall 75.77% (1817/2398), candidate pairs 1826.
- Discussione trade-off:
  - B1: più coppie da valutare ma recall elevata.
  - B2: meno coppie e più “precision-oriented”, ma forte perdita di recall già in fase di blocking.
- Output di blocking (per pipeline):
  - `data/blocking/candidate_pairs_B1.csv`, `candidate_pairs_B2.csv` come input per inferenza/evaluazione pipeline.

## 7. Modello 1 — Record Linkage con feature engineering + Logistic Regression
- Obiettivo: baseline supervisionata su feature di similarità calcolate con `recordlinkage`.
- Implementazione (da `src/trainings/record_linkage.py`):
  - Comparator setup:
    - Stringhe (jarowinkler): manufacturer/model/location (thr 0.85), cylinders (thr 0.70).
    - Exact: year, fuel_type, traction, body_type, main_color, transmission.
    - Numeric gaussian: price, mileage, latitude, longitude (offset/scale definiti nel codice).
  - Training:
    - Hyperparameter tuning su validation (grid su C e class_weight), poi training modello migliore.
    - Persistenza modello: `models/recordlinkage_model.joblib`.
- Valutazione modello (test set):
  - Metriche salvate in `output/result_models/evaluate_record_linkage.json`:
    - precision 0.9872, recall 0.9936, f1 0.9904.
- Cosa mostrare in relazione:
  - Elenco feature e razionale (perché jarowinkler, perché exact su alcune categorie).
  - Risultati e considerazioni (robustezza ai missing vs sensibilità a normalizzazione).

## 8. Modello 2 — Dedupe (training in-memory + ottimizzazione soglia)
- Obiettivo: modello supervisionato con variabili eterogenee e gestione dei missing.
- Implementazione training (da `src/trainings/train_dedupe.py`):
  - Variabili Dedupe (String/Price/Categorical) su campi mediati (manufacturer, model, year, location, cylinders, body_type, main_color, lat/lon, price, mileage, transmission/fuel_type/traction).
  - Setup prestazionale: `in_memory=True`, `num_cores=None`, campionamento per blocking statistics.
  - Esclusione del test set dal set di ID usati per training (per purezza metodologica).
  - Persistenza modello: `models/dedupe_model.pickle`.
- Ottimizzazione e valutazione (da `src/trainings/evaluate_dedupe.py`):
  - Scoring su val e ricerca soglia in [0.1, 1.0) step 0.05.
  - Metriche salvate in `output/result_models/evaluate_dedupe.json`:
    - best_threshold 0.30, precision 0.9936, recall 0.9968, f1 0.9952.
- Punti da discutere:
  - Differenza tra “score probabilistico” e decisione binaria (ruolo della soglia).
  - Costi computazionali: memory-bound vs CPU-bound, benefici del multi-core e RAM 64GB.

## 9. Modello 3 — Ditto (Transformer Entity Matching) e integrazione con pipeline
- Razionale: modello deep che sfrutta rappresentazioni testuali e contesto multi-attributo.
- Preparazione dati per training Ditto (da `src/pipelines/prepare_training_ditto.py`):
  - Serializzazione di record A/B in formato Ditto: tokenizzazione “COL … VAL …” per ciascun campo mediato.
  - Conversione `data/ground_truth/GT_train/{train,val,test}.csv` → file `.txt` in `src/pipelines/ditto/FAIR-DA4ER/ditto/data/ditto_data/`.
- Preparazione dati per inferenza Ditto su coppie bloccate (da `src/pipelines/prepare_inference_ditto.py`):
  - Conversione `data/blocking/candidate_pairs_B1.csv` e `candidate_pairs_B2.csv` → `.txt` per scoring Ditto.
- Training/inferenza su GPU (Requisito hardware):
  - Motivare uso Lightning AI Studio (16GB VRAM) per training intensivo.
  - Specificare come verrà gestito il run (config, checkpoint, batch size, max_len) e come si salvano/leggono risultati.
- Stato attività rispetto alla traccia:
  - Training Ditto e valutazione pipeline con Ditto: da completare/estendere per ottenere anche B1-Ditto e B2-Ditto nel report end-to-end.

## 10. Valutazione end-to-end delle pipeline (blocking + modello)
- Definizione delle metriche “di sistema” (da `src/pipelines/evaluate_pipelines.py`):
  - recall_total = recall_blocking × recall_model.
  - f1_total ricalcolata su precision e recall_total.
  - Misura dei tempi di inferenza (sulle sole candidate pairs).
- Risultati sperimentali (da `output/pipeline_performance_report.json`):
  - B1 + RecordLinkage:
    - precision 0.9924, recall_blocking 0.9808, recall_model 0.9962, recall_total 0.9770, f1_total 0.9847, inference_time ~0.0012s
  - B1 + Dedupe:
    - precision 0.9970, recall_total 0.9779, f1_total 0.9874, inference_time ~4.54s
  - B2 + RecordLinkage:
    - precision 0.9962, recall_total 0.7564, f1_total 0.8599, inference_time ~0.0011s
  - B2 + Dedupe:
    - precision 0.9973, recall_total 0.7564, f1_total 0.8603, inference_time ~7.51s
- Analisi comparativa:
  - Effetto dominante del blocking: B2 degrada recall_total nonostante recall_model ~0.998.
  - Trade-off prestazioni/tempo: RecordLinkage rapidissimo sulle candidate pairs; Dedupe più lento ma leggermente migliore su precision/F1 in B1.
- Cosa includere come tabelle/grafici:
  - Tabella riepilogo pipeline (precision, recall_blocking, recall_model, recall_total, f1_total, inference_time).
  - Grafico “recall_total vs #candidate_pairs” per evidenziare il compromesso B1/B2.

## 11. Discussione: scelte progettuali, error analysis e scalabilità
- Perché lo schema mediato è una condizione necessaria per RL multi-sorgente.
- Lezioni dal GT su VIN:
  - Benefici (labeling affidabile) vs limiti (VIN incompleti/errati; bias verso record con VIN valido).
- Blocking:
  - Dove si perdono match (false negative strutturali) e come si potrebbe recuperare recall (blocking multiplo, OR tra chiavi, LSH, canopy clustering).
- Modelli:
  - Quando preferire feature-based (RecordLinkage) vs probabilistico (Dedupe) vs deep (Ditto).
  - Impatto delle normalizzazioni e dei campi testuali (description) sulle prestazioni.
- Scalabilità futura su full dataset:
  - Passaggio da “oracle blocking evaluation” a blocking su tutto il dataset mediato e scoring dei modelli sulle candidate pairs reali.

## 12. Conclusioni e sviluppi futuri
- Sintesi risultati:
  - B1 risulta la strategia di blocking più bilanciata (recall_total ~0.98) a fronte di più candidate pairs.
  - RecordLinkage e Dedupe mostrano performance alte sul test; differenze principali su costi/tempi.
- Sviluppi:
  - Completamento Ditto (training + inferenza) e confronto diretto con baselines.
  - Aggiunta di misure di training time (requisito 4.H) e profiling memoria/CPU/GPU.
  - Estensione blocking (strategie composite, tuning soglie con obiettivi su recall_total).
  - Miglioramento schema mediato (feature addizionali utili, gestione valori numerici/outlier).

---
## Riferimenti a file/artefatti usati nel progetto (per orientarsi)
- Traccia: `docs/HOMEWORK.md`
- EDA: `eda/eda_cars.ipynb`, `eda/eda_vehicles.ipynb`
- Schema mediato: `src/schema/mediated_schema.ipynb` → `data/mediated_schema/mediated_schema_normalized.csv`
- Ground truth: `src/schema/ground_truth.ipynb` → `data/ground_truth/...`
- Blocking: `src/pipelines/pairs_blocking.py` → `output/blocking/stats_blocking_*.json`
- Modelli:
  - RecordLinkage: `src/trainings/record_linkage.py` → `output/result_models/evaluate_record_linkage.json`
  - Dedupe: `src/trainings/train_dedupe.py`, `src/trainings/evaluate_dedupe.py` → `output/result_models/evaluate_dedupe.json`
  - Ditto: `src/pipelines/prepare_training_ditto.py`, `src/pipelines/prepare_inference_ditto.py`, repo `src/pipelines/ditto/FAIR-DA4ER/ditto/`
- Valutazione pipeline: `src/pipelines/evaluate_pipelines.py` → `output/pipeline_performance_report.json`