# Homework 6 - Ingegneria dei Dati 2025/2026

**Docente:** Paolo Merialdo

## Obiettivo

L'obiettivo del progetto è integrare i dati su automobili disponibili da diverse sorgenti:

- [Craigslist Cars/Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- [US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

---

## Task

### 1. Analisi Sorgenti
Per ciascuna sorgente analizzare la percentuale di valori nulli e di valori unici di ciascun attributo.

### 2. Schema Mediato
Definire uno schema mediato.

### 3. Allineamento Sorgenti
Allineare le sorgenti allo schema mediato.

### 4. Record Linkage

| Step | Descrizione | Status |
|------|-------------|--------|
| **4.A** | Generare ground-truth usando VIN (numero telaio). Definire strategia per verifiche ad hoc e pulizia dati. Valutare uso di Label Studio. | ✅ Done |
| **4.B** | Eliminare attributi VIN dai due dataset e dalla ground-truth | ✅ Done |
| **4.C** | Dalla ground-truth creare tre dataset (training, validation, test) | ✅ Done |
| **4.D** | Definire due strategie di blocking B1 e B2 | ✅ Done |
| **4.E** | Definire regole di record linkage con libreria Python Record Linkage | ✅ Done |
| **4.F** | Addestrare un modello usando la libreria Python Dedupe | ✅ Done |
| **4.G** | Addestrare un modello con Ditto (https://github.com/MarcoNapoleone/FAIR-DA4ER) | ✅ Done |
| **4.H** | Valutare prestazioni pipeline (B1-dedupe, B2-dedupe, B1-RecordLinkage, B2-RecordLinkage, B1-ditto, B2-ditto) in termini di precision, recall, F1-measure, tempi di training, tempi di inferenza | ✅ Done |

---

## Deliverables

1. **Relazione** (~10 pagine): principali sfide, caratterizzazione fonti, valutazione sperimentale
2. **Presentazione** (20 min): architettura e valutazione sperimentale

### Consegna
- **Deadline:** Il giorno prima dell'esame
- **Link:** https://forms.office.com/e/EbqbR7dvK4
