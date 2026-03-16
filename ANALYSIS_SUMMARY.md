# CHICO PoseForecasting — Analisi Completa ✅

**Data**: 2026-03-16 | **Stato**: Analisi e Piano completati

---

## 📋 LAVORO COMPLETATO

### ✅ Fase 1: Mappatura Struttura Repository
Hai una **pipeline di 5 fasi** per conformal prediction su pose umana 3D:

```
DATI CALIBRAZIONE (433 MB)
        ↓
Fase 1-2: Setup offline (BallTree, Scaling) [data_prep.py]
        ↓
Fase 2.5: Non-conformity scores Mahalanobis [score_extraction.py]
        ↓
Fase 3: Addestramento 375 regressori quantile (XGB/QRF/TabPFN) [train_regressor.py]
        ↓
Fase 3.5: Validazione FCL sul validation set (5% calib) [evaluate_fcl.py]
        ↓
DATI TEST (879 MB) → Fase 4: Inferenza batch [test_inference.py]
        ↓
Fase 5: Valutazione collisioni FCL su test [evaluate_fcl.py]
        ↓
RISULTATI (metriche, report timing)
```

**Struttura file principale**:
- `main.py` — Coordinatore (modalità single/batch)
- `pipeline_utils.py` — Helper condivisi (feature engineering, SVD)
- `config.yaml` — Configurazione (ablation, iperparametri, collision tuning)
- `models/` — Reti GCN student/teacher per pose forecasting
- `utils_chico/` — Dataset loader e utility specifiche

---

### ✅ Fase 2: Analisi Dettagliata del Codice

#### **Problemi Identificati**:

1. **Duplicazione Codice** (HIGH IMPACT)
   - `compute_15_distances()` — chiamato 2 volte identicamente
   - `build_feature_matrix()` — logica duplicata train vs inference
   - Calcolo covarianze/thresholds — 150 linee duplicate

2. **Incoerenza Strutturale** (MEDIUM IMPACT)
   - `nc_scores[j][h]` vs `storici_residui[h][j]` — ordini diversi! 🚨
   - Configurazione senza validazione
   - Nessun logging strutturato (solo `print()`)

3. **Performance Bottleneck** (MEDIUM IMPACT)
   - Split indici recalcolato 375 volte con stesso seed
   - KNN path: loop su N_test con SVD per ogni campione
   - Memoria: ~400MB per distanze quando H=25

4. **Manutenibilità** (LOW IMPACT but annoying)
   - Nessun type hint
   - Nessun unit test
   - Docstring incompleti

---

### ✅ Fase 3: Piano di Refactoring Dettagliato

Ho creato un **piano strutturato** in:
📄 **`.claude/projects/CHICO-PoseForecasting/memory/REFACTORING_PLAN.md`**

#### **Sommario del Piano**:

| Priorità | Task | Impact | Time | Risk |
|----------|------|--------|------|------|
| 🔴 ALTA | Centralizzare helper (load, validate, logging) | 30% | 1h | Basso |
| 🔴 ALTA | Standardizzare [j][h] indexing | 25% | 30min | Basso |
| 🔴 ALTA | Extract split indici (una volta) | 15% | 5min | Minimo |
| 🟡 MEDIA | Logging strutturato (logging module) | 10% | 2h | Basso |
| 🟡 MEDIA | Type hints completi | 5% | 3h | Minimo |
| 🟡 MEDIA | Config validation early-fail | 8% | 1h | Basso |
| 🟢 BASSA | Vettorizzare KNN path | 12% | 1h | Medio |
| 🟢 BASSA | Cache validation set | 3% | 30min | Basso |

**TOTALE: ~8-10 ore di refactoring**

---

## 🎯 COME PROCEDERE

### **Session 1: FONDAMENTI** (~1.5h) — CONSIGLIATO INIZIARE DA QUI
```
□ Fix split indici → max 5 min
  Modifica train_regressor.py: calcola val_indices una volta fuori dal loop

□ Standardizza [j][h]
  Modifica data_prep.py, train_regressor.py, test_inference.py
  Riordina storici_residui da [h][j] → [j][h]
  Test: 1 single_run per verificare coerenza

RISULTATO: Codice più coerente + 10% speedup training
```

### **Session 2: MANUTENIBILITÀ** (~2h)
```
□ Aggiungi helper centralizzati in pipeline_utils.py
  - load_dataset(path) → dict
  - validate_config(config) → (bool, list[str])
  - setup_logging(exp_dir) → logger

□ Config validation in main.py
  Fail fast prima di launch training

□ Logging strutturato
  Sostituisci print() con logger.info/debug

RISULTATO: Fail fast, structured logs, better debug
```

### **Session 3: PERFORMANCE** (~2h, opzionale)
```
□ Vettorizzare KNN path in test_inference.py
  5-10x speedup su inference con use_knn=True

□ Cache validation set
  Faster re-run se cambiano solo iperparametri

RISULTATO: Inferenza 5-10x più veloce
```

---

## 📊 COSA FARE ADESSO

### **Opzione A: Vuoi che inizi subito il refactoring?**
→ Dimi quale Session preferisci (1, 2, o 3) e procedo con **implementazione + testing**

### **Opzione B: Vuoi revisione/feedback sul piano?**
→ Dammi feedback su:
- Priorità (agree con ordine proposto?)
- Scope (quale task saltare?)
- Timeline (troppo/poco aggressivo?)

### **Opzione C: Vuoi solo run le fasi attuali?**
→ Posso aiutarti a:
- Lanciare `python main.py` in single_run
- Debuggare risultati
- Analizzare metriche FCL

---

## 📌 KEY INSIGHTS

1. **Il codice funziona** ✅ ma ha **incoerenze** che rallentano development futuro
2. **Stanno [j][h] vs [h][j]** è la **fonte d'errore cognitiva più grande** — fix prioritario
3. **Nessuna validazione config** → errori dopo 30 min di training (💥) → fix early
4. **Performance non è bottleneck** attualmente, ma KNN vettorizzazione darebbe 5-10x speedup easy
5. **Type hints + logging** danno ROI altissimo per manutenibilità futura

---

## 🔗 DOCUMENTI RIFERIMENTO

- **Mappaatura completa**: In questa sessione (sopra nella chat)
- **Piano dettagliato**: `.claude/projects/.../REFACTORING_PLAN.md`
- **Memoria progetto**: `.claude/projects/.../MEMORY.md`

---

**Cosa vuoi fare adesso?** 👇
