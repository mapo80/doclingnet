# Piano di Intervento - Docling.NET Quality Improvement

## 🔍 PROBLEMI IDENTIFICATI

### **CRITICO 1: Layout SDK Fallback**
- **Sintomo**: Layout SDK primario restituisce 0 detections
- **Conseguenza**: Usa fallback ONNX che produce 184 detections invece di 13-14
- **Impatto**: Output markdown illeggibile, OCR 98% del tempo (145s)
- **Log Evidence**:
  ```
  [07:44:45 DBG] Layout SDK returned 0 raw detections before reprojection.
  [07:44:46 DBG] Fallback reprojection mapped 184 detections to 1275x1650
  [07:44:46 INF] Fallback ONNX inference produced 184 detections.
  ```

### **CRITICO 2: OCR Quality**
- **Sintomo**: Anche con detections corrette (13), output frammentato
- **Evidenza**: Run del 2025-10-01 aveva 13 detections ma markdown scarso
- **Impatto**: Testo duplicato, caratteri errati, struttura persa

### **ALTO 3: Table Detection**
- **Sintomo**: Tabella non identificata (tutto classificato come "Text")
- **Impatto**: TableFormer skippato, contenuto tabella frammentato
- **Consistenza**: Problema presente anche in Python CLI

### **MEDIO 4: Page Assembly Logic**
- **Sintomo**: Ordinamento e raggruppamento elementi non ottimale
- **Impatto**: Struttura documento finale non corretta

---

## 📋 PIANO DETTAGLIATO

### **FASE 1: DIAGNOSI LAYOUT SDK** ⚠️ PRIORITÀ MASSIMA

#### Step 1.1: Analisi LayoutSdk Source
**Obiettivo**: Capire perché `LayoutSdk.Process()` restituisce 0 boxes
**Azioni**:
- [x] Esaminare `LayoutSdk.cs` in `src/submodules/ds4sd-docling-layout-heron-onnx/dotnet/LayoutSdk/`
- [x] Identificare il metodo `Process()` e flow di inferenza
- [x] Verificare gestione runtime (Ort vs OpenVino)
- [x] Controllare preprocessing immagine (normalizzazione, letterboxing)
- [x] Analizzare log/exception handling

**Output Atteso**: Identificare punto esatto di failure

#### Step 1.2: Confronto con Python Implementation
**Obiettivo**: Capire differenze implementative
**Azioni**:
- [ ] Scaricare Docling Python (`git clone https://github.com/DS4SD/docling`)
- [ ] Localizzare layout detection in Python
- [ ] Confrontare preprocessing pipeline
- [ ] Confrontare post-processing (NMS, filtering)
- [ ] Documentare differenze chiave

**Output Atteso**: Lista differenze .NET vs Python

#### Step 1.3: Test Isolato LayoutSdk
**Obiettivo**: Riprodurre problema in isolamento
**Azioni**:
- [ ] Creare unit test standalone con `dataset/2305.03393v1-pg9-img.png`
- [ ] Testare con Runtime.Ort
- [ ] Testare con Runtime.OpenVino (se disponibile su macOS)
- [ ] Aggiungere logging dettagliato
- [ ] Verificare output ONNX raw (prima di NMS/filtering)

**Output Atteso**: Test che riproduce il problema

#### Step 1.4: Fix LayoutSdk
**Obiettivo**: Risolvere il problema di 0 detections
**Azioni**:
- [x] Implementare fix basato su diagnosi
- [x] Testare con immagine di test
- [ ] Verificare output: 13-14 detections (come Python)
- [ ] Ricompilare LayoutSdk come NuGet locale
- [ ] Aggiornare riferimenti in Docling.Core

**Output Atteso**: Layout SDK funzionante con detections corrette

**Success Criteria**:
✅ Layout SDK primario produce 13-14 detections (non 0)
✅ Fallback ONNX NON viene più usato
✅ Layout detection time < 2s

---

### **FASE 2: MIGLIORAMENTO OCR**

#### Step 2.1: Analisi EasyOcrNet
**Obiettivo**: Capire qualità OCR attuale
**Azioni**:
- [ ] Esaminare `EasyOcrNet.csproj` in `src/submodules/easyocrnet/`
- [ ] Verificare versione modelli OCR utilizzati
- [ ] Controllare preprocessing bbox (padding, resize)
- [ ] Analizzare confidence threshold
- [ ] Verificare language model (English)

**Output Atteso**: Configurazione OCR attuale

#### Step 2.2: Confronto con Python OCR
**Obiettivo**: Identificare differenze OCR
**Azioni**:
- [ ] Localizzare OCR in Docling Python
- [ ] Confrontare modelli utilizzati
- [ ] Confrontare preprocessing
- [ ] Confrontare parametri inferenza
- [ ] Documentare differenze

**Output Atteso**: Lista differenze OCR .NET vs Python

#### Step 2.3: Test Isolato OCR
**Obiettivo**: Testare qualità OCR su singole bbox
**Azioni**:
- [ ] Estrarre bbox campione dalla tabella
- [ ] Testare OCR su bbox in isolamento
- [ ] Confrontare con output Python
- [ ] Verificare encoding caratteri
- [ ] Testare diverse configurazioni

**Output Atteso**: Test qualità OCR

#### Step 2.4: Ottimizzazione OCR
**Obiettivo**: Migliorare qualità riconoscimento
**Azioni**:
- [ ] Implementare preprocessing migliore (se necessario)
- [ ] Aggiustare confidence threshold
- [ ] Verificare post-processing (text cleaning)
- [ ] Testare con immagine completa
- [ ] Ricompilare EasyOcrNet come NuGet locale

**Output Atteso**: OCR con qualità comparabile a Python

**Success Criteria**:
✅ Testo riconosciuto senza duplicazioni
✅ Caratteri speciali corretti
✅ Confidence scores simili a Python

---

### **FASE 3: TABLE DETECTION INVESTIGATION**

#### Step 3.1: Analisi Table Detection Failure
**Obiettivo**: Capire perché tabella non viene identificata
**Azioni**:
- [ ] Verificare se problema presente anche in Python (già confermato: sì)
- [ ] Esaminare modello `heron-optimized.onnx`
- [ ] Controllare class mapping (label indexes)
- [ ] Verificare confidence threshold per "Table" class
- [ ] Analizzare preprocessing specifico per tabelle

**Output Atteso**: Root cause table detection failure

#### Step 3.2: Possibili Soluzioni Table Detection
**Obiettivo**: Esplorare opzioni di fix
**Azioni**:
- [ ] Opzione A: Aggiustare threshold detection
- [ ] Opzione B: Usare modello layout diverso (se disponibile)
- [ ] Opzione C: Implementare table detection euristica supplementare
- [ ] Testare soluzioni con immagine campione
- [ ] Valutare trade-off

**Output Atteso**: Soluzione table detection implementata (se possibile)

**Success Criteria** (opzionale, se problem è nel modello):
✅ Tabella identificata come "Table" invece di "Text"
✅ TableFormer invocato correttamente
✅ Struttura tabella preservata in markdown

---

### **FASE 4: PAGE ASSEMBLY OPTIMIZATION**

#### Step 4.1: Analisi Page Assembly Logic
**Obiettivo**: Verificare logica ordinamento/raggruppamento
**Azioni**:
- [ ] Esaminare `PageAssemblyStage.cs` in Docling.Pipelines
- [ ] Verificare ordinamento bbox (reading order)
- [ ] Controllare merge di text adiacenti
- [ ] Verificare gestione line breaks/paragraphs
- [ ] Confrontare con logica Python

**Output Atteso**: Comprensione logica assembly

#### Step 4.2: Miglioramento Assembly
**Obiettivo**: Ottimizzare costruzione documento finale
**Azioni**:
- [ ] Implementare miglioramenti identificati
- [ ] Testare con output OCR corretto (post Fase 2)
- [ ] Verificare struttura markdown
- [ ] Validare reading order

**Output Atteso**: Markdown ben strutturato

**Success Criteria**:
✅ Paragrafi ordinati correttamente
✅ Line breaks appropriati
✅ Nessuna duplicazione testo

---

### **FASE 5: INTEGRAZIONE E VALIDAZIONE**

#### Step 5.1: Build con Submodule Modificati
**Obiettivo**: Integrare tutte le modifiche
**Azioni**:
- [ ] Configurare project references ai submodule invece di NuGet packages
- [ ] Build completa soluzione
- [ ] Risolvere breaking changes (se presenti)
- [ ] Aggiornare configurazione NuGet locale

**Output Atteso**: Build success con submodule

#### Step 5.2: Test End-to-End
**Obiettivo**: Validare intera pipeline
**Azioni**:
- [ ] Run pipeline su `dataset/2305.03393v1-pg9-img.png`
- [ ] Verificare layout: 13-14 detections
- [ ] Verificare OCR time: < 30s (invece di 145s)
- [ ] Verificare markdown quality
- [ ] Confrontare con output Python

**Output Atteso**: Pipeline completa funzionante

#### Step 5.3: Regression Testing
**Obiettivo**: Verificare non ci siano regressioni
**Azioni**:
- [ ] Testare su altri documenti del dataset
- [ ] Verificare diversi formati (PDF, DOCX se supportati)
- [ ] Run test suite completa
- [ ] Verificare performance

**Output Atteso**: Tutti i test passano

**Success Criteria**:
✅ Output markdown leggibile e corretto
✅ Performance comparabile a Python
✅ No regressioni su altri documenti

---

## 📊 METRICHE DI SUCCESSO COMPLESSIVE

### Performance
- Layout detection time: < 2s (attuale: ~1.9s ✓)
- OCR time: < 30s (attuale: 145s ❌)
- Total pipeline time: < 35s (attuale: 147s ❌)

### Quality
- Layout detections: 13-14 (attuale: 0 primario, 184 fallback ❌)
- Markdown readability: Comparabile a Python (attuale: illeggibile ❌)
- OCR accuracy: > 95% (attuale: ~60% stimato ❌)
- Table detection: Identificata (attuale: no, ma anche Python no ⚠️)

### Stability
- No fallback ONNX trigger (attuale: sempre fallback ❌)
- No duplicazioni testo (attuale: massivo ❌)
- Consistent results across runs (da verificare)

---

## 🚀 ESECUZIONE

**Approccio**: Procediamo FASE per FASE, step by step
**Priorità**: Fase 1 → Fase 2 → Fase 5 (Fase 3 e 4 se tempo permette)

**Prossimo Step**: **FASE 1 - Step 1.1** - Analisi LayoutSdk Source

Pronto a partire?
