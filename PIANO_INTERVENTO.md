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
- [x] Scaricare Docling Python (`git clone https://github.com/DS4SD/docling`)
- [x] Localizzare layout detection in Python
- [x] Confrontare preprocessing pipeline
- [x] Confrontare post-processing (NMS, filtering)
- [x] Documentare differenze chiave

**Output Atteso**: Lista differenze .NET vs Python

**Osservazioni raccolte**:
- Python usa `LayoutPredictor` (Torch + `RTDetrImageProcessor`) che esegue automaticamente resize/letterboxing a 640x640 e normalizzazione (rescale + mean/std) secondo `preprocessor_config.json`. In .NET la normalizzazione avviene in `LayoutSdkRunner.NormaliseForModel` + `SkiaImagePreprocessor`, con sola divisione per 255 e senza mean/std.
- Il modello Python applica `post_process_object_detection` di HuggingFace che riporta i bbox nelle coordinate originali; l'ONNX backend .NET replica manualmente il parsing di `logits`/`pred_boxes` applicando sigmoid e clamp dei bounding box.
- Python applica un ricco `LayoutPostprocessor` (merge Union-Find, indici spaziali, soglie di confidenza per etichetta, gestione wrapper/picture e assegnazione celle). In .NET il risultato del backend viene usato direttamente (solo reproiezione e clamp), senza filtri per etichetta né deduplicazioni.
- Le etichette Python provengono da `LayoutLabels.shifted_canonical_categories()` (background + 17 classi). L’implementazione .NET mantiene un dizionario equivalente in `OnnxRuntimeBackend.ParseOutputs`, ma con soglia fissa 0.3 per tutte le classi.
- In Python non è previsto un fallback ONNX: l’errore in inferenza fa fallire lo stage. Il runner .NET aveva un fallback (ora rimosso) e dipende dal preprocessing bespoke per evitare il caso 0 detections.

#### Step 1.3: Test Isolato LayoutSdk
**Obiettivo**: Riprodurre problema in isolamento
**Azioni**:
- [x] Creare unit test standalone con `dataset/2305.03393v1-pg9-img.png`
- [x] Testare con Runtime.Ort
- [ ] Testare con Runtime.OpenVino (n/a su macOS)
- [x] Aggiungere logging dettagliato
- [x] Verificare output ONNX raw (prima di NMS/filtering)

**Output Atteso**: Test che riproduce il problema

**Note test**:
- Aggiunto `DatasetFixture` per risolvere asset path in test e helper `CreateNormalizedImage` (Skia) che replica il letterboxing 640x640 usato dal runner .NET.
- Nuovo test `LayoutSdkIntegrationTests.OnnxRuntime_ProducesBoundingBoxes_ForSampleImage` conferma 9 detection con `LayoutRuntime.Onnx` (log xUnit).
- Il backend ONNX fallisce con input non normalizzato: l'helper evidenzia la dipendenza dall'image preprocessing corretto.

#### Step 1.4: Fix LayoutSdk
**Obiettivo**: Risolvere il problema di 0 detections
**Azioni**:
- [x] Implementare fix basato su diagnosi
- [x] Testare con immagine di test
- [x] Verificare output: 13-14 detections (come Python)

**Output Atteso**: Layout SDK funzionante con detections corrette

**Note Step 1.4**:
- L'ONNX backend ora usa softmax per i logits, soglie per etichetta (allineate ai valori Python) e una NMS semplice (IoU 0.7, same-label). L'integrazione `LayoutSdkIntegrationTests` segnala 13 box per `2305.03393v1-pg9-img.png`.

**Success Criteria**:
✅ Layout SDK primario produce 13-14 detections (non 0)
✅ Fallback ONNX NON viene più usato
✅ Layout detection time < 2s

#### Step 1.5: Post-processing Alignment
**Azioni**:
- [ ] Analizzare `LayoutPostprocessor` Python (merge, union-find, soglie per etichetta).
- [ ] Definire strategia .NET (replica completa vs modulo di raffinamento esterno).
- [ ] Implementare prototipo e validarlo contro il dataset di riferimento.

#### Step 1.6: Packaging & References
- [ ] Ricompilare LayoutSdk come NuGet locale
- [ ] Aggiornare riferimenti in Docling.Core

---

### **FASE 2: MIGLIORAMENTO OCR** 🔄 IN CORSO

#### Step 2.1: Analisi EasyOcrNet ✅ COMPLETATO
**Obiettivo**: Capire qualità OCR attuale
**Azioni**:
- [x] Esaminare `EasyOcrNet.csproj` in `src/submodules/easyocrnet/`
- [x] Verificare versione modelli OCR utilizzati
- [x] Controllare preprocessing bbox (padding, resize)
- [x] Analizzare confidence threshold
- [x] Verificare language model (English)

**Output**: Configurazione OCR attuale documentata

**Risultati**:
- **Confidence threshold**: 0.5 (default)
- **Lingue**: `["fr", "de", "es", "en"]`
- **Modelli**: `detection.onnx`, `english_g2_rec.onnx` / `latin_g2_rec.onnx`
- **Preprocessing detector**: Resize fisso 800x608, normalizzazione mean/std
- **Preprocessing recognizer**: Grayscale, resize height 64px, padding a 1000px

#### Step 2.2: Confronto con Python OCR ✅ COMPLETATO
**Obiettivo**: Identificare differenze OCR
**Azioni**:
- [x] Localizzare OCR in Docling Python (`easyocr_model.py`)
- [x] Confrontare modelli utilizzati
- [x] Confrontare preprocessing
- [x] Confrontare parametri inferenza
- [x] Documentare differenze

**Output**: Lista differenze OCR .NET vs Python documentata

**Differenze Critiche Identificate**:

| Aspetto | .NET | Python | Impatto |
|---------|------|--------|---------|
| **🔴 Risultati multipli** | `return new[] { new OcrResult(text, bbox) };` **UNA SOLA DETECTION** | `reader.readtext(im)` **LISTA COMPLETA** | ❌ CRITICO |
| **Scale/DPI** | Default | `scale=3` (216 DPI) | ⚠️ Qualità inferiore |
| **Preprocessing** | No upscaling | `get_page_image(scale=3)` | ⚠️ Qualità inferiore |
| **Confidence** | ❌ Nessun campo | ✅ `confidence=line[2]` | ⚠️ Metadati persi |

#### Step 2.3: Analisi CRAFT Detector ✅ COMPLETATO
**Obiettivo**: Capire perché EasyOcrNet ritorna solo 1 risultato
**Azioni**:
- [x] Investigare output format detector ONNX
- [x] Analizzare `GetBboxFromDetector` in `EasyOcr.cs`
- [x] Studiare algoritmo `getDetBoxes` da Python EasyOCR
- [x] Documentare algoritmo corretto

**Output**: Root cause identificato

**🔴 BUG CRITICO CONFERMATO**:

**File**: `src/submodules/easyocrnet/EasyOcrNet/EasyOcr.cs:276-311`

**Problema**:
```csharp
private static SKRect GetBboxFromDetector(ModelOutput output, int width, int height)
{
    // ... calcola UN SOLO bounding box min/max di TUTTI i pixel con score > 0.3
    for (int y = 0; y < detH; y++)
    {
        for (int x = 0; x < detW; x++)
        {
            if (score > 0.3f)  // ← Usa solo channel 0, ignora channel 1 (linkmap)!
            {
                if (x < minX) minX = x;  // ← UN SOLO box globale!
                // ...
            }
        }
    }
    return new SKRect(minX, minY, maxX, maxY);  // ← SINGOLO rettangolo!
}
```

**Algoritmo Corretto** (da Python EasyOCR):
```python
def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # 1. Binarizza entrambi i canali
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    # 2. Combina text + link
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # 3. Connected Components Analysis → trova N regioni separate!
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(...)

    # 4. Per ogni regione (N boxes):
    for k in range(1, nLabels):
        # Filtra per size e threshold
        # Applica morfologia (dilatazione)
        # Calcola minAreaRect
        det.append(box)  # ← LISTA di N boxes!

    return det  # Lista, non singolo elemento!
```

**Detector Output Format**:
- Shape: `[1, H, W, 2]`
- **Channel 0** (`textmap`): Confidence map per text regions
- **Channel 1** (`linkmap`): Confidence map per character links
- **Thresholds**: text=0.7, link=0.4, low_text=0.4 (vs 0.3 in .NET)

**Impatto**:
- Immagine con 50 righe → .NET ritorna 1 box → Docling richiama OCR centinaia di volte → **145 secondi**
- Corretto: 50 boxes → Docling chiama OCR 13 volte (una per layout bbox) → **~10-15 secondi**

#### Step 2.4: Implementazione Fix CRAFT Multi-Bbox (OPZIONE A) ⏳ PIANIFICATO

**Obiettivo**: Implementare algoritmo completo CRAFT multi-bbox extraction usando **SOLO SkiaSharp** (pure .NET)

**Requisiti**:
- ❌ NO Python wrapper
- ❌ NO OpenCV.NET
- ✅ SOLO SkiaSharp + System.Drawing (se necessario)
- ✅ Pure .NET implementation

**Piano Dettagliato**:

##### **2.4.1: Implementare Connected Components Analysis** ✅ COMPLETATO
- [x] Creare `ConnectedComponentsAnalyzer.cs` in `EasyOcrNet/ImageProcessing/`
- [x] Creare `UnionFind.cs` - Union-Find data structure
- [x] Implementare algoritmo Two-Pass Labeling:
  - First pass: scansione + assegnazione label temporanee
  - Union-Find per equivalenze
  - Second pass: riassegnazione label finali
- [x] Calcolare stats per ogni componente (area, bounding box, centroid)
- [x] Test unitario con immagini sintetiche (11 test)

**Input**: `bool[,]` binary map (H x W)
**Output**: `ComponentStats[]` con (label, x, y, width, height, area, centroid)

**Risultati**:
- ✅ UnionFind.cs: 93 righe (path compression + union by rank)
- ✅ ConnectedComponentsAnalyzer.cs: 272 righe
- ✅ ConnectedComponentsTests.cs: 260 righe
- ✅ Test: **11/11 PASSED**
- ✅ 4-connectivity implementation
- ✅ Gestione edge cases (empty, single pixel, diagonal pixels)

##### **2.4.2: Implementare Morfologia (Dilatazione)** ✅ COMPLETATO
- [x] Creare `MorphologyOps.cs` in `EasyOcrNet/ImageProcessing/`
- [x] Implementare dilatazione binaria con kernel rettangolare
- [x] Implementare erosione binaria
- [x] Implementare Open (erosion + dilation)
- [x] Implementare Close (dilation + erosion)
- [x] Ottimizzare con separable kernels (horizontal + vertical)
- [x] Test con kernel 3x3, 5x5, 7x7, asimmetrici

**Risultati**:
- ✅ MorphologyOps.cs: 233 righe
- ✅ MorphologyTests.cs: 241 righe
- ✅ Test: **12/12 PASSED**
- ✅ Ottimizzazione: O(W×H×kw×kh) → O(W×H×(kw+kh))
- ✅ Operazioni: Dilate, Erode, Open, Close

##### **2.4.3: Implementare Min Area Rectangle** ✅ COMPLETATO
- [x] Creare `GeometryUtils.cs` in `EasyOcrNet/ImageProcessing/`
- [x] Implementare strutture dati: `Point`, `RotatedRect`
- [x] Estrai contorni da segmentation map (boundary pixels in 4-connectivity)
- [x] Implementare Convex Hull (Graham's scan algorithm)
- [x] Implementare Rotating Calipers algorithm per MinAreaRect
- [x] Fallback: Axis-aligned bounding box
- [x] Test con forme varie (rettangoli ruotati, poligoni, diamanti)

**Risultati**:
- ✅ GeometryUtils.cs: 330 righe
- ✅ GeometryUtilsTests.cs: 273 righe
- ✅ Test: **17/17 PASSED**
- ✅ ExtractContour: estrazione boundary pixels
- ✅ ConvexHull: Graham's scan con polar angle sorting
- ✅ MinAreaRect: Rotating Calipers per rettangolo minimo
- ✅ RotatedRect.GetCorners(): conversione a 4 corner points
- ✅ Edge cases: empty, single point, two points, axis-aligned

**Test Suite Completa ImageProcessing**: **40/40 PASSED** ✅

##### **2.4.4: Implementare GetAllBboxesFromDetector** ✅ COMPLETATO
- [x] Modificare `DetectionPostProcessor` per generare tutte le bbox (text+link map).
- [x] Usare ConnectedComponents/Morphology/GeometryUtils per min-area rect e rettangoli scalati.
- [x] Collegare `TextDetector` alla nuova pipeline (fallback già presente).
- [x] Eseguire test EasyOcrNet mirati (`DetectionPostProcessorTests`) con modelli sintetici.

**Note**:
- Il nuovo `DetectionPostProcessor` applica soglie (0.7/0.4/0.4), connected components, dilatazione adattiva e `GeometryUtils.MinAreaRect` per ottenere bounding box multiple.
- Output finale in coordinate ridimensionate, ancora raggruppato in linee dal `TextComponentGrouper`.
- Test eseguiti via `dotnet test EasyOcrNet.Tests/EasyOcrNet.Tests.csproj --filter DetectionPostProcessorTests` dopo aver bypassato `EnsureReleaseModels` tramite la proprietà `SkipEnsureReleaseModels`.

##### **2.4.5: Modificare Read() per usare multiple boxes**
- [ ] Aggiornare `EasyOcr.Read()`:
  ```csharp
  public IEnumerable<OcrResult> Read(SKBitmap image)
  {
      // ... preprocessing detector ...
      var detOutput = _backend.RunDetector(detTensor);

      // ← NUOVO: ottieni TUTTE le boxes
      var bboxes = GetAllBboxesFromDetector(detOutput, resized.Width, resized.Height);

      // ← NUOVO: loop su tutte le boxes
      var results = new List<OcrResult>();
      foreach (var bbox in bboxes)
      {
          var text = Recognize(resized, bbox);
          results.Add(new OcrResult(text, bbox));
      }

      return results;  // ← LISTA, non array con 1 elemento!
  }
  ```

##### **2.4.6: Aggiungere Confidence Score**
- [ ] Modificare `OcrResult` record:
  ```csharp
  public record OcrResult(string Text, SKRect BoundingBox, float Confidence);
  ```
- [ ] Calcolare confidence nel recognizer
- [ ] Aggiornare `EasyOcrService.cs` per usare confidence da result

##### **2.4.7: Testing & Validation**
- [ ] Creare test con immagini multi-line
- [ ] Confrontare numero di detections con Python EasyOCR
- [ ] Verificare qualità bounding boxes
- [ ] Misurare performance

**File da Creare/Modificare**:
```
src/submodules/easyocrnet/EasyOcrNet/
├── ImageProcessing/
│   ├── ConnectedComponentsAnalyzer.cs  (NUOVO)
│   ├── MorphologyOps.cs                (NUOVO)
│   └── GeometryUtils.cs                (NUOVO)
├── EasyOcr.cs                          (MODIFICARE)
└── Backends.cs                         (OK - no changes)

src/submodules/easyocrnet/EasyOcrNet.Tests/
└── ImageProcessing/
    ├── ConnectedComponentsTests.cs     (NUOVO)
    ├── MorphologyTests.cs              (NUOVO)
    └── GeometryUtilsTests.cs           (NUOVO)
```

**Stima Complessità**:
- Connected Components: ~150 righe
- Morfologia: ~80 righe
- Min Area Rect: ~120 righe
- GetAllBboxes: ~200 righe
- Totale: ~550 righe di codice nuovo
- Testing: ~300 righe
- **Totale**: ~850 righe

**Success Criteria**:
✅ Connected Components funzionante con test
✅ Morfologia (dilatazione) funzionante
✅ Min Area Rect corretto
✅ GetAllBboxesFromDetector ritorna N boxes (non 1)
✅ Read() ritorna lista completa risultati
✅ Test multi-line passa con 10+ detections
✅ Performance accettabile (< 2s per immagine)

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
- Layout detection time: < 2s (attuale: ~0.8s ✅ **MIGLIORATO**)
- OCR time: < 30s (attuale: 145s ❌ → target: ~15s con fix CRAFT)
- Total pipeline time: < 35s (attuale: 147s ❌ → target: ~20s)

### Quality
- Layout detections: 13-14 (~~attuale: 0 primario, 184 fallback~~ ✅ **FIXATO: 13 detections con 4 tables**)
- Markdown readability: Comparabile a Python (attuale: illeggibile ❌ → dipende da OCR fix)
- OCR accuracy: > 95% (attuale: ~60% stimato ❌ → dipende da multi-bbox fix)
- Table detection: Identificata (~~attuale: no~~ ✅ **FIXATO: 4 tables rilevate**)

### Stability
- No fallback ONNX trigger (~~attuale: sempre fallback~~ ✅ **FIXATO: fallback rimosso completamente**)
- No duplicazioni testo (attuale: massivo ❌ → da verificare post-OCR fix)
- Consistent results across runs (da verificare)

---

## 📈 PROGRESSI

### ✅ FASE 1: COMPLETATA (80%)
- ✅ Layout SDK fixato: 0 → 13 detections
- ✅ Fallback rimosso: -542 righe codice
- ✅ Table detection: 0 → 4 tables
- ✅ Build successful
- ⏳ Step 1.5-1.6 (post-processing, packaging) da completare

### 🔄 FASE 2: IN CORSO (60%)
- ✅ Step 2.1: Analisi EasyOcrNet completata
- ✅ Step 2.2: Confronto Python completato
- ✅ Step 2.3: Root cause identificato (single-bbox bug)
- ⏳ Step 2.4: Piano implementazione CRAFT multi-bbox definito
- ⏳ Step 2.5-2.7: Implementazione, testing, integrazione

---

## 🚀 ESECUZIONE

**Approccio**: Procediamo FASE per FASE, step by step
**Priorità**: Fase 1 ✅ → **Fase 2 (in corso)** → Fase 5

**Prossimo Step**: **FASE 2 - Step 2.4.1** - Implementare Connected Components Analysis

**Stato**: Ready to implement CRAFT multi-bbox extraction (pure .NET, SkiaSharp only)
