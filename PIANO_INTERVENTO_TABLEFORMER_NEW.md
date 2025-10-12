# Piano di Intervento - Migrazione TableFormer a Modelli Ufficiali Docling

## Obiettivo
Sostituire completamente i modelli ONNX esistenti con i modelli ufficiali PyTorch/SafeTensors di Docling, convertendoli in ONNX ottimizzati e rimuovendo tutte le vecchie implementazioni.

## Riferimenti
- Repository ufficiale: https://github.com/docling-project/docling-ibm-models
- Modelli HuggingFace: https://huggingface.co/ds4sd/docling-models/tree/main/model_artifacts/tableformer
- Modelli disponibili:
  - `tableformer_fast.safetensors` (145 MB)
  - `tableformer_accurate.safetensors` (213 MB)
  - `tm_config.json` (7 KB)

## Fase 1: Analisi e Preparazione

### 1.1 Analizzare Architettura Modello Ufficiale
**Status**: ✅ Completata

**Azioni**:
- [ ] Clonare repository `docling-ibm-models`
- [ ] Analizzare struttura del modello TableFormer in PyTorch
- [ ] Documentare:
  - Input shape e preprocessing richiesto
  - Output shape e formato
  - Architettura (encoder, decoder, head)
  - Dipendenze e versioni PyTorch/Transformers

**File da analizzare**:
```
docling-ibm-models/
├── docling_ibm_models/tableformer/
│   ├── model.py
│   ├── config.py
│   └── __init__.py
└── demo/demo_layout_predictor.py
```

### 1.2 Scaricare Modelli Ufficiali
**Modelli target**:
- `tableformer_fast.safetensors` (145 MB) - per Fast mode
- `tableformer_accurate.safetensors` (213 MB) - per Accurate mode
- `tm_config.json` (7 KB) - configurazione

**Storage**:
```
/models/tableformer/
├── fast/
│   ├── tableformer_fast.safetensors
│   └── tm_config.json
└── accurate/
    ├── tableformer_accurate.safetensors
    └── tm_config.json
```

## Fase 2: Conversione ONNX

### 2.1 Creare Script di Conversione PyTorch → ONNX

**Script**: `tools/convert_tableformer_to_onnx.py`

**Funzionalità**:
```python
# Pseudo-codice
def convert_tableformer_to_onnx():
    1. Caricare modello SafeTensors con docling-ibm-models
    2. Creare dummy input con shape corretta (batch, 3, H, W)
    3. Esportare con torch.onnx.export()
    4. Ottimizzare con onnxruntime optimizations
    5. Validare input/output shapes
    6. Test inference con sample data
    7. Salvare ONNX con metadata
```

**Output attesi**:
```
/models/tableformer-onnx/
├── tableformer_fast.onnx
├── tableformer_accurate.onnx
├── fast_config.json
└── accurate_config.json
```

**Parametri di esportazione**:
- `opset_version`: 17 o superiore
- `input_names`: ["pixel_values"]
- `output_names`: ["logits", "pred_boxes"]
- `dynamic_axes`: {
    "pixel_values": {0: "batch"},
    "logits": {0: "batch", 1: "queries"},
    "pred_boxes": {0: "batch", 1: "queries"}
  }

### 2.2 Validazione Modelli ONNX

**Test da eseguire**:
1. **Shape validation**: Verificare output shapes `[batch, queries, classes]`
2. **Numerical validation**: Confrontare output PyTorch vs ONNX (tolerance < 1e-4)
3. **Performance test**: Misurare latency e throughput
4. **Memory footprint**: Verificare utilizzo memoria

## Fase 3: Pulizia Codice Esistente

**Status**: ✅ Completata (12 Ottobre 2025)

### 3.1 Rimuovere Vecchi Modelli
- [x] Eliminati gli ONNX legacy (`encoder.onnx`, `bbox_decoder.onnx`, `decoder.onnx`, `tableformer-fast-encoder.onnx`)
- [x] Rimosse le cartelle OpenVINO (`ov-ir/`, `ov-ir-fp16/`) e i metadata associati

### 3.2 Rimuovere Backend Obsoleti
- [x] Rimosso `TableFormerPipelineBackend.cs` e la variante ottimizzata
- [x] Rimosso `OpenVinoBackend.cs` e le dipendenze OpenVINO
- [x] Aggiornato `DefaultBackendFactory` per supportare solo il backend ONNX
- [x] Introdotto invoker stub (`NullTableFormerInvoker`) per mantenere attivo il servizio in assenza del backend definitivo

**Mantenere solo**:
- `TableFormerOnnxBackend.cs` (da riscrivere)
- `TableFormerDetectionParser.cs` (da aggiornare)
- `ITableFormerBackend.cs` (interface)

## Fase 4: Nuova Implementazione .NET

### 4.1 Riscrivere TableFormerOnnxBackend

**Nuovo design**:
```csharp
public class TableFormerOnnxBackend : ITableFormerBackend
{
    // Single ONNX session (no pipeline split)
    private readonly InferenceSession _session;
    private readonly TableFormerConfig _config;
    private readonly IImagePreprocessor _preprocessor;

    public IReadOnlyList<TableCell> Infer(SKBitmap image)
    {
        // 1. Preprocess: Letterboxing + ImageNet normalization
        var tensor = _preprocessor.Preprocess(image);

        // 2. Run ONNX inference
        var outputs = RunInference(tensor);

        // 3. Parse full DETR output [batch, queries, classes]
        var detections = ParseDetections(outputs);

        // 4. Post-process: NMS, cell grouping, row/col detection
        var cells = PostProcessCells(detections, image.Size);

        return cells;
    }
}
```

**Componenti**:
1. **ImagePreprocessor**: Letterboxing, resize, normalization
2. **DetectionParser**: Parse DETR output completo (100+ queries)
3. **CellPostProcessor**: Group cells, detect rows/cols, assign spans
4. **CoordinateTransformer**: Trasformazione inversa da letterboxed a original coords

### 4.2 Aggiornare TableFormerDetectionParser

**Nuovo parser per output DETR completo**:
```csharp
public static IReadOnlyList<TableCell> Parse(
    float[] logits,      // [batch, num_queries, num_classes]
    float[] boxes,       // [batch, num_queries, 4]
    int imageWidth,
    int imageHeight,
    float scoreThreshold = 0.25f)
{
    // Parse FULL DETR output con multi-query
    // Non più [1,3] ma [1,100,3] o simile
}
```

### 4.3 Implementare Post-Processing Avanzato

**Nuove classi**:
```csharp
public class TableCellGrouper
{
    // Group detected boxes into logical cells
    // Handle overlapping boxes, merge fragments
}

public class TableStructureAnalyzer
{
    // Detect rows and columns from cell positions
    // Assign row/column spans
    // Detect header rows/columns
}
```

## Fase 5: Integrazione e Test

### 5.1 Aggiornare TableFormerTableStructureService

**Modifiche**:
- Rimuovere hardcoded paths
- Configurazione da appsettings/environment
- Supporto hot-reload modelli
- Metrics e telemetry

### 5.2 Test Suite Completa

**Test unitari**:
```csharp
TableFormerOnnxBackendTests.cs
├── TestPreprocessing()
├── TestInference()
├── TestParsing()
├── TestPostProcessing()
└── TestEndToEnd()
```

**Test di integrazione**:
```csharp
TableFormerIntegrationTests.cs
├── TestSimpleTable()
├── TestComplexTable()
├── TestSpannedCells()
├── TestNestedHeaders()
└── TestRegressionSuite()
```

**Test con documento reale**:
- Documento: `2305.03393v1-pg9`
- Atteso: ~50 celle estratte con struttura corretta
- Confronto con golden Python output

## Fase 6: Ottimizzazione e Performance

### 6.1 Ottimizzazioni ONNX Runtime
- Graph optimizations
- Quantization (INT8 se possibile)
- Provider selection (CPU/GPU)
- Batch processing

### 6.2 Profiling e Benchmarking
- Latency per tabella
- Throughput (tabelle/sec)
- Memory usage
- Confronto Fast vs Accurate

## Fase 7: Documentazione

### 7.1 Documentazione Tecnica
**File da creare**:
- `docs/TABLEFORMER_ARCHITECTURE.md`
- `docs/TABLEFORMER_ONNX_CONVERSION.md`
- `docs/TABLEFORMER_PERFORMANCE.md`

### 7.2 Guida Utente
- Come scaricare modelli
- Come configurare paths
- Come scegliere Fast vs Accurate
- Troubleshooting comune

## Fase 8: Validazione Finale

### 8.1 Test Golden Dataset
- Test su tutti i documenti di test
- Confronto con output Python
- Verifica quality metrics (TEDS, mAP)

### 8.2 Regression Testing
- Nessuna regressione su documenti già funzionanti
- Miglioramenti misurabili su documenti problematici

## Deliverables Finali

1. ✅ Modelli ONNX convertiti e validati (Fast + Accurate)
2. ✅ Backend .NET pulito e testato
3. ✅ Parser completo per DETR multi-query
4. ✅ Post-processing avanzato (row/col detection, spans)
5. ✅ Test suite completa
6. ✅ Documentazione esaustiva
7. ✅ Performance benchmarks

## Timeline Stimata

| Fase | Durata Stimata | Priorità |
|------|----------------|----------|
| 1. Analisi e Preparazione | 2-3 ore | Alta |
| 2. Conversione ONNX | 3-4 ore | Critica |
| 3. Pulizia Codice | 1 ora | Media |
| 4. Nuova Implementazione | 6-8 ore | Critica |
| 5. Integrazione e Test | 4-5 ore | Alta |
| 6. Ottimizzazione | 2-3 ore | Media |
| 7. Documentazione | 2 ore | Bassa |
| 8. Validazione | 2-3 ore | Alta |
| **TOTALE** | **22-31 ore** | |

## Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| Conversione ONNX fallisce | Media | Alto | Testare con più opset, usare torch.onnx export tools |
| Output shapes incompatibili | Alta | Alto | Analizzare attentamente model architecture prima |
| Performance peggiori | Bassa | Medio | Benchmark early, ottimizzare con ORT tools |
| Breaking changes | Media | Alto | Extensive testing prima del merge |

## Note di Implementazione

### Problema Identificato con Modelli Attuali
I modelli ONNX attuali (`encoder.onnx`, `bbox_decoder.onnx`, `decoder.onnx`) sono modelli semplificati che restituiscono:
- Output shape: `[1, 3]` logits e `[1, 4]` boxes (1 sola detection per tabella)
- Funzionalità: **Table detection** (trovare dove sono le tabelle)
- Limitazione: NON estraggono la struttura interna (celle, righe, colonne)

### Obiettivo con Nuovi Modelli
I modelli ufficiali Docling dovrebbero essere modelli DETR completi che restituiscono:
- Output shape: `[1, num_queries, num_classes]` e `[1, num_queries, 4]`
- Funzionalità: **Table structure recognition** (estrarre tutte le celle)
- Beneficio: Estrazione completa della struttura tabellare con righe, colonne e spanning

## Status Attuale

**Completato**:
- ✅ Analisi problema modelli esistenti
- ✅ Normalizzazione ImageNet implementata
- ✅ Soglia confidenza abbassata (0.5 → 0.25)
- ✅ Logging di debug avanzato
- ✅ Parser adattato per formato semplificato [1,3]

**Da fare**:
- [ ] Clonare e analizzare docling-ibm-models
- [ ] Convertire modelli ufficiali in ONNX
- [ ] Riscrivere backend per DETR completo
- [ ] Implementare post-processing avanzato

---

**Prossimo Step**: Fase 1.1 - Clonare `docling-ibm-models` e analizzare architettura del modello TableFormer
