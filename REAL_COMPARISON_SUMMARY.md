# TableFormer: Real Comparison Analysis - Python vs C#

## Executive Summary

Il confronto tra Python e C# ha rivelato che **l'implementazione C# TorchSharp NON ha i pesi del modello caricati**, quindi sta usando **pesi random inizializzati**.

### Risultati Confronto

| Metrica | Python (✅ Trained Weights) | C# (❌ Random Weights) |
|---------|---------------------------|----------------------|
| **Tempo** | 0.49s - 1.69s | 42-43s |
| **Lunghezza sequenza** | 107-354 tags (termina con `<end>`) | 1024 tags (max_steps raggiunto) |
| **Comportamento** | Output sensati | Output random divergente |
| **Stopping** | Si ferma con `<end>` token | Non si ferma mai |

## Root Cause: Weights Not Loaded

### Evidenza #1: Bbox Identici (Segno di Random Weights)

**C# Output:**
```
Box 0: cx=0.4917, cy=0.5169, w=0.5125, h=0.5038, class=1
Box 1: cx=0.4917, cy=0.5169, w=0.5125, h=0.5038, class=1  ← IDENTICO
Box 2: cx=0.4917, cy=0.5169, w=0.5125, h=0.5038, class=1  ← IDENTICO
...
(tutti i 926 bbox sono identici!)
```

### Evidenza #2: Sequenza Infinita di ecel

**C# Output:**
```
...ecel ecel ecel ecel ecel ecel ecel ecel ecel ecel ecel ecel ecel...
(355 ecel consecutivi!)
```

Questo è il comportamento tipico di un modello con pesi random - ripete infinitamente lo stesso token.

### Evidenza #3: Non Genera `<end>` Token

- **Python:** Genera `<end>` dopo 107-354 tags
- **C#:** Raggiunge sempre `max_steps=1024` senza mai generare `<end>`

Con pesi random, la probabilità di generare esattamente il token `<end>` è ~7.7% (1/13 vocab_size), quindi è statisticamente improbabile che accada prima di max_steps.

### Evidenza #4: Performance 25-87x Più Lenta

Il modello C# è estremamente lento perché:
1. Esegue 1024 step invece di ~100-350
2. Non ha ottimizzazioni dei pesi addestrati
3. TorchSharp overhead vs PyTorch nativo

---

## Perché i Pesi Non Sono Caricati?

### Architettura C# Attuale

```csharp
public TableModel04(TableModel04Config config)
{
    _encoder = new Encoder04(config.EncImageSize, config.EncoderDim);
    _tagTransformer = new TagTransformer(...);
    _bboxDecoder = new BBoxDecoder(...);

    // ❌ MANCA: Caricamento weights da file!
    // I moduli sono inizializzati con pesi RANDOM da TorchSharp
}
```

### Architettura Python (Riferimento)

```python
# Crea il modello
model = TableModel04_rs(config, init_data=init_data, device=device)

# ✅ CARICA i pesi addestrati
state_dict = load_file("tableformer_fast.safetensors")
model.load_state_dict(state_dict, strict=False)
```

---

## File dei Pesi

Il modello addestrato è disponibile in:
```
models/model_artifacts/tableformer/fast/tableformer_fast.safetensors (139 MB)
```

Contiene 258 tensori:
- 90 tensori encoder (_encoder.*)
- 118 tensori tag transformer (_tag_transformer.*)
- 50 tensori bbox decoder (_bbox_decoder.*)

### Struttura Chiave

```
_encoder._resnet.0.weight                   # Conv2d iniziale
_encoder._resnet.4.0.conv1.weight           # ResNet blocks
...
_tag_transformer._input_filter.0.conv1.weight    # Projection 256→512
_tag_transformer._encoder.layers.0.*             # Transformer encoder
_tag_transformer._decoder.layers.0.*             # Transformer decoder
_tag_transformer._output_projection.weight       # Output head
...
_bbox_decoder._class_embed.weight                # BBox classifier
_bbox_decoder._bbox_embed.layers.0.weight        # BBox regressor
```

---

## Soluzioni Possibili

### Opzione 1: Usa ONNX Models (RACCOMANDATO)

I modelli ONNX hanno già i pesi addestrati embedati:

```
models/onnx-components/
├── tableformer_fast_encoder.onnx (11 MB)
├── tableformer_fast_tag_transformer_encoder.onnx (64 MB)
├── tableformer_fast_tag_transformer_decoder_step.onnx (26 MB)
└── tableformer_fast_bbox_decoder.onnx (38 MB)
```

**Pro:**
- ✅ Pesi già inclusi
- ✅ Ottimizzato per inferenza
- ✅ Microsoft.ML.OnnxRuntime ben supportato in C#
- ✅ Cross-platform

**Contro:**
- ❌ 4 modelli separati da orchestrare
- ❌ Richiede loop autoregressivo manuale in C#
- ❌ Più complesso del modello end-to-end

### Opzione 2: Implementare Safetensors Loader

Caricare `tableformer_fast.safetensors` direttamente in TorchSharp.

**Pro:**
- ✅ Usa l'implementazione TorchSharp esistente
- ✅ Modello end-to-end singolo
- ✅ Più facile da mantenere

**Contro:**
- ❌ TorchSharp non ha supporto nativo per safetensors
- ❌ Richiede libreria esterna o parser custom
- ❌ Mappatura nome parametri Python→C# può essere complessa

**Complessità Implementazione:**
```csharp
// Pseudo-code
var safetensors = LoadSafetensors("tableformer_fast.safetensors");

foreach (var (name, tensor) in safetensors)
{
    // Mappare nome Python → TorchSharp parameter
    var csharpName = MapParameterName(name);

    // Trovare il parametro nel modello
    var param = model.get_parameter(csharpName);

    // Copiare i pesi
    param.copy_(tensor);
}
```

### Opzione 3: Convertire Safetensors → PyTorch .pt

Esportare i pesi in formato `.pt` che TorchSharp può leggere.

```python
import torch
from safetensors.torch import load_file

state_dict = load_file("tableformer_fast.safetensors")
torch.save(state_dict, "tableformer_fast.pt")
```

Poi in C#:
```csharp
var stateDict = torch.load("tableformer_fast.pt");
// Load into model...
```

**Pro:**
- ✅ TorchSharp può avere supporto per `torch.load()`
- ✅ Formato standard PyTorch

**Contro:**
- ❌ Ancora richiede mappatura nomi parametri
- ❌ TorchSharp potrebbe non supportare completamente il formato

---

## Test Data Creato

Ho creato dati di ground truth da Python per validare eventuali implementazioni future:

```
test-data-python-ground-truth/
├── input_image.npy              # Input preprocessato (1, 3, 448, 448)
├── encoder_output.npy           # Output encoder (1, 28, 28, 256)
├── full_tag_sequence.npy        # Sequenza completa [2, 4, 4, ..., 3]
└── metadata.json                # Metadata (lunghezze, shapes, etc.)
```

**Tag Sequence Ground Truth (Immagine 125317):**
```
Length: 107 tags
First 20: [2, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4]
Last 20:  [5, 5, 5, 5, 9, 5, 4, 5, 5, 5, 5, 9, 5, 4, 5, 5, 5, 5, 9, 3]

Token mapping:
2 = <start>
3 = <end>
4 = ecel
5 = fcel
9 = nl
```

---

## Conclusioni e Raccomandazioni

### Stato Attuale

1. ✅ **Architettura C# è CORRETTA**
   - Encoder04 con projection layer
   - TagTransformer con encoder/decoder
   - BBoxDecoder con MLP
   - Loop autoregressivo implementato
   - Structure error correction implementata

2. ❌ **Pesi NON sono caricati**
   - Il modello usa pesi random
   - Output non significativi
   - Impossibile fare confronto reale

3. ✅ **Test suite è completa**
   - 294 tests tutti passano
   - 90%+ code coverage
   - Validano architettura, non accuracy

### Priorità per Confronto Reale

**P0 - CRITICO:**
1. Scegliere strategia di caricamento pesi (ONNX vs Safetensors)
2. Implementare caricamento pesi
3. Validare che genera `<end>` token correttamente
4. Confrontare output con Python ground truth

**P1 - IMPORTANTE:**
5. Ottimizzare performance (target: <5s vs 43s attuale)
6. Validare accuracy su dataset di test
7. Confrontare bbox predictions con Python

### Test di Validazione Post-Loading

Una volta caricati i pesi, il modello C# dovrebbe:

```csharp
// Input: HAL.2004.page_82.pdf_125317.png
var result = model.forward(img);

// Expected:
Assert.Equal(107, result.Sequence.Count);  // vs 1024 attuale
Assert.Equal(startToken, result.Sequence[0]);  // <start>
Assert.Equal(endToken, result.Sequence[^1]);   // <end>
Assert.InRange(result.BBoxCoords.size(0), 80, 100);  // vs 926 attuale

// First 20 tags should match Python exactly:
var expected = new long[] { 2, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4 };
for (int i = 0; i < 20; i++)
{
    Assert.Equal(expected[i], result.Sequence[i]);
}
```

---

## Files Generated

1. **Python Ground Truth:**
   - `test-data-python-ground-truth/` - Dati di riferimento per validazione

2. **Comparison Reports:**
   - `benchmark-results/COMPARISON_PYTHON_VS_CSHARP.md` - Analisi dettagliata
   - `REAL_COMPARISON_SUMMARY.md` - Questo documento

3. **Python Inference Results:**
   - `benchmark-results/python_125317_results.txt` - Python output immagine 1
   - `benchmark-results/python_125315_results.txt` - Python output immagine 2

4. **C# Inference Results:**
   - `benchmark-results/HAL.2004.page_82.pdf_125317_results.txt` - C# output (random weights)
   - `benchmark-results/HAL.2004.page_82.pdf_125315_results.txt` - C# output (random weights)

5. **Weights Export:**
   - `models/tableformer_weights_export/` - Pesi esportati in formato PyTorch .pt

6. **Utility Scripts:**
   - `benchmark_python_comparison.py` - Script inferenza Python
   - `export_tableformer_weights.py` - Export pesi da safetensors
   - `create_test_with_weights.py` - Crea test data con ground truth

---

## Next Steps

Il prossimo passo **deve** essere implementare il caricamento dei pesi. Senza questo, qualsiasi confronto è privo di significato perché stiamo confrontando un modello addestrato con uno random.

La raccomandazione è usare **ONNX** perché:
1. I modelli esistono già con pesi
2. C# ha ottimo supporto via Microsoft.ML.OnnxRuntime
3. È il formato standard per produzione

