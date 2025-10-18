# TableFormer Benchmark: Python vs C# Comparison

## Executive Summary

Confronto tra l'implementazione Python (usando PyTorch + safetensors) e l'implementazione C# (usando TorchSharp) del modello TableFormer su 2 immagini del dataset FinTabNet.

### Key Findings

| Metrica | Python | C# | Differenza |
|---------|--------|-----|------------|
| **Immagine 1 (125317)** | | | |
| Tempo inferenza | 0.49s | 42.7s | **87x più lento** |
| Lunghezza sequenza | 107 tags | 1024 tags | **9.6x più lungo** |
| Numero bbox | 90 | 926 | **10.3x più bbox** |
| **Immagine 2 (125315)** | | | |
| Tempo inferenza | 1.69s | 43.4s | **25.7x più lento** |
| Lunghezza sequenza | 354 tags | 1024 tags | **2.9x più lungo** |
| Numero bbox | 308 | 811 | **2.6x più bbox** |

---

## Detailed Analysis

### Image 1: HAL.2004.page_82.pdf_125317.png
**Original size:** 186x58 pixels → Resized to 448x448

#### Python Results (Reference Implementation)
```
Inference Time: 0.49s
Tag Sequence Length: 107
Number of Bounding Boxes: 90

First tags: <start> ecel ecel ecel ecel ecel ecel nl ecel ecel ecel ecel ecel ecel nl...
Last tags: ...fcel ecel fcel fcel fcel fcel nl fcel ecel fcel fcel fcel fcel nl <end>
```

**Tag Distribution:**
- `ecel` (end cell): Dominant
- `fcel` (first cell): Present
- `nl` (new line): Row delimiters
- Sequenza termina con `<end>` token

#### C# Results (TorchSharp Implementation)
```
Inference Time: 42,718.21ms (42.7 seconds)
Tag Sequence Length: 1024 (MAX_STEPS reached!)
Number of Bounding Boxes: 926

First tags: lcel fcel lcel <start> <pad> lcel fcel lcel <start> <pad> nl nl nl nl...
Pattern: Massive repetition of ecel tokens (355 consecutive ecel!)
```

**PROBLEMA CRITICO:**
1. ❌ Il modello C# **non si ferma** - raggiunge max_steps=1024 invece di generare `<end>`
2. ❌ Genera **335 ecel consecutivi** (caratteristico di divergenza del modello)
3. ❌ Genera **10x più bounding boxes** del necessario
4. ❌ Tag sequence contiene `<start>` e `<pad>` nella sequenza (anomalo)
5. ❌ **87x più lento** dell'implementazione Python

---

### Image 2: HAL.2004.page_82.pdf_125315.png
**Original size:** 216x70 pixels → Resized to 448x448

#### Python Results
```
Inference Time: 1.69s
Tag Sequence Length: 354
Number of Bounding Boxes: 308

First tags: <start> ched ched ched ched ched ched ched nl fcel fcel fcel fcel...
Last tags: ...fcel fcel fcel fcel fcel fcel fcel nl <end>
```

**Tag Distribution:**
- `ched` (column header): Presente all'inizio
- `fcel`/`ecel`: Struttura cellule
- `nl`: Row delimiters
- Sequenza termina correttamente con `<end>`

#### C# Results
```
Inference Time: 43,444ms (43.4 seconds)
Tag Sequence Length: 1024 (MAX_STEPS reached!)
Number of Bounding Boxes: 811

First tags: lcel fcel lcel nl ecel ecel ecel ecel ecel ecel ecel ecel...
Pattern: Again massive ecel repetition
```

**PROBLEMA CRITICO:**
1. ❌ Il modello C# **non si ferma** - raggiunge max_steps=1024
2. ❌ Genera **2.9x più tags** del necessario
3. ❌ Genera **2.6x più bounding boxes**
4. ❌ **25.7x più lento** dell'implementazione Python

---

## Root Cause Analysis

### 1. Modello non caricato correttamente (CAUSA PRIMARIA)

Il modello C# **NON HA I PESI CARICATI** dal file safetensors. Evidenze:

```csharp
// TableModel04.cs - Constructor
public TableModel04(TableModel04Config config)
{
    _encoder = new Encoder04(config.EncImageSize, config.EncoderDim);
    _tagTransformer = new TagTransformer(...);
    _bboxDecoder = new BBoxDecoder(...);

    // ❌ MANCA: Caricamento weights da safetensors!
    // ❌ I moduli sono inizializzati con pesi RANDOM
}
```

**Sintomi di pesi random:**
1. Genera sequenze infinite senza `<end>` token
2. Repetizione massiva dello stesso token (ecel)
3. Bbox identici ripetuti (cx=0.4917, cy=0.5169, w=0.5125, h=0.5038)
4. Performance estremamente lenta (no ottimizzazioni GPU)

### 2. Manca implementazione del caricamento safetensors

L'implementazione Python usa:
```python
from safetensors.torch import load_file
state_dict = load_file("tableformer_fast.safetensors")
model.load_state_dict(state_dict, strict=False)
```

L'implementazione C# **NON ha equivalente**:
- Non esiste `LoadFromSafetensors()` method
- Non esiste integrazione con TorchSharp per caricare pesi esterni
- I test usano pesi random (intenzionalmente)

### 3. Performance issues

- **Python:** 0.49s - 1.69s (ottimizzato PyTorch)
- **C#:** 42-43s (**25-87x più lento**)

Cause:
1. Pesi non ottimizzati/non caricati
2. TorchSharp overhead vs PyTorch nativo
3. Nessuna ottimizzazione GPU
4. Loop autoregressivo molto più lento
5. Ogni step decoder molto più lento

---

## Comparison Table: Tag Sequences

### Image 1 Tag Patterns

| Implementation | Pattern | Observation |
|----------------|---------|-------------|
| **Python** | `ecel ecel ecel ecel ecel ecel nl` | Struttura regolare 6 colonne |
| **C#** | `ecel ecel ecel...` (335x) | **DIVERGENZA - Repetizione infinita** |

### Image 2 Tag Patterns

| Implementation | Pattern | Observation |
|----------------|---------|-------------|
| **Python** | `ched ched ched ched ched ched ched nl` | Header rilevato correttamente |
| **C#** | `lcel fcel lcel nl ecel ecel ecel...` | Pattern confuso, no header detection |

---

## Bounding Box Analysis

### Image 1 - First 3 Bboxes

**Python:**
```
Box 0: [-17.369331   3.34868   -3.34634 ]  # Formato: 3 valori (?)
Box 1: [-16.457703   5.524607  -5.568465]
Box 2: [-16.226551    7.613853   -7.5998526]
```

**C#:**
```
Box 0: cx=0.4917, cy=0.5169, w=0.5125, h=0.5038, class=1
Box 1: cx=0.4917, cy=0.5169, w=0.5125, h=0.5038, class=1  # ❌ IDENTICO!
Box 2: cx=0.4917, cy=0.5169, w=0.5125, h=0.5038, class=1  # ❌ IDENTICO!
```

**Problema C#:** Tutti i bbox sono **identici** perché il decoder ha pesi random e predice sempre lo stesso valore.

---

## Required Fixes for C# Implementation

### CRITICAL (P0) - Senza questi fix il modello è inutilizzabile

1. ✅ **Implementare caricamento safetensors**
   ```csharp
   public void LoadFromSafetensors(string path)
   {
       // Load safetensors file
       // Map parameter names to modules
       // Load state dict into TorchSharp modules
   }
   ```

2. ✅ **Verificare mappatura layer names**
   - Encoder: `resnet.*`, `projection.*`
   - TagTransformer: `_input_filter.*`, `_encoder.*`, `_decoder.*`
   - BBoxDecoder: `_class_embed.*`, `_bbox_embed.*`

3. ✅ **Implementare early stopping corretto**
   ```csharp
   if (newTag == _endToken)
   {
       break;  // Stop generation
   }
   ```

### HIGH (P1) - Performance

4. ⚠️ **Ottimizzare loop autoregressivo**
   - Usare TorchSharp tensor operations invece di CPU
   - Batch operations dove possibile
   - Evitare CPU↔GPU transfers

5. ⚠️ **Abilitare CUDA/Metal**
   ```csharp
   var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
   ```

### MEDIUM (P2) - Accuracy

6. 📋 **Validare structure error correction**
   - Verificare che le regole siano applicate correttamente
   - Confrontare con Python step-by-step

7. 📋 **Validare bbox merging per lcel**
   - Confrontare risultati span merging Python vs C#

---

## Next Steps

### Fase 1: Caricamento Pesi (PRIORITÀ MASSIMA)
1. Implementare safetensors reader per TorchSharp
2. Mappare nomi parametri Python → C#
3. Caricare e verificare pesi layer-by-layer
4. Test: Confrontare output intermedi (encoder, decoder) con Python

### Fase 2: Validazione Funzionale
1. Verificare che genera `<end>` token correttamente
2. Confrontare lunghezza sequenze con Python
3. Confrontare numero bbox con Python
4. Confrontare tag sequence character-by-character

### Fase 3: Ottimizzazione Performance
1. Profiling per identificare bottlenecks
2. Ottimizzare hot paths
3. Target: <5s per immagine (vs 43s attuale)

---

## Conclusion

L'implementazione C# di TableFormer è **strutturalmente corretta** ma **funzionalmente non utilizzabile** perché:

1. ❌ **CRITICO:** Non carica i pesi addestrati → predizioni random
2. ❌ **CRITICO:** Genera sequenze infinite senza stopping criterion
3. ❌ **CRITICO:** 25-87x più lenta dell'implementazione Python
4. ✅ **OK:** Architettura modello corretta (encoder, transformer, decoder)
5. ✅ **OK:** Preprocessing immagini corretto
6. ✅ **OK:** Struttura codice modulare e testabile

**La priorità assoluta** è implementare il caricamento dei pesi da safetensors. Senza questo, il modello genera solo output random e non può essere validato.
