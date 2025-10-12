# TableFormer Architecture Analysis

## Data: 12 Ottobre 2025

## Fonte
- Repository: https://github.com/docling-project/docling-ibm-models
- Modelli: https://huggingface.co/ds4sd/docling-models/tree/main/model_artifacts/tableformer
- Branch analizzato: main

## Architettura Generale

### Nome Modello: TableModel04_rs (OTSL)

TableFormer è un modello encoder-decoder basato su transformer per il riconoscimento della struttura delle tabelle.

### Componenti Principali

```
TableModel04_rs
├── Encoder04 (ResNet-18 based)
├── Tag_Transformer (struttura OTSL)
└── BBoxDecoder (bounding box prediction)
```

## 1. Encoder (Encoder04)

**File**: `models/table04_rs/encoder04_rs.py`

**Architettura**:
- Base: ResNet-18 (torchvision.models.resnet18())
- Rimuove ultimi 3 layer (FC layers)
- Adaptive pooling: adatta output a dimensione fissa

**Input**:
- Shape: `(batch_size, 3, H, W)` dove tipicamente H=W=448
- Format: RGB image, normalizzato

**Output**:
- Shape: `(batch_size, 28, 28, 256)`
- Encoder dim: 256 (da ResNet-18)
- Hidden dim configurabile: 512 (default)

**Preprocessing**:
```python
# Input image: (1, 3, 448, 448)
resnet_output = resnet(images)  # (1, 256, 28, 28)
adaptive_pool_output = adaptive_pool(resnet_output)  # (1, 256, 28, 28)
encoder_output = output.permute(0, 2, 3, 1)  # (1, 28, 28, 256)
```

## 2. Tag Transformer (Tag_Transformer)

**File**: `models/table04_rs/transformer_rs.py`

**Scopo**: Genera sequenza OTSL (Optimized Table Structure Language)

**Architettura**:
- Transformer encoder-decoder
- Positional encoding
- Multi-head attention
- Configurabile: layers, heads, dimensions

**Output OTSL Tags**:
- `fcel`: first cell in row
- `ecel`: empty cell
- `lcel`: last cell in horizontal span
- `xcel`: continuation cell in horizontal span
- `ucel`: cell with vertical span
- `ched`: column header
- `rhed`: row header
- `srow`: spanning row
- `nl`: newline
- `<start>`, `<end>`: sequence markers

**Processo**:
1. Encode dell'immagine (già fatto da Encoder04)
2. Decoding autoregressive: genera tag uno alla volta
3. Ferma quando genera `<end>` o raggiunge max_steps

## 3. BBox Decoder (BBoxDecoder)

**File**: `models/table04_rs/bbox_decoder_rs.py`

**Scopo**: Predice bounding box per ogni cella identificata dai tag

**Input**:
- Encoder output: `(1, 28, 28, 256)`
- Tag decoder hidden states: lista di tensori per ogni tag che richiede bbox

**Output**:
- `outputs_class`: `(num_cells, 3)` - classificazione (tipo di cella)
- `outputs_coord`: `(num_cells, 4)` - coordinate in formato [cx, cy, w, h] normalizzate

**Classi**:
- 3 classi di bounding box (da verificare nel codice se sono header/data/spanning)

**Post-processing**:
- Merge di bbox per celle con span orizzontale
- Conversione da formato centroid+size a corner format se necessario

## 4. Processo di Inference Completo

### Input
```python
image: torch.Tensor  # Shape: (1, 3, 448, 448)
# RGB, normalizzato con ImageNet mean/std
```

### Flusso
```python
def predict(image):
    # 1. Encoder
    enc_out = encoder(image)  # (1, 28, 28, 256)

    # 2. Tag Transformer
    encoder_out = tag_transformer.encode(enc_out)  # (784, 1, 512)

    # 3. Autoregressive tag generation
    tags = []
    tag_hidden_states = []
    while not end:
        tag, hidden = tag_transformer.decode_step(encoder_out, previous_tags)
        tags.append(tag)
        if tag requires bbox:
            tag_hidden_states.append(hidden)

    # 4. BBox prediction
    bbox_classes, bbox_coords = bbox_decoder(enc_out, tag_hidden_states)

    # 5. Post-process: merge spanning cells
    final_bboxes = merge_spanning_cells(bbox_coords, tags)

    return tags, bbox_classes, final_bboxes
```

### Output
```python
seq: List[int]           # Tag sequence (indices nel word_map)
outputs_class: Tensor    # (num_cells, 3)
outputs_coord: Tensor    # (num_cells, 4) in formato [cx, cy, w, h]
```

## 5. Configurazione Modello

### Parametri Chiave (da tm_config.json)

**Da verificare nel file effettivo, parametri tipici**:
```json
{
  "model": {
    "enc_image_size": 28,
    "hidden_dim": 512,
    "enc_layers": 6,
    "dec_layers": 6,
    "nheads": 8,
    "dropout": 0.1,
    "bbox_classes": 3,
    "tag_attention_dim": 512,
    "tag_embed_dim": 512,
    "tag_decoder_dim": 512,
    "bbox_attention_dim": 512,
    "bbox_embed_dim": 256,
    "bbox_decoder_dim": 512
  },
  "predict": {
    "max_steps": 1000
  }
}
```

## 6. Formato Modelli Disponibili

### Fast Model
- File: `tableformer_fast.safetensors` (145 MB)
- Configurazione: `tm_config.json`
- Probabilmente: meno layers/heads per velocità

### Accurate Model
- File: `tableformer_accurate.safetensors` (213 MB)
- Configurazione: `tm_config.json`
- Probabilmente: più layers/heads per accuratezza

## 7. Preprocessing Immagine

### Richiesto prima dell'inference:

```python
# 1. Resize a 448x448 (con letterboxing per preservare aspect ratio)
# 2. Conversione a RGB tensor
# 3. Normalizzazione ImageNet:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalized = (image / 255.0 - mean) / std
```

## 8. Post-Processing Output

### Da OTSL tags + BBoxes a Struttura Tabella:

1. **Parse OTSL sequence**: Identifica righe, celle, span
2. **Assign bboxes**: Associa bbox a ogni cella
3. **Detect structure**:
   - Numero righe: conta `nl` tags
   - Numero colonne: analizza pattern celle per riga
   - Row spans: identifica `ucel` tags
   - Col spans: identifica `lcel`/`xcel` sequences
4. **Coordinate transformation**:
   - Da [cx, cy, w, h] normalizzato
   - A [x1, y1, x2, y2] assoluto nell'immagine originale

## 9. Differenze con Implementazione Attuale

### Implementazione Attuale (da sostituire)
- Modelli: encoder.onnx, bbox_decoder.onnx, decoder.onnx (separati)
- Output: `[1, 3]` logits, `[1, 4]` boxes (1 sola detection)
- Funzionalità: Table detection, NON structure extraction

### Nuova Implementazione (target)
- Modello: tableformer_fast.onnx / tableformer_accurate.onnx (unificato)
- Output atteso:
  - Tags OTSL: sequenza variabile (fino a max_steps)
  - BBoxes: `[num_cells, 3]` classes + `[num_cells, 4]` coords
- Funzionalità: Full table structure extraction con celle, righe, colonne, spans

## 10. Conversione ONNX - Considerazioni

### Challenges

1. **Autoregressive decoding**:
   - Tag Transformer genera tag uno alla volta
   - Richiede loop in Python, non esportabile direttamente in ONNX
   - Possibili soluzioni:
     a) Esportare encoder separato + decoder stateless per single-step
     b) Usare ONNX Runtime con external loop
     c) Pre-generare multiple steps (beam search) - complesso

2. **Dynamic shapes**:
   - Numero celle varia per tabella
   - Output bbox ha shape dinamica `[num_cells, ...]`
   - ONNX richiede shape massima o dynamic axes

3. **Word map dependency**:
   - Tag decoder richiede word_map per mapping tag↔index
   - Deve essere caricato separatamente in .NET

### Strategia Proposta

**Opzione A: Esportazione Completa (Preferita)**
```python
# Esporta modello completo con fixed max_steps
class TableFormerONNX(nn.Module):
    def forward(self, image):
        # Full prediction loop inside
        tags, classes, coords = self.model.predict(image, max_steps=1000)
        # Pad to fixed size
        return padded_tags, padded_classes, padded_coords
```

**Opzione B: Esportazione Componenti**
```python
# Esporta encoder + decoder stateless
encoder_onnx = export(encoder)
decoder_step_onnx = export(decoder_single_step)
# Loop in C#
```

**Raccomandazione**: Opzione A - Più semplice da usare in .NET

## 11. Requisiti per .NET Implementation

### Nuove Classi Necessarie

1. **OTSLParser**: Parse OTSL tag sequence → struttura logica
2. **TableStructureBuilder**: Da OTSL + bboxes → TableCell list
3. **CoordinateTransformer**: Transform normalized coords → absolute
4. **RowColumnDetector**: Detect row/column structure da celle
5. **SpanResolver**: Resolve row/column spans

### Word Map
```json
{
  "word_map_tag": {
    "<start>": 0,
    "<end>": 1,
    "<pad>": 2,
    "nl": 3,
    "fcel": 4,
    "ecel": 5,
    "lcel": 6,
    "xcel": 7,
    "ucel": 8,
    "ched": 9,
    "rhed": 10,
    "srow": 11
  }
}
```

## 12. Metriche di Valutazione

### Metriche Python (da replicare)
- **TEDS** (Tree Edit Distance Score): Accuratezza struttura
- **mAP** (mean Average Precision): Accuratezza bounding box
- **Cell Detection Rate**: % celle correttamente identificate
- **Row/Column Accuracy**: % righe/colonne corrette

## 13. Prossimi Step

1. ✅ Analisi architettura completata
2. [ ] Scaricare modelli safetensors + config
3. [ ] Creare script conversione PyTorch → ONNX
4. [ ] Testare conversione con input dummy
5. [ ] Validare output shapes e numerical accuracy
6. [ ] Implementare .NET backend
7. [ ] Implementare OTSL parser e post-processing
8. [ ] Testing end-to-end

## 14. Note Implementative Importanti

### Attenzione a:
- **Coordinate format**: OTSL usa [cx, cy, w, h] normalizzato 0-1
- **Image size**: Fixed 448x448, serve letterboxing + inverse transform
- **Tag sequence**: Può essere molto lunga (1000+ token), gestire overflow
- **Empty cells**: OTSL include celle vuote (`ecel`), non saltarle
- **Spanning cells**: Logica complessa di merge, seguire esattamente algoritmo Python

---

**Documento creato**: 12 Ottobre 2025
**Autore**: Analysis dell'implementazione ufficiale Docling TableFormer
**Versione**: 1.0
