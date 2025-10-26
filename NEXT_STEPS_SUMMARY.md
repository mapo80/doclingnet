# DoclingNet - Stato Attuale e Prossimi Passi

**Data:** 2025-10-26
**Obiettivo:** Convertire immagini in Markdown

---

## 🎉 Cosa Funziona ORA

### CLI Funzionante ✅

Il progetto ha già un CLI completamente funzionante che esegue:

```bash
dotnet run --project src/Docling.Cli -- dataset/2305.03393v1-pg9-img.png
```

**Output:**
- ✅ **Layout Detection** (LayoutSdk/Heron): 12 elementi rilevati in ~460ms
- ✅ **OCR Extraction** (EasyOcrNet): Testo estratto da tutti gli elementi
- ✅ **Table Structure** (TableFormer): 46 celle estratte con row/col spans
- ✅ **JSON Output**: Risultati salvati in `dataset/ocr_results/` e `dataset/extracted_tables/`
- ✅ **Text Output**: File `.txt` formattato con testo ordinato per posizione

**File Generati:**
```
dataset/
├── 2305.03393v1-pg9-img_ocr_text.txt                    # Testo formattato
├── ocr_results/
│   └── 2305.03393v1-pg9-img_ocr_results.json            # 25KB - OCR dettagliato
└── extracted_tables/
    ├── 2305.03393v1-pg9-img_table_1.png                 # Immagine tabella estratta
    └── table_structure_results.json                      # 32KB - Struttura tabella
```

### Componenti Integrati ✅

Tutte e tre le librerie AI/ML funzionano correttamente:

| Componente | Status | Performance | Output |
|------------|--------|-------------|--------|
| **LayoutSdk** (Heron ONNX) | ✅ Funzionante | ~460ms | BoundingBox + Label + Confidence |
| **EasyOcrNet** (CRAFT+CRNN) | ✅ Funzionante | ~1-2s/elemento | Text + BBox + Confidence |
| **TableFormer** (TorchSharp) | ✅ Funzionante | ~1s/tabella | Cells con row/col/span |

**Bug Risolti:**
- ✅ SkiaSharp 3.x alpha channel (TableFormer)
- ✅ EasyOcrNet auto-download modelli
- ✅ GitHub release tag corretto (v2025.09.19)

---

## 📋 Architettura Esistente

### Progetto Docling.Pipelines

Il progetto ha già un'architettura pipeline completa ma **non ancora collegata** al CLI:

```
src/Docling.Pipelines/
├── Abstractions/          # IPipelineStage, PipelineContext, etc.
├── Assembly/              # PageAssemblyStage (952 righe)
├── Layout/                # LayoutAnalysisStage
├── Ocr/                   # OcrStage (464 righe)
├── Tables/                # TableStructureInferenceStage
├── Serialization/         # MarkdownSerializationStage
└── Export/                # ImageExportStage
```

**Stage Implementati:**
1. ✅ `PagePreprocessingStage` - Preprocessing immagini
2. ✅ `LayoutAnalysisStage` - Layout detection
3. ✅ `OcrStage` - OCR extraction (complesso, 464 righe)
4. ✅ `TableStructureInferenceStage` - Table structure
5. ✅ `PageAssemblyStage` - Assembly (952 righe!)
6. ✅ `MarkdownSerializationStage` - Export markdown

**Modelli Esistenti:**
- `LayoutItem` (PageReference, BoundingBox, Kind, Polygons)
- `OcrBlockResult` (Page, BoundingBox, Lines)
- `TableStructure` (Rows, Columns, Cells)
- `DoclingDocument` (Items, Pages, Properties)

### Progetto Docling.Core/Export

```
src/Docling.Core/
├── Documents/             # DoclingDocument, DocItem, TableItem, ParagraphItem
├── Geometry/              # BoundingBox, Polygon, Point2D, PageSize
└── Primitives/            # PageReference, ImageRef

src/Docling.Export/
├── Serialization/         # MarkdownDocSerializer
└── Imaging/               # ImageExportArtifact
```

---

## ❓ Cosa Manca per Image→Markdown Completo

### Integrazione Pipeline ⚠️

Il CLI funzionante usa codice "monolitico" in `Program.cs` (681 righe).
La pipeline in `Docling.Pipelines` è separata e usa astrazioni diverse.

**Gap Principale:**
- CLI chiama direttamente LayoutSdk, EasyOcrNet, TableFormer
- Pipeline usa `ILayoutDetectionService`, `IOcrService`, ecc.
- Nessun bridge tra i due

### Export Markdown ⚠️

Il `MarkdownDocSerializer` è implementato ma:
- Non testato end-to-end con un `DoclingDocument` reale
- Nessun esempio di utilizzo con output CLI
- Formato tabelle da verificare

### Reading Order ⚠️

- CLI ordina elementi con semplice `OrderBy(bbox.Y).ThenBy(bbox.X)`
- Python Docling usa algoritmo complesso per multi-colonna
- Da implementare per layout complessi

---

## 🚀 Prossimo Passo Consigliato

### Opzione A: Pipeline Completa (3-5 giorni)

**Obiettivo:** Implementare pipeline end-to-end con architettura pulita

**Passi:**
1. Creare adattatori per LayoutSdk/EasyOcrNet/TableFormer
2. Implementare services (`ILayoutDetectionService`, `IOcrService`)
3. Collegare stage esistenti con implementazioni reali
4. Testare pipeline completa
5. Integrare export markdown

**Pro:** Architettura scalabile, testabile, riutilizzabile
**Contro:** Richiede tempo, debug complesso

### Opzione B: CLI→Markdown Diretto (1 giorno) ⭐ RACCOMANDATO

**Obiettivo:** Aggiungere export markdown al CLI esistente che funziona

**Passi:**
1. ✅ Analizzare output CLI attuale (JSON)
2. Creare `CliToDocumentConverter` che trasforma JSON → `DoclingDocument`
3. Usare `MarkdownDocSerializer` esistente
4. Testare con `dataset/2305.03393v1-pg9-img.png`
5. Raffinare formato markdown

**Pro:** Veloce, pragmatico, risultati immediati
**Contro:** Codice meno elegante, da refactorare dopo

### Opzione C: Hybrid (2 giorni)

**Obiettivo:** Usare codice CLI funzionante dentro pipeline

**Passi:**
1. Creare `SimpleLayoutService` che wrappa chiamate dirette a LayoutSdk
2. Creare `SimpleOcrService` che wrappa EasyOcrNet
3. Creare `SimpleTableService` che wrappa TableFormer
4. Usare stage esistenti con questi servizi semplici
5. Export markdown via `MarkdownSerializationStage`

**Pro:** Bilancio tra velocità e architettura
**Contro:** Duplicazione codice temporanea

---

## 📝 Implementazione Raccomandata: Opzione B

### Codice da Creare (1 file, ~200 righe)

```csharp
// src/Docling.Cli/CliToMarkdownConverter.cs

public class CliToMarkdownConverter
{
    // 1. Legge JSON output CLI
    private OcrResults LoadOcrResults(string jsonPath) { ... }
    private TableResults LoadTableResults(string jsonPath) { ... }

    // 2. Converte in DoclingDocument
    private DoclingDocument BuildDocument(
        LayoutResult layout,
        OcrResults ocr,
        TableResults tables) { ... }

    // 3. Export markdown
    public string ConvertToMarkdown(string imageBasePath) {
        var layout = LoadLayout(imageBasePath);
        var ocr = LoadOcrResults(imageBasePath);
        var tables = LoadTableResults(imageBasePath);

        var document = BuildDocument(layout, ocr, tables);

        var serializer = new MarkdownDocSerializer();
        var result = serializer.Serialize(document);

        return result.Markdown;
    }
}
```

### Test

```bash
# 1. Genera JSON con CLI esistente
dotnet run --project src/Docling.Cli -- dataset/2305.03393v1-pg9-img.png

# 2. Converti JSON → Markdown
dotnet run --project src/Docling.Cli -- --to-markdown dataset/2305.03393v1-pg9-img.png

# Output: dataset/2305.03393v1-pg9-img.md
```

### Output Atteso

```markdown
# Optimized Label Localization for Label Structure Recognition

order to compute the LBD score: Inference timing results for all experiments...

## 5.1 Hyper Parameter Optimization

We have chosen the PubTabNet data set to perform HPO...

### Table 1

| Model | TED | mAP | Runtime |
|-------|-----|-----|---------|
| OTSL enc-6 dec-6 | 0.981 | 0.997 | 0.45s |
| HTML enc-6 dec-6 | 0.981 | 0.995 | 1.53s |
| OTSL enc-4 dec-4 | 0.976 | 0.996 | 0.31s |

## 5.2 Quantitative Results

We picked the model parameter configuration...
```

---

## 📊 Confronto Python Docling vs DoclingNet

| Feature | Python Docling | DoclingNet | Gap |
|---------|---------------|------------|-----|
| Layout Detection | ✅ Heron (PyTorch) | ✅ Heron (ONNX) | Equivalente |
| OCR | ✅ EasyOCR | ✅ EasyOcrNet | Equivalente |
| Table Structure | ✅ TableFormer | ✅ TableFormer (TorchSharp) | Equivalente |
| Reading Order | ✅ Rule-based | ⚠️ Simple sort | Da migliorare |
| Markdown Export | ✅ Completo | ⚠️ Da testare | In progress |
| Pipeline Architecture | ✅ | ✅ (non collegata) | Da integrare |
| Multi-page | ✅ | ❌ | Future |
| Batch Processing | ✅ | ❌ | Future |

---

## 📚 Documentazione Creata

Durante questa sessione sono stati creati:

1. ✅ [DOCLING_IMPLEMENTATION_STATUS.md](DOCLING_IMPLEMENTATION_STATUS.md) (15KB)
   - Analisi completa componenti esistenti
   - Statistiche codice
   - Gap analysis

2. ✅ [DOCLING_IMAGE_TO_MARKDOWN_PLAN.md](DOCLING_IMAGE_TO_MARKDOWN_PLAN.md) (25KB)
   - Analisi workflow Python Docling
   - Piano implementazione settimana per settimana
   - Codice esempio per ogni componente

3. ✅ [NEXT_STEPS_SUMMARY.md](NEXT_STEPS_SUMMARY.md) (questo file)
   - Stato attuale chiaro
   - Opzioni con pro/contro
   - Raccomandazione pragmatica

4. ✅ Modelli dati assembly:
   - `src/Docling.Pipelines/Assembly/Models/OcrCell.cs`
   - `src/Docling.Pipelines/Assembly/Models/LayoutCluster.cs`
   - `src/Docling.Pipelines/Assembly/Models/TableStructureData.cs`

5. ✅ Python Docling scaricato e analizzato:
   - `/tmp/docling-python/` (clone repository)
   - Workflow completo documentato

---

## 🎯 Decisione

**RACCOMANDO: Opzione B - CLI→Markdown Diretto**

### Perché?

1. **Velocità:** 1 giorno vs 3-5 giorni
2. **Pragmatismo:** Usa codice che GIÀ FUNZIONA
3. **Risultati:** Output markdown utilizzabile subito
4. **Iterativo:** Può essere refactorato dopo
5. **Test:** Immagine di test già disponibile e funzionante

### Prossima Azione Concreta

**Creare:** `src/Docling.Cli/CliToMarkdownConverter.cs`

**Funzione:**
```csharp
public string ConvertCliOutputToMarkdown(string imageBasePath)
{
    // 1. Legge JSON da dataset/ocr_results/ e dataset/extracted_tables/
    // 2. Costruisce DoclingDocument
    // 3. Usa MarkdownDocSerializer esistente
    // 4. Restituisce string markdown
}
```

**Test:**
```bash
# Genera output
dotnet run --project src/Docling.Cli -- dataset/2305.03393v1-pg9-img.png

# Converti in markdown
# (da implementare come flag --to-markdown)
```

---

## 📞 Domande Aperte

1. **Formato tabelle**: Markdown tables o HTML `<table>`?
2. **Immagini**: Come gestire riferimenti a figure estratte?
3. **Metadati**: Includere confidence scores nei commenti?
4. **Encoding**: UTF-8, gestione caratteri speciali?

---

## ✅ Conclusione

Abbiamo:
- ✅ Tre librerie AI/ML funzionanti e integrate
- ✅ CLI completo che produce output JSON
- ✅ Architettura pipeline ben progettata (non collegata)
- ✅ Modelli documento completi (Docling.Core)
- ✅ Serializer markdown implementato
- ✅ Documentazione esaustiva

**Manca solo:**
Un bridge di ~200 righe che trasforma JSON → DoclingDocument → Markdown

**Tempo stimato:** 1 giorno lavorativo

**Risultato finale:** Comando che converte immagine in markdown funzionante!

---

**End of Document**
